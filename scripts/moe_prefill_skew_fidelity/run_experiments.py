#!/usr/bin/env python3
import argparse, csv, json, socket, statistics, subprocess, time
from pathlib import Path
from urllib import request

ROOT = Path(__file__).resolve().parent
REPO_ROOT = ROOT.parents[1]
SIM = REPO_ROOT / 'bin' / 'llm-d-inference-sim'
ROUTERS = ['split', 'concentrate', 'heuristic']
H100 = dict(flops='990e12', bandwidth='3.35e12')


def free_port():
    s = socket.socket()
    s.bind(('127.0.0.1', 0))
    port = s.getsockname()[1]
    s.close()
    return port


def wait_ready(port, timeout=5.0):
    deadline = time.time() + timeout
    while time.time() < deadline:
        try:
            with request.urlopen(f'http://127.0.0.1:{port}/health', timeout=0.2):
                return
        except Exception:
            time.sleep(0.02)
    raise RuntimeError('simulator did not become ready')


def payload(prompt_tokens, output_tokens):
    return json.dumps({
        'model': 'dummy-model',
        'messages': [{'role': 'user', 'content': 'x ' * prompt_tokens}],
        'max_tokens': output_tokens,
        'stream': True,
    }).encode()


def measure_ttft(port, prompt_tokens, output_tokens=1):
    req = request.Request(
        f'http://127.0.0.1:{port}/v1/chat/completions',
        data=payload(prompt_tokens, output_tokens),
        headers={'Content-Type': 'application/json', 'Accept': 'text/event-stream'},
    )
    start = time.perf_counter()
    first = None
    with request.urlopen(req, timeout=30) as resp:
        for raw in resp:
            line = raw.decode().strip()
            if not line.startswith('data:'):
                continue
            data = line[5:].strip()
            if data == '[DONE]':
                break
            chunk = json.loads(data)
            for choice in chunk.get('choices', []):
                delta = choice.get('delta') or {}
                if 'content' in delta and first is None:
                    first = time.perf_counter()
    if first is None:
        raise RuntimeError('no token chunk received')
    return (first - start) * 1000.0


def modeled_ttft_ms(port):
    text = request.urlopen(f'http://127.0.0.1:{port}/metrics', timeout=2).read().decode()
    prefix = 'vllm:time_to_first_token_seconds_sum{'
    vals = [float(line.rsplit(' ', 1)[1]) for line in text.splitlines() if line.startswith(prefix)]
    if len(vals) != 1:
        raise RuntimeError(f'expected one TTFT sum, got {vals}')
    return vals[0] * 1000.0


def run_one(router, alpha, prompt_tokens, repeat):
    if not SIM.exists():
        raise RuntimeError(f'simulator binary not found at {SIM}; run make build first')
    port = free_port()
    log = ROOT / f'server_{router}_a{alpha}_n{prompt_tokens}_r{repeat}.log'
    cmd = [
        str(SIM), '--model', 'dummy-model', '--port', str(port), '--mode', 'random',
        '--max-model-len', '20000', '--enable-moe', '--moe-router', router,
        '--moe-expert-popularity-alpha', str(alpha), '--moe-expert-parallel-size', '8',
        '--moe-num-experts', '60', '--moe-physical-expert-slots', '80', '--moe-top-k', '4',
        '--moe-num-layers', '24', '--moe-hidden-size', '2048', '--moe-intermediate-size', '1408',
        '--moe-bytes-per-element', '2', '--moe-gpu-flops', H100['flops'],
        '--moe-gpu-memory-bandwidth', H100['bandwidth'], '--moe-interconnect-bandwidth', '400e9',
        '--moe-interconnect-latency', '5us', '--time-to-first-token', '0',
        '--inter-token-latency', '0', '--seed', '42',
    ]
    with log.open('wb') as lf:
        proc = subprocess.Popen(cmd, stdout=lf, stderr=subprocess.STDOUT)
        try:
            wait_ready(port)
            value = measure_ttft(port, prompt_tokens, 1)
            modeled = modeled_ttft_ms(port)
        finally:
            proc.terminate()
            try:
                proc.wait(timeout=2)
            except subprocess.TimeoutExpired:
                proc.kill()
                proc.wait()
    return {
        'router': router,
        'alpha': alpha,
        'prompt_tokens': prompt_tokens,
        'repeat': repeat,
        'ttft_ms': value,
        'modeled_ttft_ms': modeled,
    }


def write_rows(name, rows):
    path = ROOT / name
    with path.open('w', newline='') as f:
        w = csv.DictWriter(f, fieldnames=list(rows[0]))
        w.writeheader()
        w.writerows(rows)
    return path


def summarize(rows, keys):
    groups = {}
    for row in rows:
        groups.setdefault(tuple(row[k] for k in keys), []).append(row['ttft_ms'])
    out = []
    for key, vals in groups.items():
        rec = {k: v for k, v in zip(keys, key)}
        rec['n'] = len(vals)
        rec['median_ttft_ms'] = statistics.median(vals)
        rec['min_ttft_ms'] = min(vals)
        rec['max_ttft_ms'] = max(vals)
        modelvals = [r['modeled_ttft_ms'] for r in rows if tuple(r[k] for k in keys) == key]
        rec['modeled_ttft_ms'] = statistics.median(modelvals)
        out.append(rec)
    return out


def run_skew(repeats):
    rows = []
    for alpha in [0, 0.25, 0.5, 0.75, 1.0, 1.25, 1.5, 2.0]:
        for router in ROUTERS:
            for r in range(repeats):
                row = run_one(router, alpha, 8192, r)
                rows.append(row)
                print('skew', row, flush=True)
    write_rows('skew_raw.csv', rows)
    write_rows('skew_summary.csv', summarize(rows, ['alpha', 'router']))


def run_crossover(repeats):
    rows = []
    for n in [512, 1024, 2048, 4096, 8192, 16384]:
        for router in ROUTERS:
            for r in range(repeats):
                row = run_one(router, 1.0, n, r)
                rows.append(row)
                print('crossover', row, flush=True)
    write_rows('crossover_raw.csv', rows)
    write_rows('crossover_summary.csv', summarize(rows, ['prompt_tokens', 'router']))


def run_endpoints(repeats):
    rows = []
    for n in [4096, 8192, 16384]:
        for alpha in [0, 2.0]:
            for router in ROUTERS:
                for r in range(repeats):
                    row = run_one(router, alpha, n, r)
                    rows.append(row)
                    print('endpoint', row, flush=True)
    write_rows('endpoints_raw.csv', rows)
    write_rows('endpoints_summary.csv', summarize(rows, ['prompt_tokens', 'alpha', 'router']))


def main():
    p = argparse.ArgumentParser()
    p.add_argument('--experiment', choices=['skew', 'crossover', 'endpoints', 'all'], default='all')
    p.add_argument('--repeats', type=int, default=3)
    a = p.parse_args()
    if a.experiment in ('skew', 'all'):
        run_skew(a.repeats)
    if a.experiment in ('crossover', 'all'):
        run_crossover(a.repeats)
    if a.experiment in ('endpoints', 'all'):
        run_endpoints(a.repeats)


if __name__ == '__main__':
    main()
