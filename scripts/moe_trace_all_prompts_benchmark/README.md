# MoE trace all-prompts benchmark

This folder benchmarks the **HTTP trace-replay runtime** by submitting every prompt in one `.moetrace` file at once and collecting client-observed latency and throughput metrics.

> For Split/Concentrate/Heuristic **GPU-model cross-validation**, use `scripts/moe_trace_virtual_benchmark` instead. The HTTP benchmark includes real host CPU time, goroutine scheduling, queueing, HTTP/SSE work, and sleeps. Those are useful simulator-runtime measurements, but they are not pure GPU predictions. The virtual benchmark advances only modeled GPU time and uses one shared vLLM-style serving forward across active sequences.

The benchmark is intentionally an all-at-once burst. It creates one goroutine per trace prompt, waits until every goroutine is ready, then releases all of them through one start barrier. The simulator still controls how many requests execute concurrently through `--max-num-seqs`; the remaining requests wait in the simulator queue. Queueing therefore contributes to TTFT and end-to-end latency, as it does in a real overloaded serving experiment.

## End-to-end pipeline

This benchmark is a three-step workflow:

1. **Prepare a `.moetrace` file.** The benchmark does not download or generate routing traces itself. Use an existing `.moetrace`, or convert trace JSON into the binary format with `moe-trace-tool`.
2. **Start `llm-d-inference-sim` with that trace.** The simulator must load the same trace file before the benchmark starts. Configure the MoE dimensions, router, placement, and distributed-routing mode you want to measure.
3. **Run the benchmark client.** `run.sh` opens the same trace, creates one HTTP request for every prompt, and sends those requests to the already-running simulator.

A complete example is below.

### 1. Prepare the trace

If you already have a `.moetrace` file, you can use it directly. Otherwise, build the trace conversion tool:

```bash
go build -o bin/moe-trace-tool ./cmd/moe-trace-tool
```

Then convert the JSON emitted by `trace_qwen_moe.py`:

```bash
bin/moe-trace-tool convert \
  --input instructcoder_2000_both.json \
  --output instructcoder_2000_both.moetrace
```

Validate and inspect the resulting trace before starting the simulator:

```bash
bin/moe-trace-tool validate --input instructcoder_2000_both.moetrace
bin/moe-trace-tool inspect --input instructcoder_2000_both.moetrace
```

The JSON-generation step is external to this benchmark directory; this repository's trace tool converts and validates the generated JSON. See `docs/moe-trace-tool.md` for the expected format.

### 2. Build and start the simulator

Build the current branch:

```bash
make build
```

For a distributed Heuristic replay of the Qwen1.5-MoE-A2.7B trace:

```bash
ulimit -n 8192

./bin/llm-d-inference-sim \
  --model Qwen/Qwen1.5-MoE-A2.7B \
  --port 8000 \
  --enable-moe \
  --moe-trace-path /path/to/instructcoder_2000_both.moetrace \
  --moe-fixed-placement-path /path/to/fixed_dhondt_placement.json \
  --moe-expert-parallel-size 8 \
  --moe-num-experts 60 \
  --moe-physical-expert-slots 80 \
  --moe-top-k 4 \
  --moe-num-layers 24 \
  --moe-router heuristic \
  --use-distributed-routing \
  --max-model-len 16384 \
  --max-num-seqs 32 \
  --max-waiting-queue-length 2000 \
  --time-to-first-token 0 \
  --inter-token-latency 0
```

Keep the simulator running in its own terminal. The fixed-placement file is optional, but use it when you want Split, Concentrate, and Heuristic to see exactly the same physical expert replicas.

Before starting the benchmark client, verify that the server is reachable and has the expected routing mode:

```bash
curl -s http://127.0.0.1:8000/health
curl -s http://127.0.0.1:8000/admin/config | jq '{
  router: ."moe-router",
  distributed: ."use-distributed-routing"
}'
```

For the command above, the second command should report `"router": "heuristic"` and `"distributed": true`.

### 3. Send all trace prompts

In a second terminal:

```bash
./scripts/moe_trace_all_prompts_benchmark/run.sh \
  --trace /path/to/instructcoder_2000_both.moetrace \
  --base-url http://127.0.0.1:8000 \
  --label heuristic \
  --output-dir results/instructcoder-heuristic \
  --max-http-connections 512
```

The benchmark first checks `/health` and `/admin/config`, then submits every prompt in the trace. The client does not choose the routing mode; routing is configured entirely on the simulator, and the server configuration recorded in the result shows which mode was used. Each HTTP request carries the corresponding `trace_prompt_id`, so the simulator replays that prompt's recorded token IDs and per-layer expert choices rather than generating a synthetic expert mapping.

For large traces, both the client and simulator processes need enough file descriptors for the all-at-once HTTP burst. The client defaults to `--max-http-connections 512`, which limits simultaneous TCP connections while still releasing all logical benchmark requests together. Requests above the cap wait inside the Go HTTP transport, and that waiting time is included in request latency and TTFT. The `run.sh` wrapper raises the client's soft file-descriptor limit when possible; because the simulator is a separate process, set an adequate `ulimit -n` in the simulator shell as well.

## Simulator configuration reference

Build the simulator and start it with the same `.moetrace` file that will be passed to the benchmark. For a controlled Qwen1.5-MoE-A2.7B comparison, also pass the same fixed D'Hondt placement used by real vLLM:

```bash
make build

./bin/llm-d-inference-sim \
  --model Qwen/Qwen1.5-MoE-A2.7B \
  --enable-moe \
  --moe-trace-path /path/to/instructcoder_2000_both.moetrace \
  --moe-fixed-placement-path /path/to/fixed_dhondt_placement.json \
  --moe-expert-parallel-size 8 \
  --moe-num-experts 60 \
  --moe-physical-expert-slots 80 \
  --moe-top-k 4 \
  --moe-num-layers 24 \
  --moe-router heuristic \
  --use-distributed-routing \
  --max-model-len 16384 \
  --max-num-seqs 32 \
  --max-waiting-queue-length 2000 \
  --time-to-first-token 0 \
  --inter-token-latency 0
```

Set `--max-model-len` high enough for the largest `input_tokens + decode_tokens` in the trace. The benchmark checks this before launching requests and fails early if it is too small.

For an all-at-once workload, the server should normally satisfy:

```text
max-waiting-queue-length >= num_trace_prompts - max-num-seqs
```

If it does not, the benchmark prints a warning and still runs so queue-capacity failures are visible in the result. Large all-at-once runs may also require a higher process file-descriptor limit, for example `ulimit -n`, because the client can have one HTTP request in flight per trace prompt.

## Benchmark client reference

From the repository root:

```bash
scripts/moe_trace_all_prompts_benchmark/run.sh \
  --trace /path/to/instructcoder_2000_both.moetrace \
  --base-url http://127.0.0.1:8000 \
  --label heuristic \
  --output-dir results/instructcoder-heuristic \
  --max-http-connections 512
```

`--model` is optional and defaults to the model stored in the trace. `--request-timeout 0` is the default and disables the client-wide request timeout, which is useful for queue-heavy experiments. `--max-http-connections` defaults to 512 and bounds concurrent TCP connections to the simulator. Use `--progress-every 0` to disable completion progress messages.

The harness performs `/health` and `/admin/config` preflight checks before the timed benchmark. It verifies the model, expert count, top-k, MoE-layer count, and maximum context length. Trace opening, validation, preflight, and result-directory setup are outside the measured wall-time window.

## Output

The output directory contains:

- `summary.json`: machine-readable aggregate metrics plus the trace and server configuration.
- `summary.txt`: a flat text summary convenient for pasting into experiment notes.
- `requests.csv`: one row per trace prompt with status, token counts, request latency, TTFT, TPOT, phase timing, and launch offset.

The aggregate metrics mirror the metrics used by the existing GPU benchmark write-up:

- requested, successful, and failed requests;
- prompt, output, and total token counts;
- wall time, requests/s, output tokens/s, and total tokens/s;
- request latency mean/median/p95/p99;
- TTFT mean/median/p95/p99;
- TPOT mean/median/p95/p99;
- streaming inter-token latency mean/median/p95/p99;
- prefill, decode, and E2E latency distributions;
- output-length distribution.

The summary also reports `launch_start_offset_milliseconds`. This is not a model metric; it measures how far apart the client request goroutines actually entered `http.Client.Do` after the shared start barrier. It is useful for detecting client-side scheduling skew in very large bursts.

## Metric definitions

**Wall time** starts when the shared all-prompts barrier is released and ends when the final request finishes. Setup and trace parsing are excluded.

**Request latency / E2E** is measured separately for each prompt from the instant that prompt starts its HTTP request until the `[DONE]` SSE marker is received. It includes simulator queueing.

**TTFT / prefill** is measured from the HTTP request start until the first output-token SSE event. The initial assistant-role chunk is not counted as a token. Trace replay can contain token IDs whose synthesized display string is empty; those still count as output-token events because the simulator performed decode work for them.

**TPOT** is computed per successful request with at least two output tokens:

```text
TPOT_request = (last_token_time - first_token_time) / (output_tokens - 1)
```

The benchmark then reports the distribution of these per-request TPOT values.

**Streaming ITL** is different from TPOT. Every observed gap between two consecutive output-token events is pooled across all successful requests, and the distribution is computed over those individual gaps. This preserves the distinction between per-request TPOT and streaming inter-token latency.

**Decode time** is the elapsed time from the first output token to the last output token. **Prefill time** is the client-observed TTFT. These phase names are intended for comparison with the existing benchmark tables; they are wall-clock client observations, not isolated GPU-kernel timings.

Token throughput uses only successful requests. A response is marked failed if the HTTP request fails, the stream does not end in `[DONE]`, the server returns a stream error, or the number of output-token events differs from the trace's recorded `decode_tokens` for that prompt.

## Comparing routing policies

For the controlled real-GPU comparison, keep the trace, fixed placement, hardware parameters, token budget, sequence concurrency, and all non-router settings identical and change only the router. Use the virtual benchmark:

```bash
go run ./scripts/moe_trace_virtual_benchmark \
  --trace /path/to/instructcoder_2000_both.moetrace \
  --fixed-placement /path/to/fixed_dhondt_placement.json \
  --gpu a100 \
  --token-budget 1024 \
  --max-num-seqs 32
```

The HTTP benchmark can still be run once per router to study simulator/API behavior. Restart the simulator from a fresh process for each run. If a fixed placement is not supplied, online EPLB can evolve placement state, so comparisons no longer isolate replica routing alone.

## Tests

The benchmark package has unit tests for SSE token classification, percentile calculation, and the distinction between per-request TPOT and pooled streaming ITL:

```bash
go test ./scripts/moe_trace_all_prompts_benchmark
```

The virtual serving scheduler and replica-routing fidelity are tested under `pkg/simulator`.
