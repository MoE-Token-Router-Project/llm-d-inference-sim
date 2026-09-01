# MoE trace all-prompts benchmark

This folder benchmarks the trace-replay runtime by submitting every prompt in one `.moetrace` file at once and collecting client-observed latency and throughput metrics.

The benchmark is intentionally an all-at-once burst. It creates one goroutine per trace prompt, waits until every goroutine is ready, then releases all of them through one start barrier. The simulator still controls how many requests execute concurrently through `--max-num-seqs`; the remaining requests wait in the simulator queue. Queueing therefore contributes to TTFT and end-to-end latency, as it does in a real overloaded serving experiment.

## Prerequisites

Build the simulator and start it with the same `.moetrace` file that will be passed to the benchmark. For Qwen1.5-MoE-A2.7B, an example is:

```bash
make build

./bin/llm-d-inference-sim \
  --model Qwen/Qwen1.5-MoE-A2.7B \
  --enable-moe \
  --moe-trace-path /path/to/instructcoder_2000_both.moetrace \
  --moe-expert-parallel-size 8 \
  --moe-num-experts 60 \
  --moe-physical-expert-slots 80 \
  --moe-top-k 4 \
  --moe-num-layers 24 \
  --moe-router heuristic \
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

## Run

From the repository root:

```bash
scripts/moe_trace_all_prompts_benchmark/run.sh \
  --trace /path/to/instructcoder_2000_both.moetrace \
  --base-url http://127.0.0.1:8000 \
  --label heuristic \
  --output-dir results/instructcoder-heuristic
```

`--model` is optional and defaults to the model stored in the trace. `--request-timeout 0` is the default and disables the client-wide request timeout, which is useful for queue-heavy experiments. Use `--progress-every 0` to disable completion progress messages.

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

For a fair Split/Concentrate/Heuristic comparison, restart the simulator from a fresh process for each routing policy. Trace replay updates the vLLM-style EPLB load window and can trigger D'Hondt replacement, so reusing a server after one complete benchmark would make the next policy start from evolved placement state rather than the same initial state.

Keep the trace, hardware parameters, `--max-num-seqs`, queue size, and all non-router flags fixed. Change only `--moe-router` and use a separate output directory for each run.

## Tests

The benchmark package has unit tests for SSE token classification, percentile calculation, and the distinction between per-request TPOT and pooled streaming ITL:

```bash
go test ./scripts/moe_trace_all_prompts_benchmark
```
