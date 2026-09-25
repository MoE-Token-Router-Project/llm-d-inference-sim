# moe_simulated_expert_mapping_benchmark

This benchmark sends normal streaming chat-completion requests to the simulator. It reads prompt text from a local JSONL dataset and sends every selected prompt at the same time. The request body contains `model`, `messages`, `max_completion_tokens`, `stream`, and `stream_options`; it does not contain `trace_prompt_id` and does not require `--moe-trace-path` on the simulator.

The benchmark keeps the measurement definitions used by `scripts/moe_trace_all_prompts_benchmark`. It reports request throughput, output-token throughput, total-token throughput, request latency, TTFT, TPOT, streaming ITL, prefill time, decode time, E2E time, output length, and request launch offsets. Prompt and completion token counts come from the streaming usage chunk requested with `stream_options.include_usage=true`.

## End-to-end pipeline

The benchmark is a three-process-step workflow:

1. **Prepare a local JSONL dataset.** Download prompt rows from Hugging Face with `download_dataset.py`, or provide an existing file in the benchmark JSONL format.
2. **Start `llm-d-inference-sim`.** The benchmark does not start the simulator. Start the server first with the MoE/router configuration you want to measure.
3. **Run the benchmark client.** `run.sh` reads the local dataset and sends the selected prompts to the already-running simulator over `/v1/chat/completions`.

A complete example is shown below.

### 1. Download/prepare the dataset

Install the Hugging Face dataset dependency once:

```bash
python3 -m pip install datasets
```

For GSM8K:

```bash
python3 scripts/moe_simulated_expert_mapping_benchmark/download_dataset.py \
  --dataset openai/gsm8k \
  --dataset-config main \
  --split train \
  --prompt-field question \
  --limit 2000 \
  --max-completion-tokens 128 \
  --output datasets/gsm8k_2000.jsonl
```

This produces the local file consumed by the benchmark client. If you already have a normalized JSONL dataset, this download step can be skipped.

### 2. Build and start the simulator

Build the current branch:

```bash
make build
```

For the distributed heuristic router used in the examples in this directory:

```bash
ulimit -n 8192

./bin/llm-d-inference-sim \
  --model Qwen/Qwen1.5-MoE-A2.7B \
  --port 8000 \
  --enable-moe \
  --moe-expert-parallel-size 8 \
  --moe-num-experts 60 \
  --moe-physical-expert-slots 80 \
  --moe-top-k 4 \
  --moe-num-layers 24 \
  --moe-router heuristic \
  --use-distributed-routing \
  --moe-hidden-size 2048 \
  --moe-intermediate-size 1408 \
  --moe-bytes-per-element 2 \
  --moe-gpu-flops 312e12 \
  --moe-gpu-memory-bandwidth 2e12 \
  --moe-interconnect-bandwidth 400e9 \
  --moe-interconnect-latency 5us \
  --max-model-len 16384 \
  --max-num-seqs 32 \
  --max-waiting-queue-length 2000 \
  --time-to-first-token 0 \
  --inter-token-latency 0
```

Keep this process running in its own terminal. Before starting the benchmark, verify that the client can reach the expected simulator and that distributed routing is enabled:

```bash
curl -s http://127.0.0.1:8000/health
curl -s http://127.0.0.1:8000/admin/config | jq '{
  router: ."moe-router",
  distributed: ."use-distributed-routing"
}'
```

For the command above, the second command should report `"router": "heuristic"` and `"distributed": true`.

### 3. Send the benchmark requests

In a second terminal:

```bash
./scripts/moe_simulated_expert_mapping_benchmark/run.sh \
  --dataset datasets/gsm8k_2000.jsonl \
  --base-url http://127.0.0.1:8000 \
  --output-dir results/simulated-expert-mapping-heuristic \
  --label heuristic \
  --max-http-connections 512
```

The benchmark performs `/health` and `/admin/config` preflight checks, then releases one client goroutine per selected prompt and sends streaming chat-completion requests to the simulator. The client does not choose the routing mode; routing is configured entirely on the simulator, and the server configuration recorded in the result shows which mode was used. The simulator admits up to `--max-num-seqs` active requests and queues the rest up to `--max-waiting-queue-length`.

The client defaults to `--max-http-connections 512` to avoid opening thousands of TCP connections at once. All benchmark goroutines are still released together; requests above the connection cap wait inside the Go HTTP transport, and that waiting time is included in request latency and TTFT. Override the cap when needed. On macOS, if you still see `connection reset by peer` errors, first confirm the setup with a smaller burst such as:

```bash
./scripts/moe_simulated_expert_mapping_benchmark/run.sh \
  --dataset datasets/gsm8k_2000.jsonl \
  --limit 1000 \
  --base-url http://127.0.0.1:8000 \
  --output-dir results/simulated-expert-mapping-heuristic-1000 \
  --label heuristic \
  --max-http-connections 512
```

The wrapper raises its own open-file limit when possible; the simulator is a separate process, so its shell must have an adequate `ulimit -n` as well.

## Dataset format

The input is newline-delimited JSON. Each non-empty line must contain a `prompt` string and can set a row-specific `max_completion_tokens` value:

```json
{"prompt":"Write a function that reverses a linked list.","max_completion_tokens":128}
{"prompt":"Explain why a binary search is O(log n)."}
```

Rows that omit `max_completion_tokens` use the command-line `--max-completion-tokens` value, which defaults to 128. Downloaded datasets belong under the repository-level `datasets/` directory. That directory is already ignored by git.

## Dataset download reference

The helper uses the Hugging Face `datasets` Python package in streaming mode, so it can select a small benchmark subset without first saving the full source dataset locally.

```bash
python3 -m pip install datasets

python3 scripts/moe_simulated_expert_mapping_benchmark/download_dataset.py \
  --dataset YOUR_HUGGINGFACE_DATASET \
  --split train \
  --prompt-field YOUR_PROMPT_FIELD \
  --limit 2000 \
  --max-completion-tokens 128 \
  --output datasets/moe_simulated_expert_mapping_benchmark_2000.jsonl
```

The prompt field can use dot notation for nested records. If the source dataset already contains a positive integer output-token limit, pass it with `--max-completion-tokens-field FIELD`; otherwise the helper writes the constant from `--max-completion-tokens`.

## Benchmark client reference

Start the simulator with MoE simulation enabled and with the router and hardware settings you want to measure. No MoE trace file is required for this benchmark. The simulator should have enough `max-num-seqs` and `max-waiting-queue-length` capacity for the number of requests you plan to launch.

```bash
./scripts/moe_simulated_expert_mapping_benchmark/run.sh \
  --dataset datasets/moe_simulated_expert_mapping_benchmark_2000.jsonl \
  --base-url http://127.0.0.1:8000 \
  --output-dir results/simulated-expert-mapping-heuristic \
  --label heuristic \
  --max-http-connections 512
```

The benchmark uses the simulator model from `/admin/config` when `--model` is omitted. Use `--limit N` to run only the first N dataset rows, `--request-timeout DURATION` to set a per-request HTTP timeout, and `--max-http-connections N` to bound concurrent TCP connections to the simulator. The default connection cap is 512. The `run.sh` wrapper also tries to raise the open-file limit for the all-at-once request burst.

## Launch behavior

Every selected prompt gets its own goroutine. All goroutines wait on one start gate, and the gate is released once after setup. Simulator admission still follows `--max-num-seqs`, so requests beyond active capacity wait in the simulator queue when `--max-waiting-queue-length` has room. Queue time is part of TTFT and E2E latency, matching the trace benchmark.

## Metrics

Wall time starts when the shared request gate opens and ends when the final request finishes. Request throughput is successful requests divided by wall time. Output-token throughput and total-token throughput use token counts from successful requests only.

TTFT is the time from each HTTP request start to its first output-token SSE event. Prefill time uses the same measurement. Decode time is the time from the first output-token event to the last output-token event. Per-request TPOT is decode time divided by `output_tokens - 1` for requests that produce at least two output tokens. Streaming ITL pools every measured gap between consecutive output-token events across successful requests. E2E latency is the time from request start through the `[DONE]` event.

The benchmark requires the final usage chunk for each successful stream because regular requests do not have trace metadata that supplies prompt-token counts. It also checks that the completion-token count in the usage chunk matches the number of output-token SSE events used for TPOT and ITL measurements.

## Output files

Each run writes three files under `--output-dir`: `summary.json` for structured aggregate results, `summary.txt` for the printed report, and `requests.csv` for per-request measurements. The aggregate distributions report count, mean, median, p95, p99, minimum, and maximum values, matching the existing trace benchmark.

The CSV contains `prompt_id`, success status, HTTP status, error text, prompt tokens, requested `max_completion_tokens`, actual output tokens, request latency, TTFT, TPOT, prefill time, decode time, E2E time, and launch-start offset.

## Tests

Run the package tests with:

```bash
go test ./scripts/moe_simulated_expert_mapping_benchmark
```
