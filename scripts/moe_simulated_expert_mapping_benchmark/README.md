# moe_simulated_expert_mapping_benchmark

This benchmark sends normal streaming chat-completion requests to the simulator. It reads prompt text from a local JSONL dataset and sends every selected prompt at the same time. The request body contains `model`, `messages`, `max_completion_tokens`, `stream`, and `stream_options`; it does not contain `trace_prompt_id` and does not require `--moe-trace-path` on the simulator.

The benchmark keeps the measurement definitions used by `scripts/moe_trace_all_prompts_benchmark`. It reports request throughput, output-token throughput, total-token throughput, request latency, TTFT, TPOT, streaming ITL, prefill time, decode time, E2E time, output length, and request launch offsets. Prompt and completion token counts come from the streaming usage chunk requested with `stream_options.include_usage=true`.

## Dataset format

The input is newline-delimited JSON. Each non-empty line must contain a `prompt` string and can set a row-specific `max_completion_tokens` value:

```json
{"prompt":"Write a function that reverses a linked list.","max_completion_tokens":128}
{"prompt":"Explain why a binary search is O(log n)."}
```

Rows that omit `max_completion_tokens` use the command-line `--max-completion-tokens` value, which defaults to 128. Downloaded datasets belong under the repository-level `datasets/` directory. That directory is already ignored by git.

## Download from Hugging Face

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

## Run the benchmark

Start the simulator with MoE simulation enabled and with the router and hardware settings you want to measure. No MoE trace file is required for this benchmark. The simulator should have enough `max-num-seqs` and `max-waiting-queue-length` capacity for the number of requests you plan to launch.

```bash
./scripts/moe_simulated_expert_mapping_benchmark/run.sh \
  --dataset datasets/moe_simulated_expert_mapping_benchmark_2000.jsonl \
  --base-url http://127.0.0.1:8000 \
  --output-dir results/simulated-expert-mapping-heuristic \
  --label heuristic
```

The benchmark uses the simulator model from `/admin/config` when `--model` is omitted. Use `--limit N` to run only the first N dataset rows and `--request-timeout DURATION` to set a per-request HTTP timeout. The `run.sh` wrapper also tries to raise the open-file limit for the all-at-once request burst.

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
