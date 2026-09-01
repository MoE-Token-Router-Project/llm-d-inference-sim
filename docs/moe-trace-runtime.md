# MoE trace replay

The simulator can replay logical expert routing recorded in a `.moetrace` file. Trace replay is optional: requests without `trace_prompt_id` continue to use the existing synthetic MoE workload even when a trace is loaded.

## Start the simulator

Pass the trace file with the startup-only `--moe-trace-path` option together with `--enable-moe` and MoE dimensions that match the trace:

```bash
./bin/llm-d-inference-sim \
  --model Qwen/Qwen1.5-MoE-A2.7B \
  --enable-moe \
  --moe-trace-path /data/instructcoder_2000_both.moetrace \
  --moe-expert-parallel-size 8 \
  --moe-num-experts 60 \
  --moe-physical-expert-slots 80 \
  --moe-top-k 4 \
  --moe-num-layers 24
```

`--moe-trace-path` is consumed at process startup and is not an admin-configurable field. The simulator loads and validates every prompt before the HTTP server starts accepting requests. Request processing does not read the `.moetrace` file.

Startup fails if the trace model, logical expert count, top-k, or number of sparse layers does not match the simulator configuration. The physical expert slot count must also be divisible by the expert-parallel size.

## Replay a traced prompt

Add `trace_prompt_id` to a chat-completions request:

```json
{
  "model": "Qwen/Qwen1.5-MoE-A2.7B",
  "messages": [
    {"role": "user", "content": "ignored during trace replay"}
  ],
  "trace_prompt_id": 17
}
```

The supplied `messages` still satisfy the chat API shape, but they do not define the simulated model execution. For a trace request the simulator uses the selected trace prompt as the visible prompt, uses the trace's exact input token IDs without tokenizing the supplied messages, returns the recorded decode token IDs, and uses the recorded per-layer expert choices for MoE latency and EPLB load.

The first returned token is produced by the recorded prefill. If the recorded output tokens are `A B C`, the route recorded for `A` models the decode forward between `A` and `B`, and the route recorded for `B` models the forward between `B` and `C`. The route recorded for `C` is not charged unless another output token is replayed.

`max_completion_tokens` may replay a prefix of the recorded completion, but it cannot exceed the number of decode tokens in the trace. The v1 `.moetrace` format stores exact decode token IDs and the complete generated text, not a decoded string for every token. Token timing and expert routing remain exact for a truncated replay, while textual chunk boundaries are only a deterministic presentation of the recorded generated text.

## Interaction with normal simulation

The behavior is selected per request:

| Trace loaded | `trace_prompt_id` | Behavior |
| --- | --- | --- |
| No | No | Existing simulator behavior |
| Yes | No | Existing synthetic MoE behavior |
| Yes | Yes | Trace replay |
| No | Yes | Request rejected |

Trace replay reuses the existing physical expert placement, split/concentrate/heuristic replica routing, GPU cost model, communication model, D'Hondt EPLB policy, load-window timing, and migration model. Only the source of logical expert demand changes from the synthetic power-law distribution to the recorded per-layer routes.

For partially cached prompts, trace replay aggregates only the uncached prefill positions. During decode, concurrently active trace requests contribute their recorded current-token routes. Other running requests continue to contribute the existing synthetic load approximation.

The request scheduler is unchanged. Trace replay improves the expert-load input but does not turn the simulator into a cycle-accurate vLLM continuous-batching scheduler.

## Initial restrictions

Trace replay currently applies only to ordinary text `/v1/chat/completions` requests. A request using `trace_prompt_id` is rejected when combined with multiple choices (`n > 1`), tools, remote prefill/decode, `ignore_eos`, logprobs, multimodal content, LoRA models, echo mode, or image emission.
