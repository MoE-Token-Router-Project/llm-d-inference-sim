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

## Controlled fixed-placement comparison

The real vLLM routing experiments can freeze the D'Hondt placement so every routing policy sees exactly the same physical replicas. Use the same JSON mapping with:

```bash
--moe-fixed-placement-path /data/fixed_dhondt_placement.json
```

This option requires `--moe-trace-path`. The file uses the same `physical_to_logical` format as `VLLM_FIXED_EXPERT_PLACEMENT` in the patched vLLM runtime. The simulator validates the layer count, physical slot count, logical expert IDs, one-copy-per-GPU invariant, and replica order before accepting requests.

When a fixed placement is installed, trace replay does not run online EPLB or charge expert migration. Split, Concentrate, and Heuristic therefore differ only in physical-replica routing, matching the controlled real-GPU experiment. A controlled HTTP run should contain only traced requests; the virtual benchmark below is isolated from the synthetic request path entirely.

## Replica-routing fidelity

Trace replay uses integer token assignments rather than the fractional synthetic approximation.

- Split follows the real `base_router.py` round-robin mapping. If an expert has two assignments and four replicas, only the first two replicas are activated.
- Concentrate processes experts hottest-first and selects the eligible GPU with the lowest projected local routed-expert cost.
- Heuristic uses Python round-to-even for replica count, stable least-loaded replica selection, integral token chunks, and the same 20-iteration whole-chunk local search as the patched vLLM router.

The physical replica order is preserved from the fixed `physical_to_logical` mapping because Split and tie cases depend on that order.

## Hardware fidelity

For `Qwen/Qwen1.5-MoE-A2.7B`, trace replay includes the always-on shared expert with intermediate dimension 5632 even though the trace records routed experts only.

Known A100 and H100 peak configurations automatically use the hardware calibration from `moe_all_files/serve_models.py::layer_ms_hw`:

| GPU profile | HBM efficiency | Compute efficiency | GEMM row block | Layer launch |
| --- | ---: | ---: | ---: | ---: |
| A100, 312 TFLOP/s and 2.0 TB/s | 0.75 | 0.70 | 128 | 9 us |
| H100, 990 TFLOP/s and 3.35 TB/s | 0.80 | 0.80 | 128 | 6 us |

Unknown hardware retains the ideal roofline. The calibration can be overridden at startup with environment variables:

- `LLMD_MOE_SHARED_INTERMEDIATE_SIZE`
- `LLMD_MOE_MEMORY_EFFICIENCY`
- `LLMD_MOE_COMPUTE_EFFICIENCY`
- `LLMD_MOE_GEMM_BLOCK_ROWS`
- `LLMD_MOE_KERNEL_LAUNCH_OVERHEAD`
- `LLMD_MOE_TRACE_PREFILL_BATCH_TOKENS`
- `LLMD_MOE_TRACE_PREFILL_COALESCE`

The default trace prefill token budget is 1024. Concurrent trace prefills are coalesced and chunked into shared forwards. Active decode requests share one forward per decode generation. All trace MoE forwards reserve one serialized virtual GPU timeline, so two independent HTTP workers cannot make the same simulated eight-GPU model execute overlapping forwards.

The v1 trace does not record the EP source rank of each hidden state. Communication therefore assumes source tokens are balanced across EP ranks and uses the policy-dependent destination load to model the all-to-all bottleneck. Exact source-to-destination traffic requires a future trace format that records source-rank ownership.

The HTTP path still performs the routing calculation on the host before it can emit a token. Its Prometheus modeled TTFT/TPOT values use modeled duration, but client wall-clock latency can be larger when the Go calculation itself is slower than the simulated GPU. Prefill and decode also share the same serialized model timeline but are not fused into one mixed prefill-plus-decode HTTP forward. These are simulator-runtime effects, not GPU-model effects.

## Virtual cross-validation benchmark

Use `scripts/moe_trace_virtual_benchmark` when comparing routing-policy performance with real GPU runs. It does not sleep or use HTTP wall time. It advances a pure modeled clock and implements one shared vLLM-style serving loop:

1. repeat the trace prompt set `--copies` times when the target run contains more requests than the trace;
2. admit no more than `max-num-seqs` active requests and leave later requests queued;
3. one token from every active decode sequence;
4. remaining token budget filled by chunked prefill;
5. one aggregate `[layer][expert]` workload per forward;
6. one multi-GPU routing/cost calculation per forward;
7. `N-1` decode forwards for `N` visible generated tokens.

Example for the controlled A100 experiment:

```bash
go run ./scripts/moe_trace_virtual_benchmark \
  --trace /data/instructcoder_2000_both.moetrace \
  --fixed-placement /data/fixed_dhondt_placement.json \
  --gpu a100 \
  --token-budget 1024 \
  --max-num-seqs 32 \
  --copies 1
```

Both scheduler limits and the request count should match the real vLLM invocation. `max-num-seqs` must not exceed the token budget because a decode-only forward carries one token for every active sequence. `--copies` repeats the complete prompt set through the scheduler; it does not simply multiply the final latency. Thus a 2,000-prompt trace can represent a 20,000-request run from the same prompt set with `--copies 10`.

This is the preferred result for answering whether Split, Concentrate, and Heuristic have the same relative ordering as the real vLLM run. `scripts/moe_trace_all_prompts_benchmark` remains useful for HTTP/API and queueing behavior.

## Replay a traced prompt

Add `trace_prompt_id` to a chat-completions request:

```json
{
  "model": "Qwen/Qwen1.5-MoE-A2.7B",
  "messages": [
    {"role":"user","content":"ignored during trace replay"}
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
| Yes | Yes | Trace fidelity replay |
| No | Yes | Request rejected |

Trace fidelity is intentionally isolated from the legacy synthetic workload. Requests without `trace_prompt_id` retain the previous synthetic power-law routing and latency behavior.

## Initial restrictions

Trace replay currently applies only to ordinary text `/v1/chat/completions` requests. A request using `trace_prompt_id` is rejected when combined with multiple choices (`n > 1`), tools, remote prefill/decode, `ignore_eos`, logprobs, multimodal content, LoRA models, echo mode, or image emission.
