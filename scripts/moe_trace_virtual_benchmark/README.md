# MoE trace virtual benchmark

Use this benchmark for routing-policy cross-validation against the real multi-GPU vLLM runs. It consumes the `.moetrace` workload directly and advances **modeled GPU time only**; Go CPU execution time, goroutine scheduling, HTTP queueing, and sleeps cannot change the reported policy ordering.

The serving loop follows vLLM-style continuous batching while preserving the trace generator's decode semantics:

1. Repeat the complete trace set `--copies` times in stable order when the target serving run has more requests than the trace contains.
2. Admit no more than `--max-num-seqs` active sequences; later prompts remain queued in trace order.
3. Every active decode sequence contributes one token to the next forward.
4. The remaining `--token-budget` is filled with stable-order prefill chunks.
5. The exact recorded logical expert routes are aggregated into one `[layer][expert]` workload.
6. Split, Concentrate, or Heuristic maps those assignments to the configured physical replicas.
7. All EP GPUs execute one shared forward and advance together.
8. For `N` visible output tokens, only `N-1` decode forwards are charged because the first output is produced by prefill.

For the controlled experiment in `moe_all_files`, pass the same frozen D'Hondt placement and scheduler limits used by real vLLM:

```bash
go run ./scripts/moe_trace_virtual_benchmark \
  --trace /path/to/instructcoder_2000_both.moetrace \
  --fixed-placement /path/to/fixed_dhondt_placement.json \
  --gpu a100 \
  --token-budget 1024 \
  --max-num-seqs 32 \
  --copies 1
```

`--copies` repeats the whole prompt set, like `N_P` in `moe_all_files/serve_models.py`. For example, if the trace contains 2,000 InstructCoder prompts and the real benchmark served 20,000 requests from that same prompt set, use `--copies 10`. Prompt tokens, output tokens, decode forwards, queueing, and modeled serving time all scale through the actual repeated scheduler execution rather than by multiplying the final answer.

The default Qwen1.5 settings are EP=8, 80 physical slots, hidden size 2048, routed intermediate size 1408, BF16, a 1024-token forward budget, `max-num-seqs=32`, one trace-set copy, and a 400 GB/s interconnect with 5 us one-way phase latency. `--gpu a100` selects 312 TFLOP/s and 2.0 TB/s; `--gpu h100` selects 990 TFLOP/s and 3.35 TB/s. The trace fidelity layer then applies the same achieved-efficiency, 128-row GEMM padding, launch-overhead, and Qwen shared-expert model documented in `docs/moe-trace-runtime.md`.

## Attention timing estimate

The virtual benchmark also reports `ModeledAttentionTime`, `ModeledAttentionPrefillTime`, and `ModeledAttentionDecodeOnlyTime` in JSON output (and `attn_ms`, `attn_pre_ms`, and `attn_dec_ms` in the table). This estimate is intentionally separate from `ModeledTime`, which keeps its existing meaning as routed/shared-expert MoE time. Adding attention reporting therefore does not change existing MoE timing or policy ordering.

The attention estimate targets the same scope as the vLLM `Attention.forward` NVTX range used by the single-A100 fidelity experiment: KV-cache update plus the selected attention backend, excluding QKV projection, rotary embedding, and output projection. The current model is named `flashattention2-roofline-v1`. For each sequence in a virtual forward, if `p` tokens are already present and the forward contributes `q` query tokens, causal attention has `q*p + q*(q+1)/2` visible query/key pairs. The model charges approximately `4*d` FLOPs per pair for QK^T and PV, and `(2*p + 6*q)*d` element transfers for Q/output, K/V reads, and K/V cache writes. It uses the same GPU compute/memory efficiencies as the existing trace-fidelity model and charges two hardware launch-overhead terms per layer, corresponding to KV-cache update plus FlashAttention.

This is an analytical roofline estimate, not a fit to measured attention timings. In particular, it does not attempt to reproduce Python/dispatcher gaps that may be included in Nsight's GPU-projected span between the first and last GPU operation inside an NVTX range. `AttentionLayerCalls` is also reported so a measured per-layer range count can be compared directly with the virtual scheduler.

`--max-num-seqs` must not exceed `--token-budget`, because a decode-only forward contains one token for every active sequence. Change both values when reproducing a real run that used different scheduler limits.

By default all three custom policies are evaluated:

```text
split,concentrate,heuristic
```

Use `--routers split,heuristic` to select a subset or `--json` for machine-readable output.

This benchmark is the preferred tool for asking whether the **model** reproduces real GPU policy ordering. `scripts/moe_trace_all_prompts_benchmark` remains useful for testing the HTTP simulator runtime, client-observed TTFT, queueing, and API throughput, but those wall-clock values can include host-side simulator execution overhead and should not be treated as pure GPU predictions.

The `.moetrace` v1 format does not record the EP source rank that owned each hidden state. The communication model therefore assumes source tokens are balanced across EP ranks and models the policy-dependent destination bottleneck. Exact source-to-destination all-to-all traffic requires additional source-rank information that is not present in the current trace.
