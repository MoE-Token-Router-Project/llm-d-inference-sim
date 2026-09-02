# MoE trace virtual benchmark

Use this benchmark for routing-policy cross-validation against the real multi-GPU vLLM runs. It consumes the `.moetrace` workload directly and advances **modeled GPU time only**; Go CPU execution time, goroutine scheduling, HTTP queueing, and sleeps cannot change the reported policy ordering.

The serving loop follows vLLM-style continuous batching while preserving the trace generator's decode semantics:

1. Admit no more than `--max-num-seqs` active sequences; later prompts remain queued in trace order.
2. Every active decode sequence contributes one token to the next forward.
3. The remaining `--token-budget` is filled with stable-order prefill chunks.
4. The exact recorded logical expert routes are aggregated into one `[layer][expert]` workload.
5. Split, Concentrate, or Heuristic maps those assignments to the configured physical replicas.
6. All EP GPUs execute one shared forward and advance together.
7. For `N` visible output tokens, only `N-1` decode forwards are charged because the first output is produced by prefill.

For the controlled experiment in `moe_all_files`, pass the same frozen D'Hondt placement and concurrency used by real vLLM:

```bash
go run ./scripts/moe_trace_virtual_benchmark \
  --trace /path/to/instructcoder_2000_both.moetrace \
  --fixed-placement /path/to/fixed_dhondt_placement.json \
  --gpu a100 \
  --token-budget 1024 \
  --max-num-seqs 32
```

The default Qwen1.5 settings are EP=8, 80 physical slots, hidden size 2048, routed intermediate size 1408, BF16, a 1024-token forward budget, `max-num-seqs=32`, and a 400 GB/s interconnect with 5 us one-way phase latency. `--gpu a100` selects 312 TFLOP/s and 2.0 TB/s; `--gpu h100` selects 990 TFLOP/s and 3.35 TB/s. The trace fidelity layer then applies the same achieved-efficiency, 128-row GEMM padding, launch-overhead, and Qwen shared-expert model documented in `docs/moe-trace-runtime.md`.

`--max-num-seqs` must not exceed `--token-budget`, because a decode-only forward contains one token for every active sequence. Change both values when reproducing a real run that used different scheduler limits.

By default all three custom policies are evaluated:

```text
split,concentrate,heuristic
```

Use `--routers split,heuristic` to select a subset or `--json` for machine-readable output.

This benchmark is the preferred tool for asking whether the **model** reproduces real GPU policy ordering. `scripts/moe_trace_all_prompts_benchmark` remains useful for testing the HTTP simulator runtime, client-observed TTFT, queueing, and API throughput, but those wall-clock values can include host-side simulator execution overhead and should not be treated as pure GPU predictions.

The `.moetrace` v1 format does not record the EP source rank that owned each hidden state. The communication model therefore assumes source tokens are balanced across EP ranks and models the policy-dependent destination bottleneck. Exact source-to-destination all-to-all traffic requires additional source-rank information that is not present in the current trace.
