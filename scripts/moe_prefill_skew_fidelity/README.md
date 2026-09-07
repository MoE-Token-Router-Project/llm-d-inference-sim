# MoE prefill skew fidelity experiments

These experiments use the simulator's normal OpenAI-compatible `/v1/chat/completions` HTTP path. They do not use a MoE trace. The synthetic token-to-expert distribution is controlled by `--moe-expert-popularity-alpha`, where `0` is uniform and larger values are more skewed.

The simulator is restarted for every sample. This keeps the initial expert placement fixed, prevents EPLB migration from changing placement, and avoids request history effects. Each request has a large synthetic prompt and `max_tokens=1`, so the measured TTFT is dominated by prefill.

The MoE configuration follows the fidelity experiment used for Qwen1.5-MoE: 60 experts, top-4, 24 MoE layers, 8 expert-parallel GPUs, and 80 physical expert slots. The hardware knobs use H100-class values: 990 TFLOP/s and 3.35 TB/s. Base TTFT and ITL are zero.

`run_experiments.py` records two TTFT values. `modeled_ttft_ms` comes from the simulator's Prometheus `vllm:time_to_first_token_seconds_sum` and is the main fidelity metric. `ttft_ms` is client wall-clock TTFT and also contains HTTP, tokenization, host scheduling, and router execution time.

The `skew` experiment fixes the prompt at 8192 tokens and sweeps alpha over `0`, `0.25`, `0.5`, `0.75`, `1.0`, `1.25`, `1.5`, and `2.0` for Split, Concentrate, and Heuristic. This tests whether Concentrate becomes slower as expert demand becomes strongly skewed.

The `crossover` experiment fixes alpha at `1.0` and sweeps prompt length over 512, 1024, 2048, 4096, 8192, and 16384 tokens. This checks the move from the memory-bound region toward the compute-bound region.

The `endpoints` experiment compares uniform `alpha=0` with strongly skewed `alpha=2.0` at 4096, 8192, and 16384 prompt tokens. This checks convergence under even demand and separation under skew.

Build the simulator from the repository root first:

```bash
make build
```

Run all experiments with:

```bash
python3 scripts/moe_prefill_skew_fidelity/run_experiments.py --experiment all --repeats 3
```

You can also run one experiment at a time with `--experiment skew`, `--experiment crossover`, or `--experiment endpoints`.

The script writes raw CSV files, summary CSV files, and per-run simulator logs into `scripts/moe_prefill_skew_fidelity/`. The summary files are `skew_summary.csv`, `crossover_summary.csv`, and `endpoints_summary.csv`.
