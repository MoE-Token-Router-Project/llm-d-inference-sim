# MoE token-router wall-time accounting

Trace replay measures the host CPU time spent running the configured MoE token router for every MoE layer. By default, that measured router time is shown by the profiler but is excluded from the simulated request wall time, which preserves the existing trace-replay behavior.

Pass `--moe-count-router-runtime` to add the measured router CPU time to the simulated forward duration:

```bash
./bin/llm-d-inference-sim \
  --enable-moe \
  --moe-trace-path trace.moetrace \
  --moe-count-router-runtime
```

The flag also accepts an explicit boolean value. `--moe-count-router-runtime=true` counts router time and `--moe-count-router-runtime=false` excludes it. The option requires `--moe-trace-path`, because trace replay is the path that measures per-layer router runtime.

This switch changes only token-router wall-time accounting. The measured EPLB host CPU time remains excluded from the simulated forward duration, and the modeled GPU, dispatch, combine, and expert-migration costs are unchanged.
