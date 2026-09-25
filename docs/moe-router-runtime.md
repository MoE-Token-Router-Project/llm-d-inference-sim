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

## Distributed router timing

With `--use-distributed-routing`, the simulator measures each GPU-local router
instance separately and measures the aggregation step separately. The local
routers model parallel execution, so `--moe-count-router-runtime` adds the
critical-path routing time for each layer:

```text
routing_wall_time = max(local_router_times) + aggregator_time
```

The profiler places each local `MoE Router` slice on the same
`GPU N / Operations` track as that GPU's `MoE MLP` slice. The routing
aggregator remains on its own CPU track and finishes before expert dispatch.
The measured local-router and aggregator durations are still host CPU
measurements; their placement on GPU operation tracks represents the logical
per-GPU routing pipeline.
