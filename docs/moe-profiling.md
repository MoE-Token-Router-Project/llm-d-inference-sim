# MoE profiling

Trace replay can emit a Perfetto-compatible Chrome Trace JSON artifact that shows CPU routing time on the simulated system timeline, expert communication, per-GPU MoE work, compute utilization, HBM utilization and bandwidth, and persistent expert-weight VRAM.

Start trace replay with `--moe-profile-output`:

```bash
./bin/llm-d-inference-sim \
  --model Qwen/Qwen1.5-MoE-A2.7B \
  --enable-moe \
  --moe-trace-path /data/instructcoder_2000_both.moetrace \
  --moe-profile-output profile.trace.json.gz \
  --moe-expert-parallel-size 8 \
  --moe-num-experts 60 \
  --moe-physical-expert-slots 80 \
  --moe-top-k 4 \
  --moe-num-layers 24
```

Stop the simulator gracefully to finalize the artifact, then open the resulting `.json` or `.json.gz` file in the Perfetto UI. The output contains tracks for CPU routing, dispatch and combine communication, expert migration, and each simulated GPU. GPU operation events include layer, expert token counts, FLOPs, HBM traffic, modeled compute and HBM time, utilization, bottleneck type, and persistent expert-weight VRAM.

Compute and HBM utilization are derived from the simulator cost model. They are not measurements from a physical GPU. HBM GB/s is the modeled traffic divided by the modeled GPU phase duration. Persistent VRAM includes physical expert slots and the configured shared expert. KV cache, activation lifetime, communication buffers, and GEMM workspace are not included yet.

The `CPU / Router` track is part of the `Simulated system` process so CPU routing and GPU work can be viewed on one timeline. The duration of each routing span is measured from the Go routing implementation on the simulator CPU, then placed before that layer's dispatch event on the simulated timeline. The routing span includes `measured_us` and `timing_source=simulator_host_cpu` in its event arguments. The separate `Simulator host` process is kept only for simulator-side EPLB diagnostics.

The profiler uses a separate visualization schedule that includes measured router time. This keeps `Router -> Dispatch -> GPU -> Combine` causal order in the trace without adding simulator-host CPU timing noise to modeled TTFT or TPOT. Profiling therefore changes the layout of the trace only; it does not change routing decisions or modeled request latency.

Profiling is trace-replay only in this version. `--moe-profile-output` therefore requires `--moe-trace-path`. The profiler records the same per-layer routing and GPU cost data used by trace replay. The artifact is finalized on graceful simulator shutdown, so terminate the server normally before opening the trace.
