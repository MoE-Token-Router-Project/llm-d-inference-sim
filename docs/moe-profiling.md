# MoE profiling

Trace replay can emit a Perfetto-compatible Chrome Trace JSON artifact that shows simulated CPU routing markers, expert communication, per-GPU MoE work, compute utilization, HBM utilization and bandwidth, and persistent expert-weight VRAM.

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

The simulated CPU router track is a logical event because the simulator does not yet model serving-CPU latency. Each routing event includes `host_router_us` for quick inspection. The trace also has a separate `Simulator host` process with real Go routing and EPLB spans. Those host spans are diagnostics for simulator runtime and are not added to simulated TTFT or TPOT.

Profiling is trace-replay only in this version. `--moe-profile-output` therefore requires `--moe-trace-path`. Enabling profiling does not change routing decisions or modeled latency; the profiler records the same per-layer cost breakdown used by trace replay. The artifact is finalized on graceful simulator shutdown, so terminate the server normally before opening the trace.
