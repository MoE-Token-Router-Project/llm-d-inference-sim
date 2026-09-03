# MoE profiling

Trace replay can emit a Perfetto-compatible Chrome Trace JSON artifact that shows CPU routing time on the simulated system timeline, expert communication, per-GPU MoE work, compute utilization, HBM utilization and bandwidth, persistent expert-weight VRAM, and detailed token-to-expert assignments.

## Generate a profile

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

Stop the simulator gracefully to finalize the artifact. The profiler writes the artifact on graceful shutdown, so terminate the server normally before reading the trace.

## Inspect small and large profiles

Detailed token assignment data can make profile artifacts much larger than earlier profiler output. The Perfetto browser UI is convenient for smaller traces, but a large `.json` or `.json.gz` profile can become slow or fail to load because the browser must parse and display the detailed JSON data.

Use the repository's `cmd/trace_processor` wrapper for large profiles. The wrapper downloads the pinned Perfetto `trace_processor_shell` binary for the current platform on first use and caches it under `~/.local/share/perfetto/prebuilts`. Python 3 is required, and the first run also needs network access and `curl`.

Open an interactive PerfettoSQL shell with:

```bash
python3 cmd/trace_processor profile.trace.json.gz
```

Use the `query` subcommand for one-off queries that can be saved or piped into another tool:

```bash
python3 cmd/trace_processor query profile.trace.json.gz \
  "SELECT ts, dur, name FROM slice WHERE name = 'MoE MLP' LIMIT 20;"
```

Token assignment data is stored in event arguments, and Trace Processor exposes nested arguments through the `args` table. Start by listing the exact token-assignment keys present in the trace:

```bash
python3 cmd/trace_processor query profile.trace.json.gz \
  "SELECT DISTINCT key FROM args WHERE key GLOB '*token_assignments*' ORDER BY key LIMIT 200;"
```

Inspect token-assignment values together with their GPU work slices with:

```bash
python3 cmd/trace_processor query profile.trace.json.gz "
SELECT
  s.ts,
  s.dur,
  t.name AS track,
  a.key,
  a.display_value
FROM slice AS s
JOIN track AS t ON t.id = s.track_id
JOIN args AS a ON a.arg_set_id = s.arg_set_id
WHERE s.name = 'MoE MLP'
  AND a.key GLOB '*token_assignments*'
ORDER BY s.ts, a.key
LIMIT 500;
"
```

Trace Processor still parses the trace into memory. The command-line path removes the browser UI and rendering overhead, but extremely large profiles can still require substantial memory. Use a smaller profiling workload when the trace is too large for the available host memory.

## Trace contents and token assignment semantics

The profile contains tracks for CPU routing, dispatch and combine communication, expert migration, and each simulated GPU. GPU operation events include layer, expert token counts, FLOPs, HBM traffic, modeled compute and HBM time, utilization, bottleneck type, and persistent expert-weight VRAM. Trace-backed GPU events also include `phase` and a `token_assignments` argument. Each token assignment records `request_id`, `phase`, `token_position`, `moe_layer`, and `expert_id`. `moe_layer` is the zero-based MoE layer index used by the simulator. Prefill token positions are zero-based input positions; decode positions continue after the input tokens, matching the positions in the source trace.

The routing cost model operates on expert token counts. When an expert has replicas on multiple GPUs, the profiler deterministically maps the exact trace token assignments onto the routed per-GPU expert counts in stable token order. This mapping preserves the modeled GPU loads but should not be interpreted as a measured physical execution order. Synthetic non-trace decode load remains aggregate-only and therefore has no per-token assignment records.

Compute and HBM utilization are derived from the simulator cost model. They are not measurements from a physical GPU. HBM GB/s is the modeled traffic divided by the modeled GPU phase duration. Persistent VRAM includes physical expert slots and the configured shared expert. KV cache, activation lifetime, communication buffers, and GEMM workspace are not included yet.

The `CPU / Router` and `CPU / EPLB` tracks are part of the `Simulated system` process so routing, load-balancing work, network phases, and GPU work can be viewed on one timeline. Router and EPLB span durations are measured from the Go implementation on the simulator CPU and placed on the simulated timeline. Both span types include `measured_us` and `timing_source=simulator_host_cpu` in their event arguments. EPLB appears before the routing work that uses the updated placement.

The profiler uses a visualization schedule that includes measured router and EPLB time. This keeps CPU work and `Router -> Dispatch -> GPU -> Combine` order clear without adding simulator-host CPU timing noise to modeled TTFT or TPOT. Profiling therefore changes the layout of the trace only; it does not change routing decisions or modeled request latency.

Profiling is trace-replay only in this version. `--moe-profile-output` therefore requires `--moe-trace-path`. The profiler records the same per-layer routing and GPU cost data used by trace replay.
