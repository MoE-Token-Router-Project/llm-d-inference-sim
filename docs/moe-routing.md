# MoE expert-parallel routing simulation

The simulator can model a Mixture-of-Experts layer spread across multiple simulated GPUs. This is separate from data parallelism: `data-parallel-size` creates independent serving ranks, while `moe-expert-parallel-size` defines the expert-parallel GPUs inside each rank.

For example, `--data-parallel-size 2 --moe-expert-parallel-size 8` represents two serving replicas. Each replica has its own eight-GPU expert-parallel group.

MoE simulation is disabled by default. Enable it with `--enable-moe`.

## vLLM-compatible expert placement and EPLB

Expert placement follows the vLLM 0.27.1 EPLB lifecycle used by the reference D'Hondt implementation in `MoE-Token-Router-Project/moe_all_files`.

There are two distinct placement phases.

### Initial placement

Initial placement does not use D'Hondt or an expert-load trace. It reproduces vLLM's default EPLB map:

```text
physical_to_logical = [0, 1, ..., num_logical_experts-1,
                       0, 1, ..., num_redundant_experts-1]
```

The redundant suffix cycles through logical expert IDs when necessary. Physical expert slots are contiguous by EP rank, matching vLLM's default linear placement. The same initial mapping is used for every MoE layer.

For 8 logical experts, 12 physical experts, and 4 EP ranks, the global physical map is:

```text
[0, 1, 2, 3, 4, 5, 6, 7, 0, 1, 2, 3]
```

Each rank owns three consecutive physical slots:

```text
GPU0: experts 0, 1, 2
GPU1: experts 3, 4, 5
GPU2: experts 6, 7, 0
GPU3: experts 1, 2, 3
```

### Online load collection

vLLM records expert token counts from real forward passes. Its EPLB state stores a sliding window of physical-expert loads, maps those physical counters back to logical experts using the current placement, and sums the window before calling the EPLB policy.

The simulator follows the same lifecycle. Each simulated MoE forward produces logical expert assignment counts. Those counts are the workload information used both by token routing and by the EPLB load window. Since summing all physical replicas of a logical expert returns the same logical assignment count, the simulator stores the logical form directly.

The current simulator still does not execute a real Qwen gate. The per-forward logical assignments are generated from the configured synthetic popularity distribution:

```text
p_i proportional to (i + 1)^(-alpha)
```

Therefore the *source and timing* of load collection match vLLM, while the expert IDs themselves are synthetic unless a future trace-driven gate source is connected.

### Re-placement timing

The simulator intentionally matches the vLLM 0.27.1 EPLB defaults:

```text
load window size        = 1000 forward steps
rearrangement interval  = 3000 forward steps
initial counter         = 3/4 of the interval
```

As in vLLM, only forward steps close enough to the next rearrangement are recorded into the load window. The initial counter starts at 2250, so the first online D'Hondt placement is computed after 750 simulated forward steps. Later rearrangements occur every 3000 forward steps and use the most recent 1000 recorded steps.

A prefill MoE calculation counts as one simulated forward. During decode, each generated token invokes one shared forward whose active-token count is approximated by the number of running requests.

### D'Hondt re-placement

At each rearrangement, each layer's accumulated logical-expert load is passed to a direct Go port of `dhondt/dhondt_core.py`.

The algorithm starts with one physical copy of every logical expert. Remaining physical slots are allocated with the D'Hondt score:

```text
score(expert) = load(expert) / (replicas(expert) + 1)
```

The port preserves the reference implementation's behavior:

- ties are resolved by lower logical expert ID;
- a logical expert can have at most one copy on each EP rank;
- physical expert slots must be divisible by the EP size;
- every EP rank receives exactly the same number of physical expert slots;
- experts are placed in descending replica-count order;
- ties in rank occupancy are resolved by lower rank ID.

This fixes the earlier simulator approximation that could allocate more replicas than available GPUs or place duplicate copies of one logical expert on the same GPU.

## Expert migration cost

The D'Hondt policy in the reference vLLM patch uses synchronous EPLB, so the simulator models re-placement as a blocking operation.

For each layer, the simulator compares the old and new expert-to-GPU maps. A new copy on a GPU that did not already hold that logical expert is counted as one weight transfer. Expert size is:

```text
expert_weight_bytes = 3 * hidden_size * intermediate_size * bytes_per_element
```

Transfers to different destination GPUs are modeled as parallel within one layer. The layer's migration critical path is:

```text
migration_time = interconnect_latency
               + max_inbound_expert_copies_per_gpu
                 * expert_weight_bytes
                 / interconnect_bandwidth
```

Migration time is summed over layers and added to the forward that triggers the rearrangement. This is a first-order approximation of vLLM's `rearrange_expert_weights_inplace`; it does not model detailed NVLink/NVSwitch contention or asynchronous EPLB.

## Replica routing

Placement and token routing are separate decisions. Placement determines which GPUs are allowed to execute each logical expert. The replica router then selects among those available copies for the current forward.

Three replica-routing policies are available:

- `split`: divide an expert's token assignments evenly across every replica.
- `concentrate`: send all assignments for an expert to one replica selected to minimize the current GPU critical path.
- `heuristic`: choose how many replicas to activate from the compute-to-weight-loading ratio, place those shares greedily, then perform bounded local rebalancing.

## Cost model

For hidden size `d`, intermediate size `d_ff`, and `b` bytes per element, one expert uses:

```text
expert_weight_bytes = 3 * d * d_ff * b
flops_per_assignment = 2 * 3 * d * d_ff
activation_bytes_per_assignment = (2 * d + d_ff) * b
```

For each simulated GPU, expert execution time is the maximum of memory time and compute time:

```text
memory_time = (active_experts * expert_weight_bytes
               + assignments * activation_bytes_per_assignment)
              / gpu_memory_bandwidth

compute_time = assignments * flops_per_assignment / gpu_flops

gpu_time = max(memory_time, compute_time)
```

The layer compute critical path is the slowest GPU. For expert parallel size greater than one, the simulator also adds dispatch and gather all-to-all time. It assumes tokens originate uniformly across the expert-parallel group, so the expected remote fraction is `1 - 1 / EP`.

```text
bytes_per_phase = max_gpu_assignments * remote_fraction * d * b
communication_time = 2 * (interconnect_latency
                          + bytes_per_phase / interconnect_bandwidth)

layer_time = max_gpu_time + communication_time
```

The total MoE time is the sum across `moe-num-layers`.

## Prefill and decode

For prefill, the MoE model uses the number of prompt tokens that are not already in the local KV cache. A decode rank performing remote prefill does not add prompt-side MoE execution time because that work belongs to the remote prefill rank.

For decode, the simulator approximates a continuous-batching forward pass with one token from every running request:

```text
tokens_per_decode_forward = max(1, number_of_running_requests)
```

Every request in that decode step therefore observes the cost of routing the shared active-token batch.

Latency caching is keyed by both active-token count and placement generation. A re-placement therefore cannot accidentally reuse latency computed for the previous expert map.

The MoE latency is additive. Existing `time-to-first-token` or per-token prefill latency is retained and the MoE prefill cost is added to it. Existing `inter-token-latency` is retained and the MoE decode cost is added to it. The standard TTFT and TPOT metrics therefore include the MoE component.

If an existing latency profile was calibrated from an MoE model and already includes expert execution, enabling this feature can double-count that work. For routing-only experiments, use a zero or appropriately reduced base latency profile.

## Example

```bash
go run ./cmd/llm-d-inference-sim \
  --model dummy-model \
  --enable-moe \
  --moe-expert-parallel-size 8 \
  --moe-num-experts 60 \
  --moe-physical-expert-slots 80 \
  --moe-router heuristic \
  --moe-expert-popularity-alpha 0.8 \
  --time-to-first-token 0 \
  --inter-token-latency 0
```

The server starts with vLLM's default initial expert map. After 750 simulated MoE forward steps, the first D'Hondt re-placement is computed from the observed load window. The next re-placement is 3000 forward steps later.

To compare routing policies, run the same workload three times and change only `--moe-router` between `split`, `concentrate`, and `heuristic`.

## Configuration

The available MoE parameters are:

- `enable-moe`: enable expert-parallel MoE latency simulation, default `false`.
- `moe-expert-parallel-size`: number of simulated expert GPUs, default `8`.
- `moe-num-experts`: number of logical experts per layer, default `60`.
- `moe-physical-expert-slots`: number of physical expert copies across all expert GPUs, default `80`.
- `moe-top-k`: logical experts selected per token, default `4`.
- `moe-num-layers`: MoE layers included in the cost, default `24`.
- `moe-router`: `split`, `concentrate`, or `heuristic`, default `split`.
- `moe-expert-popularity-alpha`: power-law popularity alpha, default `0.8`.
- `moe-hidden-size`: model hidden size, default `2048`.
- `moe-intermediate-size`: expert intermediate size, default `1408`.
- `moe-bytes-per-element`: tensor bytes per element, default `2`.
- `moe-gpu-flops`: simulated GPU throughput in FLOP/s, default `312e12`.
- `moe-gpu-memory-bandwidth`: simulated GPU memory bandwidth in bytes/s, default `2e12`.
- `moe-interconnect-bandwidth`: per-GPU expert-parallel bandwidth in bytes/s, default `400e9`.
- `moe-interconnect-latency`: one-way latency per all-to-all phase, default `5us`.

`moe-physical-expert-slots` must be at least the logical expert count, no greater than `moe-num-experts * moe-expert-parallel-size`, and divisible by `moe-expert-parallel-size` for D'Hondt/vLLM-compatible placement.

## Scope and limitations

This feature is intended for relative routing and placement experiments, capacity studies, and cross-validation against measured results. It is not a cycle-accurate GPU simulator.

The EPLB lifecycle and D'Hondt placement now follow the referenced vLLM behavior, but the current workload source is still synthetic rather than model-generated gate decisions. Interconnect cost uses one aggregate bandwidth and latency rather than NVLink or NVSwitch topology. Communication and expert compute are serialized rather than overlapped. Synchronous migration cost is a first-order model of vLLM weight movement. Tensor parallelism remains represented by the existing aggregate latency profile rather than explicit GPU resources.

A real Qwen token-to-expert trace can be connected later by feeding per-forward, per-layer logical expert counts into the same EPLB load-window path; the initial placement, window timing, D'Hondt re-placement, and migration machinery do not need to change.
