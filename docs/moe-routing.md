# MoE expert-parallel routing simulation

The simulator can model a Mixture-of-Experts layer spread across multiple simulated GPUs. This is separate from data parallelism: `data-parallel-size` creates independent serving ranks, while `moe-expert-parallel-size` defines the expert-parallel GPUs inside each rank.

For example, `--data-parallel-size 2 --moe-expert-parallel-size 8` represents two serving replicas. Each replica has its own eight-GPU expert-parallel group.

MoE simulation is disabled by default. Enable it with `--enable-moe`.

## Routing model

The simulator does not execute a real gate network. Logical expert demand is generated from a configurable power-law distribution:

```text
p_i proportional to (i + 1)^(-alpha)
```

`moe-expert-popularity-alpha=0` produces a uniform distribution. Larger values make the lowest-numbered experts hotter.

Every logical expert receives one physical copy. Remaining `moe-physical-expert-slots` are assigned with D'Hondt allocation so hotter experts receive more replicas. Replicas are then placed across expert-parallel GPUs while balancing expected load and avoiding two copies of the same expert on one GPU.

Three replica-routing policies are available:

- `split`: divide an expert's token assignments evenly across every replica.
- `concentrate`: send all assignments for an expert to one replica selected to minimize the current GPU critical path.
- `heuristic`: choose how many replicas to activate from the compute-to-weight-loading ratio, place those shares greedily, then perform bounded local rebalancing.

Replica placement rotates by layer so the same logical expert does not always start on the same GPU.

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

Every request in that decode step therefore observes the cost of routing the shared active-token batch. Step costs are cached by token count so the routing model does not add significant CPU overhead to token streaming.

The MoE latency is additive. Existing `time-to-first-token` or per-token prefill latency is retained and the MoE prefill cost is added to it. Existing `inter-token-latency` is retained and the MoE decode cost is added to it. The standard TTFT and TPOT metrics therefore include the MoE component.

If an existing latency profile was calibrated from an MoE model and already includes expert execution, enabling this feature can double-count that work. For routing-only experiments, use a zero or appropriately reduced base latency profile.

## Example

The defaults use the same analytical parameters as the project's MoE cross-validation experiments: 60 logical experts, 80 physical expert slots, top-k 4, 24 MoE layers, hidden size 2048, intermediate size 1408, BF16-sized elements, 312 TFLOP/s, and 2 TB/s memory bandwidth.

```bash
go run ./cmd/llm-d-inference-sim \
  --model dummy-model \
  --enable-moe \
  --moe-expert-parallel-size 8 \
  --moe-router heuristic \
  --moe-expert-popularity-alpha 0.8 \
  --time-to-first-token 0 \
  --inter-token-latency 0
```

To compare routing policies, run the same workload three times and change only `--moe-router` between `split`, `concentrate`, and `heuristic`. Compare TTFT, TPOT, end-to-end latency, and throughput.

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

`moe-physical-expert-slots` must be at least the logical expert count and cannot exceed `moe-num-experts * moe-expert-parallel-size`. This keeps at most one copy of each logical expert on each GPU.

## Scope and limitations

This feature is intended for relative routing experiments, capacity studies, and cross-validation against measured results. It is not a cycle-accurate GPU simulator.

The current model uses synthetic expert popularity rather than model-generated gate decisions or recorded expert traces. Decode batching uses the running request count instead of reproducing the exact vLLM scheduler. Interconnect cost uses one aggregate bandwidth and latency rather than NVLink or NVSwitch topology. Communication and expert compute are serialized rather than overlapped. Tensor parallelism remains represented by the existing aggregate latency profile rather than explicit GPU resources.

Trace-driven expert IDs, explicit batch formation, topology-aware communication, and compute/communication overlap can be added later without changing the DP/EP separation introduced here.
