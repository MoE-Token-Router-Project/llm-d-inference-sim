/*
Copyright 2026 The llm-d-inference-sim Authors.

Licensed under the Apache License, Version 2.0 (the "License");
you may not use this file except in compliance with the License.
You may obtain a copy of the License at

    http://www.apache.org/licenses/LICENSE-2.0

Unless required by applicable law or agreed to in writing, software
distributed under the License is distributed on an "AS IS" BASIS,
WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
See the License for the specific language governing permissions and
limitations under the License.
*/

package simulator

import (
	"encoding/json"
	"fmt"
	"math"
	"os"
	"sort"
	"strconv"
	"strings"
	"sync"
	"time"

	"github.com/llm-d/llm-d-inference-sim/pkg/common"
	"github.com/llm-d/llm-d-inference-sim/pkg/moetrace"
)

const (
	defaultTracePrefillBatchTokens = 1024
	defaultTracePrefillCoalesce    = 200 * time.Microsecond
)

type traceFidelityConfig struct {
	fixedPlacement bool

	sharedWeightBytes     float64
	sharedFlopsPerToken   float64
	sharedActivationBytes float64

	memoryEfficiency  float64
	computeEfficiency float64
	gemmBlockRows     int
	kernelLaunch      time.Duration

	prefillBatchTokens int
	prefillCoalesce    time.Duration

	timelineMu    sync.Mutex
	nextAvailable time.Time
	profiler      *moeProfileRecorder
}

type fixedPlacementFile struct {
	PhysicalToLogical  [][]int `json:"physical_to_logical"`
	NumLayers          int     `json:"num_layers"`
	NumPhysicalExperts int     `json:"num_physical_experts"`
}

type traceRoutingState struct {
	expertLoads []map[int]float64
	loads       []float64
}

type tracePrefillJob struct {
	execution *traceExecution
	position  int
	end       int
	elapsed   time.Duration
	done      chan time.Duration
}

type tracePrefillBatcher struct {
	mu        sync.Mutex
	pending   []*tracePrefillJob
	running   bool
	maxTokens int
	coalesce  time.Duration
}

var (
	traceFidelityConfigs sync.Map
	tracePrefillBatchers sync.Map
)

func configureTraceFidelity(m *moeSimulator, config *common.Configuration, fixedPlacementPath string) error {
	options := &traceFidelityConfig{
		memoryEfficiency:   1,
		computeEfficiency:  1,
		prefillBatchTokens: defaultTracePrefillBatchTokens,
		prefillCoalesce:    defaultTracePrefillCoalesce,
	}

	// Qwen1.5-MoE-A2.7B has one always-on shared expert with intermediate
	// dimension 5632. The trace contains routed experts only, so add that work
	// here rather than inventing extra logical expert assignments.
	if strings.Contains(config.Model, "Qwen1.5-MoE-A2.7B") {
		setSharedExpertCost(options, config, 5632)
	}

	// These efficiencies and tile sizes are the same A100/H100 calibration
	// used by moe_all_files/serve_models.py::layer_ms_hw. Unknown hardware
	// retains the ideal roofline instead of guessing a preset.
	if closeRelative(config.MoEGPUFlops, 312e12, 0.05) &&
		closeRelative(config.MoEGPUMemoryBandwidth, 2.0e12, 0.05) {
		options.memoryEfficiency = 0.75
		options.computeEfficiency = 0.70
		options.gemmBlockRows = 128
		options.kernelLaunch = 9 * time.Microsecond
	} else if closeRelative(config.MoEGPUFlops, 990e12, 0.05) &&
		closeRelative(config.MoEGPUMemoryBandwidth, 3.35e12, 0.05) {
		options.memoryEfficiency = 0.80
		options.computeEfficiency = 0.80
		options.gemmBlockRows = 128
		options.kernelLaunch = 6 * time.Microsecond
	}

	if err := applyTraceFidelityEnvironment(options, config); err != nil {
		return err
	}
	if fixedPlacementPath != "" {
		if err := installFixedPlacement(m, fixedPlacementPath); err != nil {
			return err
		}
		options.fixedPlacement = true
	}
	traceFidelityConfigs.Store(m, options)
	return nil
}

func setSharedExpertCost(options *traceFidelityConfig, config *common.Configuration, intermediate int) {
	if intermediate <= 0 {
		options.sharedWeightBytes = 0
		options.sharedFlopsPerToken = 0
		options.sharedActivationBytes = 0
		return
	}
	d := float64(config.MoEHiddenSize)
	dff := float64(intermediate)
	bytesPerElement := float64(config.MoEBytesPerElement)
	parameters := 3 * d * dff
	options.sharedWeightBytes = parameters * bytesPerElement
	options.sharedFlopsPerToken = 2 * parameters
	options.sharedActivationBytes = (2*d + dff) * bytesPerElement
}

func applyTraceFidelityEnvironment(options *traceFidelityConfig, config *common.Configuration) error {
	if value := os.Getenv("LLMD_MOE_SHARED_INTERMEDIATE_SIZE"); value != "" {
		parsed, err := strconv.Atoi(value)
		if err != nil || parsed < 0 {
			return fmt.Errorf("LLMD_MOE_SHARED_INTERMEDIATE_SIZE must be a non-negative integer")
		}
		setSharedExpertCost(options, config, parsed)
	}
	if err := envFloat("LLMD_MOE_MEMORY_EFFICIENCY", &options.memoryEfficiency, true); err != nil {
		return err
	}
	if err := envFloat("LLMD_MOE_COMPUTE_EFFICIENCY", &options.computeEfficiency, true); err != nil {
		return err
	}
	if value := os.Getenv("LLMD_MOE_GEMM_BLOCK_ROWS"); value != "" {
		parsed, err := strconv.Atoi(value)
		if err != nil || parsed < 0 {
			return fmt.Errorf("LLMD_MOE_GEMM_BLOCK_ROWS must be a non-negative integer")
		}
		options.gemmBlockRows = parsed
	}
	if value := os.Getenv("LLMD_MOE_KERNEL_LAUNCH_OVERHEAD"); value != "" {
		parsed, err := time.ParseDuration(value)
		if err != nil || parsed < 0 {
			return fmt.Errorf("LLMD_MOE_KERNEL_LAUNCH_OVERHEAD must be a non-negative duration")
		}
		options.kernelLaunch = parsed
	}
	if value := os.Getenv("LLMD_MOE_TRACE_PREFILL_BATCH_TOKENS"); value != "" {
		parsed, err := strconv.Atoi(value)
		if err != nil || parsed < 1 {
			return fmt.Errorf("LLMD_MOE_TRACE_PREFILL_BATCH_TOKENS must be positive")
		}
		options.prefillBatchTokens = parsed
	}
	if value := os.Getenv("LLMD_MOE_TRACE_PREFILL_COALESCE"); value != "" {
		parsed, err := time.ParseDuration(value)
		if err != nil || parsed < 0 {
			return fmt.Errorf("LLMD_MOE_TRACE_PREFILL_COALESCE must be a non-negative duration")
		}
		options.prefillCoalesce = parsed
	}
	return nil
}

func envFloat(name string, destination *float64, unitInterval bool) error {
	value := os.Getenv(name)
	if value == "" {
		return nil
	}
	parsed, err := strconv.ParseFloat(value, 64)
	if err != nil || parsed <= 0 || (unitInterval && parsed > 1) {
		return fmt.Errorf("%s must be in (0,1]", name)
	}
	*destination = parsed
	return nil
}

func closeRelative(value, target, tolerance float64) bool {
	return math.Abs(value-target) <= math.Abs(target)*tolerance
}

func installFixedPlacement(m *moeSimulator, path string) error {
	contents, err := os.ReadFile(path)
	if err != nil {
		return fmt.Errorf("read fixed MoE placement: %w", err)
	}
	var data fixedPlacementFile
	if err := json.Unmarshal(contents, &data); err != nil {
		return fmt.Errorf("parse fixed MoE placement: %w", err)
	}
	if data.NumLayers != 0 && data.NumLayers != m.numLayers {
		return fmt.Errorf("fixed MoE placement has %d layers; expected %d", data.NumLayers, m.numLayers)
	}
	if data.NumPhysicalExperts != 0 && data.NumPhysicalExperts != m.physicalSlots {
		return fmt.Errorf("fixed MoE placement has %d physical experts; expected %d", data.NumPhysicalExperts, m.physicalSlots)
	}
	if len(data.PhysicalToLogical) != m.numLayers {
		return fmt.Errorf("fixed MoE placement has %d layer maps; expected %d", len(data.PhysicalToLogical), m.numLayers)
	}
	if m.physicalSlots%m.numGPUs != 0 {
		return fmt.Errorf("physical expert slots must be divisible by expert parallel size")
	}

	slotsPerGPU := m.physicalSlots / m.numGPUs
	placements := make([][][]int, m.numLayers)
	for layer, physicalToLogical := range data.PhysicalToLogical {
		if len(physicalToLogical) != m.physicalSlots {
			return fmt.Errorf("fixed MoE placement layer %d has %d slots; expected %d",
				layer, len(physicalToLogical), m.physicalSlots)
		}
		placements[layer] = make([][]int, m.numExperts)
		seen := make([]map[int]struct{}, m.numExperts)
		for expert := range seen {
			seen[expert] = make(map[int]struct{})
		}
		for physical, logical := range physicalToLogical {
			if logical < 0 || logical >= m.numExperts {
				return fmt.Errorf("fixed MoE placement layer %d slot %d has invalid logical expert %d",
					layer, physical, logical)
			}
			gpu := physical / slotsPerGPU
			if _, duplicate := seen[logical][gpu]; duplicate {
				return fmt.Errorf("fixed MoE placement layer %d maps logical expert %d more than once to GPU %d",
					layer, logical, gpu)
			}
			seen[logical][gpu] = struct{}{}
			placements[layer][logical] = append(placements[layer][logical], gpu)
		}
		for expert, replicas := range placements[layer] {
			if len(replicas) == 0 {
				return fmt.Errorf("fixed MoE placement layer %d has no replica for logical expert %d", layer, expert)
			}
		}
	}

	m.stateMu.Lock()
	m.placements = placements
	m.placementGeneration++
	m.lastReplacementMoves = 0
	m.stateMu.Unlock()
	return nil
}

func traceFidelityFor(m *moeSimulator) *traceFidelityConfig {
	if value, ok := traceFidelityConfigs.Load(m); ok {
		return value.(*traceFidelityConfig)
	}
	return &traceFidelityConfig{
		memoryEfficiency:   1,
		computeEfficiency:  1,
		prefillBatchTokens: defaultTracePrefillBatchTokens,
	}
}

func newTraceRoutingState(numGPUs int) *traceRoutingState {
	state := &traceRoutingState{
		expertLoads: make([]map[int]float64, numGPUs),
		loads:       make([]float64, numGPUs),
	}
	for gpu := range state.expertLoads {
		state.expertLoads[gpu] = make(map[int]float64)
	}
	return state
}

func (s *traceRoutingState) add(gpu, expert int, tokens float64) {
	if tokens <= 0 {
		return
	}
	s.expertLoads[gpu][expert] += tokens
	s.loads[gpu] += tokens
}

func distributeReplicaTokens(tokens float64, replicas int) []float64 {
	result := make([]float64, replicas)
	if replicas == 0 || tokens <= 0 {
		return result
	}
	rounded := math.Round(tokens)
	if math.Abs(tokens-rounded) < 1e-9 {
		n := int(rounded)
		base := n / replicas
		remainder := n % replicas
		for replica := range result {
			result[replica] = float64(base)
			if replica < remainder {
				result[replica]++
			}
		}
		return result
	}
	share := tokens / float64(replicas)
	for replica := range result {
		result[replica] = share
	}
	return result
}

func (m *moeSimulator) traceRouterGPUCost(state *traceRoutingState, gpu int) float64 {
	memorySeconds := (float64(len(state.expertLoads[gpu]))*m.expertWeightBytes +
		state.loads[gpu]*m.activationBytes) / m.gpuBandwidth
	computeSeconds := state.loads[gpu] * m.flopsPerAssignment / m.gpuFlops
	return math.Max(memorySeconds, computeSeconds)
}

func (m *moeSimulator) traceRouteSplit(counts []float64, placement [][]int) *traceRoutingState {
	state := newTraceRoutingState(m.numGPUs)
	for expert, count := range counts {
		if count <= 0 {
			continue
		}
		chunks := distributeReplicaTokens(count, len(placement[expert]))
		for index, gpu := range placement[expert] {
			state.add(gpu, expert, chunks[index])
		}
	}
	return state
}

func (m *moeSimulator) traceRouteConcentrate(counts []float64, placement [][]int) *traceRoutingState {
	state := newTraceRoutingState(m.numGPUs)
	for _, expert := range expertOrder(counts) {
		count := counts[expert]
		if count <= 0 {
			continue
		}
		bestGPU := -1
		bestCost := math.Inf(1)
		for _, gpu := range placement[expert] {
			active := len(state.expertLoads[gpu])
			if _, present := state.expertLoads[gpu][expert]; !present {
				active++
			}
			memorySeconds := (float64(active)*m.expertWeightBytes +
				(state.loads[gpu]+count)*m.activationBytes) / m.gpuBandwidth
			computeSeconds := (state.loads[gpu] + count) * m.flopsPerAssignment / m.gpuFlops
			cost := math.Max(memorySeconds, computeSeconds)
			if cost < bestCost {
				bestCost = cost
				bestGPU = gpu
			}
		}
		state.add(bestGPU, expert, count)
	}
	return state
}

func (m *moeSimulator) traceHeuristicOrder(counts []float64) []int {
	order := make([]int, 0, len(counts))
	for expert, count := range counts {
		if count > 0 {
			order = append(order, expert)
		}
	}
	sort.SliceStable(order, func(i, j int) bool {
		left := order[i]
		right := order[j]
		leftSolo := math.Max((m.expertWeightBytes+m.activationBytes*counts[left])/m.gpuBandwidth,
			m.flopsPerAssignment*counts[left]/m.gpuFlops)
		rightSolo := math.Max((m.expertWeightBytes+m.activationBytes*counts[right])/m.gpuBandwidth,
			m.flopsPerAssignment*counts[right]/m.gpuFlops)
		return leftSolo > rightSolo
	})
	return order
}

func (m *moeSimulator) traceRouteHeuristic(counts []float64, placement [][]int) *traceRoutingState {
	state := newTraceRoutingState(m.numGPUs)
	assignments := make(map[int]map[int]float64)
	order := m.traceHeuristicOrder(counts)
	weightSeconds := m.expertWeightBytes / m.gpuBandwidth

	for _, expert := range order {
		computeSeconds := m.flopsPerAssignment * counts[expert] / m.gpuFlops
		replicaCount := 1
		if computeSeconds > weightSeconds {
			replicaCount = int(math.RoundToEven(computeSeconds / weightSeconds))
			if replicaCount < 1 {
				replicaCount = 1
			}
			if replicaCount > len(placement[expert]) {
				replicaCount = len(placement[expert])
			}
		}

		candidates := append([]int(nil), placement[expert]...)
		sort.SliceStable(candidates, func(i, j int) bool {
			return m.traceRouterGPUCost(state, candidates[i]) < m.traceRouterGPUCost(state, candidates[j])
		})
		chunks := distributeReplicaTokens(counts[expert], replicaCount)
		assignments[expert] = make(map[int]float64, replicaCount)
		for index, gpu := range candidates[:replicaCount] {
			if chunks[index] <= 0 {
				continue
			}
			assignments[expert][gpu] = chunks[index]
			state.add(gpu, expert, chunks[index])
		}
	}

	for range 20 {
		slowestGPU := 0
		base := m.traceRouterGPUCost(state, 0)
		for gpu := 1; gpu < m.numGPUs; gpu++ {
			if cost := m.traceRouterGPUCost(state, gpu); cost > base {
				slowestGPU = gpu
				base = cost
			}
		}

		bestGain := 1e-12
		bestExpert := -1
		bestDestination := -1
		bestTokens := 0.0
		bestDestinationHadExpert := false
		for _, expert := range order {
			tokens, present := assignments[expert][slowestGPU]
			if !present || tokens <= 0 {
				continue
			}
			for _, destination := range placement[expert] {
				if destination == slowestGPU {
					continue
				}
				_, destinationHadExpert := assignments[expert][destination]

				sourceActive := len(state.expertLoads[slowestGPU]) - 1
				sourceLoad := state.loads[slowestGPU] - tokens
				sourceCost := math.Max((float64(sourceActive)*m.expertWeightBytes+sourceLoad*m.activationBytes)/m.gpuBandwidth,
					sourceLoad*m.flopsPerAssignment/m.gpuFlops)

				destinationActive := len(state.expertLoads[destination])
				if !destinationHadExpert {
					destinationActive++
				}
				destinationLoad := state.loads[destination] + tokens
				destinationCost := math.Max((float64(destinationActive)*m.expertWeightBytes+destinationLoad*m.activationBytes)/m.gpuBandwidth,
					destinationLoad*m.flopsPerAssignment/m.gpuFlops)

				other := 0.0
				for gpu := 0; gpu < m.numGPUs; gpu++ {
					if gpu == slowestGPU || gpu == destination {
						continue
					}
					if cost := m.traceRouterGPUCost(state, gpu); cost > other {
						other = cost
					}
				}
				gain := base - math.Max(sourceCost, math.Max(destinationCost, other))
				if gain > bestGain {
					bestGain = gain
					bestExpert = expert
					bestDestination = destination
					bestTokens = tokens
					bestDestinationHadExpert = destinationHadExpert
				}
			}
		}
		if bestExpert < 0 {
			break
		}

		delete(assignments[bestExpert], slowestGPU)
		delete(state.expertLoads[slowestGPU], bestExpert)
		state.loads[slowestGPU] -= bestTokens
		assignments[bestExpert][bestDestination] += bestTokens
		if !bestDestinationHadExpert {
			state.expertLoads[bestDestination][bestExpert] = 0
		}
		state.expertLoads[bestDestination][bestExpert] += bestTokens
		state.loads[bestDestination] += bestTokens
	}
	return state
}

func (m *moeSimulator) traceRoute(counts []float64, placement [][]int) *traceRoutingState {
	switch m.router {
	case common.MoERouterConcentrate:
		return m.traceRouteConcentrate(counts, placement)
	case common.MoERouterHeuristic:
		return m.traceRouteHeuristic(counts, placement)
	default:
		return m.traceRouteSplit(counts, placement)
	}
}

func paddedRows(tokens float64, blockRows int) float64 {
	if tokens <= 0 || blockRows <= 0 {
		return tokens
	}
	return math.Ceil(tokens/float64(blockRows)) * float64(blockRows)
}

type traceGPUExecution struct {
	gpu             int
	expertLoads     map[int]float64
	assignments     float64
	memoryBytes     float64
	computeFlops    float64
	memoryDuration  time.Duration
	computeDuration time.Duration
	duration        time.Duration
	vramBytes       float64
}

type traceLayerExecution struct {
	layer          int
	routerStarted  time.Time
	routerDuration time.Duration
	dispatch       time.Duration
	combine        time.Duration
	gpus           []traceGPUExecution
	duration       time.Duration
}

type traceModelExecution struct {
	requestIDs   []int
	layers       []traceLayerExecution
	eplbStarted  time.Time
	eplbDuration time.Duration
	migration    time.Duration
	duration     time.Duration
}

func (m *moeSimulator) traceLayerExecution(state *traceRoutingState, counts []float64) ([]traceGPUExecution, float64) {
	options := traceFidelityFor(m)
	totalAssignments := 0.0
	for _, count := range counts {
		totalAssignments += count
	}
	totalTokens := totalAssignments / float64(m.topK)
	sharedTokensPerGPU := totalTokens / float64(m.numGPUs)

	maxCost := 0.0
	executions := make([]traceGPUExecution, 0, m.numGPUs)
	for gpu := 0; gpu < m.numGPUs; gpu++ {
		memoryBytes := options.sharedWeightBytes + options.sharedActivationBytes*sharedTokensPerGPU
		computeFlops := options.sharedFlopsPerToken * sharedTokensPerGPU
		for _, tokens := range state.expertLoads[gpu] {
			memoryBytes += m.expertWeightBytes + m.activationBytes*tokens
			computeTokens := tokens
			if options.gemmBlockRows > 0 {
				computeTokens = paddedRows(tokens, options.gemmBlockRows)
			}
			computeFlops += m.flopsPerAssignment * computeTokens
		}
		if options.gemmBlockRows > 0 && options.sharedFlopsPerToken > 0 {
			computeFlops -= options.sharedFlopsPerToken * sharedTokensPerGPU
			computeFlops += options.sharedFlopsPerToken * paddedRows(sharedTokensPerGPU, options.gemmBlockRows)
		}
		memorySeconds := memoryBytes / (m.gpuBandwidth * options.memoryEfficiency)
		computeSeconds := computeFlops / (m.gpuFlops * options.computeEfficiency)
		cost := math.Max(memorySeconds, computeSeconds)
		if totalTokens > 0 {
			cost += options.kernelLaunch.Seconds()
		}
		if cost > maxCost {
			maxCost = cost
		}
		expertLoads := make(map[int]float64, len(state.expertLoads[gpu]))
		for expert, tokens := range state.expertLoads[gpu] {
			expertLoads[expert] = tokens
		}
		slotsPerGPU := m.physicalSlots / m.numGPUs
		executions = append(executions, traceGPUExecution{
			gpu: gpu, expertLoads: expertLoads, assignments: state.loads[gpu],
			memoryBytes: memoryBytes, computeFlops: computeFlops,
			memoryDuration:  time.Duration(memorySeconds * float64(time.Second)),
			computeDuration: time.Duration(computeSeconds * float64(time.Second)),
			duration:        time.Duration(cost * float64(time.Second)),
			vramBytes:       float64(slotsPerGPU)*m.expertWeightBytes + options.sharedWeightBytes,
		})
	}
	return executions, maxCost
}

func (m *moeSimulator) traceCommunicationPhaseCost(state *traceRoutingState) float64 {
	if m.numGPUs <= 1 {
		return 0
	}
	totalAssignments := 0.0
	for _, load := range state.loads {
		totalAssignments += load
	}
	if totalAssignments == 0 {
		return 0
	}

	// The v1 trace records destination experts but not the EP source rank of
	// each hidden state. Assume sources are balanced across EP ranks, then
	// model the bottleneck of the uniform sends and policy-dependent receives.
	remoteFraction := 1 - 1/float64(m.numGPUs)
	uniformSend := totalAssignments / float64(m.numGPUs) * remoteFraction
	maxRemote := uniformSend
	for _, destinationLoad := range state.loads {
		remoteReceive := destinationLoad * remoteFraction
		if remoteReceive > maxRemote {
			maxRemote = remoteReceive
		}
	}
	return maxRemote*m.networkBytes/m.interconnectBW + m.interconnectLatency.Seconds()
}

func (m *moeSimulator) traceModelExecutionForLayerCounts(counts moeLayerCounts, requestIDs ...int) traceModelExecution {
	if len(counts) != m.numLayers {
		return traceModelExecution{}
	}
	assignments := 0.0
	for layer := range counts {
		if len(counts[layer]) != m.numExperts {
			return traceModelExecution{}
		}
		for _, count := range counts[layer] {
			assignments += count
		}
	}
	if assignments == 0 {
		return traceModelExecution{}
	}

	options := traceFidelityFor(m)
	m.stateMu.Lock()
	placements := make([][][]int, m.numLayers)
	for layer := 0; layer < m.numLayers; layer++ {
		placements[layer] = clonePlacement(m.placements[layer])
	}
	migrationLatency := time.Duration(0)
	eplbStarted := time.Time{}
	eplbDuration := time.Duration(0)
	if !options.fixedPlacement {
		eplbStarted = time.Now()
		migrationLatency = m.advanceEPLBLayerCounts(counts)
		eplbDuration = time.Since(eplbStarted)
	}
	m.stateMu.Unlock()

	execution := traceModelExecution{
		requestIDs:  append([]int(nil), requestIDs...),
		eplbStarted: eplbStarted, eplbDuration: eplbDuration, migration: migrationLatency,
	}
	totalSeconds := 0.0
	for layer := 0; layer < m.numLayers; layer++ {
		routerStarted := time.Now()
		state := m.traceRoute(counts[layer], placements[layer])
		routerDuration := time.Since(routerStarted)
		gpus, maxCost := m.traceLayerExecution(state, counts[layer])
		phase := m.traceCommunicationPhaseCost(state)
		dispatch := time.Duration(phase * float64(time.Second))
		compute := time.Duration(maxCost * float64(time.Second))
		layerExecution := traceLayerExecution{
			layer: layer, routerStarted: routerStarted, routerDuration: routerDuration, dispatch: dispatch, combine: dispatch,
			gpus: gpus, duration: dispatch + compute + dispatch,
		}
		execution.layers = append(execution.layers, layerExecution)
		totalSeconds += 2*phase + maxCost
	}
	execution.duration = time.Duration(totalSeconds*float64(time.Second)) + migrationLatency
	return execution
}

func (m *moeSimulator) traceModelLatencyForLayerCounts(counts moeLayerCounts) time.Duration {
	return m.traceModelExecutionForLayerCounts(counts).duration
}

func (m *moeSimulator) traceLatencyForLayerCounts(counts moeLayerCounts, requestIDs ...int) time.Duration {
	callStarted := time.Now()
	execution := m.traceModelExecutionForLayerCounts(counts, requestIDs...)
	if execution.duration <= 0 {
		return execution.duration
	}
	options := traceFidelityFor(m)
	start, latency := options.reserveForward(callStarted, execution.duration)
	if options.profiler != nil {
		options.profiler.recordExecution(callStarted, start, execution)
	}
	return latency
}

func (options *traceFidelityConfig) reserveForward(callStarted time.Time, modelLatency time.Duration) (time.Time, time.Duration) {
	options.timelineMu.Lock()
	defer options.timelineMu.Unlock()
	start := callStarted
	if options.nextAvailable.After(start) {
		start = options.nextAvailable
	}
	finish := start.Add(modelLatency)
	options.nextAvailable = finish
	return start, finish.Sub(callStarted)
}

func tracePrefillBatcherFor(runtime *moeTraceRuntime, m *moeSimulator) *tracePrefillBatcher {
	if value, ok := tracePrefillBatchers.Load(runtime); ok {
		return value.(*tracePrefillBatcher)
	}
	options := traceFidelityFor(m)
	batcher := &tracePrefillBatcher{
		maxTokens: options.prefillBatchTokens,
		coalesce:  options.prefillCoalesce,
	}
	actual, _ := tracePrefillBatchers.LoadOrStore(runtime, batcher)
	return actual.(*tracePrefillBatcher)
}

func traceBatchedPrefillLatency(s *SimContext, runtime *moeTraceRuntime, execution *traceExecution,
	cachedPromptTokens int) time.Duration {
	if execution == nil {
		return 0
	}
	inputTokens := len(execution.prompt.data.InputTokenIDs)
	if cachedPromptTokens < 0 {
		cachedPromptTokens = 0
	}
	if cachedPromptTokens >= inputTokens {
		return 0
	}

	job := &tracePrefillJob{
		execution: execution,
		position:  cachedPromptTokens,
		end:       inputTokens,
		done:      make(chan time.Duration, 1),
	}
	batcher := tracePrefillBatcherFor(runtime, s.moe)
	batcher.mu.Lock()
	batcher.pending = append(batcher.pending, job)
	leader := !batcher.running
	if leader {
		batcher.running = true
	}
	batcher.mu.Unlock()
	if leader {
		go batcher.process(s)
	}
	return <-job.done
}

func (b *tracePrefillBatcher) process(s *SimContext) {
	if b.coalesce > 0 {
		time.Sleep(b.coalesce)
	}
	for {
		b.mu.Lock()
		if len(b.pending) == 0 {
			b.running = false
			b.mu.Unlock()
			return
		}
		snapshot := append([]*tracePrefillJob(nil), b.pending...)
		b.mu.Unlock()

		counts := newMoELayerCounts(s.moe.numLayers, s.moe.numExperts)
		budget := b.maxTokens
		if budget < 1 {
			budget = defaultTracePrefillBatchTokens
		}
		completed := make(map[*tracePrefillJob]struct{})
		requestIDs := make([]int, 0, len(snapshot))
		seenRequestIDs := make(map[int]struct{}, len(snapshot))
		for _, job := range snapshot {
			if budget == 0 {
				break
			}
			remaining := job.end - job.position
			if remaining <= 0 {
				completed[job] = struct{}{}
				continue
			}
			take := remaining
			if take > budget {
				take = budget
			}
			addTracePrefillRange(counts, job.execution.prompt.data, job.position, job.position+take, s.moe)
			requestID := job.execution.promptID
			if _, seen := seenRequestIDs[requestID]; !seen {
				seenRequestIDs[requestID] = struct{}{}
				requestIDs = append(requestIDs, requestID)
			}
			job.position += take
			budget -= take
			if job.position >= job.end {
				completed[job] = struct{}{}
			}
		}

		sort.Ints(requestIDs)
		forwardLatency := s.moe.traceLatencyForLayerCounts(counts, requestIDs...)
		for _, job := range snapshot {
			job.elapsed += forwardLatency
		}

		b.mu.Lock()
		remainingJobs := b.pending[:0]
		for _, job := range b.pending {
			if _, done := completed[job]; done {
				job.done <- job.elapsed
				continue
			}
			remainingJobs = append(remainingJobs, job)
		}
		b.pending = remainingJobs
		b.mu.Unlock()
	}
}

func addTracePrefillRange(counts moeLayerCounts, prompt *moetrace.PromptData, start, end int, m *moeSimulator) {
	inputTokens := len(prompt.InputTokenIDs)
	if start < 0 {
		start = 0
	}
	if end > inputTokens {
		end = inputTokens
	}
	for layer := 0; layer < m.numLayers; layer++ {
		for position := start; position < end; position++ {
			base := (layer*inputTokens + position) * m.topK
			for index := 0; index < m.topK; index++ {
				expert := int(prompt.PrefillRoutes[base+index])
				counts[layer][expert]++
			}
		}
	}
}
