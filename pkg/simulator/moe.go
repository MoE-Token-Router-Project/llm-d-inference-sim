/*
Copyright 2025 The llm-d-inference-sim Authors.

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
	"math"
	"sort"
	"sync"
	"time"

	"github.com/llm-d/llm-d-inference-sim/pkg/common"
)

type moeSimulator struct {
	numGPUs             int
	numExperts          int
	topK                int
	numLayers           int
	router              string
	gpuFlops            float64
	gpuBandwidth        float64
	interconnectBW      float64
	interconnectLatency time.Duration
	expertWeightBytes   float64
	flopsPerAssignment  float64
	activationBytes     float64
	networkBytes        float64
	probabilities       []float64
	replicas            []int
	placements          [][][]int
	cacheMu             sync.RWMutex
	latencyCache        map[int]time.Duration
}

type moeRoutingState struct {
	active []map[int]struct{}
	loads  []float64
}

func newMoESimulator(config *common.Configuration) *moeSimulator {
	m := &moeSimulator{
		numGPUs:             config.MoEExpertParallelSize,
		numExperts:          config.MoENumExperts,
		topK:                config.MoETopK,
		numLayers:           config.MoENumLayers,
		router:              config.MoERouter,
		gpuFlops:            config.MoEGPUFlops,
		gpuBandwidth:        config.MoEGPUMemoryBandwidth,
		interconnectBW:      config.MoEInterconnectBandwidth,
		interconnectLatency: config.MoEInterconnectLatency,
		latencyCache:        make(map[int]time.Duration),
	}

	d := float64(config.MoEHiddenSize)
	dff := float64(config.MoEIntermediateSize)
	bytesPerElement := float64(config.MoEBytesPerElement)
	m.expertWeightBytes = 3 * d * dff * bytesPerElement
	m.flopsPerAssignment = 2 * 3 * d * dff
	m.activationBytes = (2*d + dff) * bytesPerElement
	m.networkBytes = d * bytesPerElement
	m.probabilities = powerLawProbabilities(m.numExperts, config.MoEExpertPopularityAlpha)
	m.replicas = dhondtReplicaCounts(m.probabilities, config.MoEPhysicalExpertSlots)
	m.placements = make([][][]int, m.numLayers)
	for layer := range m.numLayers {
		m.placements[layer] = placeExpertReplicas(m.probabilities, m.replicas, m.numGPUs, layer)
	}
	return m
}

func powerLawProbabilities(numExperts int, alpha float64) []float64 {
	probabilities := make([]float64, numExperts)
	total := 0.0
	for expert := range numExperts {
		weight := math.Pow(float64(expert+1), -alpha)
		probabilities[expert] = weight
		total += weight
	}
	for expert := range probabilities {
		probabilities[expert] /= total
	}
	return probabilities
}

func dhondtReplicaCounts(probabilities []float64, physicalSlots int) []int {
	replicas := make([]int, len(probabilities))
	for expert := range replicas {
		replicas[expert] = 1
	}
	for slot := len(probabilities); slot < physicalSlots; slot++ {
		bestExpert := 0
		bestScore := -1.0
		for expert, probability := range probabilities {
			score := probability / float64(replicas[expert]+1)
			if score > bestScore {
				bestExpert = expert
				bestScore = score
			}
		}
		replicas[bestExpert]++
	}
	return replicas
}

func placeExpertReplicas(probabilities []float64, replicas []int, numGPUs, layer int) [][]int {
	placement := make([][]int, len(probabilities))
	gpuLoad := make([]float64, numGPUs)
	order := expertOrder(probabilities)

	for position, expert := range order {
		gpu := (position + layer) % numGPUs
		placement[expert] = append(placement[expert], gpu)
		gpuLoad[gpu] += probabilities[expert] / float64(replicas[expert])
	}

	extras := make([]int, 0)
	for expert, count := range replicas {
		for replica := 1; replica < count; replica++ {
			extras = append(extras, expert)
		}
	}
	sort.SliceStable(extras, func(i, j int) bool {
		if probabilities[extras[i]] == probabilities[extras[j]] {
			return extras[i] < extras[j]
		}
		return probabilities[extras[i]] > probabilities[extras[j]]
	})

	for _, expert := range extras {
		bestGPU := -1
		bestLoad := math.Inf(1)
		for gpu := range numGPUs {
			if containsGPU(placement[expert], gpu) {
				continue
			}
			if gpuLoad[gpu] < bestLoad {
				bestGPU = gpu
				bestLoad = gpuLoad[gpu]
			}
		}
		if bestGPU < 0 {
			for gpu := range numGPUs {
				if gpuLoad[gpu] < bestLoad {
					bestGPU = gpu
					bestLoad = gpuLoad[gpu]
				}
			}
		}
		placement[expert] = append(placement[expert], bestGPU)
		gpuLoad[bestGPU] += probabilities[expert] / float64(replicas[expert])
	}
	return placement
}

func containsGPU(gpus []int, target int) bool {
	for _, gpu := range gpus {
		if gpu == target {
			return true
		}
	}
	return false
}

func expertOrder(values []float64) []int {
	order := make([]int, len(values))
	for index := range order {
		order[index] = index
	}
	sort.SliceStable(order, func(i, j int) bool {
		if values[order[i]] == values[order[j]] {
			return order[i] < order[j]
		}
		return values[order[i]] > values[order[j]]
	})
	return order
}

func newMoERoutingState(numGPUs int) *moeRoutingState {
	state := &moeRoutingState{
		active: make([]map[int]struct{}, numGPUs),
		loads:  make([]float64, numGPUs),
	}
	for gpu := range numGPUs {
		state.active[gpu] = make(map[int]struct{})
	}
	return state
}

func (s *moeRoutingState) clone() *moeRoutingState {
	clone := newMoERoutingState(len(s.loads))
	copy(clone.loads, s.loads)
	for gpu := range s.active {
		for expert := range s.active[gpu] {
			clone.active[gpu][expert] = struct{}{}
		}
	}
	return clone
}

func (m *moeSimulator) gpuCost(activeExperts int, assignments float64) float64 {
	memorySeconds := (float64(activeExperts)*m.expertWeightBytes + assignments*m.activationBytes) / m.gpuBandwidth
	computeSeconds := assignments * m.flopsPerAssignment / m.gpuFlops
	return math.Max(memorySeconds, computeSeconds)
}

func (m *moeSimulator) maxGPUCost(state *moeRoutingState) float64 {
	maxCost := 0.0
	for gpu := range m.numGPUs {
		cost := m.gpuCost(len(state.active[gpu]), state.loads[gpu])
		if cost > maxCost {
			maxCost = cost
		}
	}
	return maxCost
}

func (m *moeSimulator) routeSplit(counts []float64, placement [][]int) *moeRoutingState {
	state := newMoERoutingState(m.numGPUs)
	for expert, count := range counts {
		if count <= 0 {
			continue
		}
		share := count / float64(len(placement[expert]))
		for _, gpu := range placement[expert] {
			state.active[gpu][expert] = struct{}{}
			state.loads[gpu] += share
		}
	}
	return state
}

func (m *moeSimulator) routeConcentrate(counts []float64, placement [][]int) *moeRoutingState {
	state := newMoERoutingState(m.numGPUs)
	for _, expert := range expertOrder(counts) {
		count := counts[expert]
		if count <= 0 {
			continue
		}
		bestGPU := -1
		bestCost := math.Inf(1)
		for _, gpu := range placement[expert] {
			candidate := state.clone()
			candidate.active[gpu][expert] = struct{}{}
			candidate.loads[gpu] += count
			cost := m.maxGPUCost(candidate)
			if cost < bestCost || (cost == bestCost && gpu < bestGPU) {
				bestGPU = gpu
				bestCost = cost
			}
		}
		state.active[bestGPU][expert] = struct{}{}
		state.loads[bestGPU] += count
	}
	return state
}

func (m *moeSimulator) routeHeuristic(counts []float64, placement [][]int) *moeRoutingState {
	state := newMoERoutingState(m.numGPUs)
	chosen := make(map[int][]int)
	weightSeconds := m.expertWeightBytes / m.gpuBandwidth
	computeSeconds := m.flopsPerAssignment / m.gpuFlops

	for _, expert := range expertOrder(counts) {
		count := counts[expert]
		if count <= 0 {
			continue
		}
		replicaCount := int(math.Ceil(count * computeSeconds / weightSeconds))
		if replicaCount < 1 {
			replicaCount = 1
		}
		if replicaCount > len(placement[expert]) {
			replicaCount = len(placement[expert])
		}
		share := count / float64(replicaCount)
		selected := make([]int, 0, replicaCount)
		for range replicaCount {
			bestGPU := -1
			bestCost := math.Inf(1)
			for _, gpu := range placement[expert] {
				if containsGPU(selected, gpu) {
					continue
				}
				candidate := state.clone()
				candidate.active[gpu][expert] = struct{}{}
				candidate.loads[gpu] += share
				cost := m.maxGPUCost(candidate)
				if cost < bestCost || (cost == bestCost && gpu < bestGPU) {
					bestGPU = gpu
					bestCost = cost
				}
			}
			selected = append(selected, bestGPU)
			state.active[bestGPU][expert] = struct{}{}
			state.loads[bestGPU] += share
		}
		chosen[expert] = selected
	}

	for range 20 {
		before := m.maxGPUCost(state)
		slowGPU := 0
		slowCost := -1.0
		for gpu := range m.numGPUs {
			cost := m.gpuCost(len(state.active[gpu]), state.loads[gpu])
			if cost > slowCost {
				slowGPU = gpu
				slowCost = cost
			}
		}
		improved := false
		activeExperts := make([]int, 0, len(state.active[slowGPU]))
		for expert := range state.active[slowGPU] {
			activeExperts = append(activeExperts, expert)
		}
		sort.Ints(activeExperts)
		for _, expert := range activeExperts {
			selected := chosen[expert]
			if len(selected) != 1 || selected[0] != slowGPU {
				continue
			}
			for _, destination := range placement[expert] {
				if destination == slowGPU {
					continue
				}
				candidate := state.clone()
				delete(candidate.active[slowGPU], expert)
				candidate.loads[slowGPU] -= counts[expert]
				candidate.active[destination][expert] = struct{}{}
				candidate.loads[destination] += counts[expert]
				if m.maxGPUCost(candidate)+1e-15 < before {
					state = candidate
					chosen[expert] = []int{destination}
					improved = true
					break
				}
			}
			if improved {
				break
			}
		}
		if !improved {
			break
		}
	}
	return state
}

func (m *moeSimulator) route(counts []float64, placement [][]int) *moeRoutingState {
	switch m.router {
	case common.MoERouterConcentrate:
		return m.routeConcentrate(counts, placement)
	case common.MoERouterHeuristic:
		return m.routeHeuristic(counts, placement)
	default:
		return m.routeSplit(counts, placement)
	}
}

func (m *moeSimulator) communicationCost(state *moeRoutingState) float64 {
	if m.numGPUs <= 1 {
		return 0
	}
	maxAssignments := 0.0
	for _, load := range state.loads {
		maxAssignments = math.Max(maxAssignments, load)
	}
	if maxAssignments == 0 {
		return 0
	}
	remoteFraction := 1 - 1/float64(m.numGPUs)
	bytesPerPhase := maxAssignments * remoteFraction * m.networkBytes
	phaseSeconds := bytesPerPhase/m.interconnectBW + m.interconnectLatency.Seconds()
	return 2 * phaseSeconds
}

func (m *moeSimulator) latencyForTokens(tokens int) time.Duration {
	if tokens <= 0 {
		return 0
	}
	m.cacheMu.RLock()
	cached, found := m.latencyCache[tokens]
	m.cacheMu.RUnlock()
	if found {
		return cached
	}

	counts := make([]float64, m.numExperts)
	for expert, probability := range m.probabilities {
		counts[expert] = float64(tokens*m.topK) * probability
	}
	totalSeconds := 0.0
	for layer := range m.numLayers {
		state := m.route(counts, m.placements[layer])
		totalSeconds += m.maxGPUCost(state) + m.communicationCost(state)
	}
	latency := time.Duration(totalSeconds * float64(time.Second))

	m.cacheMu.Lock()
	m.latencyCache[tokens] = latency
	m.cacheMu.Unlock()
	return latency
}

func (s *SimContext) moePrefillLatency(params *TTFTParams) time.Duration {
	if s.moe == nil || params.DoRemotePrefill {
		return 0
	}
	tokens := params.PromptTokens - params.CachedPromptTokens
	if tokens < 0 {
		tokens = 0
	}
	return s.moe.latencyForTokens(tokens)
}

func (s *SimContext) moeDecodeLatency(params *InterTokenParams) time.Duration {
	if s.moe == nil {
		return 0
	}
	tokens := int(params.RunningReqs)
	if tokens < 1 {
		tokens = 1
	}
	return s.moe.latencyForTokens(tokens)
}
