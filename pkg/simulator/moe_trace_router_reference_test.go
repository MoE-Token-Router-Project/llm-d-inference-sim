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
	"math"
	"math/rand"
	"sort"
	"testing"
)

func TestTraceRoutersMatchIntegerReferenceImplementations(t *testing.T) {
	rng := rand.New(rand.NewSource(91))
	for trial := 0; trial < 100; trial++ {
		config := testMoEConfig("split")
		model := newMoESimulator(config)
		placementLoad := make([]float64, model.numExperts)
		counts := make([]float64, model.numExperts)
		for expert := range counts {
			placementLoad[expert] = float64(rng.Intn(1000) + 1)
			counts[expert] = float64(rng.Intn(500))
		}
		placement, _ := dhondtPlaceOneLayer(placementLoad, model.physicalSlots, model.numGPUs)

		assertTraceRoutingStateEqual(t, trial, "split",
			model.traceRouteSplit(counts, placement), referenceTraceSplit(model, counts, placement))
		assertTraceRoutingStateEqual(t, trial, "concentrate",
			model.traceRouteConcentrate(counts, placement), referenceTraceConcentrate(model, counts, placement))
		assertTraceRoutingStateEqual(t, trial, "heuristic",
			model.traceRouteHeuristic(counts, placement), referenceTraceHeuristic(model, counts, placement))
	}
}

func referenceTraceSplit(model *moeSimulator, counts []float64, placement [][]int) *traceRoutingState {
	state := newTraceRoutingState(model.numGPUs)
	for expert, count := range counts {
		for ordinal := 0; ordinal < int(count); ordinal++ {
			gpu := placement[expert][ordinal%len(placement[expert])]
			state.add(gpu, expert, 1)
		}
	}
	return state
}

func referenceTraceConcentrate(model *moeSimulator, counts []float64, placement [][]int) *traceRoutingState {
	state := newTraceRoutingState(model.numGPUs)
	memory := make([]float64, model.numGPUs)
	compute := make([]float64, model.numGPUs)
	order := make([]int, model.numExperts)
	for expert := range order {
		order[expert] = expert
	}
	sort.SliceStable(order, func(i, j int) bool {
		return counts[order[i]] > counts[order[j]]
	})
	for _, expert := range order {
		count := counts[expert]
		if count <= 0 {
			continue
		}
		bestGPU := -1
		bestCost := math.Inf(1)
		for _, gpu := range placement[expert] {
			cost := math.Max(
				(memory[gpu]+model.expertWeightBytes+model.activationBytes*count)/model.gpuBandwidth,
				(compute[gpu]+model.flopsPerAssignment*count)/model.gpuFlops,
			)
			if cost < bestCost {
				bestCost = cost
				bestGPU = gpu
			}
		}
		state.add(bestGPU, expert, count)
		memory[bestGPU] += model.expertWeightBytes + model.activationBytes*count
		compute[bestGPU] += model.flopsPerAssignment * count
	}
	return state
}

func referenceTraceHeuristic(model *moeSimulator, counts []float64, placement [][]int) *traceRoutingState {
	memory := make([]float64, model.numGPUs)
	compute := make([]float64, model.numGPUs)
	gpuLatency := func(gpu int) float64 {
		return math.Max(memory[gpu]/model.gpuBandwidth, compute[gpu]/model.gpuFlops)
	}
	solo := func(expert int) float64 {
		return math.Max(
			(model.expertWeightBytes+model.activationBytes*counts[expert])/model.gpuBandwidth,
			model.flopsPerAssignment*counts[expert]/model.gpuFlops,
		)
	}

	order := make([]int, 0, model.numExperts)
	for expert, count := range counts {
		if count > 0 {
			order = append(order, expert)
		}
	}
	sort.SliceStable(order, func(i, j int) bool {
		return solo(order[i]) > solo(order[j])
	})

	assignments := make(map[int]map[int]float64, len(order))
	weightSeconds := model.expertWeightBytes / model.gpuBandwidth
	for _, expert := range order {
		computeSeconds := model.flopsPerAssignment * counts[expert] / model.gpuFlops
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
			return gpuLatency(candidates[i]) < gpuLatency(candidates[j])
		})
		chunks := make([]int, replicaCount)
		for ordinal := 0; ordinal < int(counts[expert]); ordinal++ {
			chunks[ordinal%replicaCount]++
		}
		assignments[expert] = make(map[int]float64, replicaCount)
		for index, gpu := range candidates[:replicaCount] {
			tokens := float64(chunks[index])
			if tokens == 0 {
				continue
			}
			assignments[expert][gpu] = tokens
			memory[gpu] += model.expertWeightBytes + model.activationBytes*tokens
			compute[gpu] += model.flopsPerAssignment * tokens
		}
	}

	for range 20 {
		slowestGPU := 0
		for gpu := 1; gpu < model.numGPUs; gpu++ {
			if gpuLatency(gpu) > gpuLatency(slowestGPU) {
				slowestGPU = gpu
			}
		}
		base := gpuLatency(slowestGPU)
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
				sourceCost := math.Max(
					(memory[slowestGPU]-(model.expertWeightBytes+model.activationBytes*tokens))/model.gpuBandwidth,
					(compute[slowestGPU]-model.flopsPerAssignment*tokens)/model.gpuFlops,
				)
				extraWeight := model.expertWeightBytes
				if destinationHadExpert {
					extraWeight = 0
				}
				destinationCost := math.Max(
					(memory[destination]+extraWeight+model.activationBytes*tokens)/model.gpuBandwidth,
					(compute[destination]+model.flopsPerAssignment*tokens)/model.gpuFlops,
				)
				other := 0.0
				for gpu := 0; gpu < model.numGPUs; gpu++ {
					if gpu == slowestGPU || gpu == destination {
						continue
					}
					if cost := gpuLatency(gpu); cost > other {
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

		memory[slowestGPU] -= model.expertWeightBytes + model.activationBytes*bestTokens
		compute[slowestGPU] -= model.flopsPerAssignment * bestTokens
		extraWeight := model.expertWeightBytes
		if bestDestinationHadExpert {
			extraWeight = 0
		}
		memory[bestDestination] += extraWeight + model.activationBytes*bestTokens
		compute[bestDestination] += model.flopsPerAssignment * bestTokens
		delete(assignments[bestExpert], slowestGPU)
		assignments[bestExpert][bestDestination] += bestTokens
	}

	state := newTraceRoutingState(model.numGPUs)
	for expert, byGPU := range assignments {
		for gpu, tokens := range byGPU {
			state.add(gpu, expert, tokens)
		}
	}
	return state
}

func assertTraceRoutingStateEqual(t *testing.T, trial int, router string,
	got, want *traceRoutingState) {
	t.Helper()
	if len(got.loads) != len(want.loads) {
		t.Fatalf("trial %d %s GPU count=%d want=%d", trial, router, len(got.loads), len(want.loads))
	}
	for gpu := range got.loads {
		if math.Abs(got.loads[gpu]-want.loads[gpu]) > 1e-9 {
			t.Fatalf("trial %d %s GPU %d load=%f want=%f", trial, router, gpu, got.loads[gpu], want.loads[gpu])
		}
		if len(got.expertLoads[gpu]) != len(want.expertLoads[gpu]) {
			t.Fatalf("trial %d %s GPU %d experts=%v want=%v", trial, router, gpu,
				got.expertLoads[gpu], want.expertLoads[gpu])
		}
		for expert, wantTokens := range want.expertLoads[gpu] {
			if math.Abs(got.expertLoads[gpu][expert]-wantTokens) > 1e-9 {
				t.Fatalf("trial %d %s GPU %d expert %d tokens=%f want=%f", trial, router,
					gpu, expert, got.expertLoads[gpu][expert], wantTokens)
			}
		}
	}
}
