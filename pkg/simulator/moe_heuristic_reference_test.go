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

func referenceHeuristicState(m *moeSimulator, counts []float64, placement [][]int) *moeRoutingState {
	mem := make([]float64, m.numGPUs)
	comp := make([]float64, m.numGPUs)
	assignments := make(map[int]map[int]float64)
	gpuLatency := func(g int) float64 { return math.Max(mem[g]/m.gpuBandwidth, comp[g]/m.gpuFlops) }
	solo := func(e int) float64 {
		return math.Max((m.expertWeightBytes+m.activationBytes*counts[e])/m.gpuBandwidth,
			m.flopsPerAssignment*counts[e]/m.gpuFlops)
	}

	order := make([]int, 0, len(counts))
	for e, n := range counts {
		if n > 0 {
			order = append(order, e)
		}
	}
	sort.SliceStable(order, func(i, j int) bool {
		li, lj := solo(order[i]), solo(order[j])
		if li == lj {
			return order[i] < order[j]
		}
		return li > lj
	})

	w := m.expertWeightBytes / m.gpuBandwidth
	for _, e := range order {
		ce := m.flopsPerAssignment * counts[e] / m.gpuFlops
		j := 1
		if ce > w {
			j = int(math.RoundToEven(ce / w))
			if j < 1 {
				j = 1
			}
			if j > len(placement[e]) {
				j = len(placement[e])
			}
		}
		replicas := append([]int(nil), placement[e]...)
		sort.SliceStable(replicas, func(i, j int) bool { return gpuLatency(replicas[i]) < gpuLatency(replicas[j]) })
		share := counts[e] / float64(j)
		assignments[e] = make(map[int]float64)
		for _, g := range replicas[:j] {
			assignments[e][g] += share
			mem[g] += m.expertWeightBytes + m.activationBytes*share
			comp[g] += m.flopsPerAssignment * share
		}
	}

	for range 20 {
		b := 0
		for g := 1; g < m.numGPUs; g++ {
			if gpuLatency(g) > gpuLatency(b) {
				b = g
			}
		}
		base := gpuLatency(b)
		bestGain := 1e-12
		bestE, bestG := -1, -1
		bestTok := 0.0
		bestHad := false
		for _, e := range order {
			tok, ok := assignments[e][b]
			if !ok || tok <= 0 {
				continue
			}
			for _, g2 := range placement[e] {
				if g2 == b {
					continue
				}
				_, had := assignments[e][g2]
				nb := math.Max((mem[b]-(m.expertWeightBytes+m.activationBytes*tok))/m.gpuBandwidth,
					(comp[b]-m.flopsPerAssignment*tok)/m.gpuFlops)
				extraWeight := m.expertWeightBytes
				if had {
					extraWeight = 0
				}
				ng := math.Max((mem[g2]+extraWeight+m.activationBytes*tok)/m.gpuBandwidth,
					(comp[g2]+m.flopsPerAssignment*tok)/m.gpuFlops)
				other := 0.0
				for g := 0; g < m.numGPUs; g++ {
					if g == b || g == g2 {
						continue
					}
					if v := gpuLatency(g); v > other {
						other = v
					}
				}
				gain := base - math.Max(nb, math.Max(ng, other))
				if gain > bestGain {
					bestGain, bestE, bestG, bestTok, bestHad = gain, e, g2, tok, had
				}
			}
		}
		if bestE < 0 {
			break
		}
		mem[b] -= m.expertWeightBytes + m.activationBytes*bestTok
		comp[b] -= m.flopsPerAssignment * bestTok
		extraWeight := m.expertWeightBytes
		if bestHad {
			extraWeight = 0
		}
		mem[bestG] += extraWeight + m.activationBytes*bestTok
		comp[bestG] += m.flopsPerAssignment * bestTok
		delete(assignments[bestE], b)
		assignments[bestE][bestG] += bestTok
	}

	state := newMoERoutingState(m.numGPUs)
	for e, byGPU := range assignments {
		for g, tok := range byGPU {
			if tok <= 0 {
				continue
			}
			state.active[g][e] = struct{}{}
			state.loads[g] += tok
		}
	}
	return state
}

func statesEqual(a, b *moeRoutingState) bool {
	if len(a.loads) != len(b.loads) {
		return false
	}
	for g := range a.loads {
		if math.Abs(a.loads[g]-b.loads[g]) > 1e-9 {
			return false
		}
		if len(a.active[g]) != len(b.active[g]) {
			return false
		}
		for e := range a.active[g] {
			if _, ok := b.active[g][e]; !ok {
				return false
			}
		}
	}
	return true
}

func TestHeuristicMatchesCanonicalReference(t *testing.T) {
	m := newMoESimulator(testMoEConfig("heuristic"))
	placement := m.placements[0]
	rng := rand.New(rand.NewSource(7))
	for trial := 0; trial < 250; trial++ {
		counts := make([]float64, m.numExperts)
		for e := range counts {
			if rng.Intn(5) != 0 {
				counts[e] = float64(rng.Intn(1200))
			}
		}
		got := m.routeHeuristic(counts, placement)
		want := referenceHeuristicState(m, counts, placement)
		if !statesEqual(got, want) {
			t.Fatalf("trial %d differs from canonical r_heur: counts=%v got=%+v want=%+v", trial, counts, got, want)
		}
	}
}

func TestHeuristicReplicaCountUsesPythonRoundToEven(t *testing.T) {
	m := newMoESimulator(testMoEConfig("heuristic"))
	crossover := (m.expertWeightBytes / m.gpuBandwidth) / (m.flopsPerAssignment / m.gpuFlops)
	cases := []struct {
		ratio float64
		want  int
	}{{1.0, 1}, {1.5, 2}, {2.5, 2}, {3.5, 4}, {9.0, 4}}
	for _, tc := range cases {
		got := m.heuristicReplicaCount(tc.ratio*crossover, 4)
		if got != tc.want {
			t.Fatalf("ratio %.1f replicas=%d want=%d", tc.ratio, got, tc.want)
		}
	}
}

func TestHeuristicPreservesPlacementOrderOnLoadTies(t *testing.T) {
	m := newMoESimulator(testMoEConfig("heuristic"))
	counts := make([]float64, m.numExperts)
	counts[0] = 1
	placement := clonePlacement(m.placements[0])
	placement[0] = []int{2, 0, 1}
	state := m.routeHeuristic(counts, placement)
	if _, ok := state.active[2][0]; !ok {
		t.Fatalf("expert 0 was not assigned to first tied replica: %+v", state.active)
	}
	if _, ok := state.active[0][0]; ok {
		t.Fatalf("expert 0 incorrectly broke tie by GPU id: %+v", state.active)
	}
}
