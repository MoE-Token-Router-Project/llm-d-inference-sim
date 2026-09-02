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
	"os"
	"path/filepath"
	"testing"

	"github.com/llm-d/llm-d-inference-sim/pkg/common"
)

func TestTraceSplitUsesIntegerRoundRobin(t *testing.T) {
	m := newMoESimulator(testMoEConfig(common.MoERouterSplit))
	placement := make([][]int, m.numExperts)
	placement[0] = []int{0, 1, 2, 3}
	counts := make([]float64, m.numExperts)
	counts[0] = 2

	state := m.traceRouteSplit(counts, placement)
	if got := len(state.expertLoads[0]) + len(state.expertLoads[1]) +
		len(state.expertLoads[2]) + len(state.expertLoads[3]); got != 2 {
		t.Fatalf("active expert copies=%d, want 2", got)
	}
	if state.loads[0] != 1 || state.loads[1] != 1 || state.loads[2] != 0 || state.loads[3] != 0 {
		t.Fatalf("split loads=%v, want [1 1 0 0]", state.loads)
	}
}

func TestTraceHeuristicUsesPythonReplicaCountAndIntegerChunks(t *testing.T) {
	m := newMoESimulator(testMoEConfig(common.MoERouterHeuristic))
	placement := make([][]int, m.numExperts)
	placement[0] = []int{0, 1, 2, 3}
	counts := make([]float64, m.numExperts)

	// The A100 crossover in this test configuration is about 156 assignments.
	// Python round(200/156) selects one replica, whereas ceil would select two.
	counts[0] = 200
	state := m.traceRouteHeuristic(counts, placement)
	active := 0
	for gpu := range state.expertLoads {
		if state.expertLoads[gpu][0] > 0 {
			active++
		}
	}
	if active != 1 {
		t.Fatalf("200 assignments used %d replicas, want 1", active)
	}

	counts[0] = 400
	state = m.traceRouteHeuristic(counts, placement)
	total := 0.0
	active = 0
	for gpu := range state.expertLoads {
		load := state.expertLoads[gpu][0]
		if load == 0 {
			continue
		}
		active++
		if load != float64(int(load)) {
			t.Fatalf("non-integral real-token chunk %f", load)
		}
		total += load
	}
	if active != 3 || total != 400 {
		t.Fatalf("400 assignments produced active=%d total=%f, want active=3 total=400", active, total)
	}
}

func TestFixedPlacementPreservesReplicaOrderAndFreezesEPLB(t *testing.T) {
	config := testMoEConfig(common.MoERouterSplit)
	m := newMoESimulator(config)
	physical := []int{0, 1, 2, 3, 4, 5, 6, 7, 0, 1, 2, 3}
	layers := make([][]int, config.MoENumLayers)
	for layer := range layers {
		layers[layer] = append([]int(nil), physical...)
	}
	data := fixedPlacementFile{
		PhysicalToLogical:  layers,
		NumLayers:          config.MoENumLayers,
		NumPhysicalExperts: config.MoEPhysicalExpertSlots,
	}
	contents, err := json.Marshal(data)
	if err != nil {
		t.Fatal(err)
	}
	path := filepath.Join(t.TempDir(), "placement.json")
	if err := os.WriteFile(path, contents, 0o600); err != nil {
		t.Fatal(err)
	}
	if err := configureTraceFidelity(m, config, path); err != nil {
		t.Fatal(err)
	}
	if got := m.placements[0][0]; len(got) != 2 || got[0] != 0 || got[1] != 2 {
		t.Fatalf("expert 0 replica order=%v, want [0 2]", got)
	}

	counts := newMoELayerCounts(m.numLayers, m.numExperts)
	for layer := range counts {
		counts[layer][0] = 1
	}
	before := m.expertRearrangementStep
	_ = m.traceLatencyForLayerCounts(counts)
	if m.expertRearrangementStep != before {
		t.Fatalf("fixed placement advanced EPLB from %d to %d", before, m.expertRearrangementStep)
	}
}

func TestQwenA100TraceFidelityAddsSharedExpertAndKernelCalibration(t *testing.T) {
	config := testMoEConfig(common.MoERouterSplit)
	config.Model = "Qwen/Qwen1.5-MoE-A2.7B"
	m := newMoESimulator(config)
	if err := configureTraceFidelity(m, config, ""); err != nil {
		t.Fatal(err)
	}
	options := traceFidelityFor(m)
	if options.sharedWeightBytes <= 0 || options.sharedFlopsPerToken <= 0 || options.sharedActivationBytes <= 0 {
		t.Fatalf("Qwen shared expert was not configured: %+v", options)
	}
	if options.memoryEfficiency != 0.75 || options.computeEfficiency != 0.70 ||
		options.gemmBlockRows != 128 || options.kernelLaunch <= 0 {
		t.Fatalf("unexpected A100 calibration: %+v", options)
	}
}

func TestParseStartOptionsAcceptsFixedPlacement(t *testing.T) {
	options, remaining, err := ParseStartOptions([]string{
		"--moe-trace-path", "trace.moetrace",
		"--moe-fixed-placement-path=placement.json",
		"--model", "dummy",
	})
	if err != nil {
		t.Fatal(err)
	}
	if options.MoETracePath != "trace.moetrace" || options.MoEFixedPlacementPath != "placement.json" {
		t.Fatalf("options=%+v", options)
	}
	if len(remaining) != 2 || remaining[0] != "--model" || remaining[1] != "dummy" {
		t.Fatalf("remaining=%v", remaining)
	}
}
