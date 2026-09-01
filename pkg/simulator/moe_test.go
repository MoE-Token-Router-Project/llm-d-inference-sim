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
	"time"

	"github.com/llm-d/llm-d-inference-sim/pkg/common"
	. "github.com/onsi/ginkgo/v2"
	. "github.com/onsi/gomega"
)

func testMoEConfig(router string) *common.Configuration {
	return &common.Configuration{
		EnableMoE:                true,
		MoEExpertParallelSize:    4,
		MoENumExperts:            8,
		MoEPhysicalExpertSlots:   12,
		MoETopK:                  2,
		MoENumLayers:             6,
		MoERouter:                router,
		MoEExpertPopularityAlpha: 0.8,
		MoEHiddenSize:            2048,
		MoEIntermediateSize:      1408,
		MoEBytesPerElement:       2,
		MoEGPUFlops:              312e12,
		MoEGPUMemoryBandwidth:    2e12,
		MoEInterconnectBandwidth: 400e9,
		MoEInterconnectLatency:   5 * time.Microsecond,
	}
}

func activeExpertCopies(state *moeRoutingState) int {
	total := 0
	for _, experts := range state.active {
		total += len(experts)
	}
	return total
}

func countPlacementSlots(placement [][]int, numGPUs int) []int {
	slots := make([]int, numGPUs)
	for _, ranks := range placement {
		for _, rank := range ranks {
			slots[rank]++
		}
	}
	return slots
}

var _ = Describe("MoE expert-parallel simulation", func() {
	It("matches vLLM's initial physical-to-logical placement", func() {
		placement := vllmInitialPlacementOneLayer(8, 12, 4)
		Expect(placement).To(Equal([][]int{
			{0, 2},
			{0, 3},
			{0, 3},
			{1, 3},
			{1},
			{1},
			{2},
			{2},
		}))
		Expect(countPlacementSlots(placement, 4)).To(Equal([]int{3, 3, 3, 3}))
	})

	It("matches the reference D'Hondt allocation and placement", func() {
		placement, replicas := dhondtPlaceOneLayer([]float64{100, 50, 25, 10}, 6, 2)
		Expect(replicas).To(Equal([]int{2, 2, 1, 1}))
		Expect(placement).To(Equal([][]int{
			{0, 1},
			{0, 1},
			{0},
			{1},
		}))
		Expect(countPlacementSlots(placement, 2)).To(Equal([]int{3, 3}))
	})

	It("caps D'Hondt replicas at one copy per GPU and fills every rank equally", func() {
		probabilities := powerLawProbabilities(8, 0.8)
		placement, replicas := dhondtPlaceOneLayer(probabilities, 12, 4)
		total := 0
		for expert, count := range replicas {
			Expect(count).To(BeNumerically(">=", 1))
			Expect(count).To(BeNumerically("<=", 4))
			Expect(placement[expert]).To(HaveLen(count))
			seen := make(map[int]struct{})
			for _, gpu := range placement[expert] {
				_, duplicate := seen[gpu]
				Expect(duplicate).To(BeFalse())
				seen[gpu] = struct{}{}
			}
			total += count
		}
		Expect(total).To(Equal(12))
		Expect(countPlacementSlots(placement, 4)).To(Equal([]int{3, 3, 3, 3}))
	})

	It("starts from vLLM placement and replaces experts from observed load", func() {
		model := newMoESimulator(testMoEConfig(common.MoERouterSplit))
		initial := clonePlacement(model.placements[0])
		Expect(model.expertRearrangementStep).To(Equal(
			vllmEPLBStepInterval - vllmEPLBStepInterval/4,
		))

		// Put the model one step before rearrangement so this test exercises the
		// same record -> increment -> rearrange ordering as vLLM without running
		// thousands of forward passes.
		model.expertRearrangementStep = vllmEPLBStepInterval - 1
		latency := model.latencyForTokens(128)

		Expect(latency).To(BeNumerically(">", 0))
		Expect(model.placementGeneration).To(Equal(uint64(1)))
		Expect(model.placements[0]).NotTo(Equal(initial))
		Expect(model.lastReplacementMoves).To(BeNumerically(">", 0))
		Expect(countPlacementSlots(model.placements[0], model.numGPUs)).To(
			Equal([]int{3, 3, 3, 3}),
		)
	})

	It("keeps cached latency separate across placement generations", func() {
		model := newMoESimulator(testMoEConfig(common.MoERouterSplit))
		model.expertRearrangementStep = vllmEPLBStepInterval - 1
		_ = model.latencyForTokens(64)
		Expect(model.placementGeneration).To(Equal(uint64(1)))
		Expect(model.latencyCache).To(HaveLen(1))

		_ = model.latencyForTokens(64)
		Expect(model.latencyCache).To(HaveLen(2))
	})

	It("keeps split, concentrate, and heuristic replica usage distinct", func() {
		model := newMoESimulator(testMoEConfig(common.MoERouterSplit))
		counts := make([]float64, model.numExperts)
		for expert, probability := range model.probabilities {
			counts[expert] = 64 * float64(model.topK) * probability
		}
		placement := model.placements[0]
		split := model.routeSplit(counts, placement)
		concentrate := model.routeConcentrate(counts, placement)
		heuristic := model.routeHeuristic(counts, placement)

		Expect(activeExpertCopies(split)).To(Equal(12))
		Expect(activeExpertCopies(concentrate)).To(Equal(8))
		Expect(activeExpertCopies(heuristic)).To(BeNumerically(">=", 8))
		Expect(activeExpertCopies(heuristic)).To(BeNumerically("<=", 12))
	})

	It("accounts for interconnect bandwidth", func() {
		fastConfig := testMoEConfig(common.MoERouterSplit)
		slowConfig := testMoEConfig(common.MoERouterSplit)
		slowConfig.MoEInterconnectBandwidth = fastConfig.MoEInterconnectBandwidth / 10

		fast := newMoESimulator(fastConfig).latencyForTokens(128)
		slow := newMoESimulator(slowConfig).latencyForTokens(128)
		Expect(fast).To(BeNumerically(">", 0))
		Expect(slow).To(BeNumerically(">", fast))
	})

	It("uses uncached prompt tokens and skips remote prefill", func() {
		ctx := &SimContext{moe: newMoESimulator(testMoEConfig(common.MoERouterHeuristic))}
		full := ctx.moePrefillLatency(&TTFTParams{PromptTokens: 128})
		cached := ctx.moePrefillLatency(&TTFTParams{PromptTokens: 128, CachedPromptTokens: 96})
		remote := ctx.moePrefillLatency(&TTFTParams{PromptTokens: 128, DoRemotePrefill: true})

		Expect(full).To(BeNumerically(">", cached))
		Expect(cached).To(BeNumerically(">", 0))
		Expect(remote).To(BeZero())
	})

	It("models a shared decode forward from the running request count", func() {
		ctx := &SimContext{moe: newMoESimulator(testMoEConfig(common.MoERouterHeuristic))}
		oneRequest := ctx.moeDecodeLatency(&InterTokenParams{RunningReqs: 1})
		eightRequests := ctx.moeDecodeLatency(&InterTokenParams{RunningReqs: 8})
		zeroObserved := ctx.moeDecodeLatency(&InterTokenParams{RunningReqs: 0})

		Expect(oneRequest).To(BeNumerically(">", 0))
		Expect(eightRequests).To(BeNumerically(">=", oneRequest))
		Expect(zeroObserved).To(Equal(oneRequest))
	})
})
