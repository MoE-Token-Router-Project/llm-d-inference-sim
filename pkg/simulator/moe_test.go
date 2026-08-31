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

var _ = Describe("MoE expert-parallel simulation", func() {
	It("allocates every physical expert slot with D'Hondt", func() {
		probabilities := powerLawProbabilities(8, 0.8)
		replicas := dhondtReplicaCounts(probabilities, 12)
		total := 0
		for _, count := range replicas {
			Expect(count).To(BeNumerically(">=", 1))
			total += count
		}
		Expect(total).To(Equal(12))
		Expect(replicas[0]).To(BeNumerically(">=", replicas[len(replicas)-1]))
	})

	It("places replicas on distinct GPUs for each expert", func() {
		probabilities := powerLawProbabilities(8, 0.8)
		replicas := dhondtReplicaCounts(probabilities, 12)
		placement := placeExpertReplicas(probabilities, replicas, 4, 0)
		for expert, gpus := range placement {
			Expect(gpus).To(HaveLen(replicas[expert]))
			seen := make(map[int]struct{})
			for _, gpu := range gpus {
				Expect(gpu).To(BeNumerically(">=", 0))
				Expect(gpu).To(BeNumerically("<", 4))
				_, duplicate := seen[gpu]
				Expect(duplicate).To(BeFalse())
				seen[gpu] = struct{}{}
			}
		}
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
