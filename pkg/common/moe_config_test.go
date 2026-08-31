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

package common

import (
	"time"

	. "github.com/onsi/ginkgo/v2"
	. "github.com/onsi/gomega"
)

var _ = Describe("MoE configuration", func() {
	It("parses expert-parallel routing flags", func() {
		config, err := createSimConfig([]string{
			"cmd", "--model", TestModelName, "--enable-moe",
			"--moe-expert-parallel-size", "4",
			"--moe-num-experts", "8",
			"--moe-physical-expert-slots", "12",
			"--moe-top-k", "2",
			"--moe-num-layers", "6",
			"--moe-router", MoERouterHeuristic,
			"--moe-expert-popularity-alpha", "1.2",
			"--moe-hidden-size", "1024",
			"--moe-intermediate-size", "4096",
			"--moe-bytes-per-element", "2",
			"--moe-gpu-flops", "1000000",
			"--moe-gpu-memory-bandwidth", "2000000",
			"--moe-interconnect-bandwidth", "3000000",
			"--moe-interconnect-latency", "7us",
		})

		Expect(err).NotTo(HaveOccurred())
		Expect(config.EnableMoE).To(BeTrue())
		Expect(config.MoEExpertParallelSize).To(Equal(4))
		Expect(config.MoENumExperts).To(Equal(8))
		Expect(config.MoEPhysicalExpertSlots).To(Equal(12))
		Expect(config.MoETopK).To(Equal(2))
		Expect(config.MoENumLayers).To(Equal(6))
		Expect(config.MoERouter).To(Equal(MoERouterHeuristic))
		Expect(config.MoEExpertPopularityAlpha).To(BeNumerically("==", 1.2))
		Expect(config.MoEHiddenSize).To(Equal(1024))
		Expect(config.MoEIntermediateSize).To(Equal(4096))
		Expect(config.MoEInterconnectLatency).To(Equal(7 * time.Microsecond))
	})

	DescribeTable("rejects invalid enabled MoE configurations",
		func(update func(*Configuration), expected string) {
			config := newConfig()
			config.Model = TestModelName
			config.EnableMoE = true
			update(config)
			Expect(config.validate()).To(MatchError(expected))
		},
		Entry("too few physical slots", func(c *Configuration) {
			c.MoEPhysicalExpertSlots = c.MoENumExperts - 1
		}, "moe physical expert slots cannot be less than the number of logical experts"),
		Entry("too many physical slots", func(c *Configuration) {
			c.MoEPhysicalExpertSlots = c.MoENumExperts*c.MoEExpertParallelSize + 1
		}, "moe physical expert slots cannot exceed experts times expert parallel size"),
		Entry("invalid top-k", func(c *Configuration) {
			c.MoETopK = c.MoENumExperts + 1
		}, "moe top-k must be between 1 and the number of experts"),
		Entry("invalid router", func(c *Configuration) {
			c.MoERouter = "unknown"
		}, "unknown moe router unknown, supported routers are: split, concentrate, and heuristic"),
		Entry("invalid interconnect", func(c *Configuration) {
			c.MoEInterconnectBandwidth = 0
		}, "moe interconnect bandwidth must be positive for expert parallel size greater than 1"),
	)
})
