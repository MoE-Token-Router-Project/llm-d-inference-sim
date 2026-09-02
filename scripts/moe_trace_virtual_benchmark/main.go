// Copyright 2026 The llm-d-inference-sim Authors.
//
// Licensed under the Apache License, Version 2.0 (the "License");
// you may not use this file except in compliance with the License.
// You may obtain a copy of the License at
//
//     http://www.apache.org/licenses/LICENSE-2.0
//
// Unless required by applicable law or agreed to in writing, software
// distributed under the License is distributed on an "AS IS" BASIS,
// WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
// See the License for the specific language governing permissions and
// limitations under the License.

package main

import (
	"encoding/json"
	"flag"
	"fmt"
	"os"
	"strings"
	"time"

	"github.com/llm-d/llm-d-inference-sim/pkg/common"
	"github.com/llm-d/llm-d-inference-sim/pkg/moetrace"
	"github.com/llm-d/llm-d-inference-sim/pkg/simulator"
)

type output struct {
	GPU     string                            `json:"gpu"`
	Results []simulator.MoETraceVirtualResult `json:"results"`
}

func main() {
	var tracePath string
	var placementPath string
	var routers string
	var gpu string
	var tokenBudget int
	var epSize int
	var physicalSlots int
	var hiddenSize int
	var intermediateSize int
	var bytesPerElement int
	var interconnectBandwidth float64
	var interconnectLatency time.Duration
	var jsonOutput bool

	flag.StringVar(&tracePath, "trace", "", "path to .moetrace workload")
	flag.StringVar(&placementPath, "fixed-placement", "", "fixed physical_to_logical JSON used by real vLLM")
	flag.StringVar(&routers, "routers", "split,concentrate,heuristic", "comma-separated routing policies")
	flag.StringVar(&gpu, "gpu", "a100", "hardware preset: a100 or h100")
	flag.IntVar(&tokenBudget, "token-budget", 1024, "maximum tokens in one virtual forward")
	flag.IntVar(&epSize, "ep-size", 8, "expert-parallel GPU count")
	flag.IntVar(&physicalSlots, "physical-slots", 80, "physical expert slots per layer")
	flag.IntVar(&hiddenSize, "hidden-size", 2048, "model hidden size")
	flag.IntVar(&intermediateSize, "intermediate-size", 1408, "routed expert intermediate size")
	flag.IntVar(&bytesPerElement, "bytes-per-element", 2, "weight/activation element width")
	flag.Float64Var(&interconnectBandwidth, "interconnect-bandwidth", 400e9, "per-GPU EP interconnect bandwidth in bytes/s")
	flag.DurationVar(&interconnectLatency, "interconnect-latency", 5*time.Microsecond, "one-way all-to-all phase latency")
	flag.BoolVar(&jsonOutput, "json", false, "emit JSON instead of a table")
	flag.Parse()

	if tracePath == "" {
		fmt.Fprintln(os.Stderr, "--trace is required")
		os.Exit(2)
	}
	if tokenBudget < 1 || epSize < 1 || physicalSlots < 1 {
		fmt.Fprintln(os.Stderr, "token budget, EP size, and physical slots must be positive")
		os.Exit(2)
	}

	reader, err := moetrace.Open(tracePath)
	if err != nil {
		fmt.Fprintf(os.Stderr, "open trace: %v\n", err)
		os.Exit(1)
	}
	metadata := reader.Metadata()
	_ = reader.Close()

	gpuFlops, gpuBandwidth, err := gpuPreset(gpu)
	if err != nil {
		fmt.Fprintln(os.Stderr, err)
		os.Exit(2)
	}

	results := make([]simulator.MoETraceVirtualResult, 0, 3)
	for _, router := range strings.Split(routers, ",") {
		router = strings.TrimSpace(router)
		if router == "" {
			continue
		}
		config := &common.Configuration{
			Model:                    metadata.Model,
			EnableMoE:                true,
			MoEExpertParallelSize:    epSize,
			MoENumExperts:            metadata.NumExperts,
			MoEPhysicalExpertSlots:   physicalSlots,
			MoETopK:                  metadata.TopK,
			MoENumLayers:             len(metadata.SparseLayers),
			MoERouter:                router,
			MoEExpertPopularityAlpha: 0.8,
			MoEHiddenSize:            hiddenSize,
			MoEIntermediateSize:      intermediateSize,
			MoEBytesPerElement:       bytesPerElement,
			MoEGPUFlops:              gpuFlops,
			MoEGPUMemoryBandwidth:    gpuBandwidth,
			MoEInterconnectBandwidth: interconnectBandwidth,
			MoEInterconnectLatency:   interconnectLatency,
		}
		result, err := simulator.RunMoETraceVirtualBenchmark(simulator.MoETraceVirtualOptions{
			TracePath:          tracePath,
			FixedPlacementPath: placementPath,
			Config:             config,
			TokenBudget:        tokenBudget,
		})
		if err != nil {
			fmt.Fprintf(os.Stderr, "%s: %v\n", router, err)
			os.Exit(1)
		}
		results = append(results, result)
	}

	if jsonOutput {
		encoder := json.NewEncoder(os.Stdout)
		encoder.SetIndent("", "  ")
		if err := encoder.Encode(output{GPU: gpu, Results: results}); err != nil {
			fmt.Fprintf(os.Stderr, "encode result: %v\n", err)
			os.Exit(1)
		}
		return
	}

	fmt.Printf("GPU=%s prompts=%d token_budget=%d fixed_placement=%t\n",
		gpu, metadata.NumPrompts, tokenBudget, placementPath != "")
	fmt.Printf("%-12s %12s %12s %12s %8s %8s %12s\n",
		"router", "total_ms", "prefill_ms", "decode_ms", "steps", "mixed", "output_tok/s")
	for _, result := range results {
		fmt.Printf("%-12s %12.3f %12.3f %12.3f %8d %8d %12.1f\n",
			result.Router,
			float64(result.ModeledTime)/float64(time.Millisecond),
			float64(result.ModeledPrefillTime)/float64(time.Millisecond),
			float64(result.ModeledDecodeOnlyTime)/float64(time.Millisecond),
			result.Steps,
			result.MixedSteps,
			result.OutputTokensPerSecond)
	}
}

func gpuPreset(name string) (float64, float64, error) {
	switch strings.ToLower(name) {
	case "a100":
		return 312e12, 2.0e12, nil
	case "h100":
		return 990e12, 3.35e12, nil
	default:
		return 0, 0, fmt.Errorf("unknown --gpu %q; expected a100 or h100", name)
	}
}
