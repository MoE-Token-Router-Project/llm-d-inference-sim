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
	"errors"
	"fmt"
	"time"

	"github.com/llm-d/llm-d-inference-sim/pkg/common"
)

const (
	virtualPhasePrefill = iota
	virtualPhaseDecode
	virtualPhaseDone
)

// MoETraceVirtualOptions configures an offline trace-serving run. Unlike the
// HTTP replay path, the virtual run advances only modeled time; host CPU time
// spent calculating a routing policy cannot affect the reported result.
type MoETraceVirtualOptions struct {
	TracePath          string
	FixedPlacementPath string
	Config             *common.Configuration
	TokenBudget        int
}

// MoETraceVirtualResult reports modeled time for the routed MoE stack.
type MoETraceVirtualResult struct {
	Model                  string
	Router                 string
	Requests               int
	PromptTokens           int
	OutputTokens           int
	DecodeForwards         int
	Steps                  int
	PrefillSteps           int
	DecodeOnlySteps        int
	MixedSteps             int
	ModeledTime            time.Duration
	ModeledPrefillTime     time.Duration
	ModeledDecodeOnlyTime  time.Duration
	OutputTokensPerSecond  float64
}

type virtualTraceSequence struct {
	prompt         *moeTracePrompt
	phase          int
	prefillPos     int
	decodePosition int
}

// RunMoETraceVirtualBenchmark replays all prompts using vLLM-style chunked
// serving: every active decode sequence contributes one token first, then the
// remaining token budget is filled with prefill chunks. One aggregate
// [layer][expert] workload is routed exactly once per model forward.
func RunMoETraceVirtualBenchmark(options MoETraceVirtualOptions) (MoETraceVirtualResult, error) {
	if options.Config == nil {
		return MoETraceVirtualResult{}, errors.New("virtual MoE trace benchmark requires a configuration")
	}
	if options.TracePath == "" {
		return MoETraceVirtualResult{}, errors.New("virtual MoE trace benchmark requires a trace path")
	}
	config := *options.Config
	if !config.EnableMoE {
		return MoETraceVirtualResult{}, errors.New("virtual MoE trace benchmark requires MoE simulation")
	}
	if config.MoEPhysicalExpertSlots%config.MoEExpertParallelSize != 0 {
		return MoETraceVirtualResult{}, errors.New("physical expert slots must be divisible by expert parallel size")
	}

	store, err := loadMoETraceStore(options.TracePath, &config)
	if err != nil {
		return MoETraceVirtualResult{}, fmt.Errorf("load MoE trace: %w", err)
	}
	model := newMoESimulator(&config)
	if err := configureTraceFidelity(model, &config, options.FixedPlacementPath); err != nil {
		return MoETraceVirtualResult{}, fmt.Errorf("configure trace fidelity: %w", err)
	}

	budget := options.TokenBudget
	if budget <= 0 {
		budget = traceFidelityFor(model).prefillBatchTokens
	}
	if budget <= 0 {
		budget = defaultTracePrefillBatchTokens
	}

	sequences := make([]virtualTraceSequence, len(store.prompts))
	result := MoETraceVirtualResult{
		Model:    store.model,
		Router:   config.MoERouter,
		Requests: len(sequences),
	}
	for index, prompt := range store.prompts {
		sequence := virtualTraceSequence{prompt: prompt}
		inputTokens := len(prompt.data.InputTokenIDs)
		outputTokens := len(prompt.data.DecodeTokenIDs)
		result.PromptTokens += inputTokens
		result.OutputTokens += outputTokens
		if inputTokens == 0 {
			if outputTokens > 1 {
				sequence.phase = virtualPhaseDecode
			} else {
				sequence.phase = virtualPhaseDone
			}
		} else {
			sequence.phase = virtualPhasePrefill
		}
		sequences[index] = sequence
	}

	for hasVirtualWork(sequences) {
		decodeIndices := make([]int, 0, len(sequences))
		for index := range sequences {
			sequence := &sequences[index]
			if sequence.phase != virtualPhaseDecode {
				continue
			}
			decodeForwards := len(sequence.prompt.data.DecodeTokenIDs) - 1
			if sequence.decodePosition >= decodeForwards {
				sequence.phase = virtualPhaseDone
				continue
			}
			decodeIndices = append(decodeIndices, index)
		}
		if len(decodeIndices) > budget {
			return MoETraceVirtualResult{}, fmt.Errorf(
				"%d active decode sequences exceed token budget %d", len(decodeIndices), budget)
		}

		counts := newMoELayerCounts(model.numLayers, model.numExperts)
		for _, index := range decodeIndices {
			sequence := &sequences[index]
			addTraceDecodePosition(counts, sequence.prompt, sequence.decodePosition, model)
		}
		remainingBudget := budget - len(decodeIndices)
		prefillTokens := 0
		prefillTouched := make([]int, 0, len(sequences))
		for index := range sequences {
			if remainingBudget == 0 {
				break
			}
			sequence := &sequences[index]
			if sequence.phase != virtualPhasePrefill {
				continue
			}
			inputTokens := len(sequence.prompt.data.InputTokenIDs)
			remaining := inputTokens - sequence.prefillPos
			if remaining <= 0 {
				finishVirtualPrefill(sequence)
				continue
			}
			take := remaining
			if take > remainingBudget {
				take = remainingBudget
			}
			addTracePrefillRange(counts, sequence.prompt.data,
				sequence.prefillPos, sequence.prefillPos+take, model)
			sequence.prefillPos += take
			prefillTokens += take
			remainingBudget -= take
			prefillTouched = append(prefillTouched, index)
		}

		forwardTokens := len(decodeIndices) + prefillTokens
		if forwardTokens == 0 {
			return MoETraceVirtualResult{}, errors.New("virtual MoE trace scheduler made no progress")
		}
		modeled := model.traceModelLatencyForLayerCounts(counts)
		result.ModeledTime += modeled
		result.Steps++
		result.DecodeForwards += len(decodeIndices)
		if prefillTokens > 0 && len(decodeIndices) > 0 {
			result.MixedSteps++
			result.ModeledPrefillTime += modeled
		} else if prefillTokens > 0 {
			result.PrefillSteps++
			result.ModeledPrefillTime += modeled
		} else {
			result.DecodeOnlySteps++
			result.ModeledDecodeOnlyTime += modeled
		}

		for _, index := range decodeIndices {
			sequence := &sequences[index]
			sequence.decodePosition++
			if sequence.decodePosition >= len(sequence.prompt.data.DecodeTokenIDs)-1 {
				sequence.phase = virtualPhaseDone
			}
		}
		for _, index := range prefillTouched {
			sequence := &sequences[index]
			if sequence.prefillPos >= len(sequence.prompt.data.InputTokenIDs) {
				finishVirtualPrefill(sequence)
			}
		}
	}

	if result.ModeledTime > 0 {
		result.OutputTokensPerSecond = float64(result.OutputTokens) / result.ModeledTime.Seconds()
	}
	return result, nil
}

func hasVirtualWork(sequences []virtualTraceSequence) bool {
	for index := range sequences {
		if sequences[index].phase != virtualPhaseDone {
			return true
		}
	}
	return false
}

func finishVirtualPrefill(sequence *virtualTraceSequence) {
	if len(sequence.prompt.data.DecodeTokenIDs) > 1 {
		sequence.phase = virtualPhaseDecode
	} else {
		sequence.phase = virtualPhaseDone
	}
}

func addTraceDecodePosition(counts moeLayerCounts, prompt *moeTracePrompt,
	position int, model *moeSimulator) {
	for layer := 0; layer < model.numLayers; layer++ {
		base := (position*model.numLayers + layer) * model.topK
		for index := 0; index < model.topK; index++ {
			expert := int(prompt.data.DecodeRoutes[base+index])
			counts[layer][expert]++
		}
	}
}
