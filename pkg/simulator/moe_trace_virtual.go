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
	virtualPhaseWaiting = iota
	virtualPhasePrefill
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
	MaxNumSeqs         int
	Copies             int
}

// MoETraceVirtualResult reports modeled time for the routed MoE stack.
type MoETraceVirtualResult struct {
	Model                 string
	Router                string
	Requests              int
	PromptTokens          int
	OutputTokens          int
	DecodeForwards        int
	Steps                 int
	PrefillSteps          int
	DecodeOnlySteps       int
	MixedSteps            int
	ModeledTime           time.Duration
	ModeledPrefillTime    time.Duration
	ModeledDecodeOnlyTime time.Duration
	OutputTokensPerSecond float64
}

type virtualTraceSequence struct {
	prompt         *moeTracePrompt
	phase          int
	prefillPos     int
	decodePosition int
}

// RunMoETraceVirtualBenchmark replays all prompts using vLLM-style chunked
// serving: admit at most MaxNumSeqs active sequences, let every active decode
// sequence contribute one token first, then fill the remaining token budget
// with prefill chunks. One aggregate [layer][expert] workload is routed exactly
// once per model forward. Copies repeats the complete trace prompt set in
// stable order so a smaller trace can reproduce a larger serving run.
func RunMoETraceVirtualBenchmark(options MoETraceVirtualOptions) (MoETraceVirtualResult, error) {
	if options.Config == nil {
		return MoETraceVirtualResult{}, errors.New("virtual MoE trace benchmark requires a configuration")
	}
	if options.TracePath == "" {
		return MoETraceVirtualResult{}, errors.New("virtual MoE trace benchmark requires a trace path")
	}
	config := *options.Config
	if err := validateVirtualMoEConfig(&config); err != nil {
		return MoETraceVirtualResult{}, err
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
	maxNumSeqs := options.MaxNumSeqs
	if maxNumSeqs <= 0 {
		maxNumSeqs = 32
	}
	if maxNumSeqs > budget {
		return MoETraceVirtualResult{}, fmt.Errorf(
			"max-num-seqs %d exceeds token budget %d; a decode-only forward could not fit", maxNumSeqs, budget)
	}
	copies := options.Copies
	if copies <= 0 {
		copies = 1
	}

	sequences := make([]virtualTraceSequence, 0, len(store.prompts)*copies)
	result := MoETraceVirtualResult{
		Model:  store.model,
		Router: config.MoERouter,
	}
	for copyIndex := 0; copyIndex < copies; copyIndex++ {
		for _, prompt := range store.prompts {
			sequences = append(sequences, virtualTraceSequence{
				prompt: prompt,
				phase:  virtualPhaseWaiting,
			})
			result.PromptTokens += len(prompt.data.InputTokenIDs)
			result.OutputTokens += len(prompt.data.DecodeTokenIDs)
		}
	}
	result.Requests = len(sequences)

	for hasVirtualWork(sequences) {
		admitVirtualSequences(sequences, maxNumSeqs)

		decodeIndices := make([]int, 0, maxNumSeqs)
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
		prefillTouched := make([]int, 0, maxNumSeqs)
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
			// Completing a zero-work sequence above may have freed a sequence slot.
			// Admit again before declaring the scheduler stuck.
			before := activeVirtualSequences(sequences)
			admitVirtualSequences(sequences, maxNumSeqs)
			if activeVirtualSequences(sequences) != before {
				continue
			}
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

func validateVirtualMoEConfig(config *common.Configuration) error {
	if !config.EnableMoE {
		return errors.New("virtual MoE trace benchmark requires MoE simulation")
	}
	if config.MoEExpertParallelSize < 1 || config.MoENumExperts < 1 || config.MoENumLayers < 1 {
		return errors.New("EP size, expert count, and MoE layer count must be positive")
	}
	if config.MoEPhysicalExpertSlots < config.MoENumExperts ||
		config.MoEPhysicalExpertSlots > config.MoENumExperts*config.MoEExpertParallelSize {
		return errors.New("physical expert slots are outside the valid replica range")
	}
	if config.MoEPhysicalExpertSlots%config.MoEExpertParallelSize != 0 {
		return errors.New("physical expert slots must be divisible by expert parallel size")
	}
	if config.MoETopK < 1 || config.MoETopK > config.MoENumExperts {
		return errors.New("top-k must be between one and the logical expert count")
	}
	if config.MoERouter != common.MoERouterSplit &&
		config.MoERouter != common.MoERouterConcentrate &&
		config.MoERouter != common.MoERouterHeuristic {
		return fmt.Errorf("unknown MoE router %q", config.MoERouter)
	}
	if config.MoEHiddenSize < 1 || config.MoEIntermediateSize < 1 || config.MoEBytesPerElement < 1 {
		return errors.New("MoE dimensions and bytes per element must be positive")
	}
	if config.MoEGPUFlops <= 0 || config.MoEGPUMemoryBandwidth <= 0 {
		return errors.New("GPU FLOPS and memory bandwidth must be positive")
	}
	if config.MoEExpertParallelSize > 1 && config.MoEInterconnectBandwidth <= 0 {
		return errors.New("interconnect bandwidth must be positive for multi-GPU EP")
	}
	return nil
}

func admitVirtualSequences(sequences []virtualTraceSequence, maxNumSeqs int) {
	active := activeVirtualSequences(sequences)
	if active >= maxNumSeqs {
		return
	}
	for index := range sequences {
		if active >= maxNumSeqs {
			return
		}
		sequence := &sequences[index]
		if sequence.phase != virtualPhaseWaiting {
			continue
		}
		inputTokens := len(sequence.prompt.data.InputTokenIDs)
		outputTokens := len(sequence.prompt.data.DecodeTokenIDs)
		switch {
		case inputTokens > 0:
			sequence.phase = virtualPhasePrefill
			active++
		case outputTokens > 1:
			sequence.phase = virtualPhaseDecode
			active++
		default:
			sequence.phase = virtualPhaseDone
		}
	}
}

func activeVirtualSequences(sequences []virtualTraceSequence) int {
	active := 0
	for index := range sequences {
		switch sequences[index].phase {
		case virtualPhasePrefill, virtualPhaseDecode:
			active++
		}
	}
	return active
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
