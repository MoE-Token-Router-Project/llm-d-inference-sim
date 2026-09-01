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

package moetrace

import (
	"encoding/json"
	"errors"
	"fmt"
)

type sourcePromptMetadata struct {
	Index         int    `json:"index"`
	Prompt        string `json:"prompt"`
	InputTokens   int    `json:"input_tokens"`
	DecodeTokens  int    `json:"decode_tokens"`
	GeneratedText string `json:"generated_text"`
}

type sourceTraceRecord struct {
	Prompt      int       `json:"prompt"`
	Phase       string    `json:"phase"`
	Position    int       `json:"position"`
	TokenID     int64     `json:"token_id"`
	Layer       int       `json:"layer"`
	Experts     []int     `json:"experts"`
	GateWeights []float64 `json:"gate_weights"`
}

type sourceMetadata struct {
	Model           string
	NumExperts      int
	TopK            int
	SparseLayers    []int
	NumSparseLayers int
	NumPrompts      int
	Prompts         []sourcePromptMetadata
}

func validateSourceMetadata(meta sourceMetadata) error {
	if meta.Model == "" {
		return errors.New("trace model is empty")
	}
	if meta.NumExperts < 1 || meta.NumExperts > 65536 {
		return fmt.Errorf("invalid num_experts %d", meta.NumExperts)
	}
	if meta.TopK < 1 || meta.TopK > meta.NumExperts {
		return fmt.Errorf("invalid top_k %d for %d experts", meta.TopK, meta.NumExperts)
	}
	if meta.NumSparseLayers != len(meta.SparseLayers) || meta.NumSparseLayers < 1 {
		return fmt.Errorf("num_sparse_layers %d does not match sparse_layers length %d", meta.NumSparseLayers, len(meta.SparseLayers))
	}
	layerSeen := make(map[int]struct{}, len(meta.SparseLayers))
	for _, layer := range meta.SparseLayers {
		if layer < 0 {
			return fmt.Errorf("invalid sparse layer %d", layer)
		}
		if _, exists := layerSeen[layer]; exists {
			return fmt.Errorf("duplicate sparse layer %d", layer)
		}
		layerSeen[layer] = struct{}{}
	}
	if meta.NumPrompts != len(meta.Prompts) || meta.NumPrompts < 1 {
		return fmt.Errorf("num_prompts %d does not match prompts length %d", meta.NumPrompts, len(meta.Prompts))
	}
	for i, prompt := range meta.Prompts {
		if prompt.Index != i {
			return fmt.Errorf("prompt index %d at prompts[%d]; prompt indices must be sequential", prompt.Index, i)
		}
		if prompt.InputTokens < 0 || prompt.DecodeTokens < 0 {
			return fmt.Errorf("prompt %d has negative token counts", prompt.Index)
		}
		if uint64(prompt.InputTokens) > uint64(^uint32(0)) || uint64(prompt.DecodeTokens) > uint64(^uint32(0)) {
			return fmt.Errorf("prompt %d token counts exceed uint32 format limits", prompt.Index)
		}
	}
	return nil
}

func convertTraceArray(decoder *json.Decoder, writer *traceWriter, source sourceMetadata, options ConvertOptions) (ConversionSummary, error) {
	token, err := decoder.Token()
	if err != nil {
		return ConversionSummary{}, fmt.Errorf("start trace array: %w", err)
	}
	if delim, ok := token.(json.Delim); !ok || delim != '[' {
		return ConversionSummary{}, errors.New("trace field must be an array")
	}

	layerSlots := make(map[int]int, len(source.SparseLayers))
	for slot, layer := range source.SparseLayers {
		layerSlots[layer] = slot
	}

	summary := ConversionSummary{
		Model:      source.Model,
		NumPrompts: source.NumPrompts,
		NumExperts: source.NumExperts,
		TopK:       source.TopK,
		NumLayers:  source.NumSparseLayers,
	}
	for _, prompt := range source.Prompts {
		summary.PrefillTokens += uint64(prompt.InputTokens)
		summary.DecodeTokens += uint64(prompt.DecodeTokens)
	}

	nextPrompt := 0
	var current *promptAccumulator
	for decoder.More() {
		var record sourceTraceRecord
		if err := decoder.Decode(&record); err != nil {
			return ConversionSummary{}, fmt.Errorf("decode trace record %d: %w", summary.TraceRecords, err)
		}
		if record.Prompt < 0 || record.Prompt >= source.NumPrompts {
			return ConversionSummary{}, fmt.Errorf("trace record %d has invalid prompt %d", summary.TraceRecords, record.Prompt)
		}
		if current == nil || record.Prompt != current.meta.Index {
			if current != nil {
				if record.Prompt < current.meta.Index {
					return ConversionSummary{}, fmt.Errorf("trace records are not grouped by prompt: prompt %d appears after prompt %d", record.Prompt, current.meta.Index)
				}
				if err := writer.writePrompt(current); err != nil {
					return ConversionSummary{}, err
				}
				nextPrompt = current.meta.Index + 1
				current = nil
			}
			for nextPrompt < record.Prompt {
				empty, err := newPromptAccumulator(source.Prompts[nextPrompt], source, layerSlots)
				if err != nil {
					return ConversionSummary{}, err
				}
				if empty.expectedRecords() != 0 {
					return ConversionSummary{}, fmt.Errorf("missing trace records for prompt %d", nextPrompt)
				}
				if err := writer.writePrompt(empty); err != nil {
					return ConversionSummary{}, err
				}
				nextPrompt++
			}
			if record.Prompt != nextPrompt {
				return ConversionSummary{}, fmt.Errorf("trace records are not grouped by prompt: expected prompt %d, got %d", nextPrompt, record.Prompt)
			}
			var err error
			current, err = newPromptAccumulator(source.Prompts[record.Prompt], source, layerSlots)
			if err != nil {
				return ConversionSummary{}, err
			}
		}
		if err := current.add(record); err != nil {
			return ConversionSummary{}, fmt.Errorf("trace record %d: %w", summary.TraceRecords, err)
		}
		summary.TraceRecords++
		if options.Progress != nil && options.ProgressEvery > 0 && summary.TraceRecords%options.ProgressEvery == 0 {
			options.Progress(summary.TraceRecords)
		}
	}
	if _, err := decoder.Token(); err != nil {
		return ConversionSummary{}, fmt.Errorf("finish trace array: %w", err)
	}

	if current != nil {
		if err := writer.writePrompt(current); err != nil {
			return ConversionSummary{}, err
		}
		nextPrompt = current.meta.Index + 1
	}
	for nextPrompt < source.NumPrompts {
		empty, err := newPromptAccumulator(source.Prompts[nextPrompt], source, layerSlots)
		if err != nil {
			return ConversionSummary{}, err
		}
		if empty.expectedRecords() != 0 {
			return ConversionSummary{}, fmt.Errorf("missing trace records for prompt %d", nextPrompt)
		}
		if err := writer.writePrompt(empty); err != nil {
			return ConversionSummary{}, err
		}
		nextPrompt++
	}

	expected := uint64(0)
	for _, prompt := range source.Prompts {
		expected += uint64(prompt.InputTokens+prompt.DecodeTokens) * uint64(source.NumSparseLayers)
	}
	if summary.TraceRecords != expected {
		return ConversionSummary{}, fmt.Errorf("trace contains %d records; expected %d", summary.TraceRecords, expected)
	}
	return summary, nil
}
