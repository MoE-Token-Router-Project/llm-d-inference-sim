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

import "fmt"

type promptAccumulator struct {
	meta            PromptMetadata
	numLayers       int
	numExperts      int
	topK            int
	layerSlots      map[int]int
	inputTokenIDs   []uint32
	decodeTokenIDs  []uint32
	inputTokenSeen  []bool
	decodeTokenSeen []bool
	prefillSeen     []bool
	decodeSeen      []bool
	prefillCounts   []uint32
	prefillRoutes   []uint16
	decodeRoutes    []uint16
}

func newPromptAccumulator(meta sourcePromptMetadata, source sourceMetadata, layerSlots map[int]int) (*promptAccumulator, error) {
	p := meta.InputTokens
	d := meta.DecodeTokens
	l := source.NumSparseLayers
	k := source.TopK
	if p < 0 || d < 0 {
		return nil, fmt.Errorf("prompt %d has negative token counts", meta.Index)
	}
	return &promptAccumulator{
		meta: PromptMetadata{
			Index:         meta.Index,
			Prompt:        meta.Prompt,
			InputTokens:   p,
			DecodeTokens:  d,
			GeneratedText: meta.GeneratedText,
		},
		numLayers:       l,
		numExperts:      source.NumExperts,
		topK:            k,
		layerSlots:      layerSlots,
		inputTokenIDs:   make([]uint32, p),
		decodeTokenIDs:  make([]uint32, d),
		inputTokenSeen:  make([]bool, p),
		decodeTokenSeen: make([]bool, d),
		prefillSeen:     make([]bool, p*l),
		decodeSeen:      make([]bool, d*l),
		prefillCounts:   make([]uint32, l*source.NumExperts),
		prefillRoutes:   make([]uint16, l*p*k),
		decodeRoutes:    make([]uint16, d*l*k),
	}, nil
}

func (p *promptAccumulator) expectedRecords() int {
	return (p.meta.InputTokens + p.meta.DecodeTokens) * p.numLayers
}

func (p *promptAccumulator) add(record sourceTraceRecord) error {
	layerSlot, ok := p.layerSlots[record.Layer]
	if !ok {
		return fmt.Errorf("prompt %d uses non-sparse layer %d", p.meta.Index, record.Layer)
	}
	if record.TokenID < 0 || record.TokenID > int64(^uint32(0)) {
		return fmt.Errorf("prompt %d has token_id %d outside uint32 range", p.meta.Index, record.TokenID)
	}
	if len(record.Experts) != p.topK {
		return fmt.Errorf("prompt %d layer %d position %d has %d experts; expected %d", p.meta.Index, record.Layer, record.Position, len(record.Experts), p.topK)
	}
	if len(record.GateWeights) != 0 && len(record.GateWeights) != p.topK {
		return fmt.Errorf("prompt %d layer %d position %d has %d gate weights; expected %d", p.meta.Index, record.Layer, record.Position, len(record.GateWeights), p.topK)
	}
	expertSeen := make(map[int]struct{}, p.topK)
	experts := make([]uint16, p.topK)
	for i, expert := range record.Experts {
		if expert < 0 || expert >= p.numExperts {
			return fmt.Errorf("prompt %d has expert %d outside [0,%d)", p.meta.Index, expert, p.numExperts)
		}
		if _, duplicate := expertSeen[expert]; duplicate {
			return fmt.Errorf("prompt %d layer %d position %d repeats expert %d", p.meta.Index, record.Layer, record.Position, expert)
		}
		expertSeen[expert] = struct{}{}
		experts[i] = uint16(expert)
	}

	tokenID := uint32(record.TokenID)
	switch record.Phase {
	case "prefill":
		if record.Position < 0 || record.Position >= p.meta.InputTokens {
			return fmt.Errorf("prompt %d has invalid prefill position %d", p.meta.Index, record.Position)
		}
		seenIndex := layerSlot*p.meta.InputTokens + record.Position
		if p.prefillSeen[seenIndex] {
			return fmt.Errorf("duplicate prefill route for prompt %d layer %d position %d", p.meta.Index, record.Layer, record.Position)
		}
		p.prefillSeen[seenIndex] = true
		if p.inputTokenSeen[record.Position] && p.inputTokenIDs[record.Position] != tokenID {
			return fmt.Errorf("prompt %d prefill position %d has inconsistent token IDs %d and %d", p.meta.Index, record.Position, p.inputTokenIDs[record.Position], tokenID)
		}
		p.inputTokenSeen[record.Position] = true
		p.inputTokenIDs[record.Position] = tokenID
		routeBase := seenIndex * p.topK
		copy(p.prefillRoutes[routeBase:routeBase+p.topK], experts)
		countBase := layerSlot * p.numExperts
		for _, expert := range record.Experts {
			if p.prefillCounts[countBase+expert] == ^uint32(0) {
				return fmt.Errorf("prompt %d prefill count overflow for expert %d", p.meta.Index, expert)
			}
			p.prefillCounts[countBase+expert]++
		}
	case "decode":
		decodePosition := record.Position - p.meta.InputTokens
		if decodePosition < 0 || decodePosition >= p.meta.DecodeTokens {
			return fmt.Errorf("prompt %d has invalid decode position %d", p.meta.Index, record.Position)
		}
		seenIndex := decodePosition*p.numLayers + layerSlot
		if p.decodeSeen[seenIndex] {
			return fmt.Errorf("duplicate decode route for prompt %d layer %d position %d", p.meta.Index, record.Layer, record.Position)
		}
		p.decodeSeen[seenIndex] = true
		if p.decodeTokenSeen[decodePosition] && p.decodeTokenIDs[decodePosition] != tokenID {
			return fmt.Errorf("prompt %d decode position %d has inconsistent token IDs %d and %d", p.meta.Index, record.Position, p.decodeTokenIDs[decodePosition], tokenID)
		}
		p.decodeTokenSeen[decodePosition] = true
		p.decodeTokenIDs[decodePosition] = tokenID
		routeBase := seenIndex * p.topK
		copy(p.decodeRoutes[routeBase:routeBase+p.topK], experts)
	default:
		return fmt.Errorf("prompt %d has unsupported phase %q", p.meta.Index, record.Phase)
	}
	return nil
}

func (p *promptAccumulator) validateComplete() error {
	for i, seen := range p.inputTokenSeen {
		if !seen {
			return fmt.Errorf("prompt %d is missing prefill token_id for position %d", p.meta.Index, i)
		}
	}
	for i, seen := range p.decodeTokenSeen {
		if !seen {
			return fmt.Errorf("prompt %d is missing decode token_id for position %d", p.meta.Index, i)
		}
	}
	for i, seen := range p.prefillSeen {
		if !seen {
			layer := i / max(1, p.meta.InputTokens)
			position := 0
			if p.meta.InputTokens > 0 {
				position = i % p.meta.InputTokens
			}
			return fmt.Errorf("prompt %d is missing prefill route for sparse layer slot %d position %d", p.meta.Index, layer, position)
		}
	}
	for i, seen := range p.decodeSeen {
		if !seen {
			position := i / p.numLayers
			layer := i % p.numLayers
			return fmt.Errorf("prompt %d is missing decode route for position %d sparse layer slot %d", p.meta.Index, position, layer)
		}
	}
	return nil
}
