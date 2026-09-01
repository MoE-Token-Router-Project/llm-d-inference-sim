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
	"os"
	"path/filepath"
	"strings"
	"testing"
)

type testSource struct {
	Model           string                 `json:"model"`
	NumExperts      int                    `json:"num_experts"`
	TopK            int                    `json:"top_k"`
	SparseLayers    []int                  `json:"sparse_layers"`
	NumSparseLayers int                    `json:"num_sparse_layers"`
	NumPrompts      int                    `json:"num_prompts"`
	Prompts         []sourcePromptMetadata `json:"prompts"`
	Trace           []sourceTraceRecord    `json:"trace"`
}

func TestConvertAndRead(t *testing.T) {
	source := makeTestSource()
	dir := t.TempDir()
	input := filepath.Join(dir, "trace.json")
	output := filepath.Join(dir, "trace.moetrace")
	writeTestSource(t, input, source)

	summary, err := Convert(input, output, ConvertOptions{})
	if err != nil {
		t.Fatalf("Convert() error = %v", err)
	}
	if summary.NumPrompts != 2 || summary.TraceRecords != 12 {
		t.Fatalf("unexpected summary: %+v", summary)
	}

	reader, err := Open(output)
	if err != nil {
		t.Fatalf("Open() error = %v", err)
	}
	defer reader.Close()
	if err := reader.Validate(); err != nil {
		t.Fatalf("Validate() error = %v", err)
	}
	metadata := reader.Metadata()
	if metadata.Model != source.Model || metadata.NumExperts != 4 || metadata.TopK != 2 {
		t.Fatalf("unexpected metadata: %+v", metadata)
	}

	prompt, err := reader.ReadPrompt(0)
	if err != nil {
		t.Fatalf("ReadPrompt() error = %v", err)
	}
	if got, want := prompt.InputTokenIDs, []uint32{100, 101}; !equalUint32(got, want) {
		t.Fatalf("input token IDs = %v, want %v", got, want)
	}
	if got, want := prompt.DecodeTokenIDs, []uint32{200, 201}; !equalUint32(got, want) {
		t.Fatalf("decode token IDs = %v, want %v", got, want)
	}
	experts, err := prompt.PrefillExperts(1, 1)
	if err != nil {
		t.Fatal(err)
	}
	if got, want := experts, []uint16{2, 3}; !equalUint16(got, want) {
		t.Fatalf("prefill experts = %v, want %v", got, want)
	}
	experts, err = prompt.DecodeExperts(1, 0)
	if err != nil {
		t.Fatal(err)
	}
	if got, want := experts, []uint16{1, 2}; !equalUint16(got, want) {
		t.Fatalf("decode experts = %v, want %v", got, want)
	}
	count, err := prompt.PrefillExpertCount(0, 1)
	if err != nil {
		t.Fatal(err)
	}
	if count != 2 {
		t.Fatalf("prefill expert count = %d, want 2", count)
	}
}

func TestConvertRejectsMissingRoute(t *testing.T) {
	source := makeTestSource()
	source.Trace = source.Trace[:len(source.Trace)-1]
	dir := t.TempDir()
	input := filepath.Join(dir, "trace.json")
	output := filepath.Join(dir, "trace.moetrace")
	writeTestSource(t, input, source)

	_, err := Convert(input, output, ConvertOptions{})
	if err == nil || !strings.Contains(err.Error(), "missing decode route") {
		t.Fatalf("Convert() error = %v, want missing decode route", err)
	}
	if _, statErr := os.Stat(output); !os.IsNotExist(statErr) {
		t.Fatalf("partial output should not be published, stat error = %v", statErr)
	}
}

func TestConvertDoesNotReplaceExistingOutput(t *testing.T) {
	source := makeTestSource()
	dir := t.TempDir()
	input := filepath.Join(dir, "trace.json")
	output := filepath.Join(dir, "trace.moetrace")
	writeTestSource(t, input, source)
	if err := os.WriteFile(output, []byte("keep"), 0o644); err != nil {
		t.Fatal(err)
	}

	_, err := Convert(input, output, ConvertOptions{})
	if err == nil || !strings.Contains(err.Error(), "already exists") {
		t.Fatalf("Convert() error = %v, want existing output error", err)
	}
	data, err := os.ReadFile(output)
	if err != nil {
		t.Fatal(err)
	}
	if string(data) != "keep" {
		t.Fatalf("existing output was modified: %q", data)
	}
}

func makeTestSource() testSource {
	prompts := []sourcePromptMetadata{
		{Index: 0, Prompt: "first", InputTokens: 2, DecodeTokens: 2, GeneratedText: "ab"},
		{Index: 1, Prompt: "second", InputTokens: 1, DecodeTokens: 1, GeneratedText: "c"},
	}
	layers := []int{1, 3}
	var records []sourceTraceRecord
	for _, prompt := range prompts {
		for layerSlot, layer := range layers {
			for position := 0; position < prompt.InputTokens; position++ {
				records = append(records, sourceTraceRecord{
					Prompt: prompt.Index, Phase: "prefill", Position: position,
					TokenID: int64(100 + prompt.Index*10 + position), Layer: layer,
					Experts:     []int{(position + layerSlot) % 4, (position + layerSlot + 1) % 4},
					GateWeights: []float64{0.6, 0.4},
				})
			}
		}
		for decode := 0; decode < prompt.DecodeTokens; decode++ {
			for layerSlot, layer := range layers {
				records = append(records, sourceTraceRecord{
					Prompt: prompt.Index, Phase: "decode", Position: prompt.InputTokens + decode,
					TokenID: int64(200 + prompt.Index*10 + decode), Layer: layer,
					Experts:     []int{(decode + layerSlot) % 4, (decode + layerSlot + 1) % 4},
					GateWeights: []float64{0.7, 0.3},
				})
			}
		}
	}
	return testSource{
		Model: "test/moe", NumExperts: 4, TopK: 2, SparseLayers: layers,
		NumSparseLayers: len(layers), NumPrompts: len(prompts), Prompts: prompts, Trace: records,
	}
}

func writeTestSource(t *testing.T, path string, source testSource) {
	t.Helper()
	data, err := json.MarshalIndent(source, "", "  ")
	if err != nil {
		t.Fatal(err)
	}
	if err := os.WriteFile(path, data, 0o644); err != nil {
		t.Fatal(err)
	}
}

func equalUint32(a, b []uint32) bool {
	if len(a) != len(b) {
		return false
	}
	for i := range a {
		if a[i] != b[i] {
			return false
		}
	}
	return true
}

func equalUint16(a, b []uint16) bool {
	if len(a) != len(b) {
		return false
	}
	for i := range a {
		if a[i] != b[i] {
			return false
		}
	}
	return true
}
