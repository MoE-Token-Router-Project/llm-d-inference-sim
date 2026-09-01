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
	"os"
	"path/filepath"
	"strings"
	"testing"

	"github.com/llm-d/llm-d-inference-sim/pkg/common"
	"github.com/llm-d/llm-d-inference-sim/pkg/moetrace"
)

func TestParseStartOptions(t *testing.T) {
	tests := []struct {
		name      string
		args      []string
		wantPath  string
		wantArgs  []string
		wantError bool
	}{
		{
			name:     "separate value",
			args:     []string{"--model", "test/moe", "--moe-trace-path", "/tmp/a.moetrace", "--port", "9000"},
			wantPath: "/tmp/a.moetrace",
			wantArgs: []string{"--model", "test/moe", "--port", "9000"},
		},
		{
			name:     "equals value",
			args:     []string{"--moe-trace-path=/tmp/a.moetrace", "--model", "test/moe"},
			wantPath: "/tmp/a.moetrace",
			wantArgs: []string{"--model", "test/moe"},
		},
		{
			name:     "argument terminator",
			args:     []string{"--model", "test/moe", "--", "--moe-trace-path", "literal"},
			wantArgs: []string{"--model", "test/moe", "--", "--moe-trace-path", "literal"},
		},
		{
			name:      "duplicate",
			args:      []string{"--moe-trace-path=a", "--moe-trace-path=b"},
			wantError: true,
		},
		{
			name:      "missing value",
			args:      []string{"--moe-trace-path"},
			wantError: true,
		},
	}

	for _, test := range tests {
		t.Run(test.name, func(t *testing.T) {
			options, remaining, err := ParseStartOptions(test.args)
			if test.wantError {
				if err == nil {
					t.Fatal("ParseStartOptions() error = nil, want error")
				}
				return
			}
			if err != nil {
				t.Fatalf("ParseStartOptions() error = %v", err)
			}
			if options.MoETracePath != test.wantPath {
				t.Fatalf("MoETracePath = %q, want %q", options.MoETracePath, test.wantPath)
			}
			if strings.Join(remaining, "\x00") != strings.Join(test.wantArgs, "\x00") {
				t.Fatalf("remaining args = %q, want %q", remaining, test.wantArgs)
			}
		})
	}
}

func TestLoadMoETraceStore(t *testing.T) {
	path, config := writeTinyMoETrace(t)
	store, err := loadMoETraceStore(path, config)
	if err != nil {
		t.Fatalf("loadMoETraceStore() error = %v", err)
	}
	if store.model != "test/moe" || store.numExperts != 4 || store.topK != 2 || store.numLayers != 2 {
		t.Fatalf("unexpected store metadata: %+v", store)
	}
	if len(store.prompts) != 1 {
		t.Fatalf("prompts = %d, want 1", len(store.prompts))
	}
	prompt := store.prompts[0]
	if got := prompt.data.InputTokenIDs; len(got) != 2 || got[0] != 100 || got[1] != 101 {
		t.Fatalf("input token IDs = %v", got)
	}
	if got := prompt.data.DecodeTokenIDs; len(got) != 3 || got[0] != 200 || got[2] != 202 {
		t.Fatalf("decode token IDs = %v", got)
	}
	if got := strings.Join(prompt.responseStrings, ""); got != "abc" {
		t.Fatalf("joined response text = %q, want %q", got, "abc")
	}
	if got := prompt.prefillCounts[0]; !equalFloatSlices(got, []float64{1, 2, 1, 0}) {
		t.Fatalf("layer 0 prefill counts = %v", got)
	}
	if got := prompt.prefillCounts[1]; !equalFloatSlices(got, []float64{1, 0, 1, 2}) {
		t.Fatalf("layer 1 prefill counts = %v", got)
	}
}

func TestBindTraceChatRequestUsesRecordedInputTokens(t *testing.T) {
	path, config := writeTinyMoETrace(t)
	store, err := loadMoETraceStore(path, config)
	if err != nil {
		t.Fatal(err)
	}
	runtime := &moeTraceRuntime{
		store: store,
		active: traceActiveRegistry{
			requests: make(map[string]*activeTraceRequest),
		},
	}
	ctx := &SimContext{
		dataset: &traceAwareDataset{runtime: runtime},
	}
	config.Mode = common.ModeRandom
	config.MaxModelLen = 100
	ctx.SetConfig(config)

	var req ChatCompletionsRequest
	if err := req.Unmarshal([]byte(`{"model":"test/moe","messages":[{"role":"user","content":"ignored"}],"trace_prompt_id":0,"max_completion_tokens":2}`)); err != nil {
		t.Fatal(err)
	}
	req.SetDisplayedModel("test/moe")
	if traceErr := ctx.bindTraceChatRequest(&req); traceErr != nil {
		t.Fatalf("bindTraceChatRequest() error = %v", traceErr)
	}
	if got := req.TokenizedPrompt().Tokens; len(got) != 2 || got[0] != 100 || got[1] != 101 {
		t.Fatalf("bound prompt tokens = %v", got)
	}
	if len(req.TokenizedPrompt().Strings) != 0 {
		t.Fatalf("trace binding unexpectedly tokenized prompt strings: %v", req.TokenizedPrompt().Strings)
	}
	if len(req.Messages) != 1 || req.Messages[0].Content.Raw != "first" {
		t.Fatalf("bound messages = %+v", req.Messages)
	}
	if req.trace == nil || req.trace.outputTokens != 2 {
		t.Fatalf("trace execution = %+v", req.trace)
	}
	if got := req.trace.response.Tokens; len(got) != 2 || got[0] != 200 || got[1] != 201 {
		t.Fatalf("bound response tokens = %v", got)
	}
}

func TestTraceActiveRegistryUsesRecordedDecodeRoutes(t *testing.T) {
	path, config := writeTinyMoETrace(t)
	store, err := loadMoETraceStore(path, config)
	if err != nil {
		t.Fatal(err)
	}
	model := newMoESimulator(config)
	execution := &traceExecution{prompt: store.prompts[0], outputTokens: 3}
	registry := traceActiveRegistry{requests: make(map[string]*activeTraceRequest)}
	registry.activate("req", execution)

	counts, ok := registry.decodeCounts(model, 1)
	if !ok {
		t.Fatal("decodeCounts() returned no trace load")
	}
	if !equalFloatSlices(counts[0], []float64{1, 0, 1, 0}) ||
		!equalFloatSlices(counts[1], []float64{0, 1, 0, 1}) {
		t.Fatalf("decode position 0 counts = %v", counts)
	}

	registry.advance("req")
	counts, ok = registry.decodeCounts(model, 1)
	if !ok {
		t.Fatal("decodeCounts() returned no trace load at position 1")
	}
	if !equalFloatSlices(counts[0], []float64{0, 1, 1, 0}) ||
		!equalFloatSlices(counts[1], []float64{1, 0, 0, 1}) {
		t.Fatalf("decode position 1 counts = %v", counts)
	}

	registry.advance("req")
	if _, ok := registry.decodeCounts(model, 1); ok {
		t.Fatal("final recorded route must not create an extra decode step")
	}
}

func TestTracePrefillCountsUsesOnlyUncachedSuffix(t *testing.T) {
	path, config := writeTinyMoETrace(t)
	store, err := loadMoETraceStore(path, config)
	if err != nil {
		t.Fatal(err)
	}
	ctx := &SimContext{moe: newMoESimulator(config)}
	execution := &traceExecution{prompt: store.prompts[0]}
	counts := ctx.tracePrefillCounts(execution, 1)
	if !equalFloatSlices(counts[0], []float64{0, 1, 1, 0}) ||
		!equalFloatSlices(counts[1], []float64{1, 0, 0, 1}) {
		t.Fatalf("uncached prefill counts = %v", counts)
	}
}

func TestTraceEPLBRecordsLayerSpecificLoads(t *testing.T) {
	config := tinyMoEConfig()
	model := newMoESimulator(config)
	counts := moeLayerCounts{
		{3, 0, 0, 1},
		{0, 2, 2, 0},
	}
	model.recordEPLBLayerCounts(counts)
	loads := model.summedEPLBLogicalLoad()
	if !equalFloatSlices(loads[0], counts[0]) || !equalFloatSlices(loads[1], counts[1]) {
		t.Fatalf("recorded loads = %v, want %v", loads, counts)
	}
}

func TestTraceLatencyDoesNotUseSyntheticTokenCache(t *testing.T) {
	model := newMoESimulator(tinyMoEConfig())
	counts := moeLayerCounts{
		{1, 2, 0, 1},
		{0, 1, 2, 1},
	}
	latency := model.latencyForLayerCounts(counts)
	if latency <= 0 {
		t.Fatalf("latency = %s, want positive", latency)
	}
	if len(model.latencyCache) != 0 {
		t.Fatalf("trace latency populated synthetic token-count cache: %v", model.latencyCache)
	}
}

func TestTracePromptIDZeroUnmarshals(t *testing.T) {
	var req ChatCompletionsRequest
	if err := req.Unmarshal([]byte(`{"model":"test/moe","messages":[{"role":"user","content":"ignored"}],"trace_prompt_id":0}`)); err != nil {
		t.Fatal(err)
	}
	if req.TracePromptID == nil || *req.TracePromptID != 0 {
		t.Fatalf("TracePromptID = %v, want pointer to 0", req.TracePromptID)
	}
}

func TestDistributeTraceTextPreservesCompleteText(t *testing.T) {
	pieces := distributeTraceText("hello world", 7)
	if len(pieces) != 7 {
		t.Fatalf("pieces = %d, want 7", len(pieces))
	}
	if got := strings.Join(pieces, ""); got != "hello world" {
		t.Fatalf("joined pieces = %q", got)
	}
}

func writeTinyMoETrace(t *testing.T) (string, *common.Configuration) {
	t.Helper()
	dir := t.TempDir()
	input := filepath.Join(dir, "trace.json")
	output := filepath.Join(dir, "trace.moetrace")
	const source = `{
  "model": "test/moe",
  "num_experts": 4,
  "top_k": 2,
  "sparse_layers": [1, 3],
  "num_sparse_layers": 2,
  "num_prompts": 1,
  "prompts": [
    {"index": 0, "prompt": "first", "input_tokens": 2, "decode_tokens": 3, "generated_text": "abc"}
  ],
  "trace": [
    {"prompt":0,"phase":"prefill","position":0,"token_id":100,"token":"a","layer":1,"experts":[0,1],"gate_weights":[0.6,0.4]},
    {"prompt":0,"phase":"prefill","position":1,"token_id":101,"token":"b","layer":1,"experts":[1,2],"gate_weights":[0.6,0.4]},
    {"prompt":0,"phase":"prefill","position":0,"token_id":100,"token":"a","layer":3,"experts":[2,3],"gate_weights":[0.6,0.4]},
    {"prompt":0,"phase":"prefill","position":1,"token_id":101,"token":"b","layer":3,"experts":[3,0],"gate_weights":[0.6,0.4]},
    {"prompt":0,"phase":"decode","position":2,"token_id":200,"token":"a","layer":1,"experts":[0,2],"gate_weights":[0.7,0.3]},
    {"prompt":0,"phase":"decode","position":2,"token_id":200,"token":"a","layer":3,"experts":[1,3],"gate_weights":[0.7,0.3]},
    {"prompt":0,"phase":"decode","position":3,"token_id":201,"token":"b","layer":1,"experts":[1,2],"gate_weights":[0.7,0.3]},
    {"prompt":0,"phase":"decode","position":3,"token_id":201,"token":"b","layer":3,"experts":[0,3],"gate_weights":[0.7,0.3]},
    {"prompt":0,"phase":"decode","position":4,"token_id":202,"token":"c","layer":1,"experts":[2,3],"gate_weights":[0.7,0.3]},
    {"prompt":0,"phase":"decode","position":4,"token_id":202,"token":"c","layer":3,"experts":[0,1],"gate_weights":[0.7,0.3]}
  ]
}`
	if err := os.WriteFile(input, []byte(source), 0o644); err != nil {
		t.Fatal(err)
	}
	if _, err := moetrace.Convert(input, output, moetrace.ConvertOptions{}); err != nil {
		t.Fatalf("moetrace.Convert() error = %v", err)
	}
	return output, tinyMoEConfig()
}

func tinyMoEConfig() *common.Configuration {
	config := testMoEConfig(common.MoERouterSplit)
	config.Model = "test/moe"
	config.MoEExpertParallelSize = 2
	config.MoENumExperts = 4
	config.MoEPhysicalExpertSlots = 4
	config.MoETopK = 2
	config.MoENumLayers = 2
	return config
}

func equalFloatSlices(a, b []float64) bool {
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
