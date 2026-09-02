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
	"math"
	"testing"
	"time"
)

func TestClassifyStreamChunk(t *testing.T) {
	tests := []struct {
		name      string
		chunk     string
		wantToken bool
		wantError string
	}{
		{
			name:      "assistant role is not a token",
			chunk:     `{"choices":[{"delta":{"role":"assistant"},"finish_reason":null}]}`,
			wantToken: false,
		},
		{
			name:      "content is a token",
			chunk:     `{"choices":[{"delta":{"content":"hello"},"finish_reason":null}]}`,
			wantToken: true,
		},
		{
			name:      "empty trace text is still a token",
			chunk:     `{"choices":[{"delta":{},"finish_reason":null}]}`,
			wantToken: true,
		},
		{
			name:      "finish chunk is not a token",
			chunk:     `{"choices":[{"delta":{},"finish_reason":"stop"}]}`,
			wantToken: false,
		},
		{
			name:      "usage chunk is not a token",
			chunk:     `{"choices":[],"usage":{"completion_tokens":2}}`,
			wantToken: false,
		},
		{
			name:      "stream error is surfaced",
			chunk:     `{"error":{"message":"boom"}}`,
			wantToken: false,
			wantError: "boom",
		},
	}

	for _, test := range tests {
		t.Run(test.name, func(t *testing.T) {
			gotToken, gotError, err := classifyStreamChunk([]byte(test.chunk))
			if err != nil {
				t.Fatalf("classifyStreamChunk() error = %v", err)
			}
			if gotToken != test.wantToken {
				t.Fatalf("token = %v, want %v", gotToken, test.wantToken)
			}
			if gotError != test.wantError {
				t.Fatalf("stream error = %q, want %q", gotError, test.wantError)
			}
		})
	}
}

func TestDescribeUsesInterpolatedPercentiles(t *testing.T) {
	got := describe([]float64{4, 1, 3, 2})
	if got.Count != 4 {
		t.Fatalf("Count = %d, want 4", got.Count)
	}
	assertClose(t, "mean", got.Mean, 2.5)
	assertClose(t, "median", got.Median, 2.5)
	assertClose(t, "p95", got.P95, 3.85)
	assertClose(t, "p99", got.P99, 3.97)
	assertClose(t, "min", got.Min, 1)
	assertClose(t, "max", got.Max, 4)
}

func TestSummarizeSeparatesPerRequestTPOTFromPooledITL(t *testing.T) {
	ttftA := 10.0
	tpotA := 20.0
	prefillA := 10.0
	decodeA := 40.0
	ttftB := 30.0
	tpotB := 100.0
	prefillB := 30.0
	decodeB := 100.0

	results := []requestResult{
		{
			PromptID: 0, Success: true, StatusCode: 200,
			PromptTokens: 10, ExpectedOutputTokens: 3, OutputTokens: 3,
			LatencySeconds: 0.050, TTFTMilliseconds: &ttftA, TPOTMilliseconds: &tpotA,
			PrefillMilliseconds: &prefillA, DecodeMilliseconds: &decodeA, E2EMilliseconds: 50,
			StartOffsetMS: 1, ITLMilliseconds: []float64{10, 30},
		},
		{
			PromptID: 1, Success: true, StatusCode: 200,
			PromptTokens: 20, ExpectedOutputTokens: 2, OutputTokens: 2,
			LatencySeconds: 0.130, TTFTMilliseconds: &ttftB, TPOTMilliseconds: &tpotB,
			PrefillMilliseconds: &prefillB, DecodeMilliseconds: &decodeB, E2EMilliseconds: 130,
			StartOffsetMS: 2, ITLMilliseconds: []float64{100},
		},
		{
			PromptID: 2, Success: false, StatusCode: 429, Error: "queue full",
			PromptTokens: 50, ExpectedOutputTokens: 9, OutputTokens: 0, StartOffsetMS: 3,
		},
	}

	summary := summarize(
		options{baseURL: "http://127.0.0.1:8000", label: "test"},
		"http://127.0.0.1:8000/v1/chat/completions",
		traceSummary{NumPrompts: 3}, serverConfig{MoERouter: "heuristic"},
		results, 2*time.Second, time.Unix(0, 0).UTC(),
	)
	if summary.Requested != 3 || summary.Successful != 2 || summary.Failed != 1 {
		t.Fatalf("request counts = %d/%d/%d, want 3/2/1", summary.Requested, summary.Successful, summary.Failed)
	}
	if summary.TokenCounts.PromptTokens != 30 || summary.TokenCounts.OutputTokens != 5 || summary.TokenCounts.TotalTokens != 35 {
		t.Fatalf("unexpected token counts: %+v", summary.TokenCounts)
	}
	assertClose(t, "requests/s", summary.Throughput.RequestsPerSecond, 1)
	assertClose(t, "tpot mean", summary.TPOTMilliseconds.Mean, 60)
	assertClose(t, "streaming ITL mean", summary.StreamingITLMilliseconds.Mean, 140.0/3.0)
	if summary.StreamingITLMilliseconds.Count != 3 {
		t.Fatalf("streaming ITL samples = %d, want 3", summary.StreamingITLMilliseconds.Count)
	}
	if len(summary.FailureExamples) != 1 || summary.FailureExamples[0].PromptID != 2 {
		t.Fatalf("unexpected failure examples: %+v", summary.FailureExamples)
	}
}

func TestDescribeEmpty(t *testing.T) {
	got := describe(nil)
	if got != (distribution{}) {
		t.Fatalf("describe(nil) = %+v, want zero distribution", got)
	}
}

func assertClose(t *testing.T, name string, got, want float64) {
	t.Helper()
	if math.Abs(got-want) > 1e-9 {
		t.Fatalf("%s = %.12f, want %.12f", name, got, want)
	}
}
