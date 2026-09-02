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

import "testing"

func TestTraceDecodeStepIsSharedAcrossActiveRequests(t *testing.T) {
	path, config := writeTinyMoETrace(t)
	store, err := loadMoETraceStore(path, config)
	if err != nil {
		t.Fatal(err)
	}
	model := newMoESimulator(config)
	execution := &traceExecution{prompt: store.prompts[0], outputTokens: 3}
	registry := traceActiveRegistry{requests: make(map[string]*activeTraceRequest)}
	registry.activate("a", execution)
	registry.activate("b", execution)

	before := model.expertRearrangementStep
	a, ok := registry.acquireDecodeStep("a", model, 2, 0)
	if !ok {
		t.Fatal("request a did not acquire decode step")
	}
	b, ok := registry.acquireDecodeStep("b", model, 2, 0)
	if !ok {
		t.Fatal("request b did not acquire decode step")
	}
	if a.generation != b.generation {
		t.Fatalf("different shared generations: %d vs %d", a.generation, b.generation)
	}
	if a.modeledLatency != b.modeledLatency || !a.finishAt.Equal(b.finishAt) {
		t.Fatalf("participants got different timing: %+v %+v", a, b)
	}
	if model.expertRearrangementStep != before+1 {
		t.Fatalf("EPLB advanced %d steps, want 1", model.expertRearrangementStep-before)
	}

	registry.completeDecodeStep("a", a.generation)
	if registry.current == nil {
		t.Fatal("shared step ended before all participants completed")
	}
	registry.completeDecodeStep("b", b.generation)
	if registry.current != nil {
		t.Fatal("shared step remained after all participants completed")
	}

	registry.mu.Lock()
	posa := registry.requests["a"].decodePosition
	posb := registry.requests["b"].decodePosition
	registry.mu.Unlock()
	if posa != 1 || posb != 1 {
		t.Fatalf("decode positions=%d,%d want 1,1", posa, posb)
	}

	a2, ok := registry.acquireDecodeStep("a", model, 2, 0)
	if !ok {
		t.Fatal("request a did not acquire second decode step")
	}
	b2, ok := registry.acquireDecodeStep("b", model, 2, 0)
	if !ok {
		t.Fatal("request b did not acquire second decode step")
	}
	if a2.generation == a.generation || a2.generation != b2.generation {
		t.Fatalf("bad second generation: first=%d a2=%d b2=%d", a.generation, a2.generation, b2.generation)
	}
	if model.expertRearrangementStep != before+2 {
		t.Fatalf("EPLB total advance=%d want 2", model.expertRearrangementStep-before)
	}
	registry.completeDecodeStep("a", a2.generation)
	registry.completeDecodeStep("b", b2.generation)
}
