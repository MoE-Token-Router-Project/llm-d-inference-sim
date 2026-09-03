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
	"encoding/json"
	"os"
	"path/filepath"
	"testing"
	"time"
)

func TestMoEProfileRecorderWritesPerfettoTrace(t *testing.T) {
	path := filepath.Join(t.TempDir(), "profile.trace.json")
	recorder, err := newMoEProfileRecorder(path)
	if err != nil {
		t.Fatal(err)
	}
	now := time.Now()
	recorder.recordExecution(now, now, traceModelExecution{
		layers: []traceLayerExecution{{
			layer:          3,
			routerStarted:  now,
			routerDuration: 12 * time.Microsecond,
			dispatch:       4 * time.Microsecond,
			combine:        5 * time.Microsecond,
			duration:       49 * time.Microsecond,
			gpus: []traceGPUExecution{{
				gpu: 0, expertLoads: map[int]float64{2: 8, 7: 4}, assignments: 12,
				memoryBytes: 64 << 20, computeFlops: 1.2e12,
				memoryDuration: 30 * time.Microsecond, computeDuration: 20 * time.Microsecond,
				duration: 40 * time.Microsecond, vramBytes: 24 << 30,
			}},
		}},
		eplbStarted: now, eplbDuration: 3 * time.Microsecond,
		duration: 49 * time.Microsecond,
	})
	if err := recorder.Flush(); err != nil {
		t.Fatal(err)
	}

	contents, err := os.ReadFile(path)
	if err != nil {
		t.Fatal(err)
	}
	var trace chromeTraceFile
	if err := json.Unmarshal(contents, &trace); err != nil {
		t.Fatal(err)
	}
	if len(trace.TraceEvents) == 0 {
		t.Fatal("profile contained no trace events")
	}
	assertProfileEvent(t, trace.TraceEvents, "MoE MLP", "X")
	assertProfileEvent(t, trace.TraceEvents, "Compute utilization", "C")
	assertProfileEvent(t, trace.TraceEvents, "HBM utilization", "C")
	assertProfileEvent(t, trace.TraceEvents, "VRAM bytes", "C")
	assertProfileEvent(t, trace.TraceEvents, "Route layer", "i")
	assertProfileEventForPID(t, trace.TraceEvents, "Route layer", "X", profilePIDHost)
	assertProfileEventForPID(t, trace.TraceEvents, "EPLB update", "X", profilePIDHost)
	assertFlowEndsBindToEnclosingSlice(t, trace.TraceEvents)
}

func assertProfileEvent(t *testing.T, events []chromeTraceEvent, name, phase string) {
	t.Helper()
	for _, event := range events {
		if event.Name == name && event.Ph == phase {
			return
		}
	}
	t.Fatalf("profile did not contain %q phase %q", name, phase)
}

func assertProfileEventForPID(t *testing.T, events []chromeTraceEvent, name, phase string, pid int) {
	t.Helper()
	for _, event := range events {
		if event.Name == name && event.Ph == phase && event.Pid == pid {
			return
		}
	}
	t.Fatalf("profile did not contain %q phase %q on pid %d", name, phase, pid)
}

func assertFlowEndsBindToEnclosingSlice(t *testing.T, events []chromeTraceEvent) {
	t.Helper()
	found := false
	for _, event := range events {
		if event.Ph != "f" {
			continue
		}
		found = true
		if event.BP != "e" {
			t.Fatalf("flow end %q had bp=%q, want %q", event.Name, event.BP, "e")
		}
	}
	if !found {
		t.Fatal("profile contained no flow-end events")
	}
}
