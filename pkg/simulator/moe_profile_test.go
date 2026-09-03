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
	"math"
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

func TestMoEProfileRecorderTimelineMatchesExecution(t *testing.T) {
	path := filepath.Join(t.TempDir(), "profile.trace.json")
	recorder, err := newMoEProfileRecorder(path)
	if err != nil {
		t.Fatal(err)
	}
	base := recorder.origin.Add(time.Millisecond)
	execution := traceModelExecution{
		layers: []traceLayerExecution{
			{
				layer: 0, dispatch: 10 * time.Microsecond, combine: 8 * time.Microsecond,
				gpus: []traceGPUExecution{
					{gpu: 0, duration: 30 * time.Microsecond, memoryDuration: 20 * time.Microsecond, computeDuration: 30 * time.Microsecond},
					{gpu: 1, duration: 20 * time.Microsecond, memoryDuration: 20 * time.Microsecond, computeDuration: 10 * time.Microsecond},
				},
				duration: 48 * time.Microsecond,
			},
			{
				layer: 1, dispatch: 6 * time.Microsecond, combine: 4 * time.Microsecond,
				gpus: []traceGPUExecution{
					{gpu: 0, duration: 12 * time.Microsecond, memoryDuration: 12 * time.Microsecond, computeDuration: 8 * time.Microsecond},
					{gpu: 1, duration: 15 * time.Microsecond, memoryDuration: 10 * time.Microsecond, computeDuration: 15 * time.Microsecond},
				},
				duration: 25 * time.Microsecond,
			},
		},
		duration: 73 * time.Microsecond,
	}
	recorder.recordExecution(base, base, execution)

	find := func(name, cat string, layer int, tid int) chromeTraceEvent {
		t.Helper()
		for _, event := range recorder.events {
			if event.Name != name || event.Cat != cat || event.Tid != tid {
				continue
			}
			if got, ok := event.Args["layer"]; ok && int(got.(int)) == layer {
				return event
			}
		}
		t.Fatalf("missing %s/%s layer=%d tid=%d", name, cat, layer, tid)
		return chromeTraceEvent{}
	}

	startUS := durationMicros(base.Sub(recorder.origin))
	for layer, want := range []struct {
		offset, dispatch, gpuMax, combine time.Duration
	}{{0, 10 * time.Microsecond, 30 * time.Microsecond, 8 * time.Microsecond}, {48 * time.Microsecond, 6 * time.Microsecond, 15 * time.Microsecond, 4 * time.Microsecond}} {
		route := find("Route layer", "cpu.router", layer, profileTIDRouter)
		if route.Ts != startUS+durationMicros(want.offset) {
			t.Fatalf("layer %d route ts=%v", layer, route.Ts)
		}
		dispatch := find("Expert dispatch", "network.dispatch", layer, profileTIDDispatch)
		if dispatch.Ts != route.Ts || dispatch.Dur != durationMicros(want.dispatch) {
			t.Fatalf("layer %d dispatch got ts=%v dur=%v", layer, dispatch.Ts, dispatch.Dur)
		}
		for gpu := 0; gpu < 2; gpu++ {
			mlp := find("MoE MLP", "gpu.mlp", layer, profileTIDGPUBase+gpu*10)
			if mlp.Ts != dispatch.Ts+dispatch.Dur {
				t.Fatalf("layer %d gpu %d starts at %v, dispatch ends at %v", layer, gpu, mlp.Ts, dispatch.Ts+dispatch.Dur)
			}
		}
		combine := find("Expert combine", "network.combine", layer, profileTIDCombine)
		if combine.Ts != dispatch.Ts+dispatch.Dur+durationMicros(want.gpuMax) || combine.Dur != durationMicros(want.combine) {
			t.Fatalf("layer %d combine got ts=%v dur=%v", layer, combine.Ts, combine.Dur)
		}
	}
}

func TestMoEProfileFlowEndpointsFallInsideIntendedSlices(t *testing.T) {
	path := filepath.Join(t.TempDir(), "profile.trace.json")
	recorder, err := newMoEProfileRecorder(path)
	if err != nil {
		t.Fatal(err)
	}
	base := recorder.origin.Add(time.Millisecond)
	recorder.recordExecution(base, base, traceModelExecution{layers: []traceLayerExecution{{
		layer: 0, dispatch: 10 * time.Microsecond, combine: 8 * time.Microsecond,
		gpus: []traceGPUExecution{{gpu: 0, duration: 30 * time.Microsecond}, {gpu: 1, duration: 20 * time.Microsecond}},
	}}})

	slices := make([]chromeTraceEvent, 0)
	for _, event := range recorder.events {
		if event.Ph == "X" && event.Pid == profilePIDSimulated && event.Dur > 0 {
			slices = append(slices, event)
		}
	}
	contains := func(tid int, ts float64) bool {
		for _, slice := range slices {
			if slice.Tid == tid && ts > slice.Ts && ts < slice.Ts+slice.Dur {
				return true
			}
		}
		return false
	}
	for _, event := range recorder.events {
		if event.Cat != "flow" {
			continue
		}
		if event.Name == "Route to dispatch" && event.Ph == "s" {
			continue // source is the logical router instant
		}
		if !contains(event.Tid, event.Ts) {
			t.Fatalf("flow %s phase=%s tid=%d ts=%v is not strictly inside a target/source slice", event.Name, event.Ph, event.Tid, event.Ts)
		}
	}
}

func TestMoEProfileOriginIsStableForOutOfOrderRecordings(t *testing.T) {
	path := filepath.Join(t.TempDir(), "profile.trace.json")
	recorder, err := newMoEProfileRecorder(path)
	if err != nil {
		t.Fatal(err)
	}
	origin := recorder.origin
	later := origin.Add(2 * time.Millisecond)
	earlier := origin.Add(time.Millisecond)
	execution := traceModelExecution{layers: []traceLayerExecution{{layer: 0, dispatch: time.Microsecond, combine: time.Microsecond, gpus: []traceGPUExecution{{gpu: 0, duration: time.Microsecond}}}}}
	recorder.recordExecution(later, later, execution)
	recorder.recordExecution(earlier, earlier, execution)
	if !recorder.origin.Equal(origin) {
		t.Fatalf("origin changed from %v to %v", origin, recorder.origin)
	}
	for _, event := range recorder.events {
		if event.Ph != "M" && event.Ts < 0 {
			t.Fatalf("negative timestamp after out-of-order recording: %+v", event)
		}
	}
}

func TestMoEProfileSkipsFlowsWhenCorrespondingSliceIsAbsent(t *testing.T) {
	path := filepath.Join(t.TempDir(), "profile.trace.json")
	recorder, err := newMoEProfileRecorder(path)
	if err != nil {
		t.Fatal(err)
	}
	base := recorder.origin.Add(time.Millisecond)
	recorder.recordExecution(base, base, traceModelExecution{layers: []traceLayerExecution{{
		layer: 0,
		gpus:  []traceGPUExecution{{gpu: 0, duration: 10 * time.Microsecond}, {gpu: 1}},
	}}})
	for _, event := range recorder.events {
		if event.Cat == "flow" {
			t.Fatalf("unexpected flow without dispatch/combine slices: %+v", event)
		}
	}
}

func TestMoEProfileGPUCountersMatchRecordedExecution(t *testing.T) {
	path := filepath.Join(t.TempDir(), "profile.trace.json")
	recorder, err := newMoEProfileRecorder(path)
	if err != nil {
		t.Fatal(err)
	}
	base := recorder.origin.Add(time.Millisecond)
	gpu := traceGPUExecution{
		gpu: 0, duration: 40 * time.Microsecond,
		computeDuration: 20 * time.Microsecond, memoryDuration: 30 * time.Microsecond,
		memoryBytes: 80e6, vramBytes: 24e9,
		expertLoads: map[int]float64{2: 8, 7: 4}, assignments: 12,
	}
	recorder.recordExecution(base, base, traceModelExecution{layers: []traceLayerExecution{{layer: 0, gpus: []traceGPUExecution{gpu}}}})

	wantAtStart := map[string]float64{
		"Compute utilization": 50,
		"HBM utilization":     75,
		"HBM GB/s":            2000,
		"VRAM bytes":          24e9,
	}
	startUS := durationMicros(base.Sub(recorder.origin))
	for name, want := range wantAtStart {
		found := false
		for _, event := range recorder.events {
			if event.Name == name && event.Ph == "C" && event.Ts == startUS {
				found = true
				if got := event.Args["value"].(float64); math.Abs(got-want) > 1e-9*math.Max(1, math.Abs(want)) {
					t.Fatalf("%s value=%v, want %v", name, got, want)
				}
			}
		}
		if !found {
			t.Fatalf("missing %s counter", name)
		}
	}
}
