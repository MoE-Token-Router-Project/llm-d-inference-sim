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
	"sort"
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
		requestIDs: []int{42},
		phase:      traceProfilePhasePrefill,
		layers: []traceLayerExecution{{
			layer:          3,
			routerStarted:  now,
			routerDuration: 12 * time.Microsecond,
			dispatch:       4 * time.Microsecond,
			combine:        5 * time.Microsecond,
			duration:       49 * time.Microsecond,
			gpus: []traceGPUExecution{{
				gpu: 0, expertLoads: map[int]float64{2: 8, 7: 4}, assignments: 12,
				tokenAssignments: []traceProfileTokenAssignment{{
					RequestID: 42, Phase: traceProfilePhasePrefill, TokenPosition: 5, MoELayer: 3, ExpertID: 7,
				}},
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
	assertProfileRequestID(t, trace.TraceEvents, "MoE MLP", 42)
	assertProfileArg(t, trace.TraceEvents, "MoE MLP", "phase", traceProfilePhasePrefill)
	assertProfileArg(t, trace.TraceEvents, "MoE MLP", "moe_layer", float64(3))
	assertProfileTokenAssignment(t, trace.TraceEvents, "MoE MLP", traceProfileTokenAssignment{
		RequestID: 42, Phase: traceProfilePhasePrefill, TokenPosition: 5, MoELayer: 3, ExpertID: 7,
	})
	assertProfileEvent(t, trace.TraceEvents, "Compute utilization", "C")
	assertProfileEvent(t, trace.TraceEvents, "HBM utilization", "C")
	assertProfileEvent(t, trace.TraceEvents, "VRAM bytes", "C")
	assertProfileEventForPID(t, trace.TraceEvents, "Route layer", "X", profilePIDSimulated)
	assertNoProfileEventForPID(t, trace.TraceEvents, "Route layer", profilePIDHost)
	assertProfileEventForPID(t, trace.TraceEvents, "EPLB update", "X", profilePIDSimulated)
	assertFlowEndsBindToEnclosingSlice(t, trace.TraceEvents)
}

func assertProfileArg(t *testing.T, events []chromeTraceEvent, name, key string, want any) {
	t.Helper()
	for _, event := range events {
		if event.Name == name && event.Args[key] == want {
			return
		}
	}
	t.Fatalf("profile event %q did not contain %s=%v", name, key, want)
}

func assertProfileTokenAssignment(t *testing.T, events []chromeTraceEvent, name string, want traceProfileTokenAssignment) {
	t.Helper()
	for _, event := range events {
		if event.Name != name {
			continue
		}
		items, ok := event.Args["token_assignments"].([]any)
		if !ok {
			continue
		}
		for _, item := range items {
			assignment, ok := item.(map[string]any)
			if !ok {
				continue
			}
			if assignment["request_id"] == float64(want.RequestID) &&
				assignment["phase"] == want.Phase &&
				assignment["token_position"] == float64(want.TokenPosition) &&
				assignment["moe_layer"] == float64(want.MoELayer) &&
				assignment["expert_id"] == float64(want.ExpertID) {
				return
			}
		}
	}
	t.Fatalf("profile event %q did not contain token assignment %+v", name, want)
}

func assertProfileRequestID(t *testing.T, events []chromeTraceEvent, name string, requestID int) {
	t.Helper()
	for _, event := range events {
		if event.Name != name {
			continue
		}
		if got, ok := event.Args["request_id"]; ok && got == float64(requestID) {
			return
		}
	}
	t.Fatalf("profile event %q did not contain request_id=%d", name, requestID)
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

func assertNoProfileEventForPID(t *testing.T, events []chromeTraceEvent, name string, pid int) {
	t.Helper()
	for _, event := range events {
		if event.Name == name && event.Pid == pid {
			t.Fatalf("profile unexpectedly contained %q on pid %d", name, pid)
		}
	}
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
				layer: 0, routerDuration: 5 * time.Microsecond, dispatch: 10 * time.Microsecond, combine: 8 * time.Microsecond,
				gpus: []traceGPUExecution{
					{gpu: 0, duration: 30 * time.Microsecond, memoryDuration: 20 * time.Microsecond, computeDuration: 30 * time.Microsecond},
					{gpu: 1, duration: 20 * time.Microsecond, memoryDuration: 20 * time.Microsecond, computeDuration: 10 * time.Microsecond},
				},
				duration: 48 * time.Microsecond,
			},
			{
				layer: 1, routerDuration: 7 * time.Microsecond, dispatch: 6 * time.Microsecond, combine: 4 * time.Microsecond,
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
		offset, router, dispatch, gpuMax, combine time.Duration
	}{{0, 5 * time.Microsecond, 10 * time.Microsecond, 30 * time.Microsecond, 8 * time.Microsecond}, {53 * time.Microsecond, 7 * time.Microsecond, 6 * time.Microsecond, 15 * time.Microsecond, 4 * time.Microsecond}} {
		route := find("Route layer", "cpu.router", layer, profileTIDRouter)
		if route.Ts != startUS+durationMicros(want.offset) || route.Dur != durationMicros(want.router) {
			t.Fatalf("layer %d route got ts=%v dur=%v", layer, route.Ts, route.Dur)
		}
		dispatch := find("Expert dispatch", "network.dispatch", layer, profileTIDDispatch)
		if dispatch.Ts != route.Ts+route.Dur || dispatch.Dur != durationMicros(want.dispatch) {
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
		layer: 0, routerDuration: 5 * time.Microsecond, dispatch: 10 * time.Microsecond, combine: 8 * time.Microsecond,
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

func TestMoEProfileFlushOrdersConcurrentExecutionsOnVisualizationTimeline(t *testing.T) {
	path := filepath.Join(t.TempDir(), "profile.trace.json")
	recorder, err := newMoEProfileRecorder(path)
	if err != nil {
		t.Fatal(err)
	}
	earlier := recorder.origin.Add(time.Millisecond)
	later := earlier.Add(3 * time.Microsecond)
	earlierExecution := traceModelExecution{layers: []traceLayerExecution{{
		layer: 0, routerDuration: 5 * time.Microsecond, dispatch: time.Microsecond, combine: time.Microsecond,
		gpus: []traceGPUExecution{{gpu: 0, duration: time.Microsecond}},
	}}, duration: 3 * time.Microsecond}
	laterExecution := traceModelExecution{layers: []traceLayerExecution{{
		layer: 0, routerDuration: 7 * time.Microsecond, dispatch: time.Microsecond, combine: time.Microsecond,
		gpus: []traceGPUExecution{{gpu: 0, duration: time.Microsecond}},
	}}, duration: 3 * time.Microsecond}

	// Record the later modeled request first to reproduce concurrent caller reordering.
	recorder.recordExecution(later, later, laterExecution)
	recorder.recordExecution(earlier, earlier, earlierExecution)
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
	routes := make([]chromeTraceEvent, 0, 2)
	for _, event := range trace.TraceEvents {
		if event.Name == "Route layer" && event.Ph == "X" && event.Pid == profilePIDSimulated {
			routes = append(routes, event)
		}
	}
	if len(routes) != 2 {
		t.Fatalf("got %d simulated router spans, want 2", len(routes))
	}
	sort.Slice(routes, func(i, j int) bool { return routes[i].Ts < routes[j].Ts })
	if routes[0].Dur != 5 || routes[1].Dur != 7 {
		t.Fatalf("router order/durations = [%v, %v], want [5, 7] us", routes[0].Dur, routes[1].Dur)
	}
	firstVisualDuration := 8.0 // 5 us router + 1 us dispatch + 1 us GPU + 1 us combine
	if routes[1].Ts < routes[0].Ts+firstVisualDuration {
		t.Fatalf("second request starts at %v before first visual execution ends at %v", routes[1].Ts, routes[0].Ts+firstVisualDuration)
	}
}
