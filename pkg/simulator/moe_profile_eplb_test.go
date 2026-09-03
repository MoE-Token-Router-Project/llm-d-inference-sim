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

func TestMoEProfilePlacesEPLBOnSimulatedTimeline(t *testing.T) {
	path := filepath.Join(t.TempDir(), "profile.trace.json")
	recorder, err := newMoEProfileRecorder(path)
	if err != nil {
		t.Fatal(err)
	}
	base := recorder.origin.Add(time.Millisecond)
	recorder.recordExecution(base, base, traceModelExecution{
		eplbDuration: 9 * time.Microsecond,
		layers: []traceLayerExecution{{
			layer: 0, routerDuration: 5 * time.Microsecond,
			dispatch: time.Microsecond, combine: time.Microsecond,
			gpus: []traceGPUExecution{{gpu: 0, duration: time.Microsecond}},
		}},
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

	var eplb, router *chromeTraceEvent
	for i := range trace.TraceEvents {
		event := &trace.TraceEvents[i]
		if event.Name == "process_name" && event.Args["name"] == "Simulator host" {
			t.Fatal("trace still contains Simulator host process")
		}
		if event.Name == "EPLB update" && event.Ph == "X" {
			eplb = event
		}
		if event.Name == "Route layer" && event.Ph == "X" {
			router = event
		}
	}
	if eplb == nil || router == nil {
		t.Fatalf("missing EPLB or router span: eplb=%v router=%v", eplb, router)
	}
	if eplb.Pid != profilePIDSimulated || eplb.Tid != profileTIDEPLB || eplb.Cat != "cpu.eplb" {
		t.Fatalf("EPLB is not on simulated CPU track: %+v", *eplb)
	}
	if eplb.Ts+eplb.Dur != router.Ts {
		t.Fatalf("router starts at %v, EPLB ends at %v", router.Ts, eplb.Ts+eplb.Dur)
	}
	if got := eplb.Args["timing_source"]; got != "simulator_host_cpu" {
		t.Fatalf("EPLB timing source=%v", got)
	}
}
