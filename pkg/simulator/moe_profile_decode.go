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
	"sort"
	"strconv"
	"strings"
)

// MarshalJSON adds decode token details after the complete profile timeline has
// been rebuilt. Prefill forwards already carry exact token assignments from the
// prefill batcher. Decode forwards can be reconstructed from their request IDs,
// forward order, and the source trace retained by the trace runtime.
func (f chromeTraceFile) MarshalJSON() ([]byte, error) {
	type traceFileAlias chromeTraceFile
	events := append([]chromeTraceEvent(nil), f.TraceEvents...)
	runtime := profileTraceRuntime()
	if runtime != nil && runtime.store != nil {
		enrichDecodeProfileEvents(events, runtime)
	}
	return json.Marshal(traceFileAlias{TraceEvents: events, DisplayTimeUnit: f.DisplayTimeUnit})
}

func profileTraceRuntime() *moeTraceRuntime {
	var runtime *moeTraceRuntime
	tracePrefillBatchers.Range(func(key, _ any) bool {
		candidate, ok := key.(*moeTraceRuntime)
		if ok && candidate != nil {
			runtime = candidate
			return false
		}
		return true
	})
	return runtime
}

type decodeProfileState struct {
	nextPosition map[int]int
	current      map[int]int
	remaining    map[string][]traceProfileTokenAssignment
}

func enrichDecodeProfileEvents(events []chromeTraceEvent, runtime *moeTraceRuntime) {
	state := decodeProfileState{
		nextPosition: make(map[int]int),
		current:      make(map[int]int),
		remaining:    make(map[string][]traceProfileTokenAssignment),
	}
	for index := range events {
		event := &events[index]
		if event.Args == nil || event.Ph == "M" || event.Ph == "C" || event.Cat == "flow" {
			continue
		}
		requestIDs := profileEventRequestIDs(event.Args)
		if len(requestIDs) == 0 || event.Args["phase"] != nil {
			continue
		}
		event.Args["phase"] = traceProfilePhaseDecode
		if event.Name != "MoE MLP" {
			continue
		}
		layer, layerOK := profileEventInt(event.Args, "moe_layer")
		gpu, gpuOK := profileEventInt(event.Args, "gpu")
		if !layerOK || !gpuOK {
			continue
		}
		if layer == 0 && gpu == 0 {
			for _, requestID := range requestIDs {
				state.current[requestID] = state.nextPosition[requestID]
				state.nextPosition[requestID]++
			}
		}
		key := decodeProfileKey(requestIDs, layer)
		assignments, ok := state.remaining[key]
		if !ok || gpu == 0 {
			assignments = decodeAssignmentsForLayer(runtime, requestIDs, state.current, layer)
		}
		assigned, remaining := consumeGPUAssignments(assignments, event.Args["expert_token_counts"])
		state.remaining[key] = remaining
		if len(assigned) > 0 {
			event.Args["token_assignments"] = assigned
		}
	}
}

func profileEventRequestIDs(args map[string]any) []int {
	if value, ok := args["request_ids"].([]int); ok {
		return append([]int(nil), value...)
	}
	if value, ok := args["request_id"].(int); ok {
		return []int{value}
	}
	return nil
}

func profileEventInt(args map[string]any, key string) (int, bool) {
	value, ok := args[key]
	if !ok {
		return 0, false
	}
	switch typed := value.(type) {
	case int:
		return typed, true
	case float64:
		return int(typed), true
	default:
		return 0, false
	}
}

func decodeProfileKey(requestIDs []int, layer int) string {
	parts := make([]string, 0, len(requestIDs)+1)
	parts = append(parts, strconv.Itoa(layer))
	for _, requestID := range requestIDs {
		parts = append(parts, strconv.Itoa(requestID))
	}
	return strings.Join(parts, ":")
}

func decodeAssignmentsForLayer(runtime *moeTraceRuntime, requestIDs []int, positions map[int]int, layer int) []traceProfileTokenAssignment {
	requestIDs = append([]int(nil), requestIDs...)
	sort.Ints(requestIDs)
	assignments := make([]traceProfileTokenAssignment, 0, len(requestIDs)*runtime.store.topK)
	for _, requestID := range requestIDs {
		if requestID < 0 || requestID >= len(runtime.store.prompts) {
			continue
		}
		prompt := runtime.store.prompts[requestID].data
		position := positions[requestID]
		if position < 0 || position >= len(prompt.DecodeTokenIDs) {
			continue
		}
		base := (position*runtime.store.numLayers + layer) * runtime.store.topK
		for topK := 0; topK < runtime.store.topK; topK++ {
			assignments = append(assignments, traceProfileTokenAssignment{
				RequestID:     requestID,
				Phase:         traceProfilePhaseDecode,
				TokenPosition: len(prompt.InputTokenIDs) + position,
				MoELayer:      layer,
				ExpertID:      int(prompt.DecodeRoutes[base+topK]),
			})
		}
	}
	return assignments
}

func consumeGPUAssignments(assignments []traceProfileTokenAssignment, loads any) ([]traceProfileTokenAssignment, []traceProfileTokenAssignment) {
	quotas := make(map[int]int)
	switch typed := loads.(type) {
	case map[int]float64:
		for expert, tokens := range typed {
			quotas[expert] = int(tokens + 1e-9)
		}
	case map[string]any:
		for expert, tokens := range typed {
			parsed, err := strconv.Atoi(expert)
			if err != nil {
				continue
			}
			if value, ok := tokens.(float64); ok {
				quotas[parsed] = int(value + 1e-9)
			}
		}
	}
	assigned := make([]traceProfileTokenAssignment, 0)
	remaining := make([]traceProfileTokenAssignment, 0, len(assignments))
	for _, assignment := range assignments {
		if quotas[assignment.ExpertID] > 0 {
			assigned = append(assigned, assignment)
			quotas[assignment.ExpertID]--
		} else {
			remaining = append(remaining, assignment)
		}
	}
	return assigned, remaining
}
