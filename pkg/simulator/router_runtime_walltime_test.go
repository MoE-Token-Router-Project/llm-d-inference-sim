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
	"strings"
	"testing"
	"time"

	"github.com/llm-d/llm-d-inference-sim/pkg/common"
)

func TestParseStartOptionsMoECountRouterRuntime(t *testing.T) {
	tests := []struct {
		name      string
		args      []string
		want      bool
		wantArgs  []string
		wantError bool
	}{
		{
			name:     "enabled",
			args:     []string{"--moe-trace-path=/tmp/a.moetrace", "--moe-count-router-runtime", "--port", "9000"},
			want:     true,
			wantArgs: []string{"--port", "9000"},
		},
		{
			name:     "explicit false",
			args:     []string{"--moe-trace-path=/tmp/a.moetrace", "--moe-count-router-runtime=false"},
			want:     false,
			wantArgs: []string{},
		},
		{
			name:      "requires trace",
			args:      []string{"--moe-count-router-runtime"},
			wantError: true,
		},
		{
			name:      "invalid boolean",
			args:      []string{"--moe-trace-path=/tmp/a.moetrace", "--moe-count-router-runtime=maybe"},
			wantError: true,
		},
		{
			name:      "duplicate",
			args:      []string{"--moe-trace-path=/tmp/a.moetrace", "--moe-count-router-runtime", "--moe-count-router-runtime=false"},
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
			if options.MoECountRouterRuntime != test.want {
				t.Fatalf("MoECountRouterRuntime = %v, want %v", options.MoECountRouterRuntime, test.want)
			}
			if strings.Join(remaining, "\x00") != strings.Join(test.wantArgs, "\x00") {
				t.Fatalf("remaining args = %q, want %q", remaining, test.wantArgs)
			}
		})
	}
}

func TestTraceRouterRuntimeWallTimeToggle(t *testing.T) {
	config := tinyMoEConfig()
	config.MoERouter = common.MoERouterHeuristic
	config.MoEExpertParallelSize = 8
	config.MoENumExperts = 64
	config.MoEPhysicalExpertSlots = 128
	config.MoETopK = 2
	config.MoENumLayers = 4

	counts := newMoELayerCounts(config.MoENumLayers, config.MoENumExperts)
	for layer := range counts {
		for expert := range counts[layer] {
			counts[layer][expert] = float64((expert % 7) + 1)
		}
	}

	const roundingTolerance = 100 * time.Nanosecond
	for _, enabled := range []bool{false, true} {
		model := newMoESimulator(config)
		traceFidelityConfigs.Store(model, &traceFidelityConfig{
			fixedPlacement:     true,
			countRouterRuntime: enabled,
			memoryEfficiency:   1,
			computeEfficiency:  1,
			prefillBatchTokens: defaultTracePrefillBatchTokens,
		})

		execution := model.traceModelExecutionForLayerCounts(counts)
		traceFidelityConfigs.Delete(model)

		modeledDuration := execution.migration
		routerDuration := time.Duration(0)
		for _, layer := range execution.layers {
			modeledDuration += layer.duration
			routerDuration += layer.routerDuration
		}
		if routerDuration <= 2*roundingTolerance {
			t.Fatalf("measured router duration %s is too small to verify wall-time accounting", routerDuration)
		}

		want := modeledDuration
		if enabled {
			want += routerDuration
		}
		delta := execution.duration - want
		if delta < 0 {
			delta = -delta
		}
		if delta > roundingTolerance {
			t.Fatalf("countRouterRuntime=%v: duration = %s, want %s within %s (router=%s)",
				enabled, execution.duration, want, roundingTolerance, routerDuration)
		}
	}
}
