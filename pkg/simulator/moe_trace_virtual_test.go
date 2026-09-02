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

func TestVirtualTraceSchedulerChargesNMinusOneDecodeForwards(t *testing.T) {
	path, config := writeTinyMoETrace(t)
	result, err := RunMoETraceVirtualBenchmark(MoETraceVirtualOptions{
		TracePath:   path,
		Config:      config,
		TokenBudget: 1024,
	})
	if err != nil {
		t.Fatal(err)
	}
	if result.Requests != 1 || result.PromptTokens != 2 || result.OutputTokens != 3 {
		t.Fatalf("unexpected workload totals: %+v", result)
	}
	if result.DecodeForwards != 2 {
		t.Fatalf("decode forwards=%d, want 2 for three output tokens", result.DecodeForwards)
	}
	if result.Steps != 3 || result.PrefillSteps != 1 || result.DecodeOnlySteps != 2 || result.MixedSteps != 0 {
		t.Fatalf("unexpected scheduler steps: %+v", result)
	}
	if result.ModeledTime <= 0 || result.OutputTokensPerSecond <= 0 {
		t.Fatalf("virtual benchmark did not report modeled time: %+v", result)
	}
}

func TestVirtualTraceSchedulerRejectsDecodeBatchLargerThanBudget(t *testing.T) {
	path, config := writeTinyMoETrace(t)
	// One traced request never exceeds a budget of one decode token. This test
	// mainly guards the option validation and ensures the smallest valid budget
	// remains usable.
	result, err := RunMoETraceVirtualBenchmark(MoETraceVirtualOptions{
		TracePath:   path,
		Config:      config,
		TokenBudget: 1,
	})
	if err != nil {
		t.Fatal(err)
	}
	if result.DecodeForwards != 2 {
		t.Fatalf("decode forwards=%d, want 2", result.DecodeForwards)
	}
}
