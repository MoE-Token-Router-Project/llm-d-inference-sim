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
	"context"
	"errors"
	"fmt"
	"strings"

	"github.com/go-logr/logr"
	"github.com/llm-d/llm-d-inference-sim/pkg/common"
	"github.com/llm-d/llm-d-inference-sim/pkg/common/logging"
)

// StartOptions contains startup-only simulator options that are intentionally
// not mutable through the runtime admin configuration API.
type StartOptions struct {
	MoETracePath string
}

// ParseStartOptions consumes startup-only command-line flags and returns the
// remaining arguments for the normal simulator configuration parser.
func ParseStartOptions(args []string) (StartOptions, []string, error) {
	var options StartOptions
	remaining := make([]string, 0, len(args))
	seenTracePath := false

	for i := 0; i < len(args); i++ {
		arg := args[i]
		if arg == "--" {
			remaining = append(remaining, args[i:]...)
			break
		}
		switch {
		case arg == "--moe-trace-path":
			if seenTracePath {
				return StartOptions{}, nil, errors.New("--moe-trace-path may only be specified once")
			}
			if i+1 >= len(args) {
				return StartOptions{}, nil, errors.New("--moe-trace-path requires a file path")
			}
			i++
			options.MoETracePath = args[i]
			seenTracePath = true
		case strings.HasPrefix(arg, "--moe-trace-path="):
			if seenTracePath {
				return StartOptions{}, nil, errors.New("--moe-trace-path may only be specified once")
			}
			options.MoETracePath = strings.TrimPrefix(arg, "--moe-trace-path=")
			seenTracePath = true
		default:
			remaining = append(remaining, arg)
		}
	}

	if seenTracePath && options.MoETracePath == "" {
		return StartOptions{}, nil, errors.New("--moe-trace-path requires a non-empty file path")
	}
	return options, remaining, nil
}

// StartWithOptions starts the simulator and installs the optional read-only
// MoE trace runtime before any communication endpoint starts accepting work.
func StartWithOptions(ctx context.Context, config *common.Configuration, logger logr.Logger,
	options StartOptions) ([]*Simulator, error) {
	var traceStore *moeTraceStore
	var err error
	if options.MoETracePath != "" {
		if !config.EnableMoE {
			return nil, errors.New("--moe-trace-path requires --enable-moe")
		}
		if config.MoEPhysicalExpertSlots%config.MoEExpertParallelSize != 0 {
			return nil, fmt.Errorf("moe physical expert slots (%d) must be divisible by expert parallel size (%d)",
				config.MoEPhysicalExpertSlots, config.MoEExpertParallelSize)
		}
		traceStore, err = loadMoETraceStore(options.MoETracePath, config)
		if err != nil {
			return nil, fmt.Errorf("load MoE trace: %w", err)
		}
		logger.V(logging.INFO).Info("MoE trace loaded",
			"path", options.MoETracePath,
			"model", traceStore.model,
			"prompts", len(traceStore.prompts),
			"experts", traceStore.numExperts,
			"top-k", traceStore.topK,
			"layers", traceStore.numLayers)
	}

	sims, err := Start(ctx, config, logger)
	if err != nil {
		return nil, err
	}
	if traceStore == nil {
		return sims, nil
	}

	for _, sim := range sims {
		sim.Context.dataset = &traceAwareDataset{
			base: sim.Context.dataset,
			runtime: &moeTraceRuntime{
				store: traceStore,
				active: traceActiveRegistry{
					requests: make(map[string]*activeTraceRequest),
				},
			},
		}
	}
	return sims, nil
}
