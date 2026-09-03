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
	"compress/gzip"
	"encoding/json"
	"fmt"
	"io"
	"os"
	"path/filepath"
	"sort"
	"strings"
	"sync"
	"time"
)

const (
	profilePIDSimulated  = 1
	profilePIDHost       = 2
	profileTIDRouter     = 100
	profileTIDDispatch   = 200
	profileTIDCombine    = 201
	profileTIDMigration  = 202
	profileTIDGPUBase    = 1000
	profileTIDHostRouter = 100
	profileTIDHostEPLB   = 101
)

type chromeTraceEvent struct {
	Name string         `json:"name"`
	Cat  string         `json:"cat,omitempty"`
	Ph   string         `json:"ph"`
	Ts   float64        `json:"ts"`
	Dur  float64        `json:"dur,omitempty"`
	Pid  int            `json:"pid"`
	Tid  int            `json:"tid"`
	ID   uint64         `json:"id,omitempty"`
	Args map[string]any `json:"args,omitempty"`
}

type chromeTraceFile struct {
	TraceEvents     []chromeTraceEvent `json:"traceEvents"`
	DisplayTimeUnit string             `json:"displayTimeUnit"`
}

type moeProfileRecorder struct {
	mu          sync.Mutex
	path        string
	origin      time.Time
	events      []chromeTraceEvent
	initialized bool
	nextFlowID  uint64
}

func newMoEProfileRecorder(path string) (*moeProfileRecorder, error) {
	file, err := os.Create(path)
	if err != nil {
		return nil, fmt.Errorf("create MoE profile: %w", err)
	}
	if err := file.Close(); err != nil {
		return nil, fmt.Errorf("close MoE profile: %w", err)
	}
	return &moeProfileRecorder{path: path}, nil
}

func (r *moeProfileRecorder) recordExecution(callStarted, start time.Time, execution traceModelExecution) {
	r.mu.Lock()
	defer r.mu.Unlock()
	if r.origin.IsZero() {
		r.origin = callStarted
	}
	if !r.initialized {
		numGPUs := 0
		if len(execution.layers) > 0 {
			numGPUs = len(execution.layers[0].gpus)
		}
		r.initializeTracks(numGPUs)
		r.initialized = true
	}
	if !execution.eplbStarted.IsZero() && execution.eplbDuration > 0 {
		r.spanOnProcess("EPLB update", "host.eplb", profilePIDHost, profileTIDHostEPLB, execution.eplbStarted, execution.eplbDuration, nil)
	}
	for _, layer := range execution.layers {
		if !layer.routerStarted.IsZero() && layer.routerDuration > 0 {
			r.spanOnProcess("Route layer", "host.router", profilePIDHost, profileTIDHostRouter, layer.routerStarted, layer.routerDuration, map[string]any{"layer": layer.layer})
		}
	}
	cursor := start
	for _, layer := range execution.layers {
		r.instant("Route layer", "cpu.router", profileTIDRouter, cursor, map[string]any{
			"layer":          layer.layer,
			"host_router_us": durationMicros(layer.routerDuration),
		})
		routeFlow := r.newFlowID()
		r.flow("Route to dispatch", "s", profileTIDRouter, cursor, routeFlow)
		r.flow("Route to dispatch", "f", profileTIDDispatch, cursor, routeFlow)
		if layer.dispatch > 0 {
			r.span("Expert dispatch", "network.dispatch", profileTIDDispatch, cursor, layer.dispatch, map[string]any{"layer": layer.layer})
			cursor = cursor.Add(layer.dispatch)
		}
		gpuStart := cursor
		gpuDuration := time.Duration(0)
		for _, gpu := range layer.gpus {
			if gpu.duration > gpuDuration {
				gpuDuration = gpu.duration
			}
		}
		combineStart := gpuStart.Add(gpuDuration)
		for _, gpu := range layer.gpus {
			dispatchFlow := r.newFlowID()
			r.flow("Dispatch to GPU", "s", profileTIDDispatch, gpuStart, dispatchFlow)
			r.flow("Dispatch to GPU", "f", profileTIDGPUBase+gpu.gpu*10, gpuStart, dispatchFlow)
			r.recordGPU(gpuStart, layer.layer, gpu)
			combineFlow := r.newFlowID()
			r.flow("GPU to combine", "s", profileTIDGPUBase+gpu.gpu*10, gpuStart.Add(gpu.duration), combineFlow)
			r.flow("GPU to combine", "f", profileTIDCombine, combineStart, combineFlow)
		}
		cursor = combineStart
		if layer.combine > 0 {
			r.span("Expert combine", "network.combine", profileTIDCombine, cursor, layer.combine, map[string]any{"layer": layer.layer})
			cursor = cursor.Add(layer.combine)
		}
	}
	if execution.migration > 0 {
		r.span("Expert migration", "network.migration", profileTIDMigration, cursor, execution.migration, nil)
	}
}

func (r *moeProfileRecorder) Flush() error {
	r.mu.Lock()
	defer r.mu.Unlock()
	return r.flushLocked()
}

func (r *moeProfileRecorder) recordGPU(start time.Time, layer int, gpu traceGPUExecution) {
	tid := profileTIDGPUBase + gpu.gpu*10
	experts := make([]int, 0, len(gpu.expertLoads))
	for expert := range gpu.expertLoads {
		experts = append(experts, expert)
	}
	sort.Ints(experts)
	args := map[string]any{
		"gpu":                 gpu.gpu,
		"layer":               layer,
		"experts":             experts,
		"expert_token_counts": gpu.expertLoads,
		"assignments":         gpu.assignments,
		"flops":               gpu.computeFlops,
		"hbm_bytes":           gpu.memoryBytes,
		"compute_us":          durationMicros(gpu.computeDuration),
		"hbm_us":              durationMicros(gpu.memoryDuration),
		"vram_bytes":          gpu.vramBytes,
	}
	if gpu.duration > 0 {
		args["compute_utilization"] = ratio(gpu.computeDuration, gpu.duration)
		args["hbm_utilization"] = ratio(gpu.memoryDuration, gpu.duration)
		if gpu.memoryDuration >= gpu.computeDuration {
			args["bottleneck"] = "hbm"
		} else {
			args["bottleneck"] = "compute"
		}
	}
	r.span("MoE MLP", "gpu.mlp", tid, start, gpu.duration, args)
	if gpu.duration <= 0 {
		return
	}
	computeUtil := 100 * ratio(gpu.computeDuration, gpu.duration)
	hbmUtil := 100 * ratio(gpu.memoryDuration, gpu.duration)
	hbmGBps := 0.0
	if gpu.duration > 0 {
		hbmGBps = gpu.memoryBytes / gpu.duration.Seconds() / 1e9
	}
	r.counter("Compute utilization", tid+1, start, computeUtil)
	r.counter("HBM utilization", tid+2, start, hbmUtil)
	r.counter("HBM GB/s", tid+3, start, hbmGBps)
	r.counter("VRAM bytes", tid+4, start, gpu.vramBytes)
	end := start.Add(gpu.duration)
	r.counter("Compute utilization", tid+1, end, 0)
	r.counter("HBM utilization", tid+2, end, 0)
	r.counter("HBM GB/s", tid+3, end, 0)
}

func (r *moeProfileRecorder) initializeTracks(numGPUs int) {
	r.metadataForProcess(profilePIDSimulated, "process_name", 0, map[string]any{"name": "Simulated system"})
	r.metadataForProcess(profilePIDHost, "process_name", 0, map[string]any{"name": "Simulator host"})
	r.threadMetadata(profileTIDRouter, map[string]any{"name": "CPU / Router (logical)"})
	r.threadMetadata(profileTIDDispatch, map[string]any{"name": "Network / Dispatch"})
	r.threadMetadata(profileTIDCombine, map[string]any{"name": "Network / Combine"})
	r.threadMetadata(profileTIDMigration, map[string]any{"name": "Network / Expert migration"})
	r.metadataForProcess(profilePIDHost, "thread_name", profileTIDHostRouter, map[string]any{"name": "Go / MoE router"})
	r.metadataForProcess(profilePIDHost, "thread_name", profileTIDHostEPLB, map[string]any{"name": "Go / EPLB"})
	for gpu := 0; gpu < numGPUs; gpu++ {
		tid := profileTIDGPUBase + gpu*10
		r.threadMetadata(tid, map[string]any{"name": fmt.Sprintf("GPU %d / Operations", gpu)})
		r.threadMetadata(tid+1, map[string]any{"name": fmt.Sprintf("GPU %d / Compute utilization", gpu)})
		r.threadMetadata(tid+2, map[string]any{"name": fmt.Sprintf("GPU %d / HBM utilization", gpu)})
		r.threadMetadata(tid+3, map[string]any{"name": fmt.Sprintf("GPU %d / HBM GB/s", gpu)})
		r.threadMetadata(tid+4, map[string]any{"name": fmt.Sprintf("GPU %d / VRAM", gpu)})
	}
}

func (r *moeProfileRecorder) threadMetadata(tid int, args map[string]any) {
	r.metadataForProcess(profilePIDSimulated, "thread_name", tid, args)
}

func (r *moeProfileRecorder) metadataForProcess(pid int, name string, tid int, args map[string]any) {
	r.events = append(r.events, chromeTraceEvent{Name: name, Ph: "M", Pid: pid, Tid: tid, Args: args})
}

func (r *moeProfileRecorder) newFlowID() uint64 {
	r.nextFlowID++
	return r.nextFlowID
}

func (r *moeProfileRecorder) flow(name, phase string, tid int, at time.Time, id uint64) {
	r.events = append(r.events, chromeTraceEvent{
		Name: name, Cat: "flow", Ph: phase, Ts: r.timestampMicros(at), Pid: profilePIDSimulated, Tid: tid, ID: id,
	})
}

func (r *moeProfileRecorder) span(name, category string, tid int, start time.Time, duration time.Duration, args map[string]any) {
	r.spanOnProcess(name, category, profilePIDSimulated, tid, start, duration, args)
}

func (r *moeProfileRecorder) spanOnProcess(name, category string, pid, tid int, start time.Time, duration time.Duration, args map[string]any) {
	r.events = append(r.events, chromeTraceEvent{
		Name: name, Cat: category, Ph: "X", Ts: r.timestampMicros(start), Dur: durationMicros(duration),
		Pid: pid, Tid: tid, Args: args,
	})
}

func (r *moeProfileRecorder) instant(name, category string, tid int, at time.Time, args map[string]any) {
	r.events = append(r.events, chromeTraceEvent{
		Name: name, Cat: category, Ph: "i", Ts: r.timestampMicros(at), Pid: profilePIDSimulated, Tid: tid, Args: args,
	})
}

func (r *moeProfileRecorder) counter(name string, tid int, at time.Time, value float64) {
	r.events = append(r.events, chromeTraceEvent{
		Name: name, Ph: "C", Ts: r.timestampMicros(at), Pid: profilePIDSimulated, Tid: tid,
		Args: map[string]any{"value": value},
	})
}

func (r *moeProfileRecorder) timestampMicros(at time.Time) float64 {
	return float64(at.Sub(r.origin)) / float64(time.Microsecond)
}

func durationMicros(duration time.Duration) float64 {
	return float64(duration) / float64(time.Microsecond)
}

func ratio(numerator, denominator time.Duration) float64 {
	if denominator <= 0 {
		return 0
	}
	value := float64(numerator) / float64(denominator)
	if value > 1 {
		return 1
	}
	if value < 0 {
		return 0
	}
	return value
}

func (r *moeProfileRecorder) flushLocked() error {
	if r.path == "" {
		return nil
	}
	directory := filepath.Dir(r.path)
	temp, err := os.CreateTemp(directory, ".moe-profile-*")
	if err != nil {
		return fmt.Errorf("create MoE profile: %w", err)
	}
	tempPath := temp.Name()
	defer func() { _ = os.Remove(tempPath) }()

	var writer io.Writer = temp
	var gzipWriter *gzip.Writer
	if strings.HasSuffix(r.path, ".gz") {
		gzipWriter = gzip.NewWriter(temp)
		writer = gzipWriter
	}
	encoder := json.NewEncoder(writer)
	if err := encoder.Encode(chromeTraceFile{TraceEvents: r.events, DisplayTimeUnit: "ms"}); err != nil {
		_ = temp.Close()
		return fmt.Errorf("write MoE profile: %w", err)
	}
	if gzipWriter != nil {
		if err := gzipWriter.Close(); err != nil {
			_ = temp.Close()
			return fmt.Errorf("close MoE profile compressor: %w", err)
		}
	}
	if err := temp.Close(); err != nil {
		return fmt.Errorf("close MoE profile: %w", err)
	}
	if err := os.Rename(tempPath, r.path); err != nil {
		return fmt.Errorf("publish MoE profile: %w", err)
	}
	return nil
}
