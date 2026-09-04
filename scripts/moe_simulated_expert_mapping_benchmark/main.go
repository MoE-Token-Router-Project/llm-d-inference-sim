// Copyright 2026 The llm-d-inference-sim Authors.
//
// Licensed under the Apache License, Version 2.0 (the "License");
// you may not use this file except in compliance with the License.
// You may obtain a copy of the License at
//
//     http://www.apache.org/licenses/LICENSE-2.0
//
// Unless required by applicable law or agreed to in writing, software
// distributed under the License is distributed on an "AS IS" BASIS,
// WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
// See the License for the specific language governing permissions and
// limitations under the License.

package main

import (
	"bufio"
	"bytes"
	"context"
	"crypto/sha256"
	"encoding/csv"
	"encoding/hex"
	"encoding/json"
	"errors"
	"flag"
	"fmt"
	"io"
	"math"
	"net/http"
	"os"
	"os/signal"
	"path/filepath"
	"sort"
	"strconv"
	"strings"
	"syscall"
	"time"
)

const maxErrorBodyBytes = 64 << 10

type options struct {
	datasetPath                string
	baseURL                    string
	model                      string
	outputDir                  string
	label                      string
	defaultMaxCompletionTokens int
	limit                      int
	requestTimeout             time.Duration
	progressEvery              int
}

type serverConfig struct {
	Model                    string  `json:"model"`
	MaxNumSeqs               int     `json:"max-num-seqs"`
	MaxWaitingQueueLength    int     `json:"max-waiting-queue-length"`
	MaxModelLen              int     `json:"max-model-len"`
	EnableMoE                bool    `json:"enable-moe"`
	MoEExpertParallelSize    int     `json:"moe-expert-parallel-size"`
	MoENumExperts            int     `json:"moe-num-experts"`
	MoEPhysicalExpertSlots   int     `json:"moe-physical-expert-slots"`
	MoETopK                  int     `json:"moe-top-k"`
	MoENumLayers             int     `json:"moe-num-layers"`
	MoERouter                string  `json:"moe-router"`
	MoEExpertPopularityAlpha float64 `json:"moe-expert-popularity-alpha"`
	MoEHiddenSize            int     `json:"moe-hidden-size"`
	MoEIntermediateSize      int     `json:"moe-intermediate-size"`
	MoEBytesPerElement       int     `json:"moe-bytes-per-element"`
	MoEGPUFlops              float64 `json:"moe-gpu-flops"`
	MoEGPUMemoryBandwidth    float64 `json:"moe-gpu-memory-bandwidth"`
	MoEInterconnectBandwidth float64 `json:"moe-interconnect-bandwidth"`
	MoEInterconnectLatency   string  `json:"moe-interconnect-latency"`
}

type datasetRecord struct {
	Prompt              string `json:"prompt"`
	MaxCompletionTokens *int   `json:"max_completion_tokens,omitempty"`
}

type benchmarkPrompt struct {
	ID                  int
	Prompt              string
	MaxCompletionTokens int
}

type datasetSummary struct {
	Path                       string `json:"path"`
	SourceSHA256               string `json:"source_sha256"`
	NumPrompts                 int    `json:"num_prompts"`
	DefaultMaxCompletionTokens int    `json:"default_max_completion_tokens"`
}

type tokenCounts struct {
	PromptTokens int64 `json:"prompt_tokens"`
	OutputTokens int64 `json:"output_tokens"`
	TotalTokens  int64 `json:"total_tokens"`
}

type throughput struct {
	RequestsPerSecond     float64 `json:"requests_per_second"`
	OutputTokensPerSecond float64 `json:"output_tokens_per_second"`
	TotalTokensPerSecond  float64 `json:"total_tokens_per_second"`
}

type distribution struct {
	Count  int     `json:"count"`
	Mean   float64 `json:"mean"`
	Median float64 `json:"median"`
	P95    float64 `json:"p95"`
	P99    float64 `json:"p99"`
	Min    float64 `json:"min"`
	Max    float64 `json:"max"`
}

type failureExample struct {
	PromptID   int    `json:"prompt_id"`
	StatusCode int    `json:"status_code"`
	Error      string `json:"error"`
}

type benchmarkSummary struct {
	Label                    string           `json:"label,omitempty"`
	StartedAt                string           `json:"started_at"`
	BaseURL                  string           `json:"base_url"`
	Endpoint                 string           `json:"endpoint"`
	LaunchMode               string           `json:"launch_mode"`
	Dataset                  datasetSummary   `json:"dataset"`
	Server                   serverConfig     `json:"server"`
	Requested                int              `json:"requested"`
	Successful               int              `json:"successful"`
	Failed                   int              `json:"failed"`
	WallTimeSeconds          float64          `json:"wall_time_seconds"`
	TokenCounts              tokenCounts      `json:"token_counts"`
	Throughput               throughput       `json:"throughput"`
	RequestLatencySeconds    distribution     `json:"request_latency_seconds"`
	TTFTMilliseconds         distribution     `json:"ttft_milliseconds"`
	TPOTMilliseconds         distribution     `json:"tpot_milliseconds"`
	StreamingITLMilliseconds distribution     `json:"streaming_itl_milliseconds"`
	PrefillMilliseconds      distribution     `json:"prefill_milliseconds"`
	DecodeMilliseconds       distribution     `json:"decode_milliseconds"`
	E2EMilliseconds          distribution     `json:"e2e_milliseconds"`
	OutputLengthTokens       distribution     `json:"output_length_tokens"`
	LaunchStartOffsetMS      distribution     `json:"launch_start_offset_milliseconds"`
	FailureExamples          []failureExample `json:"failure_examples,omitempty"`
}

type requestResult struct {
	PromptID                     int
	Success                      bool
	StatusCode                   int
	Error                        string
	PromptTokens                 int
	RequestedMaxCompletionTokens int
	OutputTokens                 int
	LatencySeconds               float64
	TTFTMilliseconds             *float64
	TPOTMilliseconds             *float64
	PrefillMilliseconds          *float64
	DecodeMilliseconds           *float64
	E2EMilliseconds              float64
	StartOffsetMS                float64
	ITLMilliseconds              []float64
}

type chatRequest struct {
	Model               string        `json:"model"`
	Messages            []chatMessage `json:"messages"`
	MaxCompletionTokens int           `json:"max_completion_tokens"`
	Stream              bool          `json:"stream"`
	StreamOptions       streamOptions `json:"stream_options"`
}

type chatMessage struct {
	Role    string `json:"role"`
	Content string `json:"content"`
}

type streamOptions struct {
	IncludeUsage bool `json:"include_usage"`
}

type streamChunk struct {
	Choices []streamChoice `json:"choices"`
	Error   *streamError   `json:"error,omitempty"`
	Usage   *streamUsage   `json:"usage,omitempty"`
}

type streamChoice struct {
	Delta        map[string]json.RawMessage `json:"delta"`
	FinishReason *string                    `json:"finish_reason"`
}

type streamError struct {
	Message string `json:"message"`
}

type streamUsage struct {
	PromptTokens     int `json:"prompt_tokens"`
	CompletionTokens int `json:"completion_tokens"`
	TotalTokens      int `json:"total_tokens"`
}

func main() {
	var opts options
	flag.StringVar(&opts.datasetPath, "dataset", "", "path to normalized JSONL prompt dataset")
	flag.StringVar(&opts.baseURL, "base-url", "http://127.0.0.1:8000", "simulator base URL")
	flag.StringVar(&opts.model, "model", "", "request model; defaults to the simulator model")
	flag.StringVar(&opts.outputDir, "output-dir", "moe-simulated-expert-mapping-benchmark-results", "directory for summary.json, summary.txt, and requests.csv")
	flag.StringVar(&opts.label, "label", "", "optional label stored with the result, for example split or heuristic")
	flag.IntVar(&opts.defaultMaxCompletionTokens, "max-completion-tokens", 128, "completion-token limit for dataset rows that omit max_completion_tokens")
	flag.IntVar(&opts.limit, "limit", 0, "maximum number of dataset rows to submit; 0 submits all rows")
	flag.DurationVar(&opts.requestTimeout, "request-timeout", 0, "per-request timeout; 0 disables the client timeout")
	flag.IntVar(&opts.progressEvery, "progress-every", 100, "print progress every N completed requests; 0 disables progress")
	flag.Parse()

	if flag.NArg() != 0 {
		fmt.Fprintf(os.Stderr, "unexpected positional arguments: %v\n", flag.Args())
		os.Exit(2)
	}
	if opts.datasetPath == "" {
		fmt.Fprintln(os.Stderr, "--dataset is required")
		os.Exit(2)
	}
	if opts.outputDir == "" {
		fmt.Fprintln(os.Stderr, "--output-dir must not be empty")
		os.Exit(2)
	}
	if opts.defaultMaxCompletionTokens <= 0 {
		fmt.Fprintln(os.Stderr, "--max-completion-tokens must be positive")
		os.Exit(2)
	}
	if opts.limit < 0 {
		fmt.Fprintln(os.Stderr, "--limit must be non-negative")
		os.Exit(2)
	}
	if opts.progressEvery < 0 {
		fmt.Fprintln(os.Stderr, "--progress-every must be non-negative")
		os.Exit(2)
	}

	ctx, stop := signal.NotifyContext(context.Background(), os.Interrupt, syscall.SIGTERM)
	defer stop()
	if err := run(ctx, opts); err != nil {
		fmt.Fprintf(os.Stderr, "moe simulated expert mapping benchmark: %v\n", err)
		os.Exit(1)
	}
}

func run(ctx context.Context, opts options) error {
	prompts, dataset, err := loadDataset(opts.datasetPath, opts.defaultMaxCompletionTokens, opts.limit)
	if err != nil {
		return err
	}
	if len(prompts) == 0 {
		return errors.New("dataset contains no prompts")
	}
	if err := os.MkdirAll(opts.outputDir, 0o755); err != nil {
		return fmt.Errorf("create output directory: %w", err)
	}

	server, err := preflight(ctx, opts.baseURL, opts.model)
	if err != nil {
		return err
	}
	if opts.model == "" {
		opts.model = server.Model
	}
	if opts.label == "" {
		opts.label = server.MoERouter
	}
	neededWaiting := len(prompts) - server.MaxNumSeqs
	if neededWaiting < 0 {
		neededWaiting = 0
	}
	if server.MaxWaitingQueueLength < neededWaiting {
		fmt.Fprintf(os.Stderr,
			"warning: all-at-once launch has %d prompts but server capacity is max-num-seqs=%d plus max-waiting-queue-length=%d; some requests may be rejected\n",
			len(prompts), server.MaxNumSeqs, server.MaxWaitingQueueLength)
	}

	endpoint := strings.TrimRight(opts.baseURL, "/") + "/v1/chat/completions"
	client := newHTTPClient(opts.requestTimeout)
	defer client.CloseIdleConnections()

	results := make([]requestResult, len(prompts))
	resultCh := make(chan requestResult, len(prompts))
	startGate := make(chan struct{})
	benchmarkStart := time.Time{}

	for _, prompt := range prompts {
		prompt := prompt
		go func() {
			<-startGate
			resultCh <- executePrompt(ctx, client, endpoint, opts.model, prompt, benchmarkStart)
		}()
	}

	startedAt := time.Now().UTC()
	benchmarkStart = time.Now()
	close(startGate)

	for completed := 1; completed <= len(prompts); completed++ {
		result := <-resultCh
		results[result.PromptID] = result
		if opts.progressEvery > 0 && (completed%opts.progressEvery == 0 || completed == len(prompts)) {
			fmt.Fprintf(os.Stderr, "completed %d/%d requests\n", completed, len(prompts))
		}
	}
	wallTime := time.Since(benchmarkStart)

	summary := summarize(opts, endpoint, dataset, server, results, wallTime, startedAt)
	text := formatSummary(summary)
	fmt.Print(text)
	if err := writeOutputs(opts.outputDir, summary, text, results); err != nil {
		return err
	}
	if summary.Failed != 0 {
		return fmt.Errorf("%d of %d requests failed; results were written to %s", summary.Failed, summary.Requested, opts.outputDir)
	}
	return nil
}

func loadDataset(path string, defaultMaxCompletionTokens, limit int) ([]benchmarkPrompt, datasetSummary, error) {
	absolutePath, err := filepath.Abs(path)
	if err != nil {
		return nil, datasetSummary{}, fmt.Errorf("resolve dataset path: %w", err)
	}
	hash, err := hashFile(path)
	if err != nil {
		return nil, datasetSummary{}, err
	}
	file, err := os.Open(path)
	if err != nil {
		return nil, datasetSummary{}, fmt.Errorf("open dataset: %w", err)
	}
	defer file.Close()

	prompts := make([]benchmarkPrompt, 0)
	scanner := bufio.NewScanner(file)
	scanner.Buffer(make([]byte, 64*1024), 16<<20)
	lineNumber := 0
	for scanner.Scan() {
		lineNumber++
		line := bytes.TrimSpace(scanner.Bytes())
		if len(line) == 0 {
			continue
		}
		var record datasetRecord
		if err := json.Unmarshal(line, &record); err != nil {
			return nil, datasetSummary{}, fmt.Errorf("decode dataset line %d: %w", lineNumber, err)
		}
		if strings.TrimSpace(record.Prompt) == "" {
			return nil, datasetSummary{}, fmt.Errorf("dataset line %d has an empty prompt", lineNumber)
		}
		maxCompletionTokens := defaultMaxCompletionTokens
		if record.MaxCompletionTokens != nil {
			if *record.MaxCompletionTokens <= 0 {
				return nil, datasetSummary{}, fmt.Errorf("dataset line %d has non-positive max_completion_tokens", lineNumber)
			}
			maxCompletionTokens = *record.MaxCompletionTokens
		}
		prompts = append(prompts, benchmarkPrompt{
			ID:                  len(prompts),
			Prompt:              record.Prompt,
			MaxCompletionTokens: maxCompletionTokens,
		})
		if limit > 0 && len(prompts) >= limit {
			break
		}
	}
	if err := scanner.Err(); err != nil {
		return nil, datasetSummary{}, fmt.Errorf("read dataset: %w", err)
	}

	return prompts, datasetSummary{
		Path:                       absolutePath,
		SourceSHA256:               hash,
		NumPrompts:                 len(prompts),
		DefaultMaxCompletionTokens: defaultMaxCompletionTokens,
	}, nil
}

func hashFile(path string) (string, error) {
	file, err := os.Open(path)
	if err != nil {
		return "", fmt.Errorf("open dataset for hashing: %w", err)
	}
	defer file.Close()
	hasher := sha256.New()
	if _, err := io.Copy(hasher, file); err != nil {
		return "", fmt.Errorf("hash dataset: %w", err)
	}
	return hex.EncodeToString(hasher.Sum(nil)), nil
}

func preflight(ctx context.Context, baseURL, requestedModel string) (serverConfig, error) {
	client := &http.Client{Timeout: 5 * time.Second}
	healthURL := strings.TrimRight(baseURL, "/") + "/health"
	req, err := http.NewRequestWithContext(ctx, http.MethodGet, healthURL, nil)
	if err != nil {
		return serverConfig{}, fmt.Errorf("create health request: %w", err)
	}
	resp, err := client.Do(req)
	if err != nil {
		return serverConfig{}, fmt.Errorf("health check %s: %w", healthURL, err)
	}
	_, _ = io.Copy(io.Discard, resp.Body)
	_ = resp.Body.Close()
	if resp.StatusCode < 200 || resp.StatusCode >= 300 {
		return serverConfig{}, fmt.Errorf("health check %s returned HTTP %d", healthURL, resp.StatusCode)
	}

	configURL := strings.TrimRight(baseURL, "/") + "/admin/config"
	req, err = http.NewRequestWithContext(ctx, http.MethodGet, configURL, nil)
	if err != nil {
		return serverConfig{}, fmt.Errorf("create admin config request: %w", err)
	}
	resp, err = client.Do(req)
	if err != nil {
		return serverConfig{}, fmt.Errorf("read server config %s: %w", configURL, err)
	}
	defer resp.Body.Close()
	if resp.StatusCode < 200 || resp.StatusCode >= 300 {
		body, _ := io.ReadAll(io.LimitReader(resp.Body, maxErrorBodyBytes))
		return serverConfig{}, fmt.Errorf("server config %s returned HTTP %d: %s", configURL, resp.StatusCode, strings.TrimSpace(string(body)))
	}
	var config serverConfig
	if err := json.NewDecoder(resp.Body).Decode(&config); err != nil {
		return serverConfig{}, fmt.Errorf("decode server config: %w", err)
	}
	if !config.EnableMoE {
		return serverConfig{}, errors.New("server does not have MoE simulation enabled")
	}
	if requestedModel != "" && config.Model != requestedModel {
		return serverConfig{}, fmt.Errorf("server model %q does not match requested model %q", config.Model, requestedModel)
	}
	return config, nil
}

func newHTTPClient(timeout time.Duration) *http.Client {
	transport := http.DefaultTransport.(*http.Transport).Clone()
	transport.MaxIdleConns = 256
	transport.MaxIdleConnsPerHost = 256
	transport.IdleConnTimeout = 90 * time.Second
	client := &http.Client{Transport: transport}
	if timeout > 0 {
		client.Timeout = timeout
	}
	return client
}

func executePrompt(ctx context.Context, client *http.Client, endpoint, model string,
	prompt benchmarkPrompt, benchmarkStart time.Time) requestResult {
	result := requestResult{
		PromptID:                     prompt.ID,
		RequestedMaxCompletionTokens: prompt.MaxCompletionTokens,
		StatusCode:                   0,
		ITLMilliseconds:              make([]float64, 0, maxInt(prompt.MaxCompletionTokens-1, 0)),
	}
	payload, err := json.Marshal(chatRequest{
		Model: model,
		Messages: []chatMessage{{
			Role:    "user",
			Content: prompt.Prompt,
		}},
		MaxCompletionTokens: prompt.MaxCompletionTokens,
		Stream:              true,
		StreamOptions: streamOptions{
			IncludeUsage: true,
		},
	})
	if err != nil {
		result.Error = "marshal request: " + err.Error()
		return result
	}

	requestStart := time.Now()
	result.StartOffsetMS = requestStart.Sub(benchmarkStart).Seconds() * 1000
	finish := func(err error) requestResult {
		result.LatencySeconds = time.Since(requestStart).Seconds()
		result.E2EMilliseconds = result.LatencySeconds * 1000
		if err != nil {
			result.Error = err.Error()
		}
		return result
	}

	req, err := http.NewRequestWithContext(ctx, http.MethodPost, endpoint, bytes.NewReader(payload))
	if err != nil {
		return finish(fmt.Errorf("create request: %w", err))
	}
	req.Header.Set("Content-Type", "application/json")
	req.Header.Set("Accept", "text/event-stream")
	resp, err := client.Do(req)
	if err != nil {
		return finish(fmt.Errorf("HTTP request: %w", err))
	}
	defer resp.Body.Close()
	result.StatusCode = resp.StatusCode
	if resp.StatusCode < 200 || resp.StatusCode >= 300 {
		body, _ := io.ReadAll(io.LimitReader(resp.Body, maxErrorBodyBytes))
		return finish(fmt.Errorf("HTTP %d: %s", resp.StatusCode, strings.TrimSpace(string(body))))
	}

	scanner := bufio.NewScanner(resp.Body)
	scanner.Buffer(make([]byte, 16*1024), 1<<20)
	var firstToken time.Time
	var previousToken time.Time
	var lastToken time.Time
	usageSeen := false
	doneSeen := false
	completionTokensFromUsage := 0

	for scanner.Scan() {
		line := strings.TrimSpace(scanner.Text())
		if !strings.HasPrefix(line, "data:") {
			continue
		}
		data := strings.TrimSpace(strings.TrimPrefix(line, "data:"))
		if data == "[DONE]" {
			doneSeen = true
			break
		}
		var chunk streamChunk
		if err := json.Unmarshal([]byte(data), &chunk); err != nil {
			return finish(fmt.Errorf("decode SSE chunk: %w", err))
		}
		if chunk.Error != nil {
			if chunk.Error.Message != "" {
				return finish(errors.New(chunk.Error.Message))
			}
			return finish(errors.New("stream returned an error"))
		}
		if chunk.Usage != nil {
			usageSeen = true
			result.PromptTokens = chunk.Usage.PromptTokens
			completionTokensFromUsage = chunk.Usage.CompletionTokens
		}
		if !isOutputToken(chunk) {
			continue
		}

		now := time.Now()
		if result.OutputTokens == 0 {
			firstToken = now
			ttft := now.Sub(requestStart).Seconds() * 1000
			result.TTFTMilliseconds = &ttft
			result.PrefillMilliseconds = floatPtr(ttft)
		} else {
			result.ITLMilliseconds = append(result.ITLMilliseconds, now.Sub(previousToken).Seconds()*1000)
		}
		previousToken = now
		lastToken = now
		result.OutputTokens++
	}
	if err := scanner.Err(); err != nil {
		return finish(fmt.Errorf("read SSE stream: %w", err))
	}
	if !doneSeen {
		return finish(errors.New("SSE stream ended without [DONE]"))
	}
	if !usageSeen {
		return finish(errors.New("SSE stream did not include usage"))
	}
	if completionTokensFromUsage != result.OutputTokens {
		return finish(fmt.Errorf("received %d output token chunks but usage reports %d completion tokens", result.OutputTokens, completionTokensFromUsage))
	}
	if result.OutputTokens > 0 {
		decodeMS := lastToken.Sub(firstToken).Seconds() * 1000
		result.DecodeMilliseconds = floatPtr(decodeMS)
		if result.OutputTokens > 1 {
			tpotMS := decodeMS / float64(result.OutputTokens-1)
			result.TPOTMilliseconds = floatPtr(tpotMS)
		}
	}
	result = finish(nil)
	result.Success = true
	return result
}

func classifyStreamChunk(data []byte) (isToken bool, streamErr string, err error) {
	var chunk streamChunk
	if err := json.Unmarshal(data, &chunk); err != nil {
		return false, "", err
	}
	if chunk.Error != nil {
		if chunk.Error.Message != "" {
			return false, chunk.Error.Message, nil
		}
		return false, "stream returned an error", nil
	}
	return isOutputToken(chunk), "", nil
}

func isOutputToken(chunk streamChunk) bool {
	if len(chunk.Choices) == 0 {
		return false
	}
	choice := chunk.Choices[0]
	if choice.FinishReason != nil {
		return false
	}
	if choice.Delta == nil {
		return false
	}
	if _, hasRole := choice.Delta["role"]; hasRole {
		return false
	}
	if _, hasToolCalls := choice.Delta["tool_calls"]; hasToolCalls {
		return false
	}
	return true
}

func summarize(opts options, endpoint string, dataset datasetSummary, server serverConfig, results []requestResult,
	wallTime time.Duration, startedAt time.Time) benchmarkSummary {
	var requestLatency []float64
	var ttft []float64
	var tpot []float64
	var streamingITL []float64
	var prefill []float64
	var decode []float64
	var e2e []float64
	var outputLength []float64
	var launchOffsets []float64
	var promptTokens int64
	var outputTokens int64
	failureExamples := make([]failureExample, 0, 20)
	successful := 0

	for _, result := range results {
		launchOffsets = append(launchOffsets, result.StartOffsetMS)
		if !result.Success {
			if len(failureExamples) < cap(failureExamples) {
				failureExamples = append(failureExamples, failureExample{
					PromptID: result.PromptID, StatusCode: result.StatusCode, Error: result.Error,
				})
			}
			continue
		}
		successful++
		promptTokens += int64(result.PromptTokens)
		outputTokens += int64(result.OutputTokens)
		requestLatency = append(requestLatency, result.LatencySeconds)
		e2e = append(e2e, result.E2EMilliseconds)
		outputLength = append(outputLength, float64(result.OutputTokens))
		if result.TTFTMilliseconds != nil {
			ttft = append(ttft, *result.TTFTMilliseconds)
		}
		if result.TPOTMilliseconds != nil {
			tpot = append(tpot, *result.TPOTMilliseconds)
		}
		if result.PrefillMilliseconds != nil {
			prefill = append(prefill, *result.PrefillMilliseconds)
		}
		if result.DecodeMilliseconds != nil {
			decode = append(decode, *result.DecodeMilliseconds)
		}
		streamingITL = append(streamingITL, result.ITLMilliseconds...)
	}

	wallSeconds := wallTime.Seconds()
	totalTokens := promptTokens + outputTokens
	requestsPerSecond := 0.0
	outputTokensPerSecond := 0.0
	totalTokensPerSecond := 0.0
	if wallSeconds > 0 {
		requestsPerSecond = float64(successful) / wallSeconds
		outputTokensPerSecond = float64(outputTokens) / wallSeconds
		totalTokensPerSecond = float64(totalTokens) / wallSeconds
	}

	return benchmarkSummary{
		Label:           opts.label,
		StartedAt:       startedAt.Format(time.RFC3339Nano),
		BaseURL:         strings.TrimRight(opts.baseURL, "/"),
		Endpoint:        endpoint,
		LaunchMode:      "all-prompts-at-once",
		Dataset:         dataset,
		Server:          server,
		Requested:       len(results),
		Successful:      successful,
		Failed:          len(results) - successful,
		WallTimeSeconds: wallSeconds,
		TokenCounts: tokenCounts{
			PromptTokens: promptTokens,
			OutputTokens: outputTokens,
			TotalTokens:  totalTokens,
		},
		Throughput: throughput{
			RequestsPerSecond:     requestsPerSecond,
			OutputTokensPerSecond: outputTokensPerSecond,
			TotalTokensPerSecond:  totalTokensPerSecond,
		},
		RequestLatencySeconds:    describe(requestLatency),
		TTFTMilliseconds:         describe(ttft),
		TPOTMilliseconds:         describe(tpot),
		StreamingITLMilliseconds: describe(streamingITL),
		PrefillMilliseconds:      describe(prefill),
		DecodeMilliseconds:       describe(decode),
		E2EMilliseconds:          describe(e2e),
		OutputLengthTokens:       describe(outputLength),
		LaunchStartOffsetMS:      describe(launchOffsets),
		FailureExamples:          failureExamples,
	}
}

func describe(values []float64) distribution {
	if len(values) == 0 {
		return distribution{}
	}
	sorted := append([]float64(nil), values...)
	sort.Float64s(sorted)
	sum := 0.0
	for _, value := range sorted {
		sum += value
	}
	return distribution{
		Count:  len(sorted),
		Mean:   sum / float64(len(sorted)),
		Median: percentile(sorted, 50),
		P95:    percentile(sorted, 95),
		P99:    percentile(sorted, 99),
		Min:    sorted[0],
		Max:    sorted[len(sorted)-1],
	}
}

func percentile(sorted []float64, p float64) float64 {
	if len(sorted) == 0 {
		return 0
	}
	if len(sorted) == 1 {
		return sorted[0]
	}
	position := p / 100 * float64(len(sorted)-1)
	lower := int(math.Floor(position))
	upper := int(math.Ceil(position))
	if lower == upper {
		return sorted[lower]
	}
	weight := position - float64(lower)
	return sorted[lower]*(1-weight) + sorted[upper]*weight
}

func formatSummary(summary benchmarkSummary) string {
	var b strings.Builder
	fmt.Fprintln(&b, "===== RESULTS =====")
	fmt.Fprintf(&b, "label: %s\n", summary.Label)
	fmt.Fprintf(&b, "dataset: %s\n", summary.Dataset.Path)
	fmt.Fprintf(&b, "dataset_sha256: %s\n", summary.Dataset.SourceSHA256)
	fmt.Fprintf(&b, "model: %s\n", summary.Server.Model)
	fmt.Fprintf(&b, "router: %s\n", summary.Server.MoERouter)
	fmt.Fprintf(&b, "launch_mode: %s\n", summary.LaunchMode)
	fmt.Fprintf(&b, "requested: %d\n", summary.Requested)
	fmt.Fprintf(&b, "successful: %d\n", summary.Successful)
	fmt.Fprintf(&b, "failed: %d\n", summary.Failed)
	fmt.Fprintf(&b, "wall_s: %.6f\n", summary.WallTimeSeconds)
	fmt.Fprintf(&b, "requests_per_s: %.6f\n", summary.Throughput.RequestsPerSecond)
	fmt.Fprintf(&b, "prompt_tokens: %d\n", summary.TokenCounts.PromptTokens)
	fmt.Fprintf(&b, "output_tokens: %d\n", summary.TokenCounts.OutputTokens)
	fmt.Fprintf(&b, "total_tokens: %d\n", summary.TokenCounts.TotalTokens)
	fmt.Fprintf(&b, "output_tokens_per_s: %.6f\n", summary.Throughput.OutputTokensPerSecond)
	fmt.Fprintf(&b, "total_tokens_per_s: %.6f\n", summary.Throughput.TotalTokensPerSecond)
	writeDistribution(&b, "request_latency", "s", summary.RequestLatencySeconds)
	writeDistribution(&b, "ttft", "ms", summary.TTFTMilliseconds)
	writeDistribution(&b, "tpot", "ms", summary.TPOTMilliseconds)
	writeDistribution(&b, "streaming_itl", "ms", summary.StreamingITLMilliseconds)
	writeDistribution(&b, "prefill", "ms", summary.PrefillMilliseconds)
	writeDistribution(&b, "decode", "ms", summary.DecodeMilliseconds)
	writeDistribution(&b, "e2e", "ms", summary.E2EMilliseconds)
	writeDistribution(&b, "output_length", "tokens", summary.OutputLengthTokens)
	writeDistribution(&b, "launch_start_offset", "ms", summary.LaunchStartOffsetMS)
	return b.String()
}

func writeDistribution(b *strings.Builder, name, unit string, values distribution) {
	fmt.Fprintf(b, "%s_samples: %d\n", name, values.Count)
	fmt.Fprintf(b, "%s_mean_%s: %.6f\n", name, unit, values.Mean)
	fmt.Fprintf(b, "%s_median_%s: %.6f\n", name, unit, values.Median)
	fmt.Fprintf(b, "%s_p95_%s: %.6f\n", name, unit, values.P95)
	fmt.Fprintf(b, "%s_p99_%s: %.6f\n", name, unit, values.P99)
	fmt.Fprintf(b, "%s_min_%s: %.6f\n", name, unit, values.Min)
	fmt.Fprintf(b, "%s_max_%s: %.6f\n", name, unit, values.Max)
}

func writeOutputs(outputDir string, summary benchmarkSummary, text string, results []requestResult) error {
	jsonBytes, err := json.MarshalIndent(summary, "", "  ")
	if err != nil {
		return fmt.Errorf("marshal summary JSON: %w", err)
	}
	jsonBytes = append(jsonBytes, '\n')
	if err := os.WriteFile(filepath.Join(outputDir, "summary.json"), jsonBytes, 0o644); err != nil {
		return fmt.Errorf("write summary.json: %w", err)
	}
	if err := os.WriteFile(filepath.Join(outputDir, "summary.txt"), []byte(text), 0o644); err != nil {
		return fmt.Errorf("write summary.txt: %w", err)
	}
	if err := writeRequestCSV(filepath.Join(outputDir, "requests.csv"), results); err != nil {
		return err
	}
	return nil
}

func writeRequestCSV(path string, results []requestResult) error {
	file, err := os.Create(path)
	if err != nil {
		return fmt.Errorf("create requests.csv: %w", err)
	}
	writer := csv.NewWriter(file)
	header := []string{
		"prompt_id", "success", "status_code", "error", "prompt_tokens", "max_completion_tokens", "output_tokens",
		"request_latency_s", "ttft_ms", "tpot_ms", "prefill_ms", "decode_ms", "e2e_ms", "launch_start_offset_ms",
	}
	if err := writer.Write(header); err != nil {
		_ = file.Close()
		return fmt.Errorf("write requests.csv header: %w", err)
	}
	for _, result := range results {
		row := []string{
			strconv.Itoa(result.PromptID),
			strconv.FormatBool(result.Success),
			strconv.Itoa(result.StatusCode),
			result.Error,
			strconv.Itoa(result.PromptTokens),
			strconv.Itoa(result.RequestedMaxCompletionTokens),
			strconv.Itoa(result.OutputTokens),
			formatFloat(result.LatencySeconds),
			formatOptionalFloat(result.TTFTMilliseconds),
			formatOptionalFloat(result.TPOTMilliseconds),
			formatOptionalFloat(result.PrefillMilliseconds),
			formatOptionalFloat(result.DecodeMilliseconds),
			formatFloat(result.E2EMilliseconds),
			formatFloat(result.StartOffsetMS),
		}
		if err := writer.Write(row); err != nil {
			_ = file.Close()
			return fmt.Errorf("write requests.csv row: %w", err)
		}
	}
	writer.Flush()
	if err := writer.Error(); err != nil {
		_ = file.Close()
		return fmt.Errorf("flush requests.csv: %w", err)
	}
	if err := file.Close(); err != nil {
		return fmt.Errorf("close requests.csv: %w", err)
	}
	return nil
}

func formatFloat(value float64) string {
	return strconv.FormatFloat(value, 'f', 6, 64)
}

func formatOptionalFloat(value *float64) string {
	if value == nil {
		return ""
	}
	return formatFloat(*value)
}

func floatPtr(value float64) *float64 {
	return &value
}

func maxInt(a, b int) int {
	if a > b {
		return a
	}
	return b
}
