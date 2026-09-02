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
	"fmt"
	"sync"
	"time"

	"github.com/llm-d/llm-d-inference-sim/pkg/api"
	"github.com/llm-d/llm-d-inference-sim/pkg/common"
	"github.com/llm-d/llm-d-inference-sim/pkg/dataset"
	"github.com/llm-d/llm-d-inference-sim/pkg/moetrace"
	"github.com/valyala/fasthttp"
)

type moeLayerCounts [][]float64

type moeTracePrompt struct {
	data            *moetrace.PromptData
	prefillCounts   moeLayerCounts
	responseStrings []string
}

type moeTraceStore struct {
	model      string
	numExperts int
	topK       int
	numLayers  int
	prompts    []*moeTracePrompt
}

type traceExecution struct {
	prompt       *moeTracePrompt
	promptID     int
	outputTokens int
	response     api.Tokenized
	finishReason string
}

type activeTraceRequest struct {
	execution      *traceExecution
	decodePosition int
}

type traceActiveRegistry struct {
	mu       sync.Mutex
	requests map[string]*activeTraceRequest
}

type moeTraceRuntime struct {
	store  *moeTraceStore
	active traceActiveRegistry
}

type traceAwareDataset struct {
	base    dataset.Dataset
	runtime *moeTraceRuntime
}

func loadMoETraceStore(path string, config *common.Configuration) (*moeTraceStore, error) {
	reader, err := moetrace.Open(path)
	if err != nil {
		return nil, err
	}
	defer reader.Close()

	metadata := reader.Metadata()
	if metadata.Model != config.Model {
		return nil, fmt.Errorf("trace model %q does not match configured model %q", metadata.Model, config.Model)
	}
	if metadata.NumExperts != config.MoENumExperts {
		return nil, fmt.Errorf("trace has %d experts; simulator is configured for %d", metadata.NumExperts, config.MoENumExperts)
	}
	if metadata.TopK != config.MoETopK {
		return nil, fmt.Errorf("trace top-k is %d; simulator is configured for %d", metadata.TopK, config.MoETopK)
	}
	if len(metadata.SparseLayers) != config.MoENumLayers {
		return nil, fmt.Errorf("trace has %d sparse layers; simulator is configured for %d",
			len(metadata.SparseLayers), config.MoENumLayers)
	}

	store := &moeTraceStore{
		model:      metadata.Model,
		numExperts: metadata.NumExperts,
		topK:       metadata.TopK,
		numLayers:  len(metadata.SparseLayers),
		prompts:    make([]*moeTracePrompt, metadata.NumPrompts),
	}
	for promptID := 0; promptID < metadata.NumPrompts; promptID++ {
		prompt, err := reader.ReadPrompt(promptID)
		if err != nil {
			return nil, fmt.Errorf("read prompt %d: %w", promptID, err)
		}
		store.prompts[promptID] = &moeTracePrompt{
			data:            prompt,
			prefillCounts:   prefillCountsAsFloat64(prompt, store.numLayers, store.numExperts),
			responseStrings: distributeTraceText(prompt.Metadata.GeneratedText, len(prompt.DecodeTokenIDs)),
		}
	}
	return store, nil
}

func prefillCountsAsFloat64(prompt *moetrace.PromptData, numLayers, numExperts int) moeLayerCounts {
	counts := newMoELayerCounts(numLayers, numExperts)
	for layer := 0; layer < numLayers; layer++ {
		base := layer * numExperts
		for expert := 0; expert < numExperts; expert++ {
			counts[layer][expert] = float64(prompt.PrefillCounts[base+expert])
		}
	}
	return counts
}

func newMoELayerCounts(numLayers, numExperts int) moeLayerCounts {
	counts := make(moeLayerCounts, numLayers)
	for layer := range counts {
		counts[layer] = make([]float64, numExperts)
	}
	return counts
}

// distributeTraceText preserves the complete generated text without invoking
// a tokenizer in the request path. The v1 trace format records exact token IDs
// and complete generated text, but not the decoded string for each token.
func distributeTraceText(text string, tokens int) []string {
	if tokens == 0 {
		return nil
	}
	pieces := make([]string, tokens)
	if text == "" {
		return pieces
	}
	runes := []rune(text)
	for i := 0; i < tokens; i++ {
		start := i * len(runes) / tokens
		end := (i + 1) * len(runes) / tokens
		pieces[i] = string(runes[start:end])
	}
	return pieces
}

func (d *traceAwareDataset) Close() error {
	return d.base.Close()
}

func (d *traceAwareDataset) GetResponseTokens(req api.Request) (*api.Tokenized, string, error) {
	if traceReq, ok := req.(*ChatCompletionsRequest); ok && traceReq.trace != nil {
		execution := traceReq.trace
		response := &api.Tokenized{
			Tokens:  append([]uint32(nil), execution.response.Tokens...),
			Strings: append([]string(nil), execution.response.Strings...),
		}
		return response, execution.finishReason, nil
	}
	return d.base.GetResponseTokens(req)
}

func (s *SimContext) traceRuntime() *moeTraceRuntime {
	traceDataset, ok := s.dataset.(*traceAwareDataset)
	if !ok {
		return nil
	}
	return traceDataset.runtime
}

func (s *SimContext) bindTraceChatRequest(req *ChatCompletionsRequest) *api.Error {
	if req.TracePromptID == nil {
		return nil
	}
	runtime := s.traceRuntime()
	if runtime == nil {
		return newTraceRequestError("trace_prompt_id requires the simulator to start with --moe-trace-path")
	}
	if s.Config().Mode == common.ModeEcho {
		return newTraceRequestError("trace_prompt_id is not supported in echo mode")
	}
	if s.isLora(req.GetDisplayedModel()) {
		return newTraceRequestError("trace_prompt_id cannot be used with a LoRA model")
	}
	if req.SendImage() {
		return newTraceRequestError("trace_prompt_id cannot be used with image emission")
	}

	promptID := *req.TracePromptID
	if promptID < 0 || promptID >= len(runtime.store.prompts) {
		return newTraceRequestError(fmt.Sprintf("trace_prompt_id %d is outside [0,%d)", promptID, len(runtime.store.prompts)))
	}
	prompt := runtime.store.prompts[promptID]
	replayTokens := len(prompt.data.DecodeTokenIDs)
	if maxTokens := req.GetMaxCompletionTokens(); maxTokens != nil {
		if *maxTokens < 1 {
			return newTraceRequestError("max_completion_tokens must be positive for trace replay")
		}
		if *maxTokens > int64(replayTokens) {
			return newTraceRequestError(fmt.Sprintf(
				"max_completion_tokens %d exceeds the %d decode tokens recorded for trace prompt %d",
				*maxTokens, replayTokens, promptID))
		}
		replayTokens = int(*maxTokens)
	}
	if len(prompt.data.InputTokenIDs)+replayTokens > s.Config().MaxModelLen {
		return newTraceRequestError(fmt.Sprintf(
			"trace prompt %d requires %d input plus %d output tokens, exceeding max-model-len %d",
			promptID, len(prompt.data.InputTokenIDs), replayTokens, s.Config().MaxModelLen))
	}

	req.Messages = []api.Message{{
		Role: api.RoleUser,
		Content: api.ChatComplContent{
			Raw: prompt.data.Metadata.Prompt,
		},
	}}
	// Numeric IDs from the trace are authoritative. Strings are deliberately
	// omitted so trace replay never invokes a tokenizer before benchmarking.
	req.SetTokenizedPrompt(&api.Tokenized{
		Tokens: append([]uint32(nil), prompt.data.InputTokenIDs...),
	})

	finishReason := common.StopFinishReason
	if req.GetMaxCompletionTokens() != nil {
		finishReason = common.LengthFinishReason
	}
	req.trace = &traceExecution{
		prompt:       prompt,
		promptID:     promptID,
		outputTokens: replayTokens,
		response: api.Tokenized{
			Tokens:  append([]uint32(nil), prompt.data.DecodeTokenIDs[:replayTokens]...),
			Strings: append([]string(nil), prompt.responseStrings[:replayTokens]...),
		},
		finishReason: finishReason,
	}
	return nil
}

func newTraceRequestError(message string) *api.Error {
	err := api.NewError(message, fasthttp.StatusBadRequest, nil)
	return &err
}

func (r *traceActiveRegistry) activate(requestID string, execution *traceExecution) {
	if execution == nil || execution.outputTokens <= 1 {
		return
	}
	r.mu.Lock()
	defer r.mu.Unlock()
	r.requests[requestID] = &activeTraceRequest{execution: execution}
}

func (r *traceActiveRegistry) deactivate(requestID string) {
	r.mu.Lock()
	defer r.mu.Unlock()
	delete(r.requests, requestID)
}

func (r *traceActiveRegistry) advance(requestID string) {
	r.mu.Lock()
	defer r.mu.Unlock()
	state, ok := r.requests[requestID]
	if !ok {
		return
	}
	state.decodePosition++
	if state.decodePosition >= state.execution.outputTokens-1 {
		delete(r.requests, requestID)
	}
}

func (r *traceActiveRegistry) decodeCounts(m *moeSimulator, runningReqs int64) (moeLayerCounts, bool) {
	r.mu.Lock()
	defer r.mu.Unlock()
	if len(r.requests) == 0 {
		return nil, false
	}

	counts := newMoELayerCounts(m.numLayers, m.numExperts)
	activeTraceTokens := 0
	for _, state := range r.requests {
		position := state.decodePosition
		if position < 0 || position >= state.execution.outputTokens-1 {
			continue
		}
		prompt := state.execution.prompt.data
		for layer := 0; layer < m.numLayers; layer++ {
			base := (position*m.numLayers + layer) * m.topK
			for index := 0; index < m.topK; index++ {
				expert := int(prompt.DecodeRoutes[base+index])
				counts[layer][expert]++
			}
		}
		activeTraceTokens++
	}
	if activeTraceTokens == 0 {
		return nil, false
	}

	// Existing non-trace requests do not expose per-request decode position.
	// Preserve their previous load contribution using the configured synthetic
	// distribution while exact trace requests contribute their recorded routes.
	syntheticTokens := int(runningReqs) - activeTraceTokens
	if syntheticTokens > 0 {
		assignments := float64(syntheticTokens * m.topK)
		for layer := 0; layer < m.numLayers; layer++ {
			for expert, probability := range m.probabilities {
				counts[layer][expert] += assignments * probability
			}
		}
	}
	return counts, true
}

func (m *moeSimulator) latencyForLayerCounts(counts moeLayerCounts) time.Duration {
	if len(counts) != m.numLayers {
		return 0
	}
	assignments := 0.0
	for layer := range counts {
		if len(counts[layer]) != m.numExperts {
			return 0
		}
		for _, count := range counts[layer] {
			assignments += count
		}
	}
	if assignments == 0 {
		return 0
	}

	m.stateMu.Lock()
	defer m.stateMu.Unlock()

	totalSeconds := 0.0
	for layer := 0; layer < m.numLayers; layer++ {
		state := m.route(counts[layer], m.placements[layer])
		totalSeconds += m.maxGPUCost(state) + m.communicationCost(state)
	}
	latency := time.Duration(totalSeconds * float64(time.Second))
	return latency + m.advanceEPLBLayerCounts(counts)
}

func (m *moeSimulator) advanceEPLBLayerCounts(counts moeLayerCounts) time.Duration {
	if m.shouldRecordEPLBLoad() {
		m.recordEPLBLayerCounts(counts)
	}
	m.expertRearrangementStep++
	if m.expertRearrangementStep < vllmEPLBStepInterval {
		return 0
	}
	m.expertRearrangementStep = 0
	return m.rebalanceExperts()
}

func (m *moeSimulator) recordEPLBLayerCounts(counts moeLayerCounts) {
	stride := m.numLayers * m.numExperts
	base := m.eplbLoadWindowStep * stride
	for layer := 0; layer < m.numLayers; layer++ {
		layerBase := base + layer*m.numExperts
		copy(m.eplbLoadWindow[layerBase:layerBase+m.numExperts], counts[layer])
	}
	m.eplbLoadWindowStep++
	if m.eplbLoadWindowStep >= vllmEPLBWindowSize {
		m.eplbLoadWindowStep = 0
	}
}

func (s *SimContext) tracePrefillCounts(execution *traceExecution, cachedPromptTokens int) moeLayerCounts {
	if execution == nil {
		return nil
	}
	prompt := execution.prompt
	if cachedPromptTokens <= 0 {
		return prompt.prefillCounts
	}
	inputTokens := len(prompt.data.InputTokenIDs)
	if cachedPromptTokens >= inputTokens {
		return nil
	}

	counts := newMoELayerCounts(s.moe.numLayers, s.moe.numExperts)
	for layer := 0; layer < s.moe.numLayers; layer++ {
		for position := cachedPromptTokens; position < inputTokens; position++ {
			base := (layer*inputTokens + position) * s.moe.topK
			for index := 0; index < s.moe.topK; index++ {
				expert := int(prompt.data.PrefillRoutes[base+index])
				counts[layer][expert]++
			}
		}
	}
	return counts
}

func (s *SimContext) simulateTraceTTFT(respCtx ResponseContext, execution *traceExecution) {
	startPrefill := time.Now()
	params := TTFTParams{
		PromptTokens:       respCtx.UsageData().PromptTokens,
		CachedPromptTokens: respCtx.NumberCachedPromptTokens(),
		DoRemotePrefill:    respCtx.doRemotePrefill(),
		RunningReqs:        s.metrics.nRunningReqs.Load(),
	}
	ttft := s.latencyCalc().GetTimeToFirstToken(&params)
	if counts := s.tracePrefillCounts(execution, params.CachedPromptTokens); counts != nil {
		ttft += s.moe.latencyForLayerCounts(counts)
	}
	time.Sleep(ttft)
	common.WriteToChannel(s.metrics.ttftChan, ttft.Seconds(), s.logger)
	common.WriteToChannel(s.metrics.reqPrefillTimeChan, time.Since(startPrefill).Seconds(), s.logger)

	if runtime := s.traceRuntime(); runtime != nil {
		runtime.active.activate(respCtx.RequestID(), execution)
	}
}

func (s *SimContext) simulateTraceInterTokenLatency(requestID string) {
	params := InterTokenParams{RunningReqs: s.metrics.nRunningReqs.Load()}
	perTokenLatency := s.latencyCalc().GetInterTokenLatency(&params)

	runtime := s.traceRuntime()
	if runtime == nil {
		perTokenLatency += s.moeDecodeLatency(&params)
	} else if counts, ok := runtime.active.decodeCounts(s.moe, params.RunningReqs); ok {
		perTokenLatency += s.moe.latencyForLayerCounts(counts)
	} else {
		perTokenLatency += s.moeDecodeLatency(&params)
	}
	time.Sleep(perTokenLatency)
	if runtime != nil {
		runtime.active.advance(requestID)
	}
	common.WriteToChannel(s.metrics.tpotChan, perTokenLatency.Seconds(), s.logger)
}

func (s *Simulator) simulateTraceResponseProcessing(respCtx ResponseContext, execution *traceExecution) {
	reqCtx := respCtx.RequestContext()
	choiceIdx := reqCtx.choiceIndex()
	if respCtx.FinishReason() != nil && *respCtx.FinishReason() == common.CacheThresholdFinishReason {
		common.WriteToChannel(reqCtx.responseChannel(),
			&ResponseInfo{RespCtx: respCtx, ChoiceIdx: choiceIdx}, s.Context.logger)
		return
	}

	requestID := respCtx.RequestID()
	if runtime := s.Context.traceRuntime(); runtime != nil {
		defer runtime.active.deactivate(requestID)
	}

	s.Context.simulateTraceTTFT(respCtx, execution)
	common.WriteToChannel(reqCtx.responseChannel(),
		&ResponseInfo{RespCtx: respCtx, Status: ResponseStatusCreated, ChoiceIdx: choiceIdx},
		s.Context.logger)

	nTokens := 0
	startDecode := time.Now()
	if !respIsEmpty(respCtx) {
		for i, token := range respCtx.responseTokens().Tokens {
			if i != 0 {
				s.Context.simulateTraceInterTokenLatency(requestID)
				nTokens++
			}

			tokens := &api.Tokenized{
				Tokens:  []uint32{token},
				Strings: []string{respCtx.responseTokens().Strings[i]},
			}
			respInfo := ResponseInfo{Tokens: tokens, RespCtx: respCtx, ChoiceIdx: choiceIdx}
			if i == len(respCtx.responseTokens().Tokens)-1 {
				respInfo.Status = ResponseEndOfTokens
			}
			common.WriteToChannel(reqCtx.responseChannel(), &respInfo, s.Context.logger)
		}
	}

	decodeTime := time.Since(startDecode).Seconds()
	meanTPOT := 0.0
	if nTokens > 0 {
		meanTPOT = decodeTime / float64(nTokens)
	}
	common.WriteToChannel(s.Context.metrics.reqTpotChan, meanTPOT, s.Context.logger)
	common.WriteToChannel(s.Context.metrics.reqDecodeTimeChan, decodeTime, s.Context.logger)
}

func traceExecutionForRequest(req Request) *traceExecution {
	chatReq, ok := req.(*ChatCompletionsRequest)
	if !ok {
		return nil
	}
	return chatReq.trace
}
