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

package moetrace

import (
	"encoding/binary"
	"encoding/json"
	"fmt"
	"io"
	"os"
)

type traceWriter struct {
	file          *os.File
	offset        uint64
	metadata      Metadata
	metadataBytes []byte
	expertBytes   int
	index         []indexEntry
}

func newTraceWriter(file *os.File, source sourceMetadata) (*traceWriter, error) {
	expertBytes, err := expertIDBytes(source.NumExperts)
	if err != nil {
		return nil, err
	}
	prompts := make([]PromptMetadata, len(source.Prompts))
	for i, prompt := range source.Prompts {
		prompts[i] = PromptMetadata{
			Index:         prompt.Index,
			Prompt:        prompt.Prompt,
			InputTokens:   prompt.InputTokens,
			DecodeTokens:  prompt.DecodeTokens,
			GeneratedText: prompt.GeneratedText,
		}
	}
	metadata := Metadata{
		FormatVersion: FormatVersion,
		Model:         source.Model,
		NumExperts:    source.NumExperts,
		TopK:          source.TopK,
		SparseLayers:  append([]int(nil), source.SparseLayers...),
		NumPrompts:    source.NumPrompts,
		ExpertIDBytes: expertBytes,
		Prompts:       prompts,
	}
	metadataBytes, err := json.Marshal(metadata)
	if err != nil {
		return nil, fmt.Errorf("encode output metadata: %w", err)
	}
	if err := writeAll(file, make([]byte, headerSize)); err != nil {
		return nil, fmt.Errorf("write header placeholder: %w", err)
	}
	if err := writeAll(file, metadataBytes); err != nil {
		return nil, fmt.Errorf("write metadata: %w", err)
	}
	return &traceWriter{
		file:          file,
		offset:        uint64(headerSize + len(metadataBytes)),
		metadata:      metadata,
		metadataBytes: metadataBytes,
		expertBytes:   expertBytes,
		index:         make([]indexEntry, 0, source.NumPrompts),
	}, nil
}

func (w *traceWriter) writePrompt(prompt *promptAccumulator) error {
	if err := prompt.validateComplete(); err != nil {
		return err
	}
	expectedID := len(w.index)
	if prompt.meta.Index != expectedID {
		return fmt.Errorf("expected prompt %d, got %d", expectedID, prompt.meta.Index)
	}
	start := w.offset
	header := make([]byte, promptBlockHeaderSize)
	binary.LittleEndian.PutUint32(header[0:4], uint32(prompt.meta.Index))
	binary.LittleEndian.PutUint32(header[4:8], uint32(prompt.meta.InputTokens))
	binary.LittleEndian.PutUint32(header[8:12], uint32(prompt.meta.DecodeTokens))
	binary.LittleEndian.PutUint32(header[12:16], uint32(prompt.numLayers))
	binary.LittleEndian.PutUint32(header[16:20], uint32(prompt.numExperts))
	binary.LittleEndian.PutUint32(header[20:24], uint32(prompt.topK))
	binary.LittleEndian.PutUint32(header[24:28], uint32(w.expertBytes))
	if err := w.write(header); err != nil {
		return err
	}
	if err := w.writeUint32Slice(prompt.inputTokenIDs); err != nil {
		return err
	}
	if err := w.writeUint32Slice(prompt.decodeTokenIDs); err != nil {
		return err
	}
	if err := w.writeUint32Slice(prompt.prefillCounts); err != nil {
		return err
	}
	if err := w.writeExpertSlice(prompt.prefillRoutes); err != nil {
		return err
	}
	if err := w.writeExpertSlice(prompt.decodeRoutes); err != nil {
		return err
	}
	length := w.offset - start
	expectedLength, err := promptBlockLength(
		uint64(prompt.meta.InputTokens), uint64(prompt.meta.DecodeTokens), uint64(prompt.numLayers),
		uint64(prompt.numExperts), uint64(prompt.topK), uint64(w.expertBytes),
	)
	if err != nil {
		return err
	}
	if length != expectedLength {
		return fmt.Errorf("internal prompt block length mismatch: wrote %d, expected %d", length, expectedLength)
	}
	w.index = append(w.index, indexEntry{
		PromptID:     uint32(prompt.meta.Index),
		Offset:       start,
		Length:       length,
		InputTokens:  uint32(prompt.meta.InputTokens),
		DecodeTokens: uint32(prompt.meta.DecodeTokens),
	})
	return nil
}

func (w *traceWriter) finish(sourceSize uint64, sourceHash [32]byte) error {
	indexOffset := w.offset
	for _, entry := range w.index {
		buf := make([]byte, indexEntrySize)
		binary.LittleEndian.PutUint32(buf[0:4], entry.PromptID)
		binary.LittleEndian.PutUint64(buf[8:16], entry.Offset)
		binary.LittleEndian.PutUint64(buf[16:24], entry.Length)
		binary.LittleEndian.PutUint32(buf[24:28], entry.InputTokens)
		binary.LittleEndian.PutUint32(buf[28:32], entry.DecodeTokens)
		if err := w.write(buf); err != nil {
			return fmt.Errorf("write prompt index: %w", err)
		}
	}
	if len(w.index) != w.metadata.NumPrompts {
		return fmt.Errorf("wrote %d prompt blocks; expected %d", len(w.index), w.metadata.NumPrompts)
	}
	h := fileHeader{
		Version:        FormatVersion,
		MetadataOffset: headerSize,
		MetadataLength: uint64(len(w.metadataBytes)),
		DataOffset:     uint64(headerSize + len(w.metadataBytes)),
		IndexOffset:    indexOffset,
		IndexLength:    uint64(len(w.index) * indexEntrySize),
		SourceSize:     sourceSize,
		NumPrompts:     uint32(w.metadata.NumPrompts),
		ExpertIDBytes:  uint32(w.expertBytes),
		SourceSHA256:   sourceHash,
	}
	if _, err := w.file.Seek(0, io.SeekStart); err != nil {
		return fmt.Errorf("seek output header: %w", err)
	}
	if err := writeHeader(w.file, h); err != nil {
		return fmt.Errorf("write output header: %w", err)
	}
	return nil
}

func (w *traceWriter) write(data []byte) error {
	if err := writeAll(w.file, data); err != nil {
		return err
	}
	w.offset += uint64(len(data))
	return nil
}

func (w *traceWriter) writeUint32Slice(values []uint32) error {
	buf := make([]byte, len(values)*4)
	for i, value := range values {
		binary.LittleEndian.PutUint32(buf[i*4:(i+1)*4], value)
	}
	return w.write(buf)
}

func (w *traceWriter) writeExpertSlice(values []uint16) error {
	if w.expertBytes == 1 {
		buf := make([]byte, len(values))
		for i, value := range values {
			if value > 255 {
				return fmt.Errorf("expert ID %d does not fit in one byte", value)
			}
			buf[i] = byte(value)
		}
		return w.write(buf)
	}
	buf := make([]byte, len(values)*2)
	for i, value := range values {
		binary.LittleEndian.PutUint16(buf[i*2:(i+1)*2], value)
	}
	return w.write(buf)
}

func writeAll(w io.Writer, data []byte) error {
	for len(data) > 0 {
		n, err := w.Write(data)
		if err != nil {
			return err
		}
		if n == 0 {
			return io.ErrShortWrite
		}
		data = data[n:]
	}
	return nil
}
