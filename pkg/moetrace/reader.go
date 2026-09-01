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
	"encoding/hex"
	"encoding/json"
	"errors"
	"fmt"
	"io"
	"os"
)

type Reader struct {
	file     *os.File
	fileSize uint64
	header   fileHeader
	metadata Metadata
	index    []indexEntry
}

type PromptData struct {
	Metadata       PromptMetadata
	InputTokenIDs  []uint32
	DecodeTokenIDs []uint32
	PrefillCounts  []uint32
	PrefillRoutes  []uint16
	DecodeRoutes   []uint16
	numLayers      int
	numExperts     int
	topK           int
}

func Open(path string) (*Reader, error) {
	file, err := os.Open(path)
	if err != nil {
		return nil, err
	}
	closeOnError := true
	defer func() {
		if closeOnError {
			_ = file.Close()
		}
	}()
	stat, err := file.Stat()
	if err != nil {
		return nil, fmt.Errorf("stat MoE trace: %w", err)
	}
	if stat.Size() < headerSize {
		return nil, errors.New("MoE trace is shorter than its header")
	}
	h, err := readHeader(file)
	if err != nil {
		return nil, err
	}
	size := uint64(stat.Size())
	if err := validateRegion(h.MetadataOffset, h.MetadataLength, size, "metadata"); err != nil {
		return nil, err
	}
	if err := validateRegion(h.IndexOffset, h.IndexLength, size, "index"); err != nil {
		return nil, err
	}
	if h.MetadataOffset != headerSize {
		return nil, fmt.Errorf("unexpected metadata offset %d", h.MetadataOffset)
	}
	if h.DataOffset != h.MetadataOffset+h.MetadataLength {
		return nil, errors.New("data offset does not follow metadata")
	}
	if h.IndexLength != uint64(h.NumPrompts)*indexEntrySize {
		return nil, fmt.Errorf("index length %d does not match %d prompts", h.IndexLength, h.NumPrompts)
	}

	metadataBytes := make([]byte, h.MetadataLength)
	if _, err := file.ReadAt(metadataBytes, int64(h.MetadataOffset)); err != nil {
		return nil, fmt.Errorf("read metadata: %w", err)
	}
	var metadata Metadata
	if err := json.Unmarshal(metadataBytes, &metadata); err != nil {
		return nil, fmt.Errorf("decode metadata: %w", err)
	}
	if metadata.FormatVersion != FormatVersion || metadata.NumPrompts != int(h.NumPrompts) || metadata.ExpertIDBytes != int(h.ExpertIDBytes) {
		return nil, errors.New("metadata does not match file header")
	}
	if len(metadata.Prompts) != metadata.NumPrompts || len(metadata.SparseLayers) == 0 {
		return nil, errors.New("invalid MoE trace metadata")
	}
	expectedExpertBytes, err := expertIDBytes(metadata.NumExperts)
	if err != nil {
		return nil, err
	}
	if expectedExpertBytes != metadata.ExpertIDBytes {
		return nil, errors.New("expert ID width does not match expert count")
	}

	index := make([]indexEntry, h.NumPrompts)
	for i := range index {
		entry, err := readIndexEntry(file, int64(h.IndexOffset)+int64(i*indexEntrySize))
		if err != nil {
			return nil, fmt.Errorf("read prompt index %d: %w", i, err)
		}
		if entry.PromptID != uint32(i) {
			return nil, fmt.Errorf("prompt index entry %d contains prompt ID %d", i, entry.PromptID)
		}
		index[i] = entry
	}

	r := &Reader{file: file, fileSize: size, header: h, metadata: metadata, index: index}
	if err := r.Validate(); err != nil {
		return nil, err
	}
	closeOnError = false
	return r, nil
}

func (r *Reader) Close() error {
	return r.file.Close()
}

func (r *Reader) Metadata() Metadata {
	return r.metadata
}

func (r *Reader) SourceSHA256() string {
	return hex.EncodeToString(r.header.SourceSHA256[:])
}

func (r *Reader) SourceSize() uint64 {
	return r.header.SourceSize
}

func (r *Reader) FileSize() uint64 {
	return r.fileSize
}

func (r *Reader) Validate() error {
	if r.header.DataOffset > r.header.IndexOffset {
		return errors.New("data region starts after prompt index")
	}
	expectedOffset := r.header.DataOffset
	for i, entry := range r.index {
		if entry.Offset != expectedOffset {
			return fmt.Errorf("prompt %d starts at offset %d; expected %d", i, entry.Offset, expectedOffset)
		}
		if err := validateRegion(entry.Offset, entry.Length, r.fileSize, fmt.Sprintf("prompt %d", i)); err != nil {
			return err
		}
		if entry.Offset+entry.Length > r.header.IndexOffset {
			return fmt.Errorf("prompt %d overlaps the prompt index", i)
		}
		promptHeader := make([]byte, promptBlockHeaderSize)
		if _, err := r.file.ReadAt(promptHeader, int64(entry.Offset)); err != nil {
			return fmt.Errorf("read prompt %d header: %w", i, err)
		}
		promptID := binary.LittleEndian.Uint32(promptHeader[0:4])
		inputTokens := binary.LittleEndian.Uint32(promptHeader[4:8])
		decodeTokens := binary.LittleEndian.Uint32(promptHeader[8:12])
		numLayers := binary.LittleEndian.Uint32(promptHeader[12:16])
		numExperts := binary.LittleEndian.Uint32(promptHeader[16:20])
		topK := binary.LittleEndian.Uint32(promptHeader[20:24])
		expertBytes := binary.LittleEndian.Uint32(promptHeader[24:28])
		if promptID != uint32(i) || inputTokens != entry.InputTokens || decodeTokens != entry.DecodeTokens {
			return fmt.Errorf("prompt %d block header does not match its index entry", i)
		}
		if int(numLayers) != len(r.metadata.SparseLayers) || int(numExperts) != r.metadata.NumExperts || int(topK) != r.metadata.TopK || int(expertBytes) != r.metadata.ExpertIDBytes {
			return fmt.Errorf("prompt %d block dimensions do not match metadata", i)
		}
		expectedLength, err := promptBlockLength(uint64(inputTokens), uint64(decodeTokens), uint64(numLayers), uint64(numExperts), uint64(topK), uint64(expertBytes))
		if err != nil {
			return err
		}
		if entry.Length != expectedLength {
			return fmt.Errorf("prompt %d block length is %d; expected %d", i, entry.Length, expectedLength)
		}
		if r.metadata.Prompts[i].Index != i || r.metadata.Prompts[i].InputTokens != int(inputTokens) || r.metadata.Prompts[i].DecodeTokens != int(decodeTokens) {
			return fmt.Errorf("prompt %d metadata does not match its data block", i)
		}
		expectedOffset += entry.Length
	}
	if expectedOffset != r.header.IndexOffset {
		return fmt.Errorf("prompt data ends at offset %d; index starts at %d", expectedOffset, r.header.IndexOffset)
	}
	if r.header.IndexOffset+r.header.IndexLength != r.fileSize {
		return errors.New("unexpected trailing data after prompt index")
	}
	return nil
}

func (r *Reader) ReadPrompt(promptID int) (*PromptData, error) {
	if promptID < 0 || promptID >= len(r.index) {
		return nil, fmt.Errorf("prompt ID %d is outside [0,%d)", promptID, len(r.index))
	}
	entry := r.index[promptID]
	section := io.NewSectionReader(r.file, int64(entry.Offset+promptBlockHeaderSize), int64(entry.Length-promptBlockHeaderSize))
	p := &PromptData{
		Metadata:       r.metadata.Prompts[promptID],
		InputTokenIDs:  make([]uint32, entry.InputTokens),
		DecodeTokenIDs: make([]uint32, entry.DecodeTokens),
		PrefillCounts:  make([]uint32, len(r.metadata.SparseLayers)*r.metadata.NumExperts),
		PrefillRoutes:  make([]uint16, len(r.metadata.SparseLayers)*int(entry.InputTokens)*r.metadata.TopK),
		DecodeRoutes:   make([]uint16, int(entry.DecodeTokens)*len(r.metadata.SparseLayers)*r.metadata.TopK),
		numLayers:      len(r.metadata.SparseLayers),
		numExperts:     r.metadata.NumExperts,
		topK:           r.metadata.TopK,
	}
	if err := readUint32Slice(section, p.InputTokenIDs); err != nil {
		return nil, fmt.Errorf("read prompt %d input tokens: %w", promptID, err)
	}
	if err := readUint32Slice(section, p.DecodeTokenIDs); err != nil {
		return nil, fmt.Errorf("read prompt %d decode tokens: %w", promptID, err)
	}
	if err := readUint32Slice(section, p.PrefillCounts); err != nil {
		return nil, fmt.Errorf("read prompt %d prefill counts: %w", promptID, err)
	}
	if err := readExpertSlice(section, p.PrefillRoutes, r.metadata.ExpertIDBytes); err != nil {
		return nil, fmt.Errorf("read prompt %d prefill routes: %w", promptID, err)
	}
	if err := readExpertSlice(section, p.DecodeRoutes, r.metadata.ExpertIDBytes); err != nil {
		return nil, fmt.Errorf("read prompt %d decode routes: %w", promptID, err)
	}
	position, err := section.Seek(0, io.SeekCurrent)
	if err != nil {
		return nil, fmt.Errorf("check prompt %d read position: %w", promptID, err)
	}
	if section.Size() != position {
		return nil, fmt.Errorf("prompt %d contains unread data", promptID)
	}
	return p, nil
}

func (p *PromptData) PrefillExpertCount(layerSlot, expert int) (uint32, error) {
	if layerSlot < 0 || layerSlot >= p.numLayers || expert < 0 || expert >= p.numExperts {
		return 0, errors.New("prefill count index out of range")
	}
	return p.PrefillCounts[layerSlot*p.numExperts+expert], nil
}

func (p *PromptData) PrefillExperts(layerSlot, position int) ([]uint16, error) {
	if layerSlot < 0 || layerSlot >= p.numLayers || position < 0 || position >= len(p.InputTokenIDs) {
		return nil, errors.New("prefill route index out of range")
	}
	base := (layerSlot*len(p.InputTokenIDs) + position) * p.topK
	return p.PrefillRoutes[base : base+p.topK], nil
}

func (p *PromptData) DecodeExperts(position, layerSlot int) ([]uint16, error) {
	if position < 0 || position >= len(p.DecodeTokenIDs) || layerSlot < 0 || layerSlot >= p.numLayers {
		return nil, errors.New("decode route index out of range")
	}
	base := (position*p.numLayers + layerSlot) * p.topK
	return p.DecodeRoutes[base : base+p.topK], nil
}

func validateRegion(offset, length, fileSize uint64, name string) error {
	if offset > fileSize || length > fileSize-offset {
		return fmt.Errorf("%s region [%d,%d) exceeds file size %d", name, offset, offset+length, fileSize)
	}
	return nil
}

func readUint32Slice(r io.Reader, values []uint32) error {
	buf := make([]byte, len(values)*4)
	if _, err := io.ReadFull(r, buf); err != nil {
		return err
	}
	for i := range values {
		values[i] = binary.LittleEndian.Uint32(buf[i*4 : (i+1)*4])
	}
	return nil
}

func readExpertSlice(r io.Reader, values []uint16, width int) error {
	if width == 1 {
		buf := make([]byte, len(values))
		if _, err := io.ReadFull(r, buf); err != nil {
			return err
		}
		for i, value := range buf {
			values[i] = uint16(value)
		}
		return nil
	}
	buf := make([]byte, len(values)*2)
	if _, err := io.ReadFull(r, buf); err != nil {
		return err
	}
	for i := range values {
		values[i] = binary.LittleEndian.Uint16(buf[i*2 : (i+1)*2])
	}
	return nil
}
