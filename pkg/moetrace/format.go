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
	"errors"
	"fmt"
	"io"
	"math"
)

const (
	FormatVersion         uint32 = 1
	headerSize                   = 112
	indexEntrySize               = 32
	promptBlockHeaderSize        = 32
)

var fileMagic = [8]byte{'M', 'O', 'E', 'T', 'R', 'C', '0', '1'}

type Metadata struct {
	FormatVersion uint32           `json:"format_version"`
	Model         string           `json:"model"`
	NumExperts    int              `json:"num_experts"`
	TopK          int              `json:"top_k"`
	SparseLayers  []int            `json:"sparse_layers"`
	NumPrompts    int              `json:"num_prompts"`
	ExpertIDBytes int              `json:"expert_id_bytes"`
	Prompts       []PromptMetadata `json:"prompts"`
}

type PromptMetadata struct {
	Index         int    `json:"index"`
	Prompt        string `json:"prompt"`
	InputTokens   int    `json:"input_tokens"`
	DecodeTokens  int    `json:"decode_tokens"`
	GeneratedText string `json:"generated_text"`
}

type fileHeader struct {
	Version        uint32
	MetadataOffset uint64
	MetadataLength uint64
	DataOffset     uint64
	IndexOffset    uint64
	IndexLength    uint64
	SourceSize     uint64
	NumPrompts     uint32
	ExpertIDBytes  uint32
	SourceSHA256   [32]byte
}

type indexEntry struct {
	PromptID     uint32
	Offset       uint64
	Length       uint64
	InputTokens  uint32
	DecodeTokens uint32
}

func expertIDBytes(numExperts int) (int, error) {
	switch {
	case numExperts < 1:
		return 0, errors.New("num_experts must be positive")
	case numExperts <= 256:
		return 1, nil
	case numExperts <= 65536:
		return 2, nil
	default:
		return 0, fmt.Errorf("num_experts %d exceeds the format limit of 65536", numExperts)
	}
}

func writeHeader(w io.Writer, h fileHeader) error {
	buf := make([]byte, headerSize)
	copy(buf[0:8], fileMagic[:])
	binary.LittleEndian.PutUint32(buf[8:12], h.Version)
	binary.LittleEndian.PutUint32(buf[12:16], headerSize)
	binary.LittleEndian.PutUint64(buf[16:24], h.MetadataOffset)
	binary.LittleEndian.PutUint64(buf[24:32], h.MetadataLength)
	binary.LittleEndian.PutUint64(buf[32:40], h.DataOffset)
	binary.LittleEndian.PutUint64(buf[40:48], h.IndexOffset)
	binary.LittleEndian.PutUint64(buf[48:56], h.IndexLength)
	binary.LittleEndian.PutUint64(buf[56:64], h.SourceSize)
	binary.LittleEndian.PutUint32(buf[64:68], h.NumPrompts)
	binary.LittleEndian.PutUint32(buf[68:72], h.ExpertIDBytes)
	binary.LittleEndian.PutUint32(buf[72:76], indexEntrySize)
	copy(buf[80:112], h.SourceSHA256[:])
	_, err := w.Write(buf)
	return err
}

func readHeader(r io.ReaderAt) (fileHeader, error) {
	buf := make([]byte, headerSize)
	if _, err := r.ReadAt(buf, 0); err != nil {
		return fileHeader{}, fmt.Errorf("read header: %w", err)
	}
	var magic [8]byte
	copy(magic[:], buf[0:8])
	if magic != fileMagic {
		return fileHeader{}, fmt.Errorf("invalid MoE trace magic %q", string(magic[:]))
	}
	if got := binary.LittleEndian.Uint32(buf[12:16]); got != headerSize {
		return fileHeader{}, fmt.Errorf("unsupported header size %d", got)
	}
	if got := binary.LittleEndian.Uint32(buf[72:76]); got != indexEntrySize {
		return fileHeader{}, fmt.Errorf("unsupported index entry size %d", got)
	}
	h := fileHeader{
		Version:        binary.LittleEndian.Uint32(buf[8:12]),
		MetadataOffset: binary.LittleEndian.Uint64(buf[16:24]),
		MetadataLength: binary.LittleEndian.Uint64(buf[24:32]),
		DataOffset:     binary.LittleEndian.Uint64(buf[32:40]),
		IndexOffset:    binary.LittleEndian.Uint64(buf[40:48]),
		IndexLength:    binary.LittleEndian.Uint64(buf[48:56]),
		SourceSize:     binary.LittleEndian.Uint64(buf[56:64]),
		NumPrompts:     binary.LittleEndian.Uint32(buf[64:68]),
		ExpertIDBytes:  binary.LittleEndian.Uint32(buf[68:72]),
	}
	copy(h.SourceSHA256[:], buf[80:112])
	if h.Version != FormatVersion {
		return fileHeader{}, fmt.Errorf("unsupported MoE trace version %d", h.Version)
	}
	if h.ExpertIDBytes != 1 && h.ExpertIDBytes != 2 {
		return fileHeader{}, fmt.Errorf("unsupported expert ID width %d", h.ExpertIDBytes)
	}
	return h, nil
}

func readIndexEntry(r io.ReaderAt, offset int64) (indexEntry, error) {
	buf := make([]byte, indexEntrySize)
	if _, err := r.ReadAt(buf, offset); err != nil {
		return indexEntry{}, err
	}
	return indexEntry{
		PromptID:     binary.LittleEndian.Uint32(buf[0:4]),
		Offset:       binary.LittleEndian.Uint64(buf[8:16]),
		Length:       binary.LittleEndian.Uint64(buf[16:24]),
		InputTokens:  binary.LittleEndian.Uint32(buf[24:28]),
		DecodeTokens: binary.LittleEndian.Uint32(buf[28:32]),
	}, nil
}

func promptBlockLength(inputTokens, decodeTokens, numLayers, numExperts, topK, expertBytes uint64) (uint64, error) {
	inputBytes, err := checkedMul(inputTokens, 4)
	if err != nil {
		return 0, err
	}
	decodeBytes, err := checkedMul(decodeTokens, 4)
	if err != nil {
		return 0, err
	}
	countEntries, err := checkedMul(numLayers, numExperts)
	if err != nil {
		return 0, err
	}
	countBytes, err := checkedMul(countEntries, 4)
	if err != nil {
		return 0, err
	}
	allTokens, err := checkedAdd(inputTokens, decodeTokens)
	if err != nil {
		return 0, err
	}
	routeEntries, err := checkedMul(allTokens, numLayers)
	if err != nil {
		return 0, err
	}
	routeEntries, err = checkedMul(routeEntries, topK)
	if err != nil {
		return 0, err
	}
	routeBytes, err := checkedMul(routeEntries, expertBytes)
	if err != nil {
		return 0, err
	}
	total := uint64(promptBlockHeaderSize)
	for _, n := range []uint64{inputBytes, decodeBytes, countBytes, routeBytes} {
		total, err = checkedAdd(total, n)
		if err != nil {
			return 0, err
		}
	}
	return total, nil
}

func checkedMul(a, b uint64) (uint64, error) {
	if a != 0 && b > math.MaxUint64/a {
		return 0, errors.New("MoE trace size overflows uint64")
	}
	return a * b, nil
}

func checkedAdd(a, b uint64) (uint64, error) {
	if b > math.MaxUint64-a {
		return 0, errors.New("MoE trace size overflows uint64")
	}
	return a + b, nil
}
