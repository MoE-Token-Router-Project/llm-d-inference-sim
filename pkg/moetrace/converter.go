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
	"bufio"
	"crypto/sha256"
	"encoding/hex"
	"encoding/json"
	"errors"
	"fmt"
	"io"
	"os"
	"path/filepath"
)

type ConvertOptions struct {
	ProgressEvery uint64
	Progress      func(records uint64)
}

type ConversionSummary struct {
	Model         string
	NumPrompts    int
	NumExperts    int
	TopK          int
	NumLayers     int
	TraceRecords  uint64
	PrefillTokens uint64
	DecodeTokens  uint64
	SourceBytes   uint64
	OutputBytes   uint64
	SourceSHA256  string
}

func Convert(inputPath, outputPath string, options ConvertOptions) (ConversionSummary, error) {
	if inputPath == "" {
		return ConversionSummary{}, errors.New("input path is required")
	}
	if outputPath == "" {
		return ConversionSummary{}, errors.New("output path is required")
	}
	if _, err := os.Stat(outputPath); err == nil {
		return ConversionSummary{}, fmt.Errorf("output file %q already exists", outputPath)
	} else if !errors.Is(err, os.ErrNotExist) {
		return ConversionSummary{}, fmt.Errorf("stat output file: %w", err)
	}

	source, err := os.Open(inputPath)
	if err != nil {
		return ConversionSummary{}, fmt.Errorf("open input: %w", err)
	}
	defer source.Close()
	stat, err := source.Stat()
	if err != nil {
		return ConversionSummary{}, fmt.Errorf("stat input: %w", err)
	}

	outputDir := filepath.Dir(outputPath)
	if err := os.MkdirAll(outputDir, 0o755); err != nil {
		return ConversionSummary{}, fmt.Errorf("create output directory: %w", err)
	}
	temp, err := os.CreateTemp(outputDir, "."+filepath.Base(outputPath)+".tmp-*")
	if err != nil {
		return ConversionSummary{}, fmt.Errorf("create temporary output: %w", err)
	}
	tempPath := temp.Name()
	if err := temp.Chmod(0o644); err != nil {
		_ = temp.Close()
		_ = os.Remove(tempPath)
		return ConversionSummary{}, fmt.Errorf("set output permissions: %w", err)
	}
	keepTemp := false
	defer func() {
		if !keepTemp {
			_ = temp.Close()
			_ = os.Remove(tempPath)
		}
	}()

	hasher := sha256.New()
	tee := io.TeeReader(bufio.NewReaderSize(source, 1<<20), hasher)
	decoder := json.NewDecoder(tee)

	first, err := decoder.Token()
	if err != nil {
		return ConversionSummary{}, fmt.Errorf("read input JSON: %w", err)
	}
	if delim, ok := first.(json.Delim); !ok || delim != '{' {
		return ConversionSummary{}, errors.New("input JSON must be a top-level object")
	}

	var sourceMeta sourceMetadata
	var writer *traceWriter
	var summary ConversionSummary
	traceSeen := false
	for decoder.More() {
		keyToken, err := decoder.Token()
		if err != nil {
			return ConversionSummary{}, fmt.Errorf("read input field: %w", err)
		}
		key, ok := keyToken.(string)
		if !ok {
			return ConversionSummary{}, errors.New("invalid top-level JSON key")
		}
		switch key {
		case "model":
			if traceSeen {
				return ConversionSummary{}, errors.New("model metadata must precede trace")
			}
			err = decoder.Decode(&sourceMeta.Model)
		case "num_experts":
			if traceSeen {
				return ConversionSummary{}, errors.New("num_experts metadata must precede trace")
			}
			err = decoder.Decode(&sourceMeta.NumExperts)
		case "top_k":
			if traceSeen {
				return ConversionSummary{}, errors.New("top_k metadata must precede trace")
			}
			err = decoder.Decode(&sourceMeta.TopK)
		case "sparse_layers":
			if traceSeen {
				return ConversionSummary{}, errors.New("sparse_layers metadata must precede trace")
			}
			err = decoder.Decode(&sourceMeta.SparseLayers)
		case "num_sparse_layers":
			if traceSeen {
				return ConversionSummary{}, errors.New("num_sparse_layers metadata must precede trace")
			}
			err = decoder.Decode(&sourceMeta.NumSparseLayers)
		case "num_prompts":
			if traceSeen {
				return ConversionSummary{}, errors.New("num_prompts metadata must precede trace")
			}
			err = decoder.Decode(&sourceMeta.NumPrompts)
		case "prompts":
			if traceSeen {
				return ConversionSummary{}, errors.New("prompts metadata must precede trace")
			}
			err = decoder.Decode(&sourceMeta.Prompts)
		case "trace":
			if traceSeen {
				return ConversionSummary{}, errors.New("input contains multiple trace arrays")
			}
			traceSeen = true
			if err := validateSourceMetadata(sourceMeta); err != nil {
				return ConversionSummary{}, err
			}
			writer, err = newTraceWriter(temp, sourceMeta)
			if err != nil {
				return ConversionSummary{}, err
			}
			summary, err = convertTraceArray(decoder, writer, sourceMeta, options)
		default:
			var ignored json.RawMessage
			err = decoder.Decode(&ignored)
		}
		if err != nil {
			return ConversionSummary{}, fmt.Errorf("decode field %q: %w", key, err)
		}
	}
	if _, err := decoder.Token(); err != nil {
		return ConversionSummary{}, fmt.Errorf("finish input JSON: %w", err)
	}
	if !traceSeen || writer == nil {
		return ConversionSummary{}, errors.New("input JSON does not contain a trace array")
	}

	var extra any
	if err := decoder.Decode(&extra); err != io.EOF {
		if err == nil {
			return ConversionSummary{}, errors.New("input contains data after the top-level JSON object")
		}
		return ConversionSummary{}, fmt.Errorf("finish input JSON: %w", err)
	}
	if _, err := io.Copy(io.Discard, tee); err != nil {
		return ConversionSummary{}, fmt.Errorf("finish hashing input: %w", err)
	}

	var sourceHash [32]byte
	copy(sourceHash[:], hasher.Sum(nil))
	if err := writer.finish(uint64(stat.Size()), sourceHash); err != nil {
		return ConversionSummary{}, err
	}
	if err := temp.Sync(); err != nil {
		return ConversionSummary{}, fmt.Errorf("sync output: %w", err)
	}
	if err := temp.Close(); err != nil {
		return ConversionSummary{}, fmt.Errorf("close output: %w", err)
	}
	if err := os.Rename(tempPath, outputPath); err != nil {
		return ConversionSummary{}, fmt.Errorf("publish output: %w", err)
	}
	keepTemp = true

	outputStat, err := os.Stat(outputPath)
	if err != nil {
		return ConversionSummary{}, fmt.Errorf("stat output: %w", err)
	}
	summary.SourceBytes = uint64(stat.Size())
	summary.OutputBytes = uint64(outputStat.Size())
	summary.SourceSHA256 = hex.EncodeToString(sourceHash[:])
	return summary, nil
}
