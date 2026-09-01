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
	"flag"
	"fmt"
	"os"

	"github.com/llm-d/llm-d-inference-sim/pkg/moetrace"
)

func main() {
	if len(os.Args) < 2 {
		usage()
		os.Exit(2)
	}

	var err error
	switch os.Args[1] {
	case "convert":
		err = runConvert(os.Args[2:])
	case "inspect":
		err = runInspect(os.Args[2:])
	case "validate":
		err = runValidate(os.Args[2:])
	case "help", "-h", "--help":
		usage()
		return
	default:
		fmt.Fprintf(os.Stderr, "unknown command %q\n\n", os.Args[1])
		usage()
		os.Exit(2)
	}
	if err != nil {
		fmt.Fprintf(os.Stderr, "moe-trace-tool: %v\n", err)
		os.Exit(1)
	}
}

func usage() {
	fmt.Fprintln(os.Stderr, `Usage:
  moe-trace-tool convert --input TRACE.json --output TRACE.moetrace
  moe-trace-tool inspect --input TRACE.moetrace [--prompt-id N]
  moe-trace-tool validate --input TRACE.moetrace

The converter accepts JSON produced by trace_qwen_moe.py and writes a compact,
versioned binary trace for simulator replay.`)
}

func runConvert(args []string) error {
	flags := flag.NewFlagSet("convert", flag.ContinueOnError)
	flags.SetOutput(os.Stderr)
	input := flags.String("input", "", "input JSON trace produced by trace_qwen_moe.py")
	output := flags.String("output", "", "output .moetrace path")
	progressEvery := flags.Uint64("progress-every", 1_000_000, "print progress after this many trace records; 0 disables progress")
	if err := flags.Parse(args); err != nil {
		return err
	}
	if flags.NArg() != 0 {
		return fmt.Errorf("unexpected positional arguments: %v", flags.Args())
	}

	summary, err := moetrace.Convert(*input, *output, moetrace.ConvertOptions{
		ProgressEvery: *progressEvery,
		Progress: func(records uint64) {
			fmt.Fprintf(os.Stderr, "processed %d trace records\n", records)
		},
	})
	if err != nil {
		return err
	}
	fmt.Printf("Model:            %s\n", summary.Model)
	fmt.Printf("Prompts:          %d\n", summary.NumPrompts)
	fmt.Printf("MoE layers:       %d\n", summary.NumLayers)
	fmt.Printf("Logical experts:  %d\n", summary.NumExperts)
	fmt.Printf("Top-k:            %d\n", summary.TopK)
	fmt.Printf("Prefill tokens:   %d\n", summary.PrefillTokens)
	fmt.Printf("Decode tokens:    %d\n", summary.DecodeTokens)
	fmt.Printf("Trace records:    %d\n", summary.TraceRecords)
	fmt.Printf("Input bytes:      %d\n", summary.SourceBytes)
	fmt.Printf("Output bytes:     %d\n", summary.OutputBytes)
	fmt.Printf("Source SHA-256:   %s\n", summary.SourceSHA256)
	return nil
}

func runInspect(args []string) error {
	flags := flag.NewFlagSet("inspect", flag.ContinueOnError)
	flags.SetOutput(os.Stderr)
	input := flags.String("input", "", "input .moetrace path")
	promptID := flags.Int("prompt-id", -1, "optional prompt ID to inspect")
	if err := flags.Parse(args); err != nil {
		return err
	}
	if *input == "" {
		return fmt.Errorf("--input is required")
	}
	reader, err := moetrace.Open(*input)
	if err != nil {
		return err
	}
	defer reader.Close()

	metadata := reader.Metadata()
	fmt.Printf("Format version:   %d\n", metadata.FormatVersion)
	fmt.Printf("Model:            %s\n", metadata.Model)
	fmt.Printf("Prompts:          %d\n", metadata.NumPrompts)
	fmt.Printf("MoE layers:       %d\n", len(metadata.SparseLayers))
	fmt.Printf("Sparse layers:    %v\n", metadata.SparseLayers)
	fmt.Printf("Logical experts:  %d\n", metadata.NumExperts)
	fmt.Printf("Top-k:            %d\n", metadata.TopK)
	fmt.Printf("Expert ID bytes:  %d\n", metadata.ExpertIDBytes)
	fmt.Printf("Source bytes:     %d\n", reader.SourceSize())
	fmt.Printf("File bytes:       %d\n", reader.FileSize())
	fmt.Printf("Source SHA-256:   %s\n", reader.SourceSHA256())

	if *promptID >= 0 {
		prompt, err := reader.ReadPrompt(*promptID)
		if err != nil {
			return err
		}
		fmt.Printf("\nPrompt %d\n", *promptID)
		fmt.Printf("Input tokens:     %d\n", len(prompt.InputTokenIDs))
		fmt.Printf("Decode tokens:    %d\n", len(prompt.DecodeTokenIDs))
		fmt.Printf("Prompt text:      %s\n", prompt.Metadata.Prompt)
		fmt.Printf("Generated text:   %s\n", prompt.Metadata.GeneratedText)
		fmt.Printf("Input token IDs:  %v\n", prefix(prompt.InputTokenIDs, 16))
		fmt.Printf("Decode token IDs: %v\n", prefix(prompt.DecodeTokenIDs, 16))
	}
	return nil
}

func runValidate(args []string) error {
	flags := flag.NewFlagSet("validate", flag.ContinueOnError)
	flags.SetOutput(os.Stderr)
	input := flags.String("input", "", "input .moetrace path")
	if err := flags.Parse(args); err != nil {
		return err
	}
	if *input == "" {
		return fmt.Errorf("--input is required")
	}
	reader, err := moetrace.Open(*input)
	if err != nil {
		return err
	}
	defer reader.Close()
	if err := reader.Validate(); err != nil {
		return err
	}
	fmt.Printf("valid MoE trace: %s\n", *input)
	return nil
}

func prefix(values []uint32, maxValues int) []uint32 {
	if len(values) <= maxValues {
		return values
	}
	return values[:maxValues]
}
