# MoE trace tool

`moe-trace-tool` converts the JSON emitted by
`trace_qwen_moe.py` into a compact binary `.moetrace` file for simulator replay.
The conversion is separate from the serving process so large JSON parsing does
not enter request latency measurements.

## Build

```bash
go build -o bin/moe-trace-tool ./cmd/moe-trace-tool
```

## Convert

```bash
bin/moe-trace-tool convert \
  --input instructcoder_2000_both.json \
  --output instructcoder_2000_both.moetrace
```

The converter uses `encoding/json.Decoder` and keeps only one prompt's routing
data in memory at a time. Input metadata and trace records must follow the
layout produced by `trace_qwen_moe.py`: prompt metadata appears before the
`trace` array, prompt indices are sequential, and trace records are grouped by
prompt.

Conversion validates model metadata, prompt token counts, sparse-layer IDs,
token IDs, top-k expert IDs, duplicate routes, missing routes, and token-ID
consistency across layers. The output is written to a temporary file and is
renamed to the requested path only after the complete input validates. Existing
output files are not overwritten.

The command prints the source SHA-256 digest after conversion. The same digest
is stored in the `.moetrace` header so benchmark results can identify the exact
source JSON.

Use `--progress-every 0` to disable progress output, or set a record interval:

```bash
bin/moe-trace-tool convert \
  --input trace.json \
  --output trace.moetrace \
  --progress-every 5000000
```

## Inspect

```bash
bin/moe-trace-tool inspect --input instructcoder_2000_both.moetrace
```

Inspect one prompt:

```bash
bin/moe-trace-tool inspect \
  --input instructcoder_2000_both.moetrace \
  --prompt-id 17
```

## Validate

```bash
bin/moe-trace-tool validate --input instructcoder_2000_both.moetrace
```

## Binary layout

Version 1 stores a fixed header, JSON metadata, contiguous prompt data blocks,
and a fixed-size prompt index. Each prompt block stores:

- input token IDs
- decode token IDs
- precomputed prefill expert counts as `[sparse_layer][logical_expert]`
- prefill routes as `[sparse_layer][position][top_k]`
- decode routes as `[decode_position][sparse_layer][top_k]`

Expert IDs use one byte when the trace has at most 256 logical experts and two
bytes otherwise. Prompt text and generated text are retained in metadata.
Per-layer token strings and gate weights are intentionally omitted from the hot
routing representation because the simulator only needs token IDs and selected
logical experts for routing and placement timing.
