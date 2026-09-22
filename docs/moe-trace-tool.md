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

The tool writes version 2 while the reader accepts versions 1 and 2. Both
versions keep the `MOETRC01` magic, fixed header, JSON metadata, contiguous
prompt data blocks, and fixed-size prompt index. Version 2 keeps the complete
version 1 payload and appends source-GPU data to every prompt block.

Each prompt block stores:

- input token IDs
- decode token IDs
- precomputed prefill expert counts as `[sparse_layer][logical_expert]`
- prefill routes as `[sparse_layer][position][top_k]`
- decode routes as `[decode_position][sparse_layer][top_k]`
- version 2 prefill source GPUs as `[sparse_layer][position]`
- version 2 decode source GPUs as `[decode_position][sparse_layer]`

Source GPU IDs use one byte. Values 0 through 127 are physical GPU IDs, 255
means unknown or not recorded, and 128 through 254 are invalid. Version 2
metadata records `source_gpu_bytes: 1`. The JSON converter accepts
`source_gpu` on each trace record and also accepts `gpu_source` as an alias;
if neither is present it writes 255.

Expert IDs use one byte when the trace has at most 256 logical experts and two
bytes otherwise. Prompt text and generated text are retained in metadata.
Per-layer token strings and gate weights are intentionally omitted from the hot
routing representation because the simulator only needs token IDs and selected
logical experts for routing and placement timing.
