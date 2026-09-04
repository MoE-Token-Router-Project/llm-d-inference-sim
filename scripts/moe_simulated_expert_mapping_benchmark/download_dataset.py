#!/usr/bin/env python3
"""Download prompt rows from Hugging Face into benchmark JSONL."""

import argparse
import json
import os
import sys
from typing import Any


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Download a Hugging Face dataset and write normalized prompt JSONL."
    )
    parser.add_argument("--dataset", required=True, help="Hugging Face dataset name")
    parser.add_argument("--dataset-config", help="optional Hugging Face dataset config")
    parser.add_argument("--split", default="train", help="dataset split, default: train")
    parser.add_argument(
        "--prompt-field",
        required=True,
        help="field containing prompt text; dot notation is supported for nested fields",
    )
    parser.add_argument(
        "--max-completion-tokens-field",
        help="optional numeric field copied into max_completion_tokens",
    )
    parser.add_argument(
        "--max-completion-tokens",
        type=int,
        default=128,
        help="constant max_completion_tokens when no token-count field is supplied",
    )
    parser.add_argument("--limit", type=int, default=2000, help="number of rows to write")
    parser.add_argument(
        "--output",
        default="datasets/moe_simulated_expert_mapping_benchmark.jsonl",
        help="output JSONL path",
    )
    return parser.parse_args()


def get_field(row: Any, path: str) -> Any:
    value = row
    for part in path.split("."):
        if not isinstance(value, dict) or part not in value:
            raise KeyError(path)
        value = value[part]
    return value


def main() -> int:
    args = parse_args()
    if args.limit <= 0:
        print("--limit must be positive", file=sys.stderr)
        return 2
    if args.max_completion_tokens <= 0:
        print("--max-completion-tokens must be positive", file=sys.stderr)
        return 2

    try:
        from datasets import load_dataset
    except ImportError:
        print(
            "missing Python package 'datasets'; install it with: "
            "python3 -m pip install datasets",
            file=sys.stderr,
        )
        return 1

    load_args = [args.dataset]
    if args.dataset_config:
        load_args.append(args.dataset_config)
    dataset = load_dataset(*load_args, split=args.split, streaming=True)

    output = os.path.abspath(args.output)
    os.makedirs(os.path.dirname(output), exist_ok=True)
    temporary = output + ".tmp"
    written = 0
    try:
        with open(temporary, "w", encoding="utf-8") as handle:
            for row_index, row in enumerate(dataset):
                try:
                    prompt = get_field(row, args.prompt_field)
                except KeyError:
                    raise ValueError(
                        f"row {row_index} does not contain prompt field {args.prompt_field!r}"
                    )
                if not isinstance(prompt, str) or not prompt.strip():
                    raise ValueError(
                        f"row {row_index} prompt field {args.prompt_field!r} is not a non-empty string"
                    )

                max_completion_tokens = args.max_completion_tokens
                if args.max_completion_tokens_field:
                    try:
                        raw_tokens = get_field(row, args.max_completion_tokens_field)
                    except KeyError:
                        raise ValueError(
                            f"row {row_index} does not contain token-count field "
                            f"{args.max_completion_tokens_field!r}"
                        )
                    if isinstance(raw_tokens, bool) or not isinstance(raw_tokens, int):
                        raise ValueError(
                            f"row {row_index} token-count field must be an integer"
                        )
                    if raw_tokens <= 0:
                        raise ValueError(
                            f"row {row_index} token-count field must be positive"
                        )
                    max_completion_tokens = raw_tokens

                json.dump(
                    {
                        "prompt": prompt,
                        "max_completion_tokens": max_completion_tokens,
                    },
                    handle,
                    ensure_ascii=False,
                )
                handle.write("\n")
                written += 1
                if written >= args.limit:
                    break

        if written == 0:
            raise ValueError("dataset produced no rows")
        os.replace(temporary, output)
    except Exception:
        try:
            os.remove(temporary)
        except FileNotFoundError:
            pass
        raise

    print(f"wrote {written} prompts to {output}")
    return 0


if __name__ == "__main__":
    try:
        raise SystemExit(main())
    except Exception as exc:
        print(f"dataset download failed: {exc}", file=sys.stderr)
        raise SystemExit(1)
