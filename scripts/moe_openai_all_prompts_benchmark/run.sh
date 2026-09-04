#!/usr/bin/env bash
set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
REPO_ROOT="$(cd "${SCRIPT_DIR}/../.." && pwd)"

DATASET_PATH=""
LIMIT=0
ARGS=("$@")
for ((i = 0; i < ${#ARGS[@]}; i++)); do
  case "${ARGS[$i]}" in
    --dataset)
      if ((i + 1 < ${#ARGS[@]})); then
        DATASET_PATH="${ARGS[$((i + 1))]}"
      fi
      ;;
    --dataset=*)
      DATASET_PATH="${ARGS[$i]#--dataset=}"
      ;;
    --limit)
      if ((i + 1 < ${#ARGS[@]})); then
        LIMIT="${ARGS[$((i + 1))]}"
      fi
      ;;
    --limit=*)
      LIMIT="${ARGS[$i]#--limit=}"
      ;;
  esac
done

if [[ -n "${DATASET_PATH}" && -f "${DATASET_PATH}" ]]; then
  NUM_PROMPTS="$(awk 'NF {count++} END {print count+0}' "${DATASET_PATH}")"
  if [[ "${LIMIT}" =~ ^[0-9]+$ ]] && ((LIMIT > 0 && LIMIT < NUM_PROMPTS)); then
    NUM_PROMPTS="${LIMIT}"
  fi
  CURRENT_NOFILE="$(ulimit -n)"
  REQUIRED_NOFILE=$((NUM_PROMPTS + 256))
  if [[ "${CURRENT_NOFILE}" =~ ^[0-9]+$ ]] && ((CURRENT_NOFILE < REQUIRED_NOFILE)); then
    if ulimit -n "${REQUIRED_NOFILE}" 2>/dev/null; then
      echo "raised open-file limit to ${REQUIRED_NOFILE} for ${NUM_PROMPTS} concurrent requests" >&2
    else
      echo "warning: open-file limit is ${CURRENT_NOFILE}; ${NUM_PROMPTS} concurrent requests may exhaust file descriptors" >&2
    fi
  fi
fi

cd "${REPO_ROOT}"
exec go run ./scripts/moe_openai_all_prompts_benchmark "$@"
