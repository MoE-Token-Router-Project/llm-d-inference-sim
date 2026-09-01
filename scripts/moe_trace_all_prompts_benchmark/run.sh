#!/usr/bin/env bash
set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
REPO_ROOT="$(cd "${SCRIPT_DIR}/../.." && pwd)"

trace_path=""
args=("$@")
for ((i = 0; i < ${#args[@]}; i++)); do
  case "${args[$i]}" in
    --trace)
      if ((i + 1 < ${#args[@]})); then
        trace_path="${args[$((i + 1))]}"
      fi
      ;;
    --trace=*)
      trace_path="${args[$i]#--trace=}"
      ;;
  esac
done

if [[ -n "${trace_path}" && -f "${trace_path}" ]]; then
  # .moetrace v1 stores NumPrompts as a little-endian uint32 at byte offset 64.
  num_prompts="$(od -An -j 64 -N 4 -tu4 "${trace_path}" | tr -d '[:space:]')"
  if [[ "${num_prompts}" =~ ^[0-9]+$ ]] && ((num_prompts > 0)); then
    required_nofile=$((num_prompts + 256))
    soft_nofile="$(ulimit -Sn)"
    hard_nofile="$(ulimit -Hn)"

    if [[ "${hard_nofile}" != "unlimited" ]] && ((hard_nofile < required_nofile)); then
      echo "error: ${num_prompts} simultaneous trace requests need at least ${required_nofile} open files, but the hard limit is ${hard_nofile}" >&2
      echo "raise the process hard open-file limit before running this benchmark" >&2
      exit 1
    fi

    if [[ "${soft_nofile}" != "unlimited" ]] && ((soft_nofile < required_nofile)); then
      if ! ulimit -Sn "${required_nofile}"; then
        echo "error: failed to raise the open-file soft limit from ${soft_nofile} to ${required_nofile}" >&2
        exit 1
      fi
      echo "raised open-file soft limit from ${soft_nofile} to ${required_nofile} for ${num_prompts} simultaneous requests" >&2
    fi
  fi
fi

cd "${REPO_ROOT}"
exec go run ./scripts/moe_trace_all_prompts_benchmark "$@"
