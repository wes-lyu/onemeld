#!/usr/bin/env bash
# SPDX-License-Identifier: Apache-2.0
# Draw, check and redraw: one task through a running launch.yaml server, in plan and iterative modes.
#
# Usage, from the repository root: examples/qwen_umm/demo.sh <base-url> <server-log> [output-dir]
# The server log holds the "UMM composition trace" entries this script checks. Iterative
# mode passes only if the planner saw the drawing before deciding, which takes two turns.
set -euo pipefail

if [ "$#" -lt 2 ]; then
  echo "Usage: $0 <base-url> <server-log> [output-dir]" >&2
  exit 2
else
  :
fi
base_url="$1"
server_log="$2"
output_dir="${3:-results/demo}"
prompt="Draw a red apple on a blue plate. Check whether the picture matches. If it does not, redraw it. Show only the final image."

for mode in iterative plan; do
  python -m examples.qwen_umm.compare \
    --base-url "$base_url" \
    --output-dir "$output_dir/$mode" \
    --mode "$mode" \
    --prompt "$prompt" >/dev/null
  report="$output_dir/$mode/report.json"
  execution_id=$(python -c 'import json, sys; print(json.load(open(sys.argv[1]))["execution_id"])' "$report")
  kinds=$(python -c 'import json, sys; print(" ".join(json.load(open(sys.argv[1]))["segment_kinds"]))' "$report")
  grep "UMM composition trace" "$server_log" | grep "\"execution_id\": \"$execution_id\"" >"$output_dir/$mode/trace.log" || true
  turns=$(grep -c '"event": "plan"' "$output_dir/$mode/trace.log" || true)
  dispatches=$(grep '"event": "dispatch"' "$output_dir/$mode/trace.log" | grep -o '"stage": "[a-z_]*"' | tr '\n' ' ' || true)
  echo "$mode: execution $execution_id, segments [$kinds], planner turns $turns, dispatches: $dispatches"
  if [[ " $kinds " != *" image "* ]]; then
    echo "FAIL $mode: no image segment" >&2
    exit 1
  elif [ "$mode" = "iterative" ] && [ "$turns" -lt 2 ]; then
    echo "FAIL iterative: expected at least two planner turns, saw $turns" >&2
    exit 1
  elif [ "$mode" = "plan" ] && [ "$turns" -ne 1 ]; then
    echo "FAIL plan: expected one planner reply, saw $turns" >&2
    exit 1
  else
    echo "PASS $mode: outputs and trace in $output_dir/$mode"
  fi
done
