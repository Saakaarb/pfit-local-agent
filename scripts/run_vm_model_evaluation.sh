#!/usr/bin/env bash
set -euo pipefail

MODEL="${1:-${PFIT_LLM_MODEL:-qwen2.5-coder:32b}}"
BASE_URL="${PFIT_LLM_BASE_URL:-http://localhost:11434}"
TIMEOUT_SECONDS="${PFIT_EVAL_TIMEOUT_SECONDS:-600}"
MAX_TOKENS="${PFIT_EVAL_MAX_TOKENS:-12000}"
TEMPERATURE="${PFIT_EVAL_TEMPERATURE:-0.1}"
MAX_REPAIR_ATTEMPTS="${PFIT_EVAL_MAX_REPAIR_ATTEMPTS:-5}"
RUN_MODE="${PFIT_EVAL_RUN_MODE:-cheap}"
DEBUG="${PFIT_EVAL_DEBUG:-0}"
EVAL_ROOT="${PFIT_EVAL_ROOT:-evaluation_runs/vm_eval_$(date +%Y%m%d_%H%M%S)}"

DEFAULT_SESSIONS=(
  theophylline
  lotka_volterra
  robertson_session
  vanderpol_session
  mapk_cascade
  oregonator
  piezo_bouc_wen
  boehm_stat5
  sliding_basepoint_headered
  ARC_fitting
  nfkb_signaling
)

CHEAP_RUN_SESSIONS=(
  theophylline
  lotka_volterra
  robertson_session
  vanderpol_session
  mapk_cascade
  piezo_bouc_wen
  sliding_basepoint_headered
  ARC_fitting
)

if [[ -n "${PFIT_EVAL_SESSIONS:-}" ]]; then
  read -r -a SESSIONS <<< "${PFIT_EVAL_SESSIONS}"
else
  SESSIONS=("${DEFAULT_SESSIONS[@]}")
fi

mkdir -p "$EVAL_ROOT"

PYTHON=".venv/bin/python"
if [[ ! -x "$PYTHON" ]]; then
  python3 -m venv .venv
  PYTHON=".venv/bin/python"
fi

"$PYTHON" -m pip install -r requirements.txt

if command -v ollama >/dev/null 2>&1; then
  ollama list | grep -F "$MODEL" >/dev/null 2>&1 || {
    echo "Model $MODEL not listed by Ollama. Pulling it now..."
    ollama pull "$MODEL"
  }
else
  echo "warning: ollama CLI not found. Continuing; the Ollama server must already be reachable at $BASE_URL."
fi

SUMMARY_TSV="$EVAL_ROOT/summary.tsv"
SUMMARY_MD="$EVAL_ROOT/summary.md"

printf "session\tnew\tcheck\tjax\trun\tdiagnose\trepair_attempts\tfinal_loss\trun_id\tseconds\n" > "$SUMMARY_TSV"

cat > "$SUMMARY_MD" <<EOF
# VM Model Evaluation

- model: \`$MODEL\`
- base_url: \`$BASE_URL\`
- run_mode: \`$RUN_MODE\`
- timeout_seconds: \`$TIMEOUT_SECONDS\`
- max_tokens: \`$MAX_TOKENS\`
- temperature: \`$TEMPERATURE\`
- max_repair_attempts: \`$MAX_REPAIR_ATTEMPTS\`
- eval_root: \`$EVAL_ROOT\`

| Session | new | check | jax | run | diagnose | repairs | final loss | run id | seconds |
| --- | --- | --- | --- | --- | --- | ---: | ---: | --- | ---: |
EOF

status_word() {
  local code="$1"
  if [[ "$code" == "0" ]]; then
    printf "pass"
  elif [[ "$code" == "skip" ]]; then
    printf "skip"
  else
    printf "fail"
  fi
}

should_run_fit() {
  local session="$1"
  case "$RUN_MODE" in
    none)
      return 1
      ;;
    all)
      return 0
      ;;
    cheap)
      for item in "${CHEAP_RUN_SESSIONS[@]}"; do
        [[ "$item" == "$session" ]] && return 0
      done
      return 1
      ;;
    *)
      echo "Unknown PFIT_EVAL_RUN_MODE=$RUN_MODE; expected none, cheap, or all." >&2
      exit 2
      ;;
  esac
}

run_step() {
  local log="$1"
  shift
  local started finished code
  started="$(date +%s)"
  set +e
  "$@" >"$log" 2>&1
  code="$?"
  set -e
  finished="$(date +%s)"
  echo "$code $((finished - started))"
}

for session in "${SESSIONS[@]}"; do
  echo "==> Evaluating $session"
  SESSION_SRC="sessions/$session"
  SESSION_DST="$EVAL_ROOT/sessions/$session"
  SESSION_LOG_DIR="$EVAL_ROOT/$session"
  mkdir -p "$SESSION_LOG_DIR" "$EVAL_ROOT/sessions"

  if [[ ! -d "$SESSION_SRC" ]]; then
    echo "missing session: $SESSION_SRC" | tee "$SESSION_LOG_DIR/missing.log"
    printf "%s\tfail\tskip\tskip\tskip\tskip\t0\t\t\t0\n" "$session" >> "$SUMMARY_TSV"
    printf "| \`%s\` | fail | skip | skip | skip | skip | 0 |  |  | 0 |\n" "$session" >> "$SUMMARY_MD"
    continue
  fi

  rm -rf "$SESSION_DST"
  cp -R "$SESSION_SRC" "$SESSION_DST"
  rm -rf "$SESSION_DST/outputs"
  mkdir -p "$SESSION_DST/outputs"

  DEBUG_FLAG=()
  if [[ "$DEBUG" == "1" ]]; then
    DEBUG_FLAG=(--debug)
  fi

  total_seconds=0
  read -r new_code new_seconds < <(run_step "$SESSION_LOG_DIR/new.log" \
    "$PYTHON" -m local_agent.cli.main new "$SESSION_DST" \
      --overwrite \
      --model "$MODEL" \
      --base-url "$BASE_URL" \
      --timeout-seconds "$TIMEOUT_SECONDS" \
      --max-tokens "$MAX_TOKENS" \
      --temperature "$TEMPERATURE" \
      --max-repair-attempts "$MAX_REPAIR_ATTEMPTS" \
      "${DEBUG_FLAG[@]}")
  total_seconds=$((total_seconds + new_seconds))

  check_code="skip"
  jax_code="skip"
  run_code="skip"
  diagnose_code="skip"
  run_id=""
  final_loss=""

  if [[ "$new_code" == "0" ]]; then
    read -r check_code check_seconds < <(run_step "$SESSION_LOG_DIR/check.log" \
      "$PYTHON" -m local_agent.cli.main check "$SESSION_DST" \
        --model "$MODEL" \
        --base-url "$BASE_URL" \
        --timeout-seconds "$TIMEOUT_SECONDS" \
        --max-tokens "$MAX_TOKENS" \
        --temperature "$TEMPERATURE" \
        "${DEBUG_FLAG[@]}")
    total_seconds=$((total_seconds + check_seconds))
  fi

  if [[ "$check_code" == "0" ]]; then
    read -r jax_code jax_seconds < <(run_step "$SESSION_LOG_DIR/jax.log" \
      "$PYTHON" -m local_agent.cli.main jax "$SESSION_DST" \
        --model "$MODEL" \
        --base-url "$BASE_URL" \
        --timeout-seconds "$TIMEOUT_SECONDS" \
        --max-tokens "$MAX_TOKENS" \
        --temperature "$TEMPERATURE" \
        --max-repair-attempts "$MAX_REPAIR_ATTEMPTS" \
        "${DEBUG_FLAG[@]}")
    total_seconds=$((total_seconds + jax_seconds))
  fi

  if [[ "$jax_code" == "0" ]] && should_run_fit "$session"; then
    read -r run_code run_seconds < <(run_step "$SESSION_LOG_DIR/run.log" \
      "$PYTHON" -m local_agent.cli.main run "$SESSION_DST")
    total_seconds=$((total_seconds + run_seconds))
    run_id="$(sed -n 's/^run directory: .*outputs\\///p' "$SESSION_LOG_DIR/run.log" | tail -1)"
    if [[ "$run_code" == "0" && -n "$run_id" ]]; then
      read -r diagnose_code diagnose_seconds < <(run_step "$SESSION_LOG_DIR/diagnose.log" \
        "$PYTHON" -m local_agent.cli.main diagnose "$SESSION_DST" "$run_id")
      total_seconds=$((total_seconds + diagnose_seconds))
    fi
  fi

  if [[ -n "$run_id" && -f "$SESSION_DST/outputs/$run_id/NODE_fitting.log" ]]; then
    final_loss="$("$PYTHON" - "$SESSION_DST/outputs/$run_id/NODE_fitting.log" <<'PY'
from pathlib import Path
import sys
values = []
for line in Path(sys.argv[1]).read_text().splitlines():
    parts = [part.strip() for part in line.split(",")]
    if len(parts) >= 2 and parts[0].isdigit():
        try:
            values.append(float(parts[1]))
        except ValueError:
            pass
print(values[-1] if values else "")
PY
)"
  fi

  repair_attempts=0
  if [[ -f "$SESSION_DST/generated/agent_logs/llm_calls.jsonl" ]]; then
    repair_attempts="$(grep -c '"step": "repair_' "$SESSION_DST/generated/agent_logs/llm_calls.jsonl" || true)"
  fi

  new_status="$(status_word "$new_code")"
  check_status="$(status_word "$check_code")"
  jax_status="$(status_word "$jax_code")"
  run_status="$(status_word "$run_code")"
  diagnose_status="$(status_word "$diagnose_code")"

  printf "%s\t%s\t%s\t%s\t%s\t%s\t%s\t%s\t%s\t%s\n" \
    "$session" "$new_status" "$check_status" "$jax_status" "$run_status" "$diagnose_status" \
    "$repair_attempts" "$final_loss" "$run_id" "$total_seconds" >> "$SUMMARY_TSV"

  printf "| \`%s\` | %s | %s | %s | %s | %s | %s | %s | \`%s\` | %s |\n" \
    "$session" "$new_status" "$check_status" "$jax_status" "$run_status" "$diagnose_status" \
    "$repair_attempts" "${final_loss:-}" "${run_id:-}" "$total_seconds" >> "$SUMMARY_MD"
done

cat >> "$SUMMARY_MD" <<EOF

## Notes

- Per-step logs live under \`$EVAL_ROOT/<session>/\`.
- Copied/evaluated sessions live under \`$EVAL_ROOT/sessions/<session>/\`.
- Agent prompt/response logs live under each copied session's
  \`generated/agent_logs/\` directory.
- A failed \`pfit new\` for \`nfkb_signaling\` is the baseline 14B failure to beat.
EOF

echo
echo "Evaluation complete."
echo "Summary: $SUMMARY_MD"
echo "TSV:     $SUMMARY_TSV"
