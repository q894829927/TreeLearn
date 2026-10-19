#!/usr/bin/env bash
# Main workflow remains in the foreground. tee mirrors every stage to terminal.
set -Eeuo pipefail
REPO_ROOT="$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")/../.." && pwd)"
cd "$REPO_ROOT"
MODE="${1:-all}"
CONFIG="${2:-configs/experiments/forest_abc/full.yaml}"
PYTHON_BIN="${PYTHON_BIN:-python}"
case "$MODE" in
  all|resume|doctor|prepare|check|train|evaluate|external|tests) ;;
  *) printf 'Usage: bash tools/forest_abc/run.sh {all|resume|doctor|prepare|check|train|evaluate|external|tests} [config]\n' >&2; exit 2 ;;
esac
mkdir -p logs/forest_abc_v1
exec 9>logs/forest_abc_v1/workflow.lock
flock -n 9 || { printf 'Another ForestABC foreground workflow holds the lock.\n' >&2; exit 1; }
RUN_DIR="$(mktemp -d "$REPO_ROOT/logs/forest_abc_v1/run_$(date +%Y%m%dT%H%M%S)_XXXXXX")"
ln -sfn "$RUN_DIR" logs/forest_abc_v1/latest
STAGE=starting
finish() {
  local result=$?
  if (( result == 0 )); then
    printf 'DONE mode=%s (inspect metrics for effectiveness)\n' "$MODE" > "$RUN_DIR/status.txt"
  else
    printf 'FAILED stage=%s exit=%s\n' "$STAGE" "$result" > "$RUN_DIR/status.txt"
  fi
  printf 'END=%s EXIT=%s\n' "$(date --iso-8601=seconds)" "$result" | tee -a "$RUN_DIR/console.log"
}
trap finish EXIT
printf 'START=%s\nPYTHON=%s\nCONFIG=%s\nRUN_DIR=%s\n' \
  "$(date --iso-8601=seconds)" "$(command -v "$PYTHON_BIN")" "$CONFIG" "$RUN_DIR" | tee "$RUN_DIR/console.log"
stage() {
  STAGE="$1"
  shift
  printf 'RUNNING %s\n' "$STAGE" > "$RUN_DIR/status.txt"
  printf '\nRUN %s — live terminal output, log: %s/%s.log\n' "$STAGE" "$RUN_DIR" "$STAGE" | tee -a "$RUN_DIR/console.log"
  if command -v /usr/bin/time >/dev/null 2>&1; then
    /usr/bin/time -v -o "$RUN_DIR/$STAGE.time.txt" "$@" 2>&1 | tee "$RUN_DIR/$STAGE.log" | tee -a "$RUN_DIR/console.log"
  else
    "$@" 2>&1 | tee "$RUN_DIR/$STAGE.log" | tee -a "$RUN_DIR/console.log"
  fi
}
run() { stage "$1" "$PYTHON_BIN" -u tools/forest_abc/run.py "$@" --config "$CONFIG"; }
tests() { stage tests "$PYTHON_BIN" -m unittest discover -s tests -p 'test_forest_abc.py' -v; }
case "$MODE" in
  all)
    run doctor
    tests
    run prepare --require-external
    run check
    run train
    run evaluate
    run external
    ;;
  resume)
    run doctor
    tests
    run prepare --require-external
    run train --resume
    run evaluate
    run external
    ;;
  tests) tests ;;
  *) run "$MODE" ;;
esac
