#!/usr/bin/env bash
# Foreground workflow; every stage is mirrored to the terminal and a log.
set -Eeuo pipefail
REPO_ROOT="$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")/../.." && pwd)"
cd "$REPO_ROOT"
MODE="${1:-all}"
CONFIG="${2:-configs/experiments/forest_abc_v2/full.yaml}"
PYTHON_BIN="${PYTHON_BIN:-python}"
case "$MODE" in
  all|resume|doctor|prepare|check|train|evaluate|external|tests|probe) ;;
  *) printf 'Usage: bash tools/forest_abc_v2/run.sh {all|resume|doctor|prepare|check|train|evaluate|external|tests|probe} [config]\n' >&2; exit 2 ;;
esac
mkdir -p logs/forest_abc_v1 logs/forest_abc_v2
# Shared GPU/cache lock with V1 and the stage diagnostic.
exec 9>logs/forest_abc_v1/workflow.lock
flock -n 9 || { printf 'Another ForestABC workflow is running.\n' >&2; exit 1; }
RUN_DIR="$(mktemp -d "$REPO_ROOT/logs/forest_abc_v2/run_$(date +%Y%m%dT%H%M%S)_XXXXXX")"
ln -sfn "$RUN_DIR" logs/forest_abc_v2/latest
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
  /usr/bin/time -v -o "$RUN_DIR/$STAGE.time.txt" "$@" 2>&1 | tee "$RUN_DIR/$STAGE.log" | tee -a "$RUN_DIR/console.log"
}
run() { stage "$1" "$PYTHON_BIN" -u tools/forest_abc_v2/run.py "$@" --config "$CONFIG"; }
tests() { stage tests "$PYTHON_BIN" -m unittest discover -s tests -p 'test_forest_abc_v2.py' -v; }
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
  probe) stage probe "$PYTHON_BIN" -u tools/forest_abc_v2/probe_scores.py ;;
  *) run "$MODE" ;;
esac
