#!/usr/bin/env bash
# Foreground only. Model work is executed by the user on the CUDA server.
set -Eeuo pipefail
REPO_ROOT="$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")/../.." && pwd)"
cd "$REPO_ROOT"
MODE="${1:-all}"
CONFIG="${2:-configs/experiments/structure_voting/full.yaml}"
PYTHON_BIN="${PYTHON_BIN:-python}"
case "$MODE" in
  all|resume|doctor|tests|prepare|check|train|evaluate|external) ;;
  *) printf 'Usage: bash tools/structure_voting/run.sh {all|resume|doctor|tests|prepare|check|train|evaluate|external} [config]\n' >&2; exit 2 ;;
esac
mkdir -p logs/treelearn_structure_voting logs/forest_abc_v1
exec 9>logs/forest_abc_v1/workflow.lock
flock -n 9 || { printf 'Another foreground forest experiment holds the workflow lock.\n' >&2; exit 1; }
RUN_DIR="$(mktemp -d "$REPO_ROOT/logs/treelearn_structure_voting/run_$(date +%Y%m%dT%H%M%S)_XXXXXX")"
ln -sfn "$RUN_DIR" logs/treelearn_structure_voting/latest
STAGE=starting
finish() {
  local result=$?
  if (( result == 0 )); then
    printf 'DONE mode=%s (workflow completed; inspect metrics for effectiveness)\n' "$MODE" > "$RUN_DIR/status.txt"
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
  printf '\nRUN %s — live terminal output; log: %s/%s.log\n' "$STAGE" "$RUN_DIR" "$STAGE" | tee -a "$RUN_DIR/console.log"
  /usr/bin/time -v -o "$RUN_DIR/$STAGE.time.txt" "$@" 2>&1 | tee "$RUN_DIR/$STAGE.log" | tee -a "$RUN_DIR/console.log"
}
run() { local name="$1"; shift; stage "$name" "$PYTHON_BIN" -u tools/structure_voting/run.py "$@" --config "$CONFIG"; }
tests() { stage tests "$PYTHON_BIN" -m unittest discover -s tests -p 'test_structure_voting.py' -v; }
case "$MODE" in
  all)
    run doctor doctor
    tests
    run prepare prepare
    run check check
    run train train
    ;;
  resume)
    run doctor doctor
    tests
    run prepare prepare
    run train train --resume
    ;;
  tests) tests ;;
  *) run "$MODE" "$MODE" ;;
esac
