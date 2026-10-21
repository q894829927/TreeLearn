#!/usr/bin/env bash
# Foreground only: every stage and pipeline subprocess streams to this terminal.
set -Eeuo pipefail
REPO_ROOT="$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")/../.." && pwd)"
cd "$REPO_ROOT"
MODE="${1:-all}"
CONFIG="${2:-configs/experiments/treelearn_residual_instance/full.yaml}"
PYTHON_BIN="${PYTHON_BIN:-python}"
case "$MODE" in
  all|resume|doctor|prepare|check|tests|train-control|train-combined|compare|external) ;;
  *) printf 'Usage: bash tools/treelearn_residual_instance/run.sh {all|resume|doctor|prepare|check|tests|train-control|train-combined|compare|external} [config]\n' >&2; exit 2 ;;
esac
mkdir -p logs/treelearn_residual_instance logs/forest_abc_v1
# Share the GPU workflow lock with the existing ForestABC launchers.
exec 9>logs/forest_abc_v1/workflow.lock
flock -n 9 || { printf 'Another foreground forest experiment holds the GPU workflow lock.\n' >&2; exit 1; }
RUN_DIR="$(mktemp -d "$REPO_ROOT/logs/treelearn_residual_instance/run_$(date +%Y%m%dT%H%M%S)_XXXXXX")"
ln -sfn "$RUN_DIR" logs/treelearn_residual_instance/latest
STAGE=starting
finish() {
  local result=$?
  if (( result == 0 )); then
    printf 'DONE mode=%s (workflow completed; inspect comparison.json for effectiveness)\n' "$MODE" > "$RUN_DIR/status.txt"
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
run() {
  local name="$1"
  shift
  stage "$name" "$PYTHON_BIN" -u tools/treelearn_residual_instance/run.py "$@" --config "$CONFIG"
}
tests() { stage tests "$PYTHON_BIN" -m unittest discover -s tests -p 'test_residual_instance.py' -v; }
case "$MODE" in
  all|resume)
    run doctor doctor
    tests
    run prepare prepare
    if [[ "$MODE" == all ]]; then run check check; fi
    extra=()
    if [[ "$MODE" == resume ]]; then extra=(--resume); fi
    run train_control train --arm control "${extra[@]}"
    run train_combined train --arm combined "${extra[@]}"
    run compare compare
    ;;
  tests) tests ;;
  train-control) run train_control train --arm control ;;
  train-combined) run train_combined train --arm combined ;;
  external)
    run external_control external --arm control
    run external_combined external --arm combined
    run compare_external compare-external
    ;;
  *) run "$MODE" "$MODE" ;;
esac
