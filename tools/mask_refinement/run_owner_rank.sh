#!/usr/bin/env bash
# Server-only A1b: tests -> smoke -> full source training -> source evaluation.
set -euo pipefail
cd -- "$(dirname -- "${BASH_SOURCE[0]}")/../.."
config=configs/experiments/candidate_mask_mlp/step1b_owner_rank.yaml
work=work_dirs/candidate_mask_mlp_a1b_owner_rank_s42
logs=logs/candidate_mask_mlp_a1b_owner_rank
mkdir -p -- "$logs" logs/candidate_mask_mlp/background

if [[ ! -x /usr/bin/time ]] || ! command -v python >/dev/null || ! command -v flock >/dev/null; then
    printf 'Activate TreeLearn on the Linux server; GNU /usr/bin/time and flock are required.\n' >&2
    exit 2
fi
exec 9>logs/candidate_mask_mlp/background/external.lock
if ! flock -n 9; then
    printf 'Another external run, diagnostic or owner-rank run holds the lock.\n' >&2
    exit 2
fi
if [[ -f "$work/last.pth" || -f "$work/smoke/last.pth" ]]; then
    printf 'A1b checkpoints already exist. Inspect them and resume the appropriate stage; see the experiment document.\n' >&2
    exit 2
fi
printf 'PYTHON=%s\nCONDA_ENV=%s\nSTART=%s\n' "$(command -v python)" "${CONDA_DEFAULT_ENV:-unset}" "$(date -Is)"
git rev-parse HEAD >"$logs/commit.txt"

run_stage() {
    local stage=$1
    shift
    printf 'RUNNING %s\n' "$stage" >"$logs/status.txt"
    printf 'RUN %s: %s/%s.log\n' "$stage" "$logs" "$stage"
    local code=0
    /usr/bin/time -v -o "$logs/$stage.time.txt" "$@" >"$logs/$stage.log" 2>&1 || code=$?
    printf '%s\n' "$code" >"$logs/$stage.exit_code.txt"
    if [[ "$code" != 0 ]]; then
        printf 'FAILED %s exit=%s\n' "$stage" "$code" >"$logs/status.txt"
        printf 'STOP %s exit=%s; inspect %s/%s.log\n' "$stage" "$code" "$logs" "$stage" >&2
    fi
    return "$code"
}

run_stage unit_tests python -m unittest discover -s tests -p test_candidate_mask_mlp.py -v
run_stage smoke python -u tools/mask_refinement/run.py train --config "$config" --pilot --smoke
run_stage train_full python -u tools/mask_refinement/run.py train --config "$config"
if run_stage evaluate python -u tools/mask_refinement/run.py evaluate --config "$config"; then
    printf 'DONE source_gate_passed\n' >"$logs/status.txt"
else
    code=$?
    # Gate failure is a completed scientific result, not necessarily a crash.
    if [[ "$code" == 2 && -f "$work/evaluation/summary.json" ]] && \
        python -c 'import json, sys; r=json.load(open(sys.argv[1])); sys.exit(0 if r["passed"] is False and not r["smoke"] else 1)' \
        "$work/evaluation/summary.json"; then
        printf 'DONE source_gate_failed\n' >"$logs/status.txt"
    fi
    exit "$code"
fi
printf 'SAVED %s/evaluation/summary.json; source evaluation only.\n' "$work"
