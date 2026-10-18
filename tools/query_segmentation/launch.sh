#!/usr/bin/env bash
# Run in the foreground on Linux; mirror live output to terminal and logs.
set -euo pipefail
cd -- "$(dirname -- "${BASH_SOURCE[0]}")/../.."
base="$PWD/logs/query_instance_decoder_v1"
config=configs/experiments/query_instance_decoder/v1.yaml
work=work_dirs/query_instance_decoder_v1_s42
mkdir -p -- "$base" logs/candidate_mask_mlp/background

if [[ ${1:-} != --worker ]]; then
    mode=${1:-all}
    case "$mode" in all|resume|evaluate|external|diagnose|filter|evaluate-filtered|external-filtered) ;; *) printf 'Use all, resume, evaluate, external, diagnose, filter, evaluate-filtered or external-filtered.\n' >&2; exit 2;; esac
    run_dir=$(mktemp -d "$base/run_$(date +%Y%m%dT%H%M%S)_XXXXXX")
    printf 'RUN_DIR=%s\nFOREGROUND: live progress is shown here and saved to logs.\n' "$run_dir"
    bash "$0" --worker "$run_dir" "$mode" 2>&1 | tee -i "$run_dir/console.log"
    exit 0
fi

run_dir=$2
mode=$3
printf '%s\n' "$$" >"$run_dir/pid.txt"
exec 9>logs/candidate_mask_mlp/background/external.lock
if ! flock -n 9; then
    printf 'LOCKED: another training/evaluation/diagnostic run holds the lock.\n' | tee "$run_dir/status.txt"
    exit 2
fi
ln -sfn -- "$run_dir" "$base/latest"
stage=start
monitor_pid=
finish() {
    code=$?
    if [[ -n "$monitor_pid" ]]; then kill "$monitor_pid" 2>/dev/null || true; fi
    if [[ "$code" == 0 ]]; then
        printf 'DONE mode=%s (workflow completed; inspect metrics for effectiveness)\n' "$mode" >"$run_dir/status.txt"
    elif [[ "$code" == 130 || "$code" == 143 ]]; then
        printf 'INTERRUPTED stage=%s exit=%s\n' "$stage" "$code" >"$run_dir/status.txt"
    else
        printf 'FAILED stage=%s exit=%s\n' "$stage" "$code" >"$run_dir/status.txt"
    fi
    cat "$run_dir/status.txt"
    printf 'END=%s EXIT=%s\n' "$(date -Is)" "$code"
}
trap finish EXIT
worker=$$
(
    while kill -0 "$worker" 2>/dev/null; do
        date -Is
        awk '/MemTotal:|MemAvailable:|SwapFree:/{print}' /proc/meminfo
        ps -eo pid,ppid,rss,pcpu,stat,comm --sort=-rss | sed -n '1,12p'
        if command -v nvidia-smi >/dev/null; then
            nvidia-smi --query-gpu=memory.used,utilization.gpu --format=csv,noheader || true
        fi
        sleep 20
    done
) >"$run_dir/resources.log" 2>&1 &
monitor_pid=$!
printf 'START=%s\nPID=%s\nPYTHON=%s\nCONDA_ENV=%s\n' "$(date -Is)" "$$" "$(command -v python)" "${CONDA_DEFAULT_ENV:-unset}"
git rev-parse HEAD >"$run_dir/commit.txt"

run_stage() {
    stage=$1
    shift
    printf 'RUNNING %s\n' "$stage" >"$run_dir/status.txt"
    printf 'RUN %s: %s/%s.log\n' "$stage" "$run_dir" "$stage"
    if /usr/bin/time -v -o "$run_dir/$stage.time.txt" "$@" 2>&1 | tee -i "$run_dir/$stage.log"; then
        printf 'FINISHED %s\n' "$stage"
    else
        stage_codes=("${PIPESTATUS[@]}")
        # A failed Python command must not be hidden by a successful tee.
        if [[ ${stage_codes[0]} != 0 ]]; then return "${stage_codes[0]}"; fi
        return "${stage_codes[1]}"
    fi
}

case "$mode" in
    all)
        if [[ -f "$work/last.pth" ]]; then printf 'Existing training checkpoint; launch with resume.\n' >&2; exit 2; fi
        run_stage tests python -m unittest discover -s tests -p test_query_segmentation.py -v
        run_stage prepare python -u tools/query_segmentation/run.py prepare --config "$config"
        if [[ ! -f "$work/smoke/best.pth" ]]; then
            run_stage smoke python -u tools/query_segmentation/run.py train --config "$config" --smoke
        fi
        run_stage smoke_evaluate python -u tools/query_segmentation/run.py evaluate --config "$config" --smoke
        run_stage train python -u tools/query_segmentation/run.py train --config "$config"
        run_stage evaluate python -u tools/query_segmentation/run.py evaluate --config "$config"
        ;;
    resume)
        run_stage tests python -m unittest discover -s tests -p test_query_segmentation.py -v
        run_stage train python -u tools/query_segmentation/run.py train --config "$config" --resume
        run_stage evaluate python -u tools/query_segmentation/run.py evaluate --config "$config"
        ;;
    evaluate)
        run_stage evaluate python -u tools/query_segmentation/run.py evaluate --config "$config"
        ;;
    external)
        run_stage external python -u tools/query_segmentation/run.py evaluate --config "$config" --external
        ;;
    diagnose)
        run_stage tests python -m unittest discover -s tests -p test_query_segmentation_diagnosis.py -v
        run_stage diagnose python -u tools/query_segmentation/diagnose.py --config "$config"
        ;;
    filter)
        run_stage tests python -m unittest discover -s tests -p test_query_segmentation_allocation.py -v
        run_stage filter python -u tools/query_segmentation/apply_allocation_filter.py --config "$config"
        ;;
    evaluate-filtered)
        run_stage tests python -m unittest discover -s tests -p test_query_segmentation_allocation.py -v
        run_stage evaluate python -u tools/query_segmentation/run.py evaluate --config "$config" --allocation-filter
        ;;
    external-filtered)
        run_stage tests python -m unittest discover -s tests -p test_query_segmentation_allocation.py -v
        run_stage external python -u tools/query_segmentation/run.py evaluate --config "$config" --external --allocation-filter
        ;;
esac
