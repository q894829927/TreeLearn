#!/usr/bin/env bash
# Run on Linux with TreeLearn activated. The launcher detaches itself safely.
set -euo pipefail
cd -- "$(dirname -- "${BASH_SOURCE[0]}")/../.."
base="$PWD/logs/query_instance_decoder_v1"
config=configs/experiments/query_instance_decoder/v1.yaml
work=work_dirs/query_instance_decoder_v1_s42
mkdir -p -- "$base" logs/candidate_mask_mlp/background

if [[ ${1:-} != --worker ]]; then
    mode=${1:-all}
    case "$mode" in all|resume|evaluate|external) ;; *) printf 'Use all, resume, evaluate or external.\n' >&2; exit 2;; esac
    run_dir=$(mktemp -d "$base/run_$(date +%Y%m%dT%H%M%S)_XXXXXX")
    nohup bash "$0" --worker "$run_dir" "$mode" >"$run_dir/console.log" 2>&1 < /dev/null &
    printf '%s\n' "$!" >"$run_dir/pid.txt"
    printf 'PID=%s\nRUN_DIR=%s\n' "$!" "$run_dir"
    printf 'tail -n 20 %s/console.log\n' "$run_dir"
    exit 0
fi

run_dir=$2
mode=$3
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
    else
        printf 'FAILED stage=%s exit=%s\n' "$stage" "$code" >"$run_dir/status.txt"
    fi
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
printf 'START=%s\nPYTHON=%s\nCONDA_ENV=%s\n' "$(date -Is)" "$(command -v python)" "${CONDA_DEFAULT_ENV:-unset}"
git rev-parse HEAD >"$run_dir/commit.txt"

run_stage() {
    stage=$1
    shift
    printf 'RUNNING %s\n' "$stage" >"$run_dir/status.txt"
    printf 'RUN %s: %s/%s.log\n' "$stage" "$run_dir" "$stage"
    /usr/bin/time -v -o "$run_dir/$stage.time.txt" "$@" >"$run_dir/$stage.log" 2>&1
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
        run_stage train python -u tools/query_segmentation/run.py train --config "$config" --resume
        run_stage evaluate python -u tools/query_segmentation/run.py evaluate --config "$config"
        ;;
    evaluate)
        run_stage evaluate python -u tools/query_segmentation/run.py evaluate --config "$config"
        ;;
    external)
        run_stage external python -u tools/query_segmentation/run.py evaluate --config "$config" --external
        ;;
esac
