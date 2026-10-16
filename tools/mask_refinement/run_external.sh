#!/usr/bin/env bash
# Run on the Linux server from an activated TreeLearn environment.
# Launch with nohup; see docs/experiments/candidate_mask_mlp_wytham_zh.md.
set -euo pipefail

stage=${1:-}
if [[ "$stage" != prepare && "$stage" != evaluate ]]; then
    printf 'Usage: bash tools/mask_refinement/run_external.sh {prepare|evaluate} [external.py options]\n' >&2
    exit 2
fi
shift
cd -- "$(dirname -- "${BASH_SOURCE[0]}")/../.."

for dependency in python flock pgrep ps awk; do
    if ! command -v "$dependency" >/dev/null 2>&1; then
        printf 'Missing command: %s. Activate the TreeLearn environment on the Linux server.\n' "$dependency" >&2
        exit 2
    fi
done
if [[ ! -x /usr/bin/time || ! -r /proc/meminfo ]]; then
    printf 'This runner requires Linux /proc and GNU /usr/bin/time.\n' >&2
    exit 2
fi
python_bin=$(command -v python)
user_id=$(id -u)
log_root=logs/candidate_mask_mlp/background
mkdir -p -- "$log_root"

# Keep the lock in descendants too: a surviving pipeline must block a duplicate.
exec 9>"$log_root/external.lock"
if ! flock -n 9; then
    printf 'An external background run is already active. Inspect %s/latest_*/console.log.\n' "$log_root" >&2
    exit 2
fi
if pgrep -u "$user_id" -f '[t]ools/(mask_refinement/(external|pipeline_streaming)|pipeline/pipeline)\.py'; then
    printf 'An existing external/pipeline process was found above. Inspect it before starting another.\n' >&2
    exit 2
fi

run_dir=$(mktemp -d "$log_root/${stage}_$(date +%Y%m%dT%H%M%S)_XXXXXX")
ln -sfn -- "$(basename -- "$run_dir")" "$log_root/latest_$stage"
printf 'RUN_DIR=%s\n' "$run_dir"
printf '%s\n' "$$" >"$run_dir/runner.pid"
exec >>"$run_dir/console.log" 2>&1
printf 'START=%s\nPYTHON=%s\nCONDA_ENV=%s\n' "$(date -Is)" "$python_bin" "${CONDA_DEFAULT_ENV:-unset}"
printf 'COMMAND='
printf '%q ' "$python_bin" -u tools/mask_refinement/external.py "$stage" "$@"
printf '\n'

monitor_pid=
cleanup() {
    local result=$?
    trap - EXIT
    if [[ -n "$monitor_pid" ]]; then
        kill "$monitor_pid" 2>/dev/null || true
        wait "$monitor_pid" 2>/dev/null || true
    fi
    printf '%s\n' "$result" >"$run_dir/exit_code.txt"
    printf 'END=%s EXIT_CODE=%s\n' "$(date -Is)" "$result"
    exit "$result"
}
trap cleanup EXIT

/usr/bin/time -v -o "$run_dir/time.txt" \
    "$python_bin" -u tools/mask_refinement/external.py "$stage" "$@" &
task_pid=$!
printf '%s\n' "$task_pid" >"$run_dir/task.pid"

# Sample host RAM/swap and this user's largest processes without recording args.
# These samples may survive even if SIGKILL prevents a final exit record.
task_cgroup=$(awk -F: '$1 == "0" {print $3; exit}' /proc/self/cgroup)
sample_resources() {
    date -Is
    awk '/^(MemTotal|MemAvailable|SwapTotal|SwapFree):/ {print}' /proc/meminfo
    if [[ -n "$task_cgroup" ]]; then
        for name in memory.current memory.max memory.events; do
            if [[ -r "/sys/fs/cgroup$task_cgroup/$name" ]]; then
                printf 'CGROUP %s: ' "$name"
                cat -- "/sys/fs/cgroup$task_cgroup/$name"
            fi
        done
    fi
    ps -u "$user_id" -o pid,ppid,rss,stat,comm --sort=-rss | sed -n '1,16p'
}
monitor_resources() {
    while kill -0 "$task_pid" 2>/dev/null; do
        sample_resources
        sleep 10
    done
}
monitor_resources >>"$run_dir/resources.log" 2>&1 &
monitor_pid=$!

if wait "$task_pid"; then
    result=0
else
    result=$?
fi
sample_resources >>"$run_dir/resources.log" 2>&1 || true
exit "$result"
