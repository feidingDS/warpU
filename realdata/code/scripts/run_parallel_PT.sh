#!/usr/bin/env bash
set -euo pipefail

# -------- Configuration --------
TOTAL_REPEATS="${TOTAL_REPEATS:-50}"           # Override with env var: TOTAL_REPEATS=80 ./run_parallel_PT.sh
RENV_PROJECT_DIR="${RENV_PROJECT_DIR:-.}"      # renv project root (defaults to current directory)
RUN_SCRIPT="code/main/figure6/PT/Comparison_estimation_real_results_PT_parallel.R"
MERGE_SCRIPT="code/main/figure6/PT/merge_results_PT.R"
LOG_DIR="output/logs"
OUT_DIR="output/results"
# -------------------------------

export OPENBLAS_NUM_THREADS=1
export MKL_NUM_THREADS=1
export BLIS_NUM_THREADS=1
export GOTO_NUM_THREADS=1



# Parallel job count (override with JOBS)
if [[ "${JOBS:-}" != "" ]]; then
  MAX_PARALLEL="$JOBS"
else
  case "$(uname -s)" in
    Darwin) MAX_PARALLEL="$(sysctl -n hw.ncpu)" ;;
    Linux)  MAX_PARALLEL="$(nproc)" ;;
    *)      MAX_PARALLEL=4 ;;
  esac
fi
echo "[info] parallel workers: $MAX_PARALLEL"


if (( MAX_PARALLEL > 10 )); then
    MAX_PARALLEL=$(( MAX_PARALLEL - 4 ))
fi

# Disable renv sandbox if needed
export RENV_CONFIG_SANDBOX_ENABLED=FALSE

# Directories and dependencies
mkdir -p "$OUT_DIR" "$LOG_DIR"
command -v Rscript >/dev/null 2>&1 || { echo "[error] Rscript not found"; exit 1; }

# Enter renv project root (renv::load depends on the working directory)
cd "$RENV_PROJECT_DIR"

# -------- Parallel execution: xargs -P controls concurrency --------
# - seq 1..N generates repeat indices
# - xargs -P $MAX_PARALLEL runs at most $MAX_PARALLEL subprocesses concurrently
# - each task's log is written to output/logs/PT_{i}.log
seq 1 "$TOTAL_REPEATS" \
| xargs -I{} -P "$MAX_PARALLEL" bash -c '
  i="$1"
  log="'"$LOG_DIR"'/PT_${i}.log"
  echo "[info] starting task $i -> $log"
  CURRENT_REPEAT="$i" \
  Rscript -e "renv::load(); current_repeat <- as.integer(Sys.getenv(\"CURRENT_REPEAT\")); source(\"'"$RUN_SCRIPT"'\")" \
    >"$log" 2>&1
' _ {}

# -------- Merge results --------
echo "[info] merging results..."
Rscript -e "renv::load(); source('$MERGE_SCRIPT')" > "$LOG_DIR/merge_PT.log" 2>&1

echo "[ok] All processes completed"
