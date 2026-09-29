#!/usr/bin/env bash
set -euo pipefail

TOTAL_REPEATS="${TOTAL_REPEATS:-50}"
MAX_PARALLEL="${JOBS:-$(command -v nproc >/dev/null && nproc || sysctl -n hw.ncpu)}"

if (( MAX_PARALLEL > 10 )); then
    MAX_PARALLEL=$(( MAX_PARALLEL - 4 ))
fi

echo "[info] parallel workers = $MAX_PARALLEL"

export OPENBLAS_NUM_THREADS=1
export MKL_NUM_THREADS=1
export BLIS_NUM_THREADS=1
export GOTO_NUM_THREADS=1
export RENV_CONFIG_SANDBOX_ENABLED=FALSE

mkdir -p output/logs output/results

# Enter renv project root if needed
cd "${RENV_PROJECT_DIR:-.}"

# Generate 1..TOTAL_REPEATS, xargs -P runs exactly P tasks in parallel
seq 1 "$TOTAL_REPEATS" \
| xargs -I{} -P "$MAX_PARALLEL" bash -c '
  i="$1"
  log="output/logs/warpU_${i}.log"
  echo "[info] starting task $i -> $log"
  CURRENT_REPEAT="$i" \
  Rscript -e "renv::load(); current_repeat <- as.integer(Sys.getenv(\"CURRENT_REPEAT\")); source(\"code/main/figure6/warpU/Comparison_estimation_real_results_warpU_parallel.R\")" \
    >"$log" 2>&1
' _ {}

# Merge
Rscript -e "renv::load(); source('code/main/figure6/warpU/merge_results_warpU.R')" > output/logs/merge_warpU.log 2>&1
echo "[ok] all done"
