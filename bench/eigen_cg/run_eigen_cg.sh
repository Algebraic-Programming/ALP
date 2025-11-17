#!/usr/bin/env bash
# Benchmark harness for Eigen CG (fixed iterations). Inherits repository license.
# Requires: cg_eigen built, matrices downloaded into data/matrices
set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
REPO_ROOT="${SCRIPT_DIR%/bench/eigen_cg}"
MATRIX_DIR="${REPO_ROOT}/data/matrices"
RESULTS_DIR="${REPO_ROOT}/results"
mkdir -p "${RESULTS_DIR}" "${MATRIX_DIR}"

MATRIX_LIST_FILE="${SCRIPT_DIR}/matrices.list"
OUTPUT_CSV="${RESULTS_DIR}/eigen_cg.csv"
: > "${OUTPUT_CSV}" # truncate

# Header
printf "matrix,threads,repetitions,niter_fixed,mean_iter_ms,stddev_ms,stderr_ms\n" >> "${OUTPUT_CSV}"

CG_BIN="${REPO_ROOT}/build/bin/cg_eigen"
# allow alternative build layout used by this repo (build/bench/eigen_cg/cg_eigen)
if [[ ! -x "${CG_BIN}" ]]; then
  if [[ -x "${REPO_ROOT}/build/bench/eigen_cg/cg_eigen" ]]; then
    CG_BIN="${REPO_ROOT}/build/bench/eigen_cg/cg_eigen"
  else
    echo "cg_eigen binary not found (looked in build/bin and build/bench/eigen_cg). Build the project first (ENABLE_EIGEN_CG_BENCH=ON)." >&2
    exit 1
  fi
fi

NITER_FIXED=${NITER_FIXED:-256}
REPETITIONS=${REPETITIONS:-32}
MAX_ITER=${MAX_ITER:-2000}
TOLERANCE=${TOLERANCE:-1e-8}

# Thread counts: powers of two up to nproc
MAX_THREADS=$(nproc)
THREADS=()
for t in 1 2 4 8 16 32 64; do
  if (( t <= MAX_THREADS )); then THREADS+=("$t"); fi
done

# Download helper
fetch_matrix() {
  local name="$1" url="$2"
  local target="${MATRIX_DIR}/${name}.mtx"
  if [[ -f "${target}" ]]; then return 0; fi
  echo "Fetching ${name} from ${url}" >&2
  tmp=$(mktemp -d)
  trap 'rm -rf "${tmp}"' RETURN

  # Download the URL to tmp
  dl="$tmp/download"
  if ! curl -L --fail -o "$dl" "$url"; then
    echo "Failed to download ${name} from ${url}" >&2
    return 1
  fi

  # If the URL looks like a tarball, try to extract .mtx files and pick the largest
  if file "$dl" | grep -qiE 'gzip compressed data|tar archive'; then
    mkdir -p "$tmp/extr"
    if ! tar -xzf "$dl" -C "$tmp/extr"; then
      echo "Failed to extract archive for ${name}" >&2
      return 1
    fi
    # find .mtx files inside extracted tree
    mtxfile=$(find "$tmp/extr" -type f -iname '*.mtx' -print0 | xargs -0 ls -1S 2>/dev/null | head -n1 || true)
    if [[ -n "$mtxfile" ]]; then
      cp "$mtxfile" "$target"
      echo "Extracted matrix to ${target}" >&2
      return 0
    else
      echo "No .mtx file found inside archive for ${name}" >&2
      return 1
    fi
  fi

  # Otherwise assume the download is already an .mtx file
  # Move it to target
  mv "$dl" "$target"
  chmod a+r "$target"
  return 0
}

# Expected format in matrices.list: name<tab>url
while IFS=$'\t' read -r name url; do
  [[ -z "$name" || "$name" =~ ^# ]] && continue
  fetch_matrix "$name" "$url" || continue
  matrix_path="${MATRIX_DIR}/${name}.mtx"
  for thr in "${THREADS[@]}"; do
    export OMP_NUM_THREADS="$thr"
    # Capture output line from binary
    line=$("${CG_BIN}" "${matrix_path}" --fixed --niter "${NITER_FIXED}" --repetitions "${REPETITIONS}" --max-iter "${MAX_ITER}" --tolerance "${TOLERANCE}" 2>&1 | tail -n1)
    # Parse key=val pairs
    # Expect: matrix=..., mode=..., niter_fixed=..., repetitions=..., mean_iter_ms=..., stddev=..., stderr=...
    mean=$(echo "$line" | sed -n 's/.*mean_iter_ms=\([^,]*\).*/\1/p')
    stddev=$(echo "$line" | sed -n 's/.*stddev=\([^,]*\).*/\1/p')
    stderr=$(echo "$line" | sed -n 's/.*stderr=\([^,]*\).*/\1/p')
    printf "%s,%d,%d,%d,%s,%s,%s\n" "$name" "$thr" "$REPETITIONS" "$NITER_FIXED" "$mean" "$stddev" "$stderr" >> "${OUTPUT_CSV}"
  done
done < "${MATRIX_LIST_FILE}"

echo "Results written to ${OUTPUT_CSV}" >&2
