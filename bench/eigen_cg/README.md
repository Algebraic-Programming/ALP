# Eigen Conjugate Gradient Benchmark (Fixed Iterations)

This benchmark integrates Eigen (header-only) into the repository to measure average iteration time of a manual Conjugate Gradient (CG) loop over a set of sparse matrices in Matrix Market (`.mtx`) format.

## Goals
- Do **not** solve the linear system; instead, run a fixed number of iterations (`NITER=256` by default) to benchmark execution characteristics.
- Report per-run average iteration time and aggregate mean, standard deviation, and standard error across repetitions.
- Scale across powers-of-two thread counts using OpenMP.

## Components
- `scripts/clone_eigen.sh`: shallow clone of Eigen into `extern/eigen` (ignored by git).
- `bench/eigen_cg/mm_loader.[hpp|cpp]`: Minimal Matrix Market loader (coordinate real matrices).
- `bench/eigen_cg/cg_eigen.cpp`: Manual CG implementation with timing per iteration.
- `bench/eigen_cg/run_eigen_cg.sh`: Harness script varying thread counts, producing CSV results.
- `bench/eigen_cg/matrices.list`: Matrix names and placeholder URLs (replace with real sources).
- Output CSV: `results/eigen_cg.csv`.

## Building
Ensure you have cloned Eigen first.

```bash
./scripts/clone_eigen.sh
cmake -S . -B build -DENABLE_EIGEN_CG_BENCH=ON
cmake --build build -j
```

Binary location: `build/bin/cg_eigen`.

## Running the Benchmark
Download or replace matrix URLs in `bench/eigen_cg/matrices.list`.

```bash
bash bench/eigen_cg/run_eigen_cg.sh
```

Environment variables to override defaults:
- `NITER_FIXED` (default 256)
- `REPETITIONS` (default 32)
- `MAX_ITER` (default 2000; ignored in fixed mode except for internal data structures)
- `TOLERANCE` (default 1e-8; ignored in fixed mode)

The harness sets `OMP_NUM_THREADS` per run.

## Output CSV Format
`matrix,threads,repetitions,niter_fixed,mean_iter_ms,stddev_ms,stderr_ms`

## Notes
- Matrices must be square; CG expects (approximately) symmetric positive definite matrices. We skip strict SPD validation because the aim is timing.
- Eigen remains unmodified; new code added here inherits the repository's license.
- Replace placeholder URLs with actual Matrix Market or dataset sources. Provide paper URL to auto-augment the matrix list.

## Extending
- Add symmetry check or residual tracking if needed.
- Introduce other iteration counts or solver variants (e.g., preconditioned CG) for comparative benchmarking.

