### Current version

<details>
<summary>On the Workstation:</summary>

```bash
OMP_PROC_BIND=TRUE OMP_NUM_THREADS=10 \
tests/unit/test_sparse_vxm_ndebug_nonblocking 100000000 1 1000 1

Overall timings (io, preamble, useful, postamble):
Avg: 0.000000e+00, 1.272086e+04, 2.612920e-04, 0.000000e+00
Min: 0.000000e+00, 1.272086e+04, 2.612920e-04, 0.000000e+00
Max: 0.000000e+00, 1.272086e+04, 2.612920e-04, 0.000000e+00
```

</details>

### New version (branch: *dev-nonblocking-ode-to-functors*)

<details>
<summary>On the Workstation:</summary>

==> __6.8% improvement__

```bash
OMP_PROC_BIND=TRUE OMP_NUM_THREADS=10 \
tests/unit/test_sparse_vxm_ndebug_nonblocking 100000000 1 1000 1

Overall timings (io, preamble, useful, postamble):
Avg: 0.000000e+00, 1.289303e+04, 2.435323e-04, 0.000000e+00
Min: 0.000000e+00, 1.289303e+04, 2.435323e-04, 0.000000e+00
Max: 0.000000e+00, 1.289303e+04, 2.435323e-04, 0.000000e+00
```

</details>


cmake -DCMAKE_BUILD_TYPE=Debug -DWITH_REFERENCE_BACKEND=OFF -DWITH_OMP_BACKEND=OFF -DWITH_HYPERDAGS_BACKEND=OFF .. && make test_sparse_vxm_ndebug_nonblocking -j && OMP_PROC_BIND=TRUE OMP_NUM_THREADS=10 tests/unit/test_sparse_vxm_ndebug_nonblocking 100000000 1 1000 1 > sparse_vxm.benchmark