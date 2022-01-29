

   Copyright 2021 Huawei Technologies Co., Ltd.

 Licensed under the Apache License, Version 2.0 (the "License");
 you may not use this file except in compliance with the License.
 You may obtain a copy of the License at

     http://www.apache.org/licenses/LICENSE-2.0

 Unless required by applicable law or agreed to in writing, software
 distributed under the License is distributed on an "AS IS" BASIS,
 WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
 See the License for the specific language governing permissions and
 limitations under the License.


# Tests

Test folder contains various tests subdivided in three categories:
1. **unit tests**
2. **smoke tests**
3. **performance tests**

Their goals, in turn:

1. code coverage, test various common intended functionalities (the so-called
``happy paths' operate as expected, and test that border cases behave as defined
in the specification. Clarifying perhaps the principal goal of code coverage:
running all unit tests should ideally execute all lines of source code at least
once. In this code base, we strive to fully cover the public interface with unit
tests. Some tests, however, may also cover internal functions and data
structures.

2. smoke tests aim to test typical usage of the library, and focus on the
``happy path' only. In ALP/GraphBLAS, this includes aiming for more ``advanced'
usages such as integration with MPI, or manual launching where multiple OS
processes are combined ad hoc to co-execute an ALP program. These tests
typically execute faster than the full unit test suite, so they may be used to
quickly verify that any code changes are sane. On errors, however, smoke tests
are also significantly less informative than unit tests and cannot be expected
to catch all possible errors.

3. performance tests have as a primary aim to check that code changes do not
introduce performance bugs. Secondarily and ideally, performance tests also can
be used to generate results that go into publications, for external parties to
validate that published results are consistent, and to easily repeat past
experiments on new platforms.

The overall target `make tests` builds and runs the unit, smoke, and performance
tests in turn. To build and run only one category of tests, use one of

1. `make unittests`,
2. `make smoketests`,
3. `make perftests` (or the more verbose equivalent `make performancetests`).


# Extending the test suite

To add new tests, simply add their source code in the folder corresponding to
the category of the test. Then edit the corresponding `CMakeLists.txt` in that
folder in order to have it built-- simply see how some of the existing tests
are declared for the exact syntax. Then, add your test to one of the following
shell scripts that are responsible for executing the tests:

1. `tests/unit/unittests.sh` for unit tests,
2. `tests/smoke/smoketests.sh` for smoke tests, or
3. `tests/performance/performancetests.sh` for performance tests.

Like for the `CMakeLists.txt`, simply see how other tests are called in order
to add new ones. Do note that depending on the position in the script, tests
can be made to execute for all backends, only for specific backends, or skip
execution for specific backends. Ideally, however, all tests execute for all
backends, though sane exceptions do exist.


# Drivers

Some ALP/GraphBLAS algorithms can be executed through drivers that are built as
part of the smoke and/or performance tests. They can be built via the
`make build_tests_category_smoke build_tests_category_performance` targets. A
list of drivers of interest would be produced in the following locations,
**relative to the build directory**:

1. (`/path/to/the/build/directory/`)`tests/smoke/hpcg*`

2. `tests/smoke/conjugate_gradient*`

3. `tests/smoke/graphchallenge_nn_single_inference*`

4. `tests/performance/driver*`

The name of each such executable (there are several, as indicated by the above
wildcard) always ends with the backend by which it was compiled. By default,
these include `reference` and `reference_omp`. If ALP was configured with LPF,
these also include `bsp1d` and `hybrid`.

To run them, please use `grbrun` which can be found in `/install/path/bin/` for
the most robust results-- please ensure to use the `-b` flag to `grbrun` to
indicate a matching backend to the executable. For example, one could issue
from the build directory the following command:

`/install/path/bin/grbrun -b reference_omp tests/smoke/hpcg_reference_omp`

Most drives require mandatory command line arguments, and all drivers support
optional arguments. Most drivers, for example, require as a minimum argument
a path to an input dataset to operate on.

The executables that explicitly start with the `driver` prefix note as an
infix which algorithms they drive, such as the pagerank or the k-NN breadth-
first search. These algorithms are defined in `include/graphblas/algorithms`.
Likewise, executables with the prefix `conjugate_gradient` have their sources
listed in the likewise named file in `include/graphblas/algorithms`.

The document closes with short notes on specific driver executables below.

## HPCG

The HPCG executables correspond very closely to the reference HPCG benchmark
for the HPCG500 rank available at

https://github.com/hpcg-benchmark/hpcg

The main difference is the usage of the Red-Black Gauss-Seidel smoother in
place of the original Gauss-Seidel one, which is inherently sequential and not
naturally expressible in GraphBLAS.

The test is written inside `hpcg_test.cpp` and uses various internal
utilities to

- parse the command line arguments
- generate a 3D HPCG problem
- run the HPCG algorithm, benchmark the time and report the results

The results are currently printed on the terminal and no automatic validation
occurs.

The binaries take several optional arguments, which can be listed with the `-h`
option. No argument is needed, in which case the test will produce a small
system of sizes `16 x 16 x 16` and run the simulation on it. An example of run
with arguments is

```bash
tests/smoke/hpcg_reference_omp --test-rep 1 --init-iter 1 --nx 16 --ny 16
    --nz 16 --smoother-steps 1 --max_iter 56 --max_coarse-levels 1
```

The arguments defaults are currently set to the default ones of the reference
HPCG test.

### Extra compile options

Macros can be injected during compilation to inspect the application while
running. The following can be defined:

- `HPCG_PRINT_SYSTEM` to print the main system elements, like the system matrix,
the constant vector `b` and the initial solution and the various coarsening
matrices; this helps debugging system generation problems; note that the number
of printed elements is limited (typically 50 elements per dimensions -
rows/columns) because of the large size of matrices and vectors
- `HPCG_PRINT_STEPS` to print the squared norms of the main vectors (solution,
residual, direction vector, ...) during the simulation, in order to check their
evolution; this is particularly helpful in case of numerical problem, as it
allows tracing the issue and drilling down to the point where the error occurs

To define these symbols, you can, for example, compile the smoke test with

```bash
make EXTRA_CFLAGS="-DHPCG_PRINT_SYSTEM -DHPCG_PRINT_STEPS"
    build_tests_category_smoke
```

