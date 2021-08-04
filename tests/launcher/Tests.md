

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

This folder contains various tests made out of basic numerical algorithms, which leverage GraphBLAS to run.
Here follows a brief description of each one.

## HPCG
It corresponds very closely to the reference HPCG benchmark for the HPCG500 rank available at

https://github.com/hpcg-benchmark/hpcg

The main difference is the usage of the Red-Black Gauss-Seidel smoother in place of the original Gauss-Seidel one, which is inherently sequential and not naturally expressible in GraphBLAS.
The test is written inside `hpcg_test.cpp` and uses various internal utilities to

- parse the command line arguments
- generate a 3D HPCG problem
- run the HPCG algorithm, benchmark the time and report the results

The results are currently printed on the terminal and no automatic validation occurs.

To compile the test from the `code` directory, run

```bash
make bin/tests/hpcg_reference_omp
```

The resulting binary can take several optional arguments, which can be listed with the `-h` option. No argument is needed, in which case the test will produce a small system of sizes `16 x 16 x 16` and run the simulation on it. An example of run with arguments is

```bash
bin/tests/hpcg_reference_omp --test-rep 1 --init-iter 1 --nx 16 --ny 16 --nz 16
	--smoother-steps 1 --max_iter 56 --max_coarse-levels 1
```

The arguments defaults are currently set to the default ones of the reference HPCG test.

### Extra compile options
They can be injected during compilation to inspect the application while running. The follwing symbols can be defined:

- `HPCG_PRINT_SYSTEM` to print the main system elements, like the system matrix, the constant vector `b` and the initial solution and the various coarsening matrices; this helps debugging system generation problems; note that the number of printed elements is limited (typically 50 elements per dimensions - rows/columns) because of the large size of matrices and vectors
- `HPCG_PRINT_STEPS` to print the squared norms of the main vectors (solution, residual, direction vector, ...) during the simulation, in order to check their evolution; this is particularly helpful in case of numerical problem, as it allows tracing the issue and drilling down to the point where the error occurs

To define these symbols, you can compile the test with

```bash
make EXTRA_CFLAGS="-DHPCG_PRINT_SYSTEM -DHPCG_PRINT_STEPS"
	bin/tests/hpcg_reference_omp
```
