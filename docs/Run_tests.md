<pre>
  Copyright 2023 Huawei Technologies Co., Ltd.

Licensed under the Apache License, Version 2.0 (the "License");
you may not use this file except in compliance with the License.
You may obtain a copy of the License at

    http://www.apache.org/licenses/LICENSE-2.0

Unless required by applicable law or agreed to in writing, software
distributed under the License is distributed on an "AS IS" BASIS,
WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
See the License for the specific language governing permissions and
limitations under the License.
</pre>

The current testing infrastructure is based on
[CTest](https://cmake.org/cmake/help/latest/manual/ctest.1.html), a
program companion to CMake that is able to launch tests, filter them and
report the results in great detail. CTest is usually installed together
with CMake and can be invoked in a similar way from the command line.
Therefore, it integrates smoothly with a CMake configuration/compilation
infrastructure, and allows defining test cases in the same CMake files
where CMake targets are defined. Indeed, CTest is *not* a test suite, but
rather a test *launcher*: tests programs are coded independently and
compiled as usual as executables and then described to CTest as
command-line programs, hence with command-line arguments if needed.
CTest simply runs this program as-is and reports the execution result.

Due to its various features, ALP/GraphBLAS developed several
facilities to generate tests, which are explained in the following.

# How to use CTest
The following examples show the most common options for ctest.

From the build directory, one can simply type

```bash
ctest
```

tu run **all** available tests (it may take some time).
To filter certain tests, one needs the `-R` (regex) option, which runs any
test matching the given regex, e.g.:

```bash
ctest -R mxv
```

runs all tests whose name contains `mxv`, while

```bash
ctest -R "mxv.*processes:1"
```

runs all tests whose name contains `mxv` *and* after `processes:1`.
CMake/CTest supports a
[regex syntax](https://cmake.org/cmake/help/latest/command/string.html#regex-specification)
very close to the UNIX Simple Regular Expressions.
To avoid running tests and only list them, e.g. to check a passed regex,
one can pass the `-N` option, e.g.:

```bash
ctest -R "mxv.*processes:1" -N
```

Tests also have *labels* in order to be grouped into categories;
for example, tests in ALP/GraphBLAS have labels corresponding to
the backend they run and the category they belong to.
One can list all the labels via

```bash
ctest --print-labels
```

and filter via the `-L` option: for example,

```bash
ctest -L reference
```

runs all tests whose label contains "reference" (here, tests for both
`reference` and `reference_omp` backend).
As usual, one can only list matched tests via the `-N` option.
Union and intersection of conditions on test names can be achieved via the
regular expression; for example:

```bash
ctest  -R "buildMatrix|mxv" -N
```

lists all tests that contain either `buildMatrix` *or* `mxv`, while

```bash
ctest  -R "buildMatrix.*processes:1" -N
```

lists all tests with `buildMatrix` *and* `processes:1` in the name, in this
specific order.
For labels, union can be achieved in the same way, while, starting from CMake
3.21, intersection can be achieved via repeated usage of the `-L` option:

```bash
ctest  -L 'mode:unit' -L 'backend:reference$' -N
```

lists all tests for the unit category *and* for the reference backend
(reference_omp is excluded -- notice the POSIX string terminator `$` at the
end).
The [official documentation](https://cmake.org/cmake/help/latest/manual/ctest.1.html#label-matching)
contains more information about this topic.

Instead, if `-R` and `-L` are used simultaneously, the *intersection* is achieved; for example:

```bash
ctest -L backend:reference_omp -R buildMatrix -N
```

lists all tests for the reference_omp backend whose name contains `buildMatrix`.
The `-U` flag instead achieves the *union* of results for this specific options
combination.

A few noticeable options also control the output:

* `-Q` suppresses any output
* `-O <file>` redirects the output to a file
* `-V` and `-VV` enable more (much more) output from tests run
* `--output-junit <file>` (from version 3.21) produces an XML output in JUnit
  format, which can thus be interpreted by many common tools and platforms
  (e.g., GitHub, GitLab, ...)

The complete synopsis is available at the
[official website](https://cmake.org/cmake/help/latest/manual/ctest.1.html#id16).

# Add ALP/GraphBLAS Tests
Dedicated facilities are present to add ALP/GraphBLAS tests easily, following
the same philosophy of the facilities to
[add new test executables](Build_and_test_infra.md#adding-a-new-test-executable).

Since a test requires an executable to run, its directive must be added after
the `add_grb_executables()` directive to create the test executable; it can be
added in the same CMake file, and it is an encouraged practice to do so that
one can immediately see how tests are built and run, and changes to one
directive can immediately be applied to the other.

Using an example from
[the CMake file for unit tests](../tests/unit/CMakeLists.txt), the test
executable for various backends is created via


```cmake
add_grb_executables( clearMatrix clearMatrix.cpp
	BACKENDS reference reference_omp bsp1d hybrid hyperdags nonblocking
)
```

where the first argument `clearMatrix` is the base name, from which the CMake
targets for each backend and mode are created and so are the corresponding
executable names (see
[Naming conventions for targets](Build_and_test_infra.md#naming-conventions-for-targets)).

Starting from this base name, one can define tests for these executables as
follows:

```cmake
add_grb_tests( clearMatrix clearMatrix
	BACKENDS reference reference_omp bsp1d hybrid hyperdags nonblocking
	ARGUMENTS 10000000
	Test_OK_SUCCESS
)
```

where:

1. `add_grb_tests` is the command to define one or multiple CTest's
2. the first `clearMatrix` argument is the base name of the test, which is used
   to generate one or more tests according to the
   [Test generation and naming conventions](#test-generation-and-naming-conventions)
3. the second `clearMatrix` argument is the base name of the CMake target, i.e.
   the first argument of the previous `add_grb_executables()` command generating
   the compilation target(s); note that the first and second argument of
   `add_grb_tests` are not mandated to match, and one may define multiple tests
   (hence with different test base names - first argument) for the same CMake
   target `clearMatrix` (second argument)
4. `BACKENDS reference ...` is the list of backends to generate the tests for;
   much like for `add_grb_executables()`, this generates one test per backend
   with appropriate naming and labels
5. `ARGUMENTS 10000000` is the list of arguments to pass to the executable for
   testing; it can be omitted, in which case the executable is called with no
   argument
6. `Test_OK_SUCCESS` instructs CTest to check the execution output for the
   string `Test OK`, a common convention in ALP/GraphBLAS

Further non-mandatory options, not exemplified above, are:

* `PROCESSES p1 [p2 ...]`: lists the number of processes to run the test with
  (if allowed by the backend)
* `THREADS t1 [t2 ...]`: lists the number of threads to run the test with (if
  allowed by the backend)
* `OUTPUT_VALIDATE <Bash command>` : is a Bash command to validate the output,
  where a `0` return code means successful validation (failure otherwise); since
  the output (stdout/stderr) of a test is stored in a file, this command can
  access it via the `@@TEST_OUTPUT_FILE@@` placeholder, which is automatically
  replaced with the absolute path of the file storing stdout/stderr


# Test generation and naming conventions
As for `add_grb_executables()`, also CTest's are generated according to certain
naming conventions.
The structure is

```cmake
<test base name>-<mode (if any)>-<backend>-processes:<num processes>-threads:<num threads>
```

which makes test runtime information explicit.
Note that each backend has a predefined list of `<num processes>` and
`<num threads>` to generate tests for, which the user can override via the
above-mentioned `PROCESSES` and `THREADS` options.
If any of them is not specified, the backend-dependent options are used and test
for all possible configurations are generated, i.e., for the cartesian product
of `PROCESSES` and `THREADS` (wherever this information comes from).
These rules apply to all backends, hence the command

```cmake
add_grb_tests( clearMatrix clearMatrix
	BACKENDS reference reference_omp bsp1d hybrid hyperdags nonblocking
	ARGUMENTS 10000000
	Test_OK_SUCCESS
	PROCESSES 1 2
	THREADS 3 4
)
```

generates 24 tests, corresponding to the cartesian products of `BACKENDS` x
`PROCESSES` x `THREADS`.
This means that, for example, the four different tests with the "reference"
(i.e., sequential) backend run exactly the same way, as this backend ignores any
number of `PROCESSES` or `THREADS`.
Hence, care should be used when overriding default values for `PROCESSES` or
`THREADS`, and one should make sure that all listed backends allow specifying
those resources.
The above example could therefore be modified as:

```cmake
add_grb_tests( clearMatrix clearMatrix
	BACKENDS reference hyperdags
	ARGUMENTS 10000000
	Test_OK_SUCCESS
)

add_grb_tests( clearMatrix clearMatrix
	BACKENDS reference_omp nonblocking
	ARGUMENTS 10000000
	Test_OK_SUCCESS
	THREADS 3 4
)

add_grb_tests( clearMatrix clearMatrix
	BACKENDS bsp1d
	ARGUMENTS 10000000
	Test_OK_SUCCESS
	PROCESSES 1 2
)

add_grb_tests( clearMatrix clearMatrix
	BACKENDS hybrid
	ARGUMENTS 10000000
	Test_OK_SUCCESS
	PROCESSES 1 2
	THREADS 3 4
)
```


# Internals
As from the
[specification](https://cmake.org/cmake/help/latest/command/add_test.html#command:add_test),
CTest essentially runs a command-line program and checks certain configurable
conditions after the command returns (return code, output, ...); as such, it has
no notion of "backend" or "launcher".
ALP/GraphBLAS then internally generates a command that creates the proper
environment to run the command by using a Python3-based test launcher, which
needs options to run the executable, all coming directly from the CMake/CTest
infrastructure:

* backend name; this can be any of ALP/GraphBLAS backends, or "none" for
  standard executables
* number of processes (single value)
* number of threads (single value)
* absolute path of file to redirect stdout and stderr to
* whether to look for a success string
* path of test executable and (optionally) its arguments

This launcher is generated during CMake configuration into
`<build directory>/tests/grb_test_runner.py` and can also be used manually;
for more details about its options, it can be invoked as

```bash
python3 <build directory>/tests/grb_test_runner.py --help
```

Since this launcher acts as "intermediate" between CTest and the executable
being run, its messages are also visible in the CTest log, while the executable
output is forwarded to a file and thus not visible in the CTest log.
However, in case of test failure, the launcher also prints this output on
stdout, making it directly available on the CTest log.
In any case, the launcher prints the file path it is redirecting to.


# Testing environment

For reproducibility of tests, the section
[Reproducible Builds section](Build_and_test_infra.md#reproducible-builds),
describes how dedicated Docker images can be built to have a reproducible
environment for building and testing.
These images store all tools, dependencies and input datasets to build and run
all backends and tests; you may refer to the
[section](Build_and_test_infra.md#reproducible-builds) for more details.
