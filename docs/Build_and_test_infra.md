<pre>
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
</pre>

## Table of Contents

- [Introduction to ALP/GraphBLAS Building and Testing Infrastructure:](#introduction-to-alpgraphblas-building-and-testing-infrastructure)
- [The Building Infrastructure](#the-building-infrastructure)
  - [Generation via the `bootstrap.sh` script](#generation-via-the-bootstrapsh-script)
    - [Direct Generation via `cmake`](#direct-generation-via-cmake)
    - [CMake Build Options, Types and Flags](#cmake-build-options-types-and-flags)
  - [Naming conventions for targets](#naming-conventions-for-targets)
  - [Adding a new test](#adding-a-new-test)
  - [Adding a new backend](#adding-a-new-backend)
    - [1. Add the related project options](#1-add-the-related-project-options)
    - [2. Add the backend-specific variables](#2-add-the-backend-specific-variables)
    - [3. Add the information to generate installation wrappers](#3-add-the-information-to-generate-installation-wrappers)
    - [4. Add the headers target](#4-add-the-headers-target)
    - [5. Add the binary target](#5-add-the-binary-target)
    - [6. Add the backend name to the relevant tests](#6-add-the-backend-name-to-the-relevant-tests)
- [Test Categories and modes](#test-categories-and-modes)
- [Reproducible Builds](#reproducible-builds)
- [The coverage infrastructure](#the-coverage-infrastructure)

# Introduction to ALP/GraphBLAS Building and Testing Infrastructure:

The current building infrastructure is based on [CMake](https://cmake.org) and
on what is commonly defined "Modern CMake". If you are new to this technology,
here are some useful materials:

* [CMake official tutorial](https://cmake.org/cmake/help/latest/guide/tutorial/index.html)
* an introduction do Modern CMake like
[Effective Modern CMake](https://gist.github.com/mbinna/c61dbb39bca0e4fb7d1f73b0d66a4fd1)
* [control structures](https://cmake.org/cmake/help/latest/manual/cmake-language.7.html#id27)
and in particular [basic expressions](https://cmake.org/cmake/help/latest/command/if.html#basic-expressions)
* [scoping rules](https://levelup.gitconnected.com/cmake-variable-scope-f062833581b7)
* [properties of targets](https://cmake.org/cmake/help/latest/manual/cmake-properties.7.html#properties-on-targets)

The current testing infrastructure is composed of several scripts invoking the
test binaries.
These scripts can be invoked either manually or via the building infrastructure.

In the following, the main steps to use the building and testing infrastructure
are discussed, together with more advanced topics.

# The Building Infrastructure

To create the building infrastructure, `cmake` **version 3.13 or higher** is
required (https://cmake.org/download/), together with GNU `make` (or `ninja`).
Not that ALP/GraphBLAS supports **only Linux** and there is no current plan to
support other operating systems.

This section details various aspects of the building infrastructure, from its
generation to its expansion with more tests or more backends.

In general, the **building infrastructure is designed to generate multiple
ALP/GraphBLAS backends at once**, according to the passed configuration options.
Some backends have their own corresponding binary library (like the bsp1d and
the hybrid backend), while other backends may be grouped into the same library;
for example, the reference and reference_omp backends live together in the
so-called *shared memory backend* library.
The building infrastructure allows users to select which backends are to be
built together with the relevant build options (dependencies,
additional compilation/optimization flags, ...).

There are **two ways to create the building infrastructure**, depending on the
level of control you want over the build options.

## Generation via the `bootstrap.sh` script

The easiest way to initialize the building infrastructure is via the
`bootstrap.sh` script in the project root. This script allows for the convenient
setting of common build options for end-users of ALP/GraphBLAS. To invoke it,
create an empty directory for the build, move into it, and then invoke the
`bootstrap.sh` script from there. For example:

```bash
cd <ALP/GraphBLAS root>
mkdir build
cd build
../bootstrap.sh --prefix=../install <other options>
```

The `bootstrap.sh` script should be invoked from an empty directory; if the
directory is not empty, it asks to delete its contents. Invoking it from the
ALP/GraphBLAS source directory is disallowed. The script accepts the following
arguments:

* `--prefix=<path/to/install/directory/>` (**mandatory**) specifies the
directory to install ALP/GraphBLAS consumable targets (binaries, headers,
wrapper scripts and configuration files for consumption via CMake); if the
directory does not exist it is built at the moment, but its parent directory
must exist
* `--with-lpf[=<path/>]` enables LPF and passes its installation directory; if
only `--with-lpf` is given without the `=<path/>` information, LPF binaries
are assumed to be available in the standard search paths and automatically read
from there (e.g., via `command -v lpfrun`)
* `--with-banshee=<path/>` to pass the tools to compile the banshee backend
(required together with `--with-snitch=<path/>`)
* `--with-snitch=<path/>` to pass the tools for Snitch support for the banshee
backend (required together with `--with-banshee=<path/>`)
* `--no-reference` disables the reference and reference_omp backends
* `--debug-build` build ALP/GraphBLAS with debug-suitable options, both backends
and tests; note that this causes tests to run much slower than with standard
options (corresponding to CMake's `Release` build type)
* `--generator=<value>` sets the generator for CMake, otherwise CMake's default
is used; example values are `Unix Makefiles` (usually CMake's default on UNIX
systems) and `Ninja` -- for more information, see
[the official documentation](https://cmake.org/cmake/help/latest/manual/cmake-generators.7.html)
* `--show` shows generation commands instead of running them; useful for dry
runs of the script
* `--delete-files` deletes all files in the current directory without asking for
confirmation; it is iseful, for example, for scripted builds
* `--with-datasets=<path/>` allows passing the path to the directory with the
  datasets required to run some tests (otherwise skipped)
* `--spblas-prefix=<prefix>` to indicate a custom prefix for the spBLAS library;
  the library (and the corresponding `make` target) will be called
  "\<prefix\>\_spblas_\<backend\>"
* `--no-solver-lib` to disable generating the target for the library of lines
  solvers (compiled against the nonblocking backend)
* `--enable-extra-solver-lib` to enable libraries for solvers compiled against
  the reference and OMP backends
* `--help` shows all available options and skips directory checks.

For a dry run, just add the `--show` option to inspect the building command on
the terminal.

### Direct Generation via `cmake`

If you want more control over the building options, you may want to invoke
`cmake` manually and choose each option.
This allows choosing the build type (e.g., `Release` or `Debug`), which
backends and dependencies are enabled and many more aspects.

Since CMake encourages out-of-tree builds, you should first create a dedicated
build directory, from which you may run `cmake`.
This way you can experiment with multiple building options and have separate
build-trees, where in case of changes you need to recompile only what is really
needed: for example, you may have a directory `build_release` to compile with
release-suitable options (`cmake` flag `-DCMAKE_BUILD_TYPE=Release`) and another
called `build_debug` to compile with debug-suitable options (`cmake` flag
`-DCMAKE_BUILD_TYPE=Debug`); if you change a single file, you must recompile
only that one inside the directory corresponding to the build type you want to
test (e.g., `build_release` if you want to assess the performance or
`build_debug` to run with a debugger).

As from above, a convenient way to start even for a custom build is from the
`bootstrap.sh` script, which can be invoked with the `--show` option to inspect
the building command and start from there with the custom options.
For example:

```cmake
mkdir build_release
cd build_release
cmake -DCMAKE_INSTALL_PREFIX=/path/to/install/dir -DCMAKE_BUILD_TYPE=Release \
  -DLPF_INSTALL_PATH='/path/to/lpf/install' <other options prefixed with -D ...> \
  ../

make -j$(nproc) unittests
```

The following section describes all available options.

### CMake Build Options, Types and Flags

Currently, the CMake infrastructure supports several options, grouped in the
following according to their scope.

To control the backends to build, the following options are available:

* `WITH_REFERENCE_BACKEND` to build the reference backend (default: `ON`)
* `WITH_OMP_BACKEND` to build the OMP backend (default: `ON`)
* `WITH_NUMA` to enable NUMA support (default: `ON`)
* `LPF_INSTALL_PATH` path to the LPF tools for the bsp1d and hybrid backends
(default: `OFF`, no LPF backend)
* `WITH_BSP1D_BACKEND` build the bsp1d backend (needs `LPF_INSTALL_PATH` set,
otherwise `OFF`)
* `WITH_HYBRID_BACKEND` build the Hybrid backend (needs `LPF_INSTALL_PATH` set,
otherwise `OFF`)
* `WITH_NONBLOCKING_BACKEND` build the non-blocking backend (default: `ON`)

When choosing, keep in mind that several constraints apply:

* the bsp1d and and hybrid backends both need LPF and NUMA support
* the hybrid backend needs the OMP backend
* the bsp1d backend requires either the reference or the OMP backend

Passing incompatible options will cause error messages and the build to stop.

The ALP/GraphBLAS building infrastructure currently supports  *Release*,
*Debug* and *Coverage* builds, on which several compilation flags depend that
are defined by default in the [main CMakeLists.txt file](../CMakeLists.txt)
inside the section `SETTINGS FOR COMPILATION`.
The build type can be chosen via the standard
[CMAKE_BUILD_TYPE](https://cmake.org/cmake/help/latest/variable/CMAKE_BUILD_TYPE.html)
option; if none is passed, *Release* is automatically set.
In particular, *Release* optimizes all targets aggressively and disables non-
mandatory sanity and run-time checks, which are enabled in the *Debug* build
and can hence result in much slower code.
Finally, the *Coverage* build type instruments backend and test binaries to
extract coverage information after running them and, to this aim, may disable
certain performance optimizations; for more information, see the
[dedicated section](#the-coverage-infrastructure).

The following options control compile definitions and options for backends and
tests and are by default empty:

* `COMMON_COMPILE_DEFINITIONS` compile definitions common to both backends and
tests; setting this option **overrides** the default definitions
* `COMMON_COMPILE_OPTIONS` compile options common to both backends and
tests; setting this option **overrides** the default options
* `ADDITIONAL_BACKEND_DEFINITIONS` compiler definitions **only for backends**
that are **appended** to default definitions, or appended to common definitions
if the defaults were overridden
* `ADDITIONAL_BACKEND_OPTIONS` compiler options **only for backends**
that are **appended** to default options, or appended to common options if the
defaults were overridden
* `ADDITIONAL_TEST_DEFINITIONS` compiler definitions **only for tests** that are
**appended** to default definitions, or to common definitions if the default
were overridden
* `ADDITIONAL_TEST_OPTIONS` compiler options applicable **only to tests**.
These options are **appended** to the default options, or appended to common
options if the defaults were overridden
* `TEST_PERFORMANCE_DEFINITIONS` compiler definitions for performance,
applicable **only to tests** that do not explicitly disable performance flags.
Setting this option **overrides** the default performance definitions, leaving
any other definitions intact
* `TEST_PERFORMANCE_OPTIONS` compiler options for performance
applicable **only to tests** that do not explicitly disable performance flags.
Setting this option **overrides** the default performance options, leaving any
other options intact

Since ALP/GraphBLAS is mostly template-based, most of the code being compiled
for a test belongs to the executable itself, rather than to the backend library
the test is linked against.
This motivates the presence of performance flags for performance tests as well as
for excessively slow unit and smoke tests. The backend libraries also use
performance flags, though most crucially, the end-user should take care to
compile ALP/GraphBLAS programs or libraries with similar performance flags-- in
particular, the `-DNDEBUG` flag should not be forgotten.
In case a user wants to control performance optimizations for both tests and
backends explicitly, she may set the `COMMON_COMPILE_{DEFINITIONS|OPTIONS}`
options to override the global flags and the `ADDITIONAL_BACKEND_{DEFINITIONS|OPTIONS}`
to append flags.
For example, one may add the `_FORTIFY_SOURCE=1` definition for backends only,
e.g. for production purposes, via

```bash
cmake -DADDITIONAL_BACKEND_DEFINITIONS="_FORTIFY_SOURCE=1" <other flags as usual> ../
```

or similarly for tests by adding the flag
`-DADDITIONAL_TEST_DEFINITIONS="_FORTIFY_SOURCE=1"` to the same command.
During the CMake configuration step a report summarizes *all* the flags for the
various kinds of targets and for each test category.

To manually tweak the various compilation flags, one can edit the
`CMakeCache.txt` file, where the related variables are stored for backends and
for the various categories of tests.
Then a `cmake .` run regenerates the compilation infrastructure and the flags
report should correctly reflect the new compilation settings.

Furthermore, one can use
[CMake Environment Variables for Languages](https://cmake.org/cmake/help/latest/manual/cmake-env-variables.7.html#environment-variables-for-languages) (like
[CXXFLAGS](https://cmake.org/cmake/help/latest/envvar/CXXFLAGS.html)) to
"inject" flags from the command line that are directly applied; the only caveat
here is that these flags may conflict with the default ones or be ignored
depending on the order they are passed to the compiler; hence, this apprach is
discouraged.
For a deep inspection of all the commands being run, the user can build the
desired target(s) via `make` and pass the `VERBOSE=3` option, for example

```bash
make test_matrixIterator_bsp1d VERBOSE=3
```

Finally, another set of options store the directories with test inputs:

* `DATASETS_DIR` Directory with datasets for tests, used for most tests
requiring an input (default: `<ALP/GraphBLAS root>/datasets`)
* `GNN_DATASET_PATH` directory with the GNN dataset, for the tests requiring it


## Naming conventions for targets

The names of targets defined inside the infrastructure follow these conventions:

* targets needed (directly or indirectly) to build backends start with
`backend_`, in particular:
  * targets referring to headers include paths (in
[<ALP/GraphBLAS root>/include/CMakeLists.txt](../include/CMakeLists.txt)) end
with `_headers`, following the pattern `backend_<backend>_headers`
  * targets to build whole backends are named
`backend_<backend>_<static|shared>`, for example `backend_reference_static`
  * targets for backends to be linked to tests are named
`backend_<backend>` and are usually an alias of a corresponding target
`backend_<backend>_<static|shared>`; **this naming scheme should not be
changed**, otherwise tests would not link to backends anymore and errors would
occur
* targets to build and run single tests start with `test_`, then have the test
name, the category name, the mode name (if any) and end with the backend name,
following the pattern `test_<test name>_<category>_<mode>_<backend>`, such as
`test_hpcg_unit_ndebug_reference_omp`; note that the actual binaries do not have
the `test_` prefix and the category name the for simplicity; for more
information about categories and modes, you may read the [Testing
Infrastructure](#the-testing-infrastructure) section
* targets to build and run a group of tests start with the prefix `tests_`, for
example
  * `tests_<category>` build and runs the tests for the given category
(`unit`, `knn`, `pagerank`, ...)
  * some target names referring to the same per-category tests are kept for
backward compatibility with the previous infrastructure, such as `unittests`,
`smoketests`, `test`, ...; these targets are listed in the
[main documentation](../README.md#overview-of-the-main-makefile-targets)
* targets to just build (without running) groups of tests (for example to test
the building infrastructure) start with `build_`, for example:
  * `build_tests_category_<category>` builds all test of the given category
  * `build_tests_backend_<backend>` builds all test of the given category
  * `build_tests_all` builds all tests
* certain targets that just list existing tests for each category start with
`list`, for example:
  * `list_tests_category_<category>` for the tests of a given category
  * `list_tests_backend_<backend>` for the tests of a given backend
  * `list_tests_categories` to list the actual test categories
  * `list_tests_all` to list all tests


Other targets:

* `libs` build the binary libraries for all backends
* `docs` builds all HTML ALP/GraphBLAS documentation in
`<ALP/GraphBLAS root>/docs/code/html/index.html` and the LaTeX source files in
`<ALP/GraphBLAS root>/docs/code/latex`; if `pdflatex`, `graphviz`, and other
standard tools are available, they are compiled into a PDF found at
`<ALP/GraphBLAS root>/docs/code/latex/refman.pdf`.

## Adding a new test

Test sources are split in categories, whose purpose is explained in the [Testing
Infrastructure](#the-testing-infrastructure) section.
For each category, the tests are compiled and run according to the file
`tests/<category>/CMakeLists.txt`, for example
[tests/unit/CMakeLists.txt](../tests/unit/CMakeLists.txt).
Adding a test thus requires adding the relevant command to the `CMakeLists.txt`
file that corresponding to the test's category, for example

```cmake
add_grb_executables( my_test
    my_test_source_1.cpp
    my_test_source_2.hpp
    my_test_source_3.hpp

    BACKENDS reference reference_omp bsp1d hybrid
    ADDITIONAL_LINK_LIBRARIES test_utils
    COMPILE_DEFINITIONS MY_TEST_KEY=VALUE ANOTHER_TEST_DEFINITION
)
```

In this example:
* `add_grb_executables` is a command similar in spirit to CMake's
[`add_executable`](https://cmake.org/cmake/help/latest/command/add_executable.html),
which adds multiple tests, one per backend
* `my_test` is the test *base name*
* `my_test_source_1.cpp`, `my_test_source_2.cpp` and `my_test_source_3.cpp` are
the test source files (at least one is required)
* `BACKENDS reference reference_omp bsp1d hybrid` is the list of all backends
the test should be compiled against (at least one is required); for each
backend, an executable target is created following the naming conventions in
[Naming conventions for targets](#naming-conventions-for-targets)
* `ADDITIONAL_LINK_LIBRARIES test_utils` (optional) lists additional libraries
to link (the backend library is linked by default)
* `COMPILE_DEFINITIONS MY_TEST_KEY=VALUE ANOTHER_TEST_DEFINITION` (optional)
lists additional compile definitions (corresponding to, e.g., gcc's definitions
`-DMY_TEST_KEY=VALUE -DANOTHER_TEST_DEFINITION`)

More options are available for the function `add_grb_executables`, which are
documented in [cmake/AddGRBTests.cmake](../cmake/AddGRBTests.cmake).

Tests may have different categories; presently, one of:

* unit,
* performance,
* smoke.

All tests belonging to each category are run via the related script in the
project root: unit tests are run via the script `unittests.sh`, while the
performance classes are run as part of the `perftests.sh` script.
When a new test is added, **its invocation(s) must be manually added to the
relevant script**.
Each script is sub-divided in several sections depending on the backend that is
assumed to run and on relevant options: hence, you should place your test
invocation in the relevant section.

Furthermore, you can achieve more control over the test target generation, i.e.,
the building of tests, by using the function `add_grb_executable_custom`, also
defined in [cmake/AddGRBTests.cmake](../cmake/AddGRBTests.cmake), which requires
to specify dependencies manually (thus, building against multiple backends needs
correspondingly multiple calls of the same function) and is therefore used only
in special cases.


## Adding a new backend

Adding a new backend requires multiple changes to the building infrastructure,
which are discussed here.
For the sake of this example, we assume that the new backend:

* is named `example`
* has several headers stored inside
`<ALP/GraphBLAS root>/include/graphblas/example`
* has several implementation files stored inside
`<ALP/GraphBLAS root>/src/graphblas/example`
* should produce a separate static library, to be linked to each test/application

Hence, the steps are as follows.

### 1. Add the related project options

The new backend may need some configuration options from the user, to be added
at the beginning of
[the main configuration file `<ALP/GraphBLAS root>/CMakeLists.txt`](../CMakeLists.txt)
with the proper validation steps; for example, you may add an option to enable
it

```cmake
option( WITH_EXAMPLE_BACKEND "Enable building New Backend" OFF )
```

Similarly, you should also add the relevant logic to check for dependencies (if
any) and for possible interactions with other backends and compile
options/definitions.
Some examples of this logic already exist in the
[root CMakeLists.txt](../CMakeLists.txt).
For example, the following code snippet

```cmake
if( NOT LPF_INSTALL_PATH AND
    (WITH_BSP1D_BACKEND OR WITH_HYBRID_BACKEND) )
    message( SEND_ERROR "The BSP1D and Hybrid backends require LPF" )
    message( SEND_ERROR "Hence, you should set LPF_INSTALL_PATH" )
    message( FATAL_ERROR "or not enable WITH_BSP1D_BACKEND or WITH_HYBRID_BACKEND")
endif()
```
checks that the install path of LPF is given if the bsp1d or hybrid backends are
enabled, because LPF is needed to build either of them.
Another example is finding the dependencies specific to your backend, which is
usually done via CMake's `find_package` command.
For example, the following logic

```cmake
if( WITH_BSP1D_BACKEND OR WITH_HYBRID_BACKEND )
    find_package( MPI REQUIRED )
    find_package( LPF REQUIRED )
endif( )
```

checks whether MPI and LPF are installed in the system, since they are both
required for either the bsp1d and the hybrid backend; if not, the `REQUIRED`
keyword instructs CMake to immediately halt and emit an error message.

In the
[Official documentation](https://cmake.org/cmake/help/latest/manual/cmake-modules.7.html#find-modules)
you can find a list of pre-defined modules to look for dependencies, or you can
easily write your own called `Find<DependencyName>.cmake` and add it to the
[cmake directory](../cmake) in the root; you may invoke it as

```cmake
if( WITH_EXAMPLE_BACKEND )
  find_package( <DependencyName> REQUIRED )
endif()
```

For an overview of importing dependencies you may refer to the official
[Using Dependencies Guide](https://cmake.org/cmake/help/latest/guide/using-dependencies/index.html).
For an example on how to write a custom module for a dependency, you may check
the internal [FindNuma module](../cmake/FindNuma.cmake).

### 2. Add the backend-specific variables

The file
[`<ALP/GraphBLAS root>/cmake/AddGRBVars.cmake`](../cmake/AddGRBVars.cmake)
stores the relevant variables for all the main steps of the compilation, like
default names of backend targets, compilation and linking options and so on.
You should add the new ones for the new backend.

Examples of needed variables:

1. a variable storing the default name of the backend target for tests,
according to the conventions explained in
[Naming conventions for targets](#naming-conventions-for-targets), for example

    ```cmake
    set( EXAMPLE_BACKEND_DEFAULT_NAME "backend_example" )
    ```

2. a variable storing the compilation definitions to include the relevant
headers, for example

    ```cmake
    set( EXAMPLE_INCLUDE_DEFS "_GRB_WITH_EXAMPLE" )
    ```

3. a variable storing the compilation definitions to select the example backend
by default when compiling executables, for example

    ```cmake
    set( EXAMPLE_SELECTION_DEFS "_GRB_BACKEND=example" )
    ```

4. the variable `ALL_BACKENDS` lists all possible backends (even if not enabled)
to detect potential configuration errors: therefore, you should always add the
new backend to this variable; on the contrary, the variable `AVAILABLE_BACKENDS`
lists only the backends actually available in the building infrastructure,
depending on the user's inputs; you may add your backend with something like

    ```cmake
    if ( WITH_EXAMPLE_BACKEND )
    	list( APPEND AVAILABLE_BACKENDS "example" )
    endif()
    ```

5. the variable `AVAILABLE_TEST_BACKENDS` lists all backends that were enabled
and for which tests are built; usually it is a subset of `AVAILABLE_BACKENDS`,
which also contains backends pulled in as dependencies of user-chosen backends;
for example, if the user enables only the hyperdags backend, the reference
backend is also listed in `AVAILABLE_BACKENDS`, while `AVAILABLE_TEST_BACKENDS`
lists only hyperdags.

For more details, you may see inside
[`<ALP/GraphBLAS root>/cmake/AddGRBVars.cmake`](../cmake/AddGRBVars.cmake) how
existing backends populate these variables.

### 3. Add the information to generate installation wrappers

The file
[`<ALP/GraphBLAS root>/cmake/AddGRBInstall.cmake`](../cmake/AddGRBInstall.cmake)
stores variables to generate the wrapper scripts for the usage of ALP/GraphBLAS
from external projects, explained in
[ALP/GraphBLAS Wrapper Scripts](Use_ALPGraphBLAS_in_your_own_project.md#via-the-wrapper-scripts).

The first variable of interest is the install location for the binary file,
which may be set via a variable like

```cmake
set( EXAMPLE_BACKEND_INSTALL_DIR "${BINARY_LIBRARIES_INSTALL_DIR}/example" )
```

used in the following steps. The same binary file may implement multiple
backends. For example, both the reference and the OMP backend share
the same binary file, i.e., the one generated for shared memory backends.

For convenience, the macro `addBackendWrapperGenOptions` is provided to
automatically generate the necessary variables according to the internal naming
conventions.
You should invoke it by listing the relevant options for compilation and
linking, after the backend name; for example

```cmake
if( WITH_EXAMPLE_BACKEND )
  addBackendWrapperGenOptions( "example"
    COMPILE_DEFINITIONS "${EXAMPLE_INCLUDE_DEFS}" "${EXAMPLE_SELECTION_DEFS}"
    COMPILE_OPTIONS "-Wall" "-Wextra"
    LINK_FLAGS "${EXAMPLE_BACKEND_INSTALL_DIR}/lib${BACKEND_LIBRARY_OUTPUT_NAME}.a"
  )
endif()
```

Note that some compilation definitions and options are already applied to all
backends, as listed in `COMMON_WRAPPER_DEFINITIONS` and `COMMON_WRAPPER_OPTIONS`.
For more options, you may read the documentation of the
`addBackendWrapperGenOptions` macro, in the same file.
Since it is practically impossible to automatically get all compilation options
and all linking dependencies from a CMake target, you should carefully add all
relevant information for the building and linking processes, pretty much as if
you were manually invoking the compiler from the command line in order to
compile an application depending on your backend.
Similarly, you should add all relevant options for the runner (e.g., the LPF
runner for distributed targets) and its environment.
For more examples, you may inspect the usages of the
`addBackendWrapperGenOptions` macro inside
[`<ALP/GraphBLAS root>/cmake/AddGRBInstall.cmake`](../cmake/AddGRBInstall.cmake).
As a final validation step, you may check the content of the wrapper scripts
described in
[ALP/GraphBLAS Wrapper Scripts](Use_ALPGraphBLAS_in_your_own_project.md#via-the-wrapper-scripts)
and even try it against a simple application (like a test).

### 4. Add the headers target

Add a target listing the headers of your new backend to the
[`<ALP/GraphBLAS root>/include/CMakeLists.txt`](../include/CMakeLists.txt) file,
usually enclosed in

```cmake
if( WITH_EXAMPLE_BACKEND )
 ...
endif()
```
This usually requires:

1. creating an `INTERFACE` library, for example

    ```cmake
    add_library( backend_example_headers INTERFACE )
    ```
2. adding a dependency on `backend_headers_nodefs` to add the global include
path (`<ALP/GraphBLAS root>/include`) all source files assume

    ```cmake
    target_link_libraries( backend_example_headers INTERFACE backend_headers_nodefs )
    ```
3. adding other dependencies (if any) as `INTERFACE`

    ```cmake
    target_link_libraries( backend_example_headers INTERFACE <cmake targets example headers depend on> )
    ```

4. adding relevant compile definitions/options (if any)

    ```cmake
    target_compile_definitions( backend_example_headers INTERFACE "KEY=VALUE" "SYMBOL_TO_BE_DEFINED" )
    target_compile_options( backend_example_headers "-Wall" "-Wextra" ) # if you want very verbose warnings
    ```
    please, note that these settings will propagate to all targets depending on
`backend_example_headers`, so you should add only what is really needed

5. adding the installation options

    ```cmake
    install( DIRECTORY "${CMAKE_CURRENT_SOURCE_DIR}/graphblas/example" # path with the headers
        DESTINATION "${GRB_INCLUDE_INSTALL_DIR}" # installation destination: you should leave the default
        FILES_MATCHING REGEX "${HEADERS_REGEX}" # regex to install headers only (in case "${CMAKE_CURRENT_SOURCE_DIR}/graphblas/example" contains other files)
    )
    install( TARGETS backend_example_headers EXPORT GraphBLASTargets )
    ```
    to copy the header files (first call) and to add the `backend_example_headers`
target to the configuration file during installation (so that users can consume
the backends directly from their CMake infrastructure), while the second call
exports the newly created target to the CMake infrastructure automatically
generated inside the installation directory


### 5. Add the binary target
To actually build the library, you should

1. create a new `CMakeLists.txt` file inside
`<ALP/GraphBLAS root>/src/graphblas/example`, where you write the compilation
instruction for the binary library of the new backend

2. include it hierarchically from
[`<ALP/GraphBLAS root>/src/graphblas/CMakeLists.txt`](../src/graphblas/CMakeLists.txt)
by adding the sub-directory at the end

    ```cmake
    if( WITH_EXAMPLE_BACKEND )
        add_subdirectory( example )
    endif()
    ```
3. inside `<ALP/GraphBLAS root>/src/graphblas/example/CMakeLists.txt`, create
the library target

    ```cmake
    add_library( backend_example_static STATIC <sources of new backend> )
    ```
    note that this creates a static library, while you may want to also create a
shared library with the `SHARED` keyword

4. link the new target against all needed targets, in particular the related
backend headers and the `backend_flags` target

    ```cmake
    target_link_libraries( backend_example_static
      PRIVATE backend_flags
      PUBLIC backend_example_headers
      PRIVATE <other dependencies, e.g. Numa::Numa>
    )
    ```

    here, it is important to note that:

    * the `backend_flags` target, which stores the default flags for all backends
and in particular the optimization flags, is linked as `PRIVATE` in order to
keep its flags only local to `backend_example_static` and not propagate them to
depending targets (which have their own flags)
    * `backend_example_headers` is linked as `PUBLIC` to expose the include
paths and the related definitions also to depending targets
    * the other dependencies are here all linked as `PRIVATE` for the sake of
example, the actual visibility though depends on the implementation of the
backend and should be evaluated for each dependency

5. add the relevant compile definitions to select the new backend by default
when compiling depending targets (like tests), for example

    ```cmake
    target_compile_definitions( backend_example_static PUBLIC "${EXAMPLE_BACKEND_SELECTION_DEFS}" )
    ```
    where `EXAMPLE_BACKEND_SELECTION_DEFS` is a variable defined in
[step 2](#2-add-the-backend-specific-variables) inside
[`<ALP/GraphBLAS root>/cmake/AddGRBVars.cmake`](../cmake/AddGRBVars.cmake),
usually something like `"_GRB_BACKEND=example"`; here, the `PUBLIC` keyword
causes the definitions to be used for both the backend library and the depending
targets

6. add the needed definitions and compile options (if any), with the usual
    * `target_compile_definitions( backend_example_static ...)`
    * `target_compile_options( backend_example_static ... )`
    * similarly, you may want to set specific target properties, like the binary
build path, with

        ```cmake
        set_target_properties( backend_example_static PROPERTIES ARCHIVE_OUTPUT_DIRECTORY "${CMAKE_CURRENT_BINARY_DIR}/example_output_dir" )
        ```

1. add the new library to the `libs` target, which allows users to compile all
backend libraries at once

    ```cmake
    add_dependencies( libs backend_example_static )
    ```
8. add the installation options

    ```cmake
    install( TARGETS backend_example_static
      EXPORT GraphBLASTargets
      ARCHIVE DESTINATION "${EXAMPLE_BACKEND_INSTALL_DIR}"
    )
    ```
    which exports the binary target to the generated CMake infrastructure in the
installation directory and the binary library to the directory
`${EXAMPLE_BACKEND_INSTALL_DIR}` defined previously

9. if the target name `backend_example_static` is different from the
corresponding default one specified during
[step 2](#2-add-the-backend-specific-variables) inside
[`<ALP/GraphBLAS root>/cmake/AddGRBVars.cmake`](../cmake/AddGRBVars.cmake),
you should create an `ALIAS` target for the default backend target to exist

    ```cmake
    add_library( "${EXAMPLE_BACKEND_DEFAULT_NAME}" ALIAS backend_example_static )
    ```
    where `EXAMPLE_BACKEND_DEFAULT_NAME` is the variable defined in
[step 2](#2-add-the-backend-specific-variables) that stores the default name
according to the convention `backend_<backend name>`, for example
`set( EXAMPLE_BACKEND_DEFAULT_NAME backend_example )` (for more information,
you may refer to
[the targets naming conventions](#naming-conventions-for-targets)); note that a
separate `ALIAS` target also allows easily switching between static and shared
libraries for the default target, e.g. if the user may want to control this from
the initial configuration (e.g., by adding a hypothetical configuration option
`-DLINK_SHARED_LIBRARIES_BY_DEFAULT=ON` to the initial cmake invocation).

### 6. Add the backend name to the relevant tests

Since not all tests may run against all backends (for example because of
functionalities still under development), you should add the `example` option to
each test you want to compile against.
Tests are listed in the `CMakeLists.txt` files under
`<ALP/GraphBLAS root>/tests/` according to their category.
For example, to compile the existing `argmax` test (defined in
[tests/unit/CMakeLists.txt](../tests/unit/CMakeLists.txt)) also against the new
backend, you may add `example` to its list of compatible backends as in the
following

```cmake
add_grb_executables( argmax argmax.cpp
    BACKENDS reference reference_omp bsp1d hybrid example
)
```

From here, the function `add_grb_executables` (defined in
[cmake/AddGRBTests.cmake](../cmake/AddGRBTests.cmake)):

1. checks that a backend named `example` is present in `ALL_BACKENDS`
(as from [step 2.](#2-add-the-backend-specific-variables)); if not, an error is
raised
2. check whether it is actually enabled, i.e. it it inside `AVAILABLE_BACKENDS`
(as from [step 2.](#2-add-the-backend-specific-variables)); if not, no binary is
built for this test and backend
3. checks that a target `backend_example` exists; if not, no default target
associated to the `example` backend exists and an error is raised
4. creates an executable target linked against `backend_example` and populates
the related variables (list of per-category tests and so on); the target is
named `test_argmax_[<test mode>_]example` according to the conventions
5. on compilation, a binary named `argmax_example` is generated in
`<build directory>/tests/unit`

Similarly, you may want to add an example built against your test to showcase
your new backend's nitty-gritties.
This is done by adding an executable target inside
[examples/CMakeLists.txt](../examples/CMakeLists.txt), for example

```cmake
if( WITH_EXAMPLE_BACKEND )
    add_executable( sp_example sp.cpp )
    target_link_libraries( sp_example backend_example )
    add_dependencies( examples sp_example )
endif()
```

# Test Categories and modes

Tests are grouped in *categories* according to what they test:

1. **unit** tests assess the functionalities of ALP/GraphBLAS and aim at broad
code coverage by testing that the code is compliant to the specification
both with the most common execution conditions and with corner cases
2. **performance** tests have as a primary aim to check that new commits do not
introduce performance bugs. Secondarily and ideally, performance tests also can
be used to generate results that go into publications, to validate published
results are correct (for external parties), and/or to easily repeat past
experiments on new platforms.
3. **smoke** tests aim to test typical usage of the library, and focus on the
"happy path" only. In ALP/GraphBLAS, this includes aiming for more "advanced"
usages such as integration with MPI, or manual launching where multiple OS
processes are combined ad hoc to co-execute an ALP program. These tests could
also be used to very quickly verify that changes are sane-- e.g., after making
changes to a backend, run the smoke tests, and only if those are OK, push them
for the CI to run the full unit test suite.

The test categories are defined in the CMake variable `TEST_CATEGORIES` inside
[the root CMakeLists.txt](../CMakeLists.txt): each test must belong to one or
more of the categories listed there; in case of unknown category specified with
a test (see [Adding a new test](#adding-a-new-test)), an error is raised.

Categories requiring a test to be compiled in multiple ways may define so-called
*modes*, i.e. set of naming conventions for the test's target name and the
corresponding executable and specific compilation flags.
The various compilation flags and modes are defined and documented in the
[CompileFlags.cmake file](../cmake/CompileFlags.cmake).
During configuration this file generates a report with all compilation flags for
the various target types and categories/modes.

# Reproducible Builds

To ease building and deploying ALP/GraphBLAS, dedicated Docker images can be
built from the `Dockerfile`s in the *ALP/ReproducibleBuild* repository

https://github.com/Algebraic-Programming/ReproducibleBuilds

The images built from these files represent the standard build environment used
by most ALP/GraphBLAS developers and contain all necessary dependencies to build
*all* ALP/GraphBLAS backends and tests, including the necessary input datasets.
You may refer to the
[README](https://github.com/Algebraic-Programming/ReproducibleBuilds#readme) for
more information about the content and how to build the images.

The same images can be used for a Continuous Integration (CI) setup, as they
provide all needed dependencies and tools.
Indeed, the file [`.gitlab-ci.yml`](../.gitlab-ci.yml) describes the CI jobs
that internally test ALP/GraphBLAS via [GitLab](https://about.gitlab.com/),
which is available [open source](https://about.gitlab.com/install/).

The here described image and CI pipeline are confirmed to work with x86 runners
with 24 virtual CPU cores and 32GB RAM.

# The coverage infrastructure

The *Coverage* build type stores coverage information into machine- or human-readable
files after running one or more test binaries.
These files can be directly read by users or can be consumed by tools to display
the coverage information in a user-friendly interface, possibly integrated
within a CI/CD infrastructure (e.g., GitHub or GitLab).

The coverage infrastructure prescribes additional dependencies:
* [gcov](https://gcc.gnu.org/onlinedocs/gcc/Gcov.html) to instrument the binary
  during compilation; it usually comes together with a GNU C/C++ compiler, in
  form of a compiler-specific library (`libgcov.a`) and a command-line tool
  (e.g., `gcov-9` for `gcc-9`/`g++-9`)
* [gcovr](https://gcovr.com/en/stable), a Python3 tool that translates gcov
  traces to multiple formats; it can be installed via `pip` as `python3 -m pip install gcovr`
  (on some distributions also via the package manager, e.g., `apt-get install gcovr`
  -- though the first method is preferable as it provides a more up-to-date
  version) and clearly requires
* [Python3](https://www.python.org/), available in most Linux distributions
  (e.g., `apt-get install python3`) or as [pre-built binary](https://github.com/indygreg/python-build-standalone/releases)
  for many OSs and architectures
  (e.g., https://github.com/indygreg/python-build-standalone/releases/download/20230116/cpython-3.11.1+20230116-x86_64_v4-unknown-linux-gnu-install_only.tar.gz)

Note that because of the *gcov* dependency **only GCC is currently supported**
as a compiler.
Also note that the coverage infrastructure is implemented only in the CMake
infrastructure.

The Coverage build mode can be enabled in two ways:
- if you use the `bootstrap.sh` script to generate the build infrastructure, you
  can add the `--coverage-build` option on invocation
- if you directly use CMake, with `-DCMAKE_BUILD_MODE=Coverage`

By doing so:
* all binary targets are compiled with the *-fprofile-arcs* and *-ftest-coverage*
  flags to enable tracing: these cause the creation of `.gcno` and `.gcda` files
  in the directory each binary runs in that store exection traces
* performance optimizations for *all* targets are much more restrictive than for
  other build types, preventing any aggressive optimization (`-O1`) and especially
  inlining in order to gather accurate coverage information
* because of these restrictions, multiple modes for a test category are not useful
  anymore (the flags are the same, and numerical precision is not being tested),
  so only one mode is enabled: for example, only the *ndebug* mode is enabled
  for unit tests
* a new folder is created inside the build directory named `coverage`, where
  coverage reports are stored

Note that, because of **very limited** performance optimizations, tests may run
*much* slower than in release.

Selecting the coverage build type adds several `make` targets to produce coverage
information; these are:
* `coverage_json`: generates *coverage/coverage.json*
* `coverage_cobertura`: generates *coverage/coverage.xml*
* `coverage_csv`: generates *coverage/coverage.csv*
* `coverage_coveralls`: generates *coverage/coveralls.json*
* `coverage_html`: generates *coverage/index.html*

These targets correspond to the output formats
[gcovr can generate](https://gcovr.com/en/stable/output/index.html).

These commands will use any  `.gcno` and `.gcda` files generated during
execution(s) of any program/test.
To clean a coverage report and all coverage information generated during the
execution of the binaries, you can use the `make coverage_clean` command; to
clean only the generated report, simply clean the content of the `coverage`
folder.
Hence, a typical workflow to extract coverage information is:
1. configure with coverage build type, e.g.:
    ```bash
    mkdir build
    cd build
    ../bootstrap.sh --prefix=./install --coverage-build
    ```
2. build and run one or more tests as usual, e.g.:
    ```bash
    make unittests -j$(nproc)
    ```
3. parse coverage information to produce a report, e.g., a human-readable HTML
  report
    ```bash
    make coverage_html
    ```
1. read the HTML report with, e.g., a browser:
    ```bash
    xdg-open coverage/index.html
    ```
2. clean coverage information and report:
    ```bash
    make coverage_clean
    ```
