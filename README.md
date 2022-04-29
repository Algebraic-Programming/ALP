<pre>
   _____  .____   __________      /\   ________                    .__   __________.____       _____    _________
  /  _  \ |    |  \______   \    / /  /  _____/___________  ______ |  |__\______   \    |     /  _  \  /   _____/
 /  /_\  \|    |   |     ___/   / /  /   \  __\_  __ \__  \ \____ \|  |  \|    |  _/    |    /  /_\  \ \_____  \
/    |    \    |___|    |      / /   \    \_\  \  | \// __ \|  |_> >   Y  \    |   \    |___/    |    \/        \
\____|__  /_______ \____|     / /     \______  /__|  (____  /   __/|___|  /______  /_______ \____|__  /_______  /
        \/        \/          \/             \/           \/|__|        \/       \/        \/       \/        \/
</pre>

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

- [Requirements](#requirements)
	- [Compilation](#compilation)
	- [Linking and run-time](#linking-and-run-time)
	- [Optionals](#optionals)
- [Quick start](#quick-start)
- [Overview of the main Makefile targets](#overview-of-the-main-makefile-targets)
	- [1. Running ALP/GraphBLAS as a standalone executable](#1-running-alpgraphblas-as-a-standalone-executable)
		- [Implementation](#implementation)
		- [Compilation](#compilation-1)
		- [Linking](#linking)
		- [Running](#running)
		- [Threading](#threading)
	- [2. Running parallel ALP/GraphBLAS programs from existing parallel contexts](#2-running-parallel-alpgraphblas-programs-from-existing-parallel-contexts)
		- [Implementation](#implementation-1)
		- [Running](#running-1)
- [Debugging](#debugging)
- [Development in ALP](#development-in-alp)
- [Acknowledgements](#acknowledgements)
- [Citing ALP/GraphBLAS](#citing-alpgraphblas)

# Requirements

## Compilation

To compile ALP/GraphBLAS, you need the following tools:

1. A C++11-capable compiler such as GCC 4.8.2 or higher, with OpenMP support
2. LibNUMA development headers
3. POSIX threads development headers
4. CMake (https://cmake.org/download/) version 3.13 or higher, with GNU Make
(CMake's default build tool on UNIX systems) or the build tool of your choice
(e.g., [Ninja](https://ninja-build.org))

## Linking and run-time
The ALP/GraphBLAS libraries link against the following libraries:

1. LibNUMA: `-lnuma`
2. Standard math library: `-lm`
3. POSIX threads: `-lpthread`
4. OpenMP: `-fopenmp` in the case of GCC

## Optionals
Required for distributed-memory auto-parallelisation:

* The Lightweight Parallel Foundations (LPF) communication layer, version 1.0 or
higher, its collectives library, and its dependences. See the LPF project
repositories:

* (https://gitee.com/CSL-ALP/lpf);
* (https://github.com/Algebraic-Programming/LPF).

This dependency applies to compilation, linking, and run-time dependences.

Additionally, to generate the code documentation:

* `doyxgen` reads code comments and generates the documentation
* `graphviz` generates various diagrams for inheritance, call paths, etc.
* `pdflatex` is required to build the PDF file out of the Latex generated
documentation


# Quick start
The compilation and testing infrastructure is based on
[CMake](https://cmake.org), which supports multiple tools for compilation.
In the following, we use it to generate a building infrastructure based on GNU
Makefile, which is CMake's default build tool on UNIX systems and is broadly
available.
However, the concepts described here apply very similarly to other compilation
backends like `ninja`, which are becoming increasingly popular: instead of
`make <target name>`, one can simply run, e.g., `ninja <target name>`.

Here are the basic steps to quickly compile and install ALP/GraphBLAS for shared
memory machines (i.e. without distributed-memory support):


```bash
cd <ALP/GraphBLAS root>
mkdir build
cd build
../bootstrap.sh --prefix=../install
make -j
```

In more detail, the steps to follow are:

1. Create an empty directory for building ALP/GraphBLAS and move into it:
`mkdir build && cd build`.
2. Invoke the `bootstrap.sh` script located inside the ALP/GraphBLAS root directory
`<ALP/GraphBLAS root>` to generate the build infrastructure via CMake inside the
 current directory:
 `<ALP/GraphBLAS root>/bootstrap.sh --prefix=</path/to/install/dir>`
    - note: add `--with-lpf=/path/to/lpf/install/dir` if you have LPF installed
and would like to use it.
3. Issue `make -j` to compile the C++11 ALP/GraphBLAS library for the configured
backends.
4. (*Optional*) To later run all unit tests, several datasets must be made
available. Please run the `<ALP/GraphBLAS root>/tools/downloadDatasets.sh`
script for

    a. an overview of datasets required for the basic tests, as well as

    b. the option to automatically download them.

5. (*Optional*) To make the ALP/GraphBLAS documentation, issue `make docs`. This
generates both

    a. PDF documentations in `<ALP/GraphBLAS root>/docs/code/latex/refman.pdf`,
and

    b. HTML documentations in `<ALP/GraphBLAS root>/docs/code/html/index.html`.

6. (*Optional*) Issue `make -j smoketests` to run a quick set of functional
   tests. Please scan the output for any failed tests.
   If you do this with LPF enabled, and LPF was configured to use an MPI engine
   (which is the default), and the MPI implementation used is _not_ MPICH, then
   the default command lines the tests script uses are likely wrong. In this
   case, please edit `tests/parse_env.sh` by searching for the MPI
   implementation you used, and uncomment the lines directly below each
   occurance.
7. (*Optional*) Issue `make -j unittests` to run an exhaustive set of unit
   tests. Please scan the output for any failed tests.
   If you do this with LPF enabled, please edit `tests/parse_env.sh` if required
   as described in step 5.
8. (*Optional*) Issue `make -j install` to install ALP/GraphBLAS into your
install directory configured during step 1.
9. (*Optional*) Issue `source </path/to/install/dir>/bin/setenv` to make available the
`grbcxx` and `grbrun` compiler wrapper and runner.

Congratulations, you are now ready for developing and integrating ALP/GraphBLAS
algorithms! Any feedback, question, problem reports are most welcome at

<div align="center">
<a href="mailto:albertjan.yzelman@huawei.com">albertjan.yzelman@huawei.com</a>
</div>
<br />

In-depth performance measurements may be obtained via the following additional
and optional step:

10. (*Optional*) To check in-depth performance of this ALP/GraphBLAS
implementation, issue `make -j perftests`. This will run several algorithms in
several ALP/GraphBLAS configurations. All output is captured in
`<ALP/GraphBLAS root>/build/tests/performance/output`. A summary of benchmark
results are found in the following locations:

    a. `<ALP/GraphBLAS root>/build/tests/performance/output/benchmarks`.

    b. `<ALP/GraphBLAS root>/build/tests/performance/output/scaling`.

If you do this with LPF enabled, please note the remark described at step 5 and
apply any necessary changes also to `tests/performance/performancetests.sh`.


# Overview of the main Makefile targets

The following table lists the main build targets of interest:

| Target                | Explanation |
|----------------------:|---------------------------------------------------|
| \[*default*\]         | builds the ALP/GraphBLAS libraries and examples   |
| `install`             | install libraries, headers and some convenience   |
|                       | scripts into the path set as `--prefix=<path>`    |
| `unittests`           | builds and runs all available unit tests          |
| `smoketests`          | builds and runs all available smoke tests         |
| `perftests`           | builds and runs all available performance tests   |
| `tests`               | builds and runs all available unit, smoke, and    |
|                       | performance tests                                 |
| `docs`                | builds all HTML and LaTeX documentation out of the|
|                       | sources inside `<ALP/GraphBLAS root>/docs/code/`  |

For more information about the testing harness, please refer to the
[related documentation](tests/Tests.md).

For more information on how the build and test infrastructure operate, please
refer to the [the related documentation](docs/Build_and_test_infra.md).

There are several use cases in which ALP/GraphBLAS can be deployed and utilized,
listed in the following. These assume that the user has installed ALP/GraphBLAS
in a dedicated directory via `make install`.

## 1. Running ALP/GraphBLAS as a standalone executable

### Implementation

The `grb::Launcher< AUTOMATIC >` class abstracts a group of user processes that
should collaboratively execute any single ALP/GraphBLAS program. The
ALP/GraphBLAS program of interest must have the following signature:
`void grb_program( const T& input_data, U& output_data )`.
The types `T` and `U` can be any plain-old-data (POD) type, including structs --
these can be used to broadcast input data from the master process to all user
processes (`input_data`) -- and for data to be sent back on exit of the parallel
ALP/GraphBLAS program.

The above sending-and-receiving across processes applies only to ALP/GraphBLAS
implementations and backends that support or require multiple user processes;
the sequential reference and shared-memory parallel reference_omp backends, for
example, support only one user process.

In case of multiple user processes, the overhead of the broadcasting of input
data is linear in P as well as linear in the byte-size of `T`, and hence should
be kept to a minimum. A recommended use of this mechanism is, e.g., to broadcast
input data location; any additional I/O should use the parallel I/O mechanisms
that ALP/GraphBLAS defines within the program itself.

Implementations may require sending back the output data to the calling
process, even if there is only one ALP/GraphBLAS user process. The data
movement cost this incurs shall be linear to the byte size of `U`.

### Compilation

Our backends auto-vectorise.
For best results, please edit the `include/graphblas/base/config.hpp` file prior
to compilation and installation. Most critically, ensure that
`config::SIMD_SIZE::bytes` defined in that file is set correctly with respect to
the target architecture.

The program may be compiled using the compiler wrapper `grbcxx` generated during
installation; for more options on using ALP/GraphBLAS in external projects, you
may read
[How-To use ALP/GraphBLAS in your own project](docs/Use_ALPGraphBLAS_in_your_own_project.md).

When using the LPF-enabled distributed-memory backend to ALP/GraphBLAS, for
example, simply use

```bash
grbcxx -b hybrid
```
as the compiler command.
Use

```bash
grbcxx -b hybrid --show <your regular compilation command>
```
to show all flags that the wrapper passes on.

This backend is one example that is capable of spawning multiple ALP/GraphBLAS
user processes. In contrast, compilation using

```bash
grbcxx -b reference
```
produces a sequential binary, while

```bash
grbcxx -b reference_omp
```
produces a shared-memory parallel binary instead.

The same ALP/GraphBLAS source code needs never change.

### Linking

The executable must be statically linked against an ALP/GraphBLAS library that
is different depending on the selected backend.
The compiler wrapper `grbcxx` takes care of all link-time dependencies
automatically.
When using the LPF-enabled BSP1D backend to ALP/GraphBLAS, for example, simply
use `grbcxx -b bsp1d` as the compiler/linker command.
Use

```bash
grbcxx -b bsp1d --show <your regular compilation command>
```
to show all flags that the wrapper passes on.

### Running

The resulting program has run-time dependencies that are taken care of by the
LPF runner `lpfrun` or by the ALP/GraphBLAS runner `grbrun`.
We recommend using the latter:

```bash
grbrun -b hybrid -np <#processes> </path/to/my/program>
```
Here, `<#processes>` is the number of requested ALP/GraphBLAS user processes.

### Threading

To employ threading in addition to distributed-memory parallelism, use the
hybrid backend instead of the bsp1d backend.

To employ threading to use all available hyper-threads or cores on a single
node, use the reference_omp backend.

In both cases, make sure that during execution the `OMP_NUM_THREADS` and
`OMP_PROC_BIND` environment variables are set appropriately on each node that
executes an ALP/GraphBLAS user process.

## 2. Running parallel ALP/GraphBLAS programs from existing parallel contexts

This, instead of automatically spawning a requested number of user processes,
assumes a number of processes already exist and that we wish those processes to
jointly execute a parallel ALP/GraphBLAS program.

### Implementation

The binary that contains the ALP/GraphBLAS program to be executed must define
the following global symbol with the given value:

```c++
const int LPF_MPI_AUTO_INITIALIZE = 0
```

A program may then again be launched via the Launcher, but in this case the
`MANUAL` template argument should be used instead.
This specialisation disallows the use of a default constructor.
Instead, construction requires four arguments as follows:

```c++
grb::Launcher< MANUAL > launcher( s, P, hostname, portname )
```

Here, `P` is the total number of processes that should jointly execute a
parallel ALP/GraphBLAS program, while `0 <= s < P` is a unique ID of this
process amongst its `P`-1 siblings.
The types of `s` and `P` are `size_t`, i.e., unsigned integers.
One of these processes must be selected as a connection broker prior to forming
a group of ALP/GraphBLAS user processes.
The remainder `P`-1 processes must first connect to the chosen broker using
TCP/IP connections.
This choice must be made outside of ALP/GraphBLAS, prior to setting up the
launcher and materialises as the hostname and portname constructor arguments.
These are strings, and must be equal across all processes.

As before, and after the successful construction of a manual launcher instance,
a parallel ALP/GraphBLAS program is launched via

```c++
grb::Launcher< MANUAL >::exec( &grb_program, input, output )
```

in exactly the same way as described earlier, though with two useful
differences:
  1. the input data struct is passed on from the original process to exactly one
corresponding ALP/GraphBLAS user process; i.e., no broadcast occurs. Since the
original process and the ALP/GraphBLAS user process are, from an operating
system point of view, the same process, input no longer needs to be a
plain-old-data type. Pointers, for example, are now perfectly valid to pass
along.
  2. the same applies on output data; these are passed from the ALP/GraphBLAS
user process to a corresponding originating process in a one-to-one fashion as
well.

### Running

The pre-existing process must have been started using an external mechanism.
This mechanism must include run-time dependence information that is normally
passed by the ALP/GraphBLAS runner whenever a distributed-memory parallel
backend is selected.

If the external mechanism by which the original processes are started allows it,
this is most easily effected by using the standard `grbcxx` launcher while
requesting only *one* process only, e.g.,

```bash
grbrun -b hybrid -n 1 </your/executable>
```

If the external mechanism does not allow this, then please execute e.g.

```bash
grbrun -b hybrid -n 1 --show </any/executable>
```

to inspect the run-time dependences and environment variables that must be made
available, resp., set, as part of the external mechanism that spawns the
original processes.


# Debugging

To debug an ALP/GraphBLAS program, please compile it using the sequential
reference backend and use standard debugging tools such as `valgrind` and `gdb`.

If bugs appear in one backend but not another, it is likely you have found a bug
in the former backend implementation. Please send a minimum working example
(MWE) that demonstrates the bug to the maintainers, in one of the following
ways:
  1. raise it as an issue at (https://github.com/Algebraic-Programming/ALP/issues);
  2. raise it as an issue at (https://gitee.com/CSL-ALP/graphblas/);
  3. send the MWE to (mailto:albertjan.yzelman@huawei.com).


# Development in ALP

Your contributions to ALP/GraphBLAS would be most welcome. Merge or Pull Requests
(MRs/PRs) can be contributed via Gitee and GitHub. See above for the links.

For the complete development documentation, you should start from the
[docs/README file](docs/README.md) and the related
[Development guide](docs/Development.md).


# Acknowledgements

The LPF communications layer was primarily authored by Wijnand Suijlen, without
whom the current ALP/GraphBLAS would not be what it is now.

The collectives library and its interface to the ALP/GraphBLAS was primarily
authored by Jonathan M. Nash.

The testing infrastructure that performs smoke, unit, and performance testing of
sequential, shared-memory parallel, and distributed-memory parallel backends was
primarily developed by Daniel Di Nardo.

ALP and ALP/GraphBLAS have since developed significantly, primarily through
efforts by researchers at the Huawei Paris and Zürich Research Centres, and the
Computing Systems Laboratory in Zürich specifically. See the NOTICE file for
individual contributors.

# Citing ALP/GraphBLAS

If you use ALP/GraphBLAS in your work, please consider citing one or more of the
following papers, as appropriate:

  - A C++ GraphBLAS: specification, implementation, parallelisation, and
evaluation by A. N. Yzelman, D. Di Nardo, J. M. Nash, and W. J. Suijlen (2020).
Pre-print.
[PDF](http://albert-jan.yzelman.net/PDFs/yzelman20.pdf),
[Bibtex](http://albert-jan.yzelman.net/BIBs/yzelman20.bib).
 - Nonblocking execution in GraphBLAS by Aristeidis Mastoras, Sotiris
Anagnostidis, and A. N. Yzelman (2022). Pre-print.
[PDF](http://albert-jan.yzelman.net/PDFs/mastoras22-pp.pdf),
[Bibtex](http://albert-jan.yzelman.net/BIBs/mastoras22.bib).

