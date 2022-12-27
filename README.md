<pre>
   _____  .____   __________      /\   ________                    .__   __________.____       _____    _________
  /  _  \ |    |  \______   \    / /  /  _____/___________  ______ |  |__\______   \    |     /  _  \  /   _____/
 /  /_\  \|    |   |     ___/   / /  /   \  __\_  __ \__  \ \____ \|  |  \|    |  _/    |    /  /_\  \ \_____  \
/    |    \    |___|    |      / /   \    \_\  \  | \// __ \|  |_> >   Y  \    |   \    |___/    |    \/        \
\____|__  /_______ \____|     / /     \______  /__|  (____  /   __/|___|  /______  /_______ \____|__  /_______  /
        \/        \/          \/             \/           \/|__|        \/       \/        \/       \/        \/

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



# Minimal requirements

We first summarise the compile-time, link-time, and run-time dependences of ALP.
The following are required for producing both sequential and shared-memory ALP
libraries and programs, using its `reference` and `reference_omp` backends.

## Compilation

To compile ALP/GraphBLAS, you need the following tools:

1. A C++11-capable compiler such as GCC 4.8.2 or higher, with OpenMP support
2. LibNUMA development headers
3. POSIX threads development headers
4. [CMake](https://cmake.org/download/) version 3.13 or higher, with GNU Make
(CMake's default build tool on UNIX systems) or any other supported build tool.

## Linking and run-time
The ALP/GraphBLAS libraries link against the following libraries:

1. LibNUMA: `-lnuma`
2. Standard math library: `-lm`
3. POSIX threads: `-lpthread`
4. OpenMP: `-fopenmp` in the case of GCC


# Optional dependences

The below summarises the dependences for optional features.

## Distributed-memory auto-parallelisation

For distributed-memory parallelisation, the Lightweight Parallel Foundations
(LPF) communication layer, version 1.0 or higher, is required. ALP makes use
of the LPF core library and its collectives library. The LPF library has its
further dependences, which are all summarised on the LPF project page:

* [Gitee](https://gitee.com/CSL-ALP/lpf);
* [Github](https://github.com/Algebraic-Programming/LPF).

The dependence on LPF applies to compilation, linking, and run-time. Fulfulling
the dependence enables the `bsp1d` and `hybrid` ALP/GraphBLAS backends.

## Code documentation

For generating the code documentations:
* `doyxgen` reads code comments and generates the documentation;
* `graphviz` generates various diagrams for inheritance, call paths, etc.;
* `pdflatex` is required to build the PDF file out of the Latex generated
  documentation.


# Very quick start

Here are example steps to compile and install ALP/GraphBLAS for shared-memory
machines, without distributed-memory support. The last three commands show-case
the compilation and execution of the `sp.cpp` example program.

```bash
cd <ALP/GraphBLAS root>
mkdir build
cd build
../bootstrap.sh --prefix=../install
make -j
make -j install
source ../install/bin/setenv
grbcxx ../examples/sp.cpp
grbrun ./a.out
```


# Quick start

In more detail, the steps to follow are:

1. Edit the `include/graphblas/base/config.hpp`. In particular, please ensure
   that `config::SIMD_SIZE::bytes` defined in that file is set correctly with
   respect to the target architecture.

2. Create an empty directory for building ALP/GraphBLAS and move into it:
   `mkdir build && cd build`.

3. Invoke the `bootstrap.sh` script located inside the ALP/GraphBLAS root directory
   `<ALP/GraphBLAS root>` to generate the build infrastructure via CMake inside the
   current directory:

   `<ALP/GraphBLAS root>/bootstrap.sh --prefix=</path/to/install/dir>`

    - note: add `--with-lpf=/path/to/lpf/install/dir` if you have LPF installed
            and would like to use it.

4. Issue `make -j` to compile the C++11 ALP/GraphBLAS library for the configured
   backends.

5. (*Optional*) To later run all unit tests, several datasets must be made
   available. Please run the `<ALP/GraphBLAS root>/tools/downloadDatasets.sh`
   script for

    a. an overview of datasets required for the basic tests, as well as

    b. the option to automatically download them.

6. (*Optional*) To make the ALP/GraphBLAS documentation, issue `make docs`. This
   generates both

    a. a PDF in `<ALP/GraphBLAS build dir>/docs/code/latex/refman.pdf`, and

    b. HTML in `<ALP/GraphBLAS build dir>/docs/code/html/index.html`.

7. (*Optional*) Issue `make -j smoketests` to run a quick set of functional
   tests. Please scan the output for any failed tests.
   If you do this with LPF enabled, and LPF was configured to use an MPI engine
   (which is the default), and the MPI implementation used is _not_ MPICH, then
   the default command lines the tests script uses are likely wrong. In this
   case, please edit `tests/parse_env.sh` by searching for the MPI
   implementation you used, and uncomment the lines directly below each
   occurance.

8. (*Optional*) Issue `make -j unittests` to run an exhaustive set of unit
   tests. Please scan the output for any failed tests.
   If you do this with LPF enabled, please edit `tests/parse_env.sh` if required
   as described in step 5.

9. Issue `make -j install` to install ALP/GraphBLAS into your
install directory configured during step 1.

10. (*Optional*) Issue `source </path/to/install/dir>/bin/setenv` to make available the
`grbcxx` and `grbrun` compiler wrapper and runner.

Congratulations, you are now ready for developing and integrating ALP/GraphBLAS
algorithms! Any feedback, question, problem reports are most welcome at

<div align="center">
<a href="mailto:albertjan.yzelman@huawei.com">albertjan.yzelman@huawei.com</a>
</div>


# Additional Contents

The remainder of this file summarises other build system targets, how to
integrate ALP algorithms into applications, debugging, development, and,
finally, acknowledges contributors and lists technical papers.

- [Overview of the main Makefile targets](#overview-of-the-main-makefile-targets)
- [Automated performance testing](#automated-performance-testing)
- [Integrating ALP with applications](#integrating-alp-with-applications)
	- [Running ALP programs as standalone executables](#running-alp-programs-as-standalone-executables)
		- [Implementation](#implementation)
		- [Compilation](#compilation-1)
		- [Linking](#linking)
		- [Running](#running)
		- [Threading](#threading)
	- [Running parallel ALP programs from existing parallel contexts](#running-parallel-alp-programs-from-existing-parallel-contexts)
		- [Implementation](#implementation-1)
		- [Running](#running-1)
	- [Integrating ALP within your coding project](#integrating-alp-within-your-coding-project)
- [Configuration](#configuration)
- [Debugging](#debugging)
- [Development in ALP](#development-in-alp)
- [Acknowledgements](#acknowledgements)
- [Citing ALP and ALP/GraphBLAS](#citing-alp-and-alpgraphblas)


# Overview of the main Makefile targets

The following table lists the main build targets of interest:

| Target                | Explanation |
|----------------------:|---------------------------------------------------|
| \[*default*\]         | builds the ALP/GraphBLAS libraries and examples   |
| `install`             | install libraries, headers and some convenience   |
|                       | scripts into the path set via `--prefix=<path>`   |
| `unittests`           | builds and runs all available unit tests          |
| `smoketests`          | builds and runs all available smoke tests         |
| `perftests`           | builds and runs all available performance tests   |
| `tests`               | builds and runs all available unit, smoke, and    |
|                       | performance tests                                 |
| `docs`                | builds HTML and LaTeX code and API documentation  |

For more information about the testing harness, please refer to the
[related documentation](tests/Tests.md).

For more information on how the build and test infrastructure operate, please
refer to the [the related documentation](docs/Build_and_test_infra.md).


# Automated performance testing

To check in-depth performance of this ALP/GraphBLAS implementation, issue
`make -j perftests`. This will run several algorithms in several ALP/GraphBLAS
configurations. This generates three main output files:

1. `<ALP/GraphBLAS build dir>/tests/performance/output`, which summarises the
   whole run;

2. `<ALP/GraphBLAS build dir>/tests/performance/output/benchmarks`, which
   summarises the performance of individual algorithms; and

3. `<ALP/GraphBLAS build dir>/tests/performance/output/scaling`, which
   summarises operator scaling results.

To ensure that all tests run, please ensure all related datasets are available
as also described at step 5 of the quick start.

With LPF enabled, please note the remark described at steps 3 and 7 of the quick
start guide. If LPF was not configured using MPICH, please review and apply any
necessary changes to `tests/performance/performancetests.sh`.


# Integrating ALP with applications

There are several use cases in which ALP can be deployed and utilized, listed
in the following. These assume that the user has installed ALP/GraphBLAS in a
dedicated directory via `make install`.

## Running ALP programs as standalone executables

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
both the sequential `reference` and the shared-memory parallel `reference_omp`
backends, for example, support only one user process.

In case of multiple user processes, the overhead of the broadcasting of input
data is linear in the number of user processes, as well as linear in the byte-
size of `T` which hence should be kept to a minimum. A recommended use of this
mechanism is, e.g., to broadcast input data locations; any additional I/O
should use the parallel I/O mechanisms that ALP/GraphBLAS exposes to the ALP
program itself.

Output data is retrieved only from the user process with ID `0`, even if
multiple user processes exist. Some implemenations or systems may require
sending back the output data to a calling process, even if there is only
one user process. The data movement cost incurred should hence be considered
linear in the byte size of `U`, and, similar to the input data broadcasting,
the use of parallel I/O facilities from the ALP program itself for storing
large outputs is strongly advisable.

### Compilation

Our backends auto-vectorise, hence please recall step 1 from the quick start
guide, and make sure the `include/graphblas/base/config.hpp` file reflects the
correct value for `config::SIMD_SIZE::bytes`. This value must be updated prior
to the compilation and installation of ALP.

When targeting different architectures with differing SIMD widths, different
ALP installations for different architectures could be maintained.

ALP programs may be compiled using the compiler wrapper `grbcxx` that is
generated during installation. To compile high-performance code when compiling
your programs using the ALP installation, the following flags are recommended:

 - `-DNDEBUG -O3 -mtune=native -march=native -funroll-loops`

Omitting these flags for brevity, some compilation examples follow.

When using the LPF-enabled hybrid shared- and distributed-memory backend of
ALP/GraphBLAS, simply use

```bash
grbcxx -b hybrid
```
as the compiler command. To show all flags that the wrapper passes on, please use

```bash
grbcxx -b hybrid --show
```
and append your regular compilation arguments.

The `hybrid` backend is capable of spawning multiple ALP/GraphBLAS user
processes. In contrast, compilation using

```bash
grbcxx -b reference
```
produces a sequential binary, while

```bash
grbcxx -b reference_omp
```
produces a shared-memory parallel binary.

Note that the ALP/GraphBLAS source code never requires change while switching
backends.

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
grbrun -b hybrid -np <P> </path/to/my/program>
```
Here, `P` is the number of requested ALP/GraphBLAS user processes.

### Threading

The `hybrid` backend employs threading in addition to distributed-memory
parallelism. To employ threading to use all available hyper-threads or cores
on a single node, the `reference_omp` backend may be selected instead.

In both cases, make sure that during execution the `OMP_NUM_THREADS` and
`OMP_PROC_BIND` environment variables are set appropriately on each node that
executes ALP/GraphBLAS user process(es).

## Running parallel ALP programs from existing parallel contexts

This, instead of automatically spawning a requested number of user processes,
assumes a number of processes already exist and that we wish those processes to
jointly execute a single parallel ALP/GraphBLAS program.

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
a group of ALP/GraphBLAS user processes. The remainder `P-1` processes must
first connect to the chosen broker using TCP/IP connections. This choice must
be made outside of ALP/GraphBLAS, prior to setting up the launcher, and
materialises as the `hostname` and `portname` Launcher constructor arguments.
The host and port name are strings, and must be equal across all processes.

As before, and after the successful construction of a manual launcher instance,
a parallel ALP/GraphBLAS program is launched via

```c++
grb::Launcher< MANUAL >::exec( &grb_program, input, output )
```

in exactly the same way as described earlier, though with the input and output
arguments now being passed in a one-to-one fashion:
  1. The input data is passed on from the original process to exactly one
     corresponding ALP/GraphBLAS user process; i.e., no broadcast occurs. The
     original process and the ALP/GraphBLAS user process are, from an operating
     system point of view, the same process. Therefore, and additionally, input
     no longer needs to be a plain-old-data (POD) type. Pointers, for example,
     are now perfectly valid to pass along, and enable sharing data between the
     original process and the ALP/GraphBLAS algorithm.
  2. The output data is passed from each ALP/GraphBLAS user process to the
     original process that called `Launcher< MANUAL >::exec`. To share
     ALP/GraphBLAS vector data, it is, for example, legal to return a
     `grb::PinnedVector< T >` as the `exec` output argument type. Doing so is
     akin to returning a pointer to output data, and does not explicitly pack
     nor transmit vector data.

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

## Integrating ALP within your coding project

Please see [this article](docs/Use_ALPGraphBLAS_in_your_own_project.md) on how
to add ALP and ALP/GraphBLAS as a dependence to your project.


# Configuration

ALP employs configuration headers that contain `constexpr` settings that take
effect every time ALP programs are compiled. Multiple object files that were
compiled using ALP must all been compiled using the same configuration
settings-- linking objects that have been compiled with a mixture of
configurations are likely to incur undefined behaviour. The recommendation is
to set a configuration before building and installing ALP, and to keep the
installation directories read-only so that configurations remain static.

There exists one main configuration file that affects all ALP backends, while
some configurations only affect a specfic backend. The main configuration file
is found in `<root>/include/graphblas/base/config.hpp`, which allows one to set
the

1. cache line size, in bytes, within the `CACHE_LINE_SIZE` class;
2. SIMD width, in bytes, within the `SIMD_SIZE` class;
3. default number of experiment repetitions during benchmarking, within the
   `BENCHMARKING` class;
4. L1 data cache size, in bytes, within `MEMORY::big_memory` class;
5. from which size onwards memory allocations will be reported, in log-2
   bytes, within `MEMORY::big_memory`;
6. index type used for row coordinates, as the `RowIndexType` typedef;
7. index type used for column coordinates, as the `ColIndexType` typedef;
8. type used for indexing nonzeroes, as the `NonzeroIndexType` typedef;
9. index type used for vector coordinates, as the `VectorIndexType` typedef.

Other main configuration values are automatically inferred, fixed
non-configurable settings, or are presently not used by any ALP backend.

## Reference and reference_omp backends

The file `include/graphblas/reference/config.hpp` contain defaults that pertain
to the auto-vectorising and sequential `reference` backend, but also to the
shared-memory auto-parallelising `reference_omp` backend. It allows one to set

1. whether prefetching is enabled in `PREFETCHING::enabled`;
2. the prefetch distance in `PREFETCHING::distance`;
3. the default memory allocation strategy for thread-local data in
   `IMPLEMENTATION::defaultAllocMode()`;
4. same, but for shared data amongst threads in
   `IMPLEMENTATION::sharedAllocMode()`;

Configuration elements not mentioned here are fixed non-user-configurable
settings. Modifying any of the above should be done with utmost care as it
typically affects the defaults across an ALP installation, and *all* programs
compiled using it.

## OpenMP backends

The file `include/graphblas/omp/config.hpp` contains some basic configuration
parameters that affect any OpenMP-based backend. However, the configuration
file does not contain any other user-modifiable setings, but rather contains
a) some utilities that OpenMP-based backends may rely on, and b) default
that are derived from other settings described in the above. These settings
should only be overridden with compelling and expert knowledge.

## LPF backends

The file `include/graphblas/bsp/config.hpp` contains some basic configuration
parameters that affect any LPF-based backend. It includes:

1. an initial maximum of LPF memory slot registrations in `LPF::regs()`;
2. an initial maximum of LPF messages in `LPF::maxh()`.

These defaults, if insufficient, will be automatically resized during execution.
Setting these large enough will therefore chiefly prevent buffer resizes at run-
time. Modifying these should normally not lead to significant performance
differences.

## Utilities

The file `include/graphblas/utils/config.hpp` details configurations of various
utility functions, including:

1. a buffer size used during reading input files, in `PARSER::bsize()`;
2. the block size of individual reads in `PARSER::read_bsize()`.

These defaults are usually fine except when reading from SSDs, which would
benefit of a larger `read_bsize`.

## Others

While there are various other configuration files (find `config.hpp`), the above
should list all user-modifiable configuration settings of interest. The
remainder pertain to configurations that are automatically deduced from the
aforementioned settings, or pertain to settings that describe how to safely
compose backends and only of interest to developers.


# Debugging

To debug an ALP/GraphBLAS program, please compile it using the sequential
reference backend and use standard debugging tools such as `valgrind` and `gdb`.
Additionally, please ensure to *not* pass the `-DNDEBUG` flag during
compilation.

If bugs appear in one backend but not another, it is likely you have found a bug
in the former backend implementation. Please send a minimum working example that
demonstrates the bug to the maintainers, either as an issue on or an email to:
  1. [GitHub](https://github.com/Algebraic-Programming/ALP/issues);
  2. [Gitee](https://gitee.com/CSL-ALP/graphblas/issues);
  3. [Albert-Jan](mailto:albertjan.yzelman@huawei.com).


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
Computing Systems Laboratory in Zürich in particular. See the [NOTICE](NOTICE)
file for individual contributors.


# Citing ALP and ALP/GraphBLAS

If you use ALP/GraphBLAS in your work, please consider citing one or more of the
following papers, as appropriate:

 - [A C++ GraphBLAS: specification, implementation, parallelisation, and evaluation](http://albert-jan.yzelman.net/PDFs/yzelman20.pdf)
   by A. N. Yzelman, D. Di Nardo, J. M. Nash, and W. J. Suijlen (2020).
   Pre-print.
   [Bibtex](http://albert-jan.yzelman.net/BIBs/yzelman20.bib).
 - [Nonblocking execution in GraphBLAS](http://albert-jan.yzelman.net/PDFs/mastoras22-pp.pdf)
   by Aristeidis Mastoras, Sotiris Anagnostidis, and A. N. Yzelman (2022).
   Pre-print.
   [Bibtex](http://albert-jan.yzelman.net/BIBs/mastoras22.bib).

