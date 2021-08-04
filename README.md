<pre>
  ________                    .__     
 /  _____/___________  ______ |  |__  
/   \  __\_  __ \__  \ \____ \|  |  \ 
\    \_\  \  | \// __ \|  |_> >   Y  \
 \______  /__|  (____  /   __/|___|  /
        \/           \/|__|        \/ 
   __________.____       _____    _________
   \______   \    |     /  _  \  /   _____/
    |    |  _/    |    /  /_\  \ \_____  \ 
    |    |   \    |___/    |    \/        \
    |______  /_______ \____|__  /_______  /
           \/        \/       \/        \/ 
</pre>


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


# Requirements

## Compilation

To compile GraphBLAS, you need the following tools:

1. A C++11-capable compiler such as GCC 4.8.2 or higher, with OpenMP support
2. LibNUMA development headers
3. POSIX threads development headers

## Linking and run-time
The GraphBLAS libraries link aginst the following libraries:

1. LibNUMA: `-lnuma`
2. Standard math library: `-lm`
3. POSIX threads: `-lpthread`
4. OpenMP: `-fopenmp` in the case of GCC

## Optionals
Required for distributed-memory parallelism:

* The Lightweight Parallel Foundations (LPF) communication layer, version 1 or higher, and its collectives library, and all LPF dependences.

This dependendy applies to compilation, linking, and run-time dependences.


# Quick start
Here are the basic steps to quickly compile and install GraphBLAS for shared memory machines (i.e. without distributed-memory support):

1. Issue `./configure --prefix=</path/to/install/dir> --no-lpf --no-banshee`
     - note: use `--with-lpf=/path/to/lpf/installation/` instead of `--no-lpf` if you have LPF installed (and would like to use it).
2. Issue `make -j` to compile the C++11 GraphBLAS library for shared memory.
3. (*Optional*) To run all unit tests, several datasets must be made available. Please run the tools/downloadDatasets.sh script for
     a) an overview of datasets required for the basic tests, as well as
     b) the option to automatically download them.
4. (*Optional*) Issue `make -j tests` to run functional and performance tests automatically. Please scan the output for any failed tests.
5. (*Optional*) To make the GraphBLAS documentation, issue `make docs`. This generates both
     a) PDF documentations in `docs/latex/refman.pdf`, and
     b) HTML documentations in `docs/html/index.html`.
6. Issue `make -j install` to install the GraphBLAS into your install directory configured during step 1.
7. Issue `source </my/install/path>/bin/setenv` to make available the grbcxx and grbrun compiler wrapper and runner.

Congratulations, you are now ready for developing and integrating GraphBLAS algorithms! Any feedback, question,
problem reports are most welcome at

                        albertjan.yzelman@huawei.com


 In-depth performance measurements may be obtained via the following additional and optional steps:

1. (*Optional*) To check in-depth performance of this GraphBLAS implementation, issue `make perftests`. This will run several algorithms in several GraphBLAS configurations. All output is captured in `bin/tests/output`, just as with `make tests`. A summary of the benchmark results is output to `bin/tests/output/benchmarks`.

2. (*Optional*) Download the com-orkut dataset from the SNAP repository and copy it into the `datasets/` directory. Benchmarks on this dataset can be started via `make perftests` The output will be caught in `bin/tests/output/` and a summary of the results will be appended to `bin/tests/output/benchmarks`.


# Overview of all Makefile targets

The following table lists the main Makefile targets of interest for users: they allow to build, test and generate the documentation of the entire GraphBLAS code-base.

| Target | Explanation |
|----------------------:|---------------------------------------------------|
| `libs` \[*default*\] | builds all required libraries in `libs/` |
| `install` | this will copy `include/` and `lib/` to the install directory set by `./configure --prefix=<path>`. If configure was not called before, make install will not copy anyting and instead prints a warning |
| `tests` | builds all test executables in `bin/tests/` and runs all tests. A summary is printed to `stdout`, full test output is retained in `bin/tests/output/` |
| `unittests` | builds and runs only the unit tests from `make tests` |
| `smoketests` | builds and runs only the smoke tests from `make tests` |
| `perftests` |this will build several benchmark apps and run these on the various datasets found in the datasets/ directory. Full benchmark output is retained in `bin/tests/output`. A summary will be appended to `bin/tests/output/benchmarks` |
| `docs` | builds all HTML GraphBLAS documentation in `docs/html/index.html`. It also generates LaTeX source files in `docs/latex`, which, if `pdflatex`, `graphviz`, and other standard tools are available, are compiled into a PDF found at `docs/latex/refman.pdf` |
| `examples` | builds a couple of example GraphBLAS codes in `bin/examples/` |
| `clean` | deletes all files the preceding make targets could have generated but retains all executables in `bin/`, as well as any compilation configurations found in `gcc-active.mk`. It also retains the compiled libraries in `lib/`. Note that this behaviour *includes* the deletion of test and benchmark output |
| `veryclean` | as `clean`, but for *all* possible generated files, thus including `deps/`, executables, libraries, compilation configurations, etc. Any user modifications outside of `bin/` and deps/ are retained, however the resulting state may thus possibly *not* correspond to the original vanilla state of this distribution |



# Deploying GraphBLAS codes

There are several use cases in which GraphBLAS can be deployed and utilized, listed in the following.

## 1. Running GraphBLAS in parallel as a standalone executable

### Implementation

We recommend the use of the `grb::Launcher< AUTOMATIC >` class for this use case since it abstracts away all LPF calls that would normally be required to start multiple processes via its exec member function. The  GraphBLAS program of interest must have the following signature: `void grb_program( const T& input_data, U& output_data )`. The types `T` and `U` can be any plain-old-data type, including structs -- these can be used to broadcast input data from the master process to all user processes (input_data) -- and for data to be sent back on exit of the parallel GraphBLAS program.

The overhead of these two communication steps are linear in P as well as linear in the byte-size of `T` and `U`, and hence should be kept to a minimum. A recommended use of this mechanism is, e.g., to broadcast input data location (one or several `std::string`s containing paths) and to receive back an error code to check the status of the computation; any additional I/O should use the parallel I/O mechanisms GraphBLAS defines within the GraphBLAS program itself.

### Compilation

The program must be compiled using the following flags:

```bash
    -D_GRB_WITH_LPF
    -D_GRB_WITH_OMP
    -D_GRB_WITH_REFERENCE
    -D_GRB_BACKEND=BSP1D
    -D_GRB_COORDINATES_BACKEND=reference
```

To ease the compilation process, the compiler wrapper `grbcxx` takes care of all compile-time dependences and macro definitions automatically. When using the LPF-enabled BSP1D backend to GraphBLAS, for example, simply use `grbcxx -b bsp1d` as the compiler command. Use `grbcxx -b bsp1d --show <your regular compilation command>` to show all flags that the wrapper passes on.

### Linking

The executable must be statically linked against `lib/spmd/libgraphblas.a` and dynamically against
- `-lpthread`
- `-lnuma`
- `-lm`

Any further dependences are determined by LPF; please see its documentation.

To ease the linking process, the compiler wrapper `grbcxx` takes care of all link-time dependencies automatically. When using the LPF-enabled BSP1D backend to GraphBLAS, for example, simply use `grbcxx -b bsp1d` as the compiler/linker command. Use `grbcxx -b bsp1d --show <your regular compilation command>` to show all flags that the wrapper passes on.

### Running

The resulting program has run-time dependencies that are taken care of by the LPF runner lpfrun or by the GraphBLAS runner grbrun. We recommend to use the latter: `grbrun -b bsp1d -np <#processes> </my/program>`.

Here, *<#processes>* is the number of requested processes.

### Threading

To employ threading in addition to distributed-memory parallelism, add the following flags during the compilation process:

```bash
    -D_GRB_BSP1D_BACKEND=reference_omp
```

and replace `-D_GRB_COORDINATES_BACKEND=reference` by `-D_GRB_COORDINATES_BACKEND=reference_omp`.

Alternatively, use the compiler wrapper via `grbcxx -b hybrid`.

During execution, make sure the `OMP_NUM_THREADS` and `OMP_PROC_BIND` environment variables are set appropriately on each node a user process runs on.

## 2. Running parallel GraphBLAS programs from existing parallel contexts

This, instead of automatically spawning a requested number of user processes, assumes a number of processes already exist and that we wish those processes to jointly execute a parallel GraphBLAS program.

### Implementation

The binary that contains the GraphBLAS program to be executed in the above described way must define the following global symbol with the given value:

```c++
       const int LPF_MPI_AUTO_INITIALIZE = 0
```

A program may then again be launced via the grb::Launcher, but in this case the MANUAL specialisation shoud be used instead. In this case its default constructor should not be used, and four arguments should be given instead:

```c++
    grb::Launcher< MANUAL > launcher( s, P, hostname, portname );
```

Here, P is the total number of processes that should jointly execute a parallel GraphBLAS program, while 0 <= s < P is a unique ID of this process amongst its P-1 siblings. s and P are of type `size_t`, i.e., unsigned integers. One of these processes must be selected as a connection broker prior to forming a group of GraphBLAS user processes, P-1 processes must first connect to a chosen process using TCP/IP connections to initialise. This choice must be made outside of GraphBLAS, prior to setting up the launcher; the hostname and portname are strings that must be equal across all processes.

As before, and after the successful construction of a manual launcher instance, a parallel GraphBLAS program is launched via 

```c++
    grb::Launcher< MANUAL >::exec( &grb_program, input, output ),
```

in exactly the same way as described earlier with two useful differences:

1. the input data struct is passed on from the original process to the GraphBLAS user process in a one-to-one fashion; i.e., no broadcast occurs. Since the original process and the GraphBLAS user process are, from an operating system point of view, in fact the same process, input no longer needs to be a plain-old-data type. Pointers, for example, are now perfectly valid to pass along.

2. the same applies on output data; these are passed from the GraphBLAS user program to the original process in a one-to-one fashion as well.

### Running

The pre-existing process must have been started using the LPF or GraphBLAS runner (i.e., using `lpfrun` or `grbrun -b <bsp1d/hybrid/...>` requesting *one* process only; e.g.,

```bash
    grbrun -b hybrid -n 1 </your/executable>
```

This is only to ensure that the required run-time paths and libraries can be found when required, which is when the parallel GraphBLAS program is called. Setting the right library paths, opening the right libraries, or preloading them can of course also be done manually. To inspect the paths and libraries the runner makes available, simply pass `--show` to the command line.

All other concerns such as compilation flags remain unchanged as per the above use case #1.

## 3. Running GraphBLAS sequentially (single process)

This is useful for debugging, working on small computations, or use of GraphBLAS within each thread of a larger (parallel) program.

### Compiling

Note that our sequential reference backend auto-vectorises. For best results, please edit `include/graphblas/config.hpp` prior to compilation.

No special compilation flags are necessary to compile a single-process GraphBLAS program, save for those to enable the C++11 language. On most compilers this is achieved via

```bash
    -std=c++11
```

Again, the compiler wrapper `grbcxx` is recommended. Purely sequential (but vectorised) code is generated via `grbcxx -b reference`.

### Linking

The program must be statically linked against `lib/sequential/libgraphblas.a` and dynamically against `-lnuma`. Again, the use of `grbcxx` is recommended as it injects the right linker commands as required.

### Running

The generated executable can be run directly, or via `grbrun -b reference`.


## 4. Single-node shared-memory parallel GraphBLAS

### Compilation

To use a single-process but multi-threaded GraphBLAS implementation that still auto-vectorises, simply add the following two compile flags:

```bash
    -D_GRB_BACKEND=reference_omp
    -fopenmp
```
  
Again, the compiler wrapper `grbcxx` is recommended. Single-node shared-memory parallel (and vectorised) code is generated via `grbcxx -b reference_omp`.

When using the multi-threaded GraphBLAS backend, one can control the number of threads and their affinity through the standard OpenMP environment variables such as `OMP_NUM_THREADS` and `OMP_PROC_BIND`.


# Developing GraphBLAS

To develop GraphBLAS, you should follow the principles and guidelines listed in the [Development guide](DEVELOPMENT.md)



# Acknowledgements

The LPF communications layer was primarily authored by Wijnand Suijlen, without which the current GraphBLAS implementation would not look as it does now.

The collectives library and its interface to the GraphBLAS is was primarily authored by Jonathan M. Nash.

For additional acknowledgements, please see the NOTICE file.

