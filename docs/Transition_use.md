<pre>
  Copyright 2024 Huawei Technologies Co., Ltd.

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

# Introduction to the transition path

ALP/GraphBLAS exposes several of its functionalities via established C
interfaces, in order to facilitate the transition of legacy software.
Ideally, users of transition interfaces need only re-compile and link their
software; in some cases, trivial modifications might be required to migrate to
transition interfaces, e.g., changing the prefix of called functions.

The current transition path interfaces is at an *experimental prototype phase*;
in particular, not all primitives in a given standard API are currently
implemented. For SparseBLAS in particular, additional support or coverage may
freely be requested in GitHub issue #14. For other standard interfaces, feel
free to open new issues or to contact the maintainers.

The currently exposed interfaces are:

* the Sparse BLAS interface as defined by the BLAS forum and in the following
  paper: Duff, Iain S., Michael A. Heroux, and Roldan Pozo. "An overview of the
  sparse basic linear algebra subprograms: The new standard from the BLAS
  technical forum." ACM Transactions on Mathematical Software (TOMS) 28(2),
  2002, pp. 239-267. Refer to either of the following for the current
  implementation status within ALP:
    - [stable](http://albert-jan.yzelman.net/alp/user/blas__sparse_8h.html);
    - [development](../include/transition/blas_sparse.h).
* the SpBLAS interface-- while not a standard proper, it may be considered a
  de-facto one as it is implemented by various vendors as well as open source
  projects and enjoys wide-spread use. Refer to either of the following for
  the current implementation status within ALP:
   - [stable](http://albert-jan.yzelman.net/alp/user/spblas_8h.html);
   - [development](../include/transition/spblas_impl.h).
  The prefix emitted to this function, necessary since this is not a properly
  standardised interface, by default starts with `alp_`, but can be configured
  during bootstrap via the `--spblas-prefix` option; see also
  [the configuration and compilation instructions](Build_and_test_infra.md#generation-via-the-bootstrapsh-script).
* a non-standard solver interface that currently only exposes the ALP conjugate
  gradient (CG) algorithm and may be used with any CRS matrix and raw C/C++
  vector data. Refer to either of the following for the current implementation
  status within ALP:
   - [stable]((http://albert-jan.yzelman.net/alp/user/solver_8h.html);
   - [development](../include/transition/solver.h).
  Based on this solver API we also expose a CG solver that matches the Kunpeng
  Library. The ALP coverage this API is nearly complete and documented here:
   - [stable]();
   - [development](../include/transition/kml_iss.h).

All of these transition libraries show-case ALP's ability to quickly wrap around
external APIs, thus simplifying integration of ALP-backed code with existing
software. We do note, however, that the direct use of the native C++ ALP API may
lead to higher performance than the use of these transition path interfaces, and
that in some cases the legacy interface itself is what makes achieving such
higher performance impossible.

The transition interfaces are built with the default `make`. Upon installation,
the headers and (static) libraries are installed in
`<installation directory>/include/transition/` and
`<installation directory>/lib/sequential/`, respectively.

The default library names for the *sequential* transition path libraries are:
 - `libsparseblas_sequential.a` for the standard Sparse BLAS implementation,
 - `libalp_cspblas_sequential.a` for the de-facto standard SpBLAS, and
 - `libspsolver_sequential.a` for the non-standard solver library.

The default shared-memory parallel transition path libraries are:
 - `libsparseblas_shmem_parallel.a` for the standard Sparse BLAS,
 - `libalp_cspblas_shmem_parallel.a` for the de-facto standard SpBLAS,
 - `libspsolver_shmem_parallel.a` for the non-standard solver library, and
 - `libksolver.a` for the ALP-generated Kunpeng Library solver implementation.

At present, no dynamic libraries are built -- if this would be useful, please
feel free to submit a feature request or to contact the maintainers.

