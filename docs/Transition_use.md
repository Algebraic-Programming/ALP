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
transition interfaces, e.g., changing functions prefix.

The exposed interfaces are:

* the **sparse CG solver** interface, offering an implementation of the
  Conjugate Gradient (CG) algorithm with the related facilities to populate the
  necessary data structures; you may check
  [the related header file](../include/transition/solver.h)
* the **KML solver** interface, exposing similar functionalities to the sparse
  CG solver and matching the interface of the
  [Kunpeng Math Library Solver](https://www.hikunpeng.com/document/detail/en/kunpengaccel/math-lib/devg-kml/kunpengaccel_kml_16_0287.html);
  you may go through the available functions
  [in the related header](../include/transition/kml_iss.h)
* the **Sparse BLAS** interface, exposing some calls of the de-facto standard
  [Sparse BLAS API](https://www.intel.com/content/www/us/en/docs/onemkl/developer-reference-dpcpp/2024-0/sparse-blas-functionality.html);
  these calls are visible
  [in the implementation header](../include/transition/spblas_impl.h),
  which exposes them via the `spblas.h` header (created during installation)
  with a configurable prefix; the prefix can be configured during bootstrap via
  the `--spblas-prefix` option (as from
  [the configuration and compilation instructions](Build_and_test_infra.md#generation-via-the-bootstrapsh-script));
  the default prefix is `kml_sparse_`, following the prefix conventions of
  [Kunpeng Math Library SPBLAS](https://www.hikunpeng.com/document/detail/en/kunpengaccel/math-lib/devg-kml/kunpengaccel_kml_16_0186.html)

These interfaces are simply built with `make` (or with the dedicated targets)
and, upon installation, the headers are available in
`<installation directory>/include/transition/`, while the (static) binaries in
`<installation directory>/lib/sequential/`.

