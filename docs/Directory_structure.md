
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

# Directory structure of the project

The directories of this project are organized as follows:

* the `<project root>` is supposed to contain only a minimal amount of files,
like the license, the root of the building and testing infrastructure and the
startup documentation
* `cmake` contains CMake files for the build and test infrastructure, in
particular the definitions of compile-time definitions and options for all
targets
* `docs` contains the documentation of the project:
  * the Markdown files at the top are the static documentation about the project
itself and the infrastructure
  * the `code` subdirectory contains the documentation generated from the code
via `doxygen`
* `examples` contains some examples to showcase the usage of ALP/GraphBLAS
* `include` contains the headers ALP/GraphBLAS is composed of; these amount to
most of the code, since ALP/GraphBLAS is template-based; in particular
  * `graphblas/algorithms` contains algorithms that are useful for end-users
  * `graphblas/utils` contains utilities that are useful for end-users,
regardless whether or not they actually use ALP/GraphBLAS
* `src` contains the implementation of the runtime functionalities of backends,
usually amounting to few initialization routines
* `test` contains the testing infrastructure, in particular
  * `performance`, `smoke` and `unit` for the three categories of tests; in
particular, `performance` contains drivers for all algorithms in
`include/graphblas/algorithms` -- no drivers should appear elsewhere at present
  * `banshee` for the test specific to the Banshee backend (experimental)
  * `utils` for utilities that are reasonably only expected to be useful for the
code in the ALP/GraphBLAS test suite
* `tools` contains some basic tools to ease development, like for linting or
downloading the common datasets required for testing