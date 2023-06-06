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

The current testing infrastructure is composed of several scripts invoking the
test binaries.
Test binaries can be generated via the testing infrastructure,
as from the [related guide](Build_and_test_infra.md#adding-a-new-test).
Tests should be added to the scripts manually, invoking the appropriate launcher
and passing the dedicated options.

# Run ALP/GraphBLAS Tests

Tests are run via dedicated scripts in the project root, which invoke the
specific test binaries.
This solution deals with the complexity of testing ALP/GraphBLAS, whose
different backends require different execution targets (shared-memory and a
distributed system with an MPI or LPF launcher).

These scripts should be invoked via the corresponding `make` targets inside the
build directory (e.g., `make unittests`): this invocation takes care of passing
the scripts the relevant parameters (location of binaries, available backends,
datasets location, output paths, ...) and, as usual, shows their output in the
`stdout`.

As from the
[Reproducible Builds section](Build_and_test_infra.md#reproducible-builds),
Docker images can be built to have a reproducible environment for building and
testing.
These images store all tools, dependencies and input datasets to build and run
all backends and tests; you may refer to the
[section](Build_and_test_infra.md#reproducible-builds) for more details.
