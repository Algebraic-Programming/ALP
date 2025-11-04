Developer guide — pyalp / alp-graphblas
=====================================

Author:
Denis Jelovina

Support:
For support or to report issues, please open an issue on the project's GitHub issue tracker. For direct contact, email denis.jelovina@gmail.com 

This document explains how the Python packaging for the pyalp bindings works, how CI builds wheels, and what to change when you add a new compiled backend (pybind11 module) or Python dependency.

C++ binding logic and Python usage (summary)
-------------------------------------------
The pyalp package exposes native C++ backends built with pybind11. Each backend is compiled as a separate Python extension module (shared object) with a canonical name like `pyalp_ref`, `pyalp_omp`, or `pyalp_nonblocking`. The packaging layout installs those compiled modules into the `pyalp` package so they are importable as `pyalp.pyalp_ref`, `pyalp.pyalp_omp`, etc.

How Python code uses the compiled backends
- Direct import: after installation you can import a backend module directly, for example:

  import pyalp.pyalp_ref
  M = pyalp.pyalp_ref.Matrix(10, 10)

- Helper API: the package also provides helper APIs that discover and return backends at runtime, e.g. `pyalp_importname.get_backend('pyalp_ref')` which returns the compiled module object. This is useful for selecting backends dynamically.

How the Python object maps to C++
- Each compiled extension is a pybind11 module which registers C++ types (Matrix, Vector, operators) and functions. The pybind11 binding code (in the `pyalp` C++ sources) defines the Python-visible class names and methods, so `pyalp.pyalp_ref.Matrix` is a python wrapper around the C++ Matrix implementation in the native backend.
- At build time, CMake compiles the C++ sources into a platform-specific shared object; the packaging step copies that shared object into the `pyalp` package so the interpreter can import it as a normal module.

Current functional limitations and caveats
- Cross-backend imports: importing different backend modules in the same Python process can cause pybind11 type-registration collisions (duplicate registrations of the same C++ types across modules). The bindings now use `py::module_local()` for many wrapper types to reduce collisions, but issues can still occur. If you need repeatable cross-backend usage, either run backends in separate processes or design a shared-registration approach (single module that dispatches to backends or explicit shared-type registration across modules).
- Cross-backend bindings: supporting full cross-backend interoperability requires either
  - a single compiled extension exporting a stable API and selecting backends internally, or
  - explicit cross-registration code that ensures each type is only registered once (or registered with module-local variants and safe conversion functions). Both approaches require C++ changes and careful testing.
- Wheel portability and optimization trade-offs:
  - Wheels are built per-ABI and per-OS (CI uses per-ABI build dirs). The project disables aggressive target-specific flags (no `-march=native`, LTO off) to improve portability, but wheels are still platform/ABI-specific (glibc versus musl, macOS SDK versions). Expect different wheel filenames per ABI/OS and possible limitations on older OS versions.
  - CI currently skips `*-musllinux*` and does not publish Windows wheels by default (see CI matrix). If you need musl or Windows support, update the CI configuration and the before-build steps to provide appropriate toolchains and packaging options.
- Size and dependency implications: bundling multiple backends increases wheel size.

If you plan to change the bindings or support cross-backend imports, read the `pybind11` docs on module-local registrations and consider writing small integration tests that import multiple backends in isolated subprocesses.


Local builds (tested with `pyalp-ci.yml`)
-----------------------------------------
If you prefer fast iteration or want to debug native build issues locally, build and test wheels on your machine. The repository provides `pyalp-ci.yml` to exercise the build steps in CI (useful to validate local changes on pull requests), but local builds let you iterate without pushing tags or waiting for remote runners.

When to build locally
- Fast iteration when changing bindings, packaging logic, or test code.
- Debugging native-build problems where you need immediate access to compiler and linker output.
- Packaging-only checks: point `pyalp/setup.py` at an existing `.so` (via `PREBUILT_PYALP_SO`) to validate wheel contents without rebuilding native code.

How to build wheels locally (quick recipe)
- Prepare a per-ABI build directory and run CMake (example for Python 3.11):
- Build a wheel from the `pyalp` package and point it at the per-ABI build dir so the generated metadata and prebuilt `.so` get picked up:

```bash
  cmake  -DENABLE_PYALP=ON -DCMAKE_BUILD_TYPE=Release $ALP_REPO_PATH
  make pyalp_ref
  # append the new path to  PYTHONPATH, ie. export PYTHONPATH=$PYTHONPATH:$(pwd)/python
```

Advantage of local builds
- Performance, active optimisations for the build architecture
- Speed: no remote queue or tag/push cycle.
- Control: change CMake flags and environment variables and rebuild immediately.
- Debuggability: full compiler/linker logs and the ability to attach tools.


Full publish pipeline (publish-to-testpypi.yml + promote-to-pypi.yml)
-----------------------------------------------------------------
The full repository publish flow is implemented in two primary workflows:

- `publish-to-testpypi.yml` — builds wheels for multiple ABIs/OSes using `cibuildwheel`, publishes them to TestPyPI, uploads wheel artifacts to a GitHub Release, and runs verification steps that install the TestPyPI package into a clean virtualenv for smoke tests. This workflow is triggered by pushing a tag matching `pyalp.v*`.

- `promote-to-pypi.yml` — a gated workflow that downloads wheel assets from a GitHub Release and uploads them to PyPI. This job requires the `production` environment and uses the `PYPI_API_TOKEN` secret; the environment gating ensures human approval before the token is available to the workflow.

Key differences vs local builds
- Scope: the publish pipelines run multiple ABIs and platforms, produce canonical release artifacts, and publish them to TestPyPI/PyPI.
- Reproducibility: CI uses standard manylinux containers and controlled macOS runners to produce wheels intended for distribution; this reduces host-specific variation.
- Approval and secrets: promote-to-pypi requires an environment approval to access the PyPI token, preventing accidental publishes.

When to use the publish pipeline
- After local validation and CI runs (e.g., `pyalp-ci.yml` for PRs), create an annotated tag `pyalp.vX.Y.Z` and push it to trigger `publish-to-testpypi.yml`.
- Once TestPyPI artifacts are validated, run `promote-to-pypi.yml` (workflow dispatch) to publish to PyPI; this step requires environment approval and the presence of the `PYPI_API_TOKEN` secret.

Operational note: TestPyPI propagation and verification
- The verification step that installs wheels from TestPyPI can occasionally fail due to propagation delays between upload and index availability. If the TestPyPI install step fails transiently, re-run the workflow or re-trigger the release; the promote job should only be run once test artifacts are available and verified.




High-level contract
- Inputs: CMake-based native backends built by the top-level CMake tree, a generated Python metadata file produced by CMake, and the Python package source in `pyalp/src`.
- Output: Platform-specific wheels that contain the compiled shared object(s) and a generated `_metadata.py` file. The published PyPI project name is `alp-graphblas`, but the import name inside Python remains `pyalp`.
- Success criteria: pip install alp-graphblas (from TestPyPI or PyPI) yields a package exposing `pyalp.get_build_metadata()` and one or more backend modules accessible via `pyalp.get_backend(<name>)`.

Where things live (important files)
- `pyalp/pyproject.toml` — project metadata used by CI and for the package release (project name, version, runtime dependencies such as numpy).
- `pyalp/setup.py` — custom setuptools glue. It either copies prebuilt shared objects from the CMake build tree into the wheel (preferred for CI-built wheels) or builds from source with pybind11 when no prebuilt artifact is present.
- `pyalp/src/pyalp/_metadata.py.in` — CMake template used to generate `pyalp_metadata.py` (copied into wheels as `_metadata.py`). If you change the runtime metadata shape, update this file and the code that reads it.
- Top-level CMake files (`CMakeLists.txt` and `src/…`) — define native targets such as `pyalp_ref`, `pyalp_omp`, `pyalp_nonblocking`. CI runs a top-level CMake configure/build per-Python-ABI and produces the native `.so` files and the generated metadata file.
- `.github/workflows/publish-to-testpypi.yml` — builds wheels with cibuildwheel and publishes to TestPyPI (trigger: push tag `pyalp.v*`). This workflow also creates a GitHub Release with wheel assets.
- `.github/workflows/promote-to-pypi.yml` — promotes a GitHub Release's wheel assets to PyPI. The job requires `environment: production` (see repository settings) and uses the secret `PYPI_API_TOKEN`.
- `.github/scripts/` — helper scripts used by CI (e.g., verification and TestPyPI wait scripts).

How the CI build produces a wheel (brief)
- cibuildwheel is used to produce wheels for multiple Python ABIs and OSes.
- Before building each wheel, CI runs a `CIBW_BEFORE_BUILD` script which:
  - Installs CMake + Ninja inside the build container.
  - Derives Git and package version metadata and sets environment variables.
  - Configures a per-ABI CMake build directory (e.g. `build/cp311`) and runs CMake to produce the compiled backends and a generated `pyalp_metadata.py` file inside that build dir.
  - Exports `CMAKE_BUILD_DIR` pointing to the per-ABI build directory so `pyalp/setup.py` can locate the generated outputs.
- The packaging step runs `pyalp/setup.py` (setup.py will copy discovered prebuilt `.so` files and the generated metadata file into the package build directory). The wheel built by cibuildwheel therefore contains the prebuilt, ABI-specific `.so` and `_metadata.py`.

How `pyalp/setup.py` cooperates with CMake
- By default `setup.py` searches the repo `../build/**` tree for prebuilt shared objects named like the native targets (`pyalp_ref`, `pyalp_omp`, `pyalp_nonblocking`). If it finds them it adds Extension entries with empty sources and uses a custom `build_ext` to copy the prebuilt library into the wheel.
- `setup.py` looks for the generated metadata file in the directory pointed to by the `CMAKE_BUILD_DIR` environment variable (set by the CI before_build script). If present it copies `pyalp_metadata.py` -> `_metadata.py` next to the extension in the wheel.
- If no prebuilt modules are detected and `pybind11` is available, `setup.py` will fall back to building from sources with pybind11.
- Environment variables you can use locally:
  - `CMAKE_BUILD_DIR` — path to the per-ABI CMake build dir that contains `pyalp_metadata.py` and the built `.so` files.
  - `PREBUILT_PYALP_SO` or `PYALP_PREBUILT_SO` — point to a single prebuilt shared object to include in the wheel (helpful for local testing).

Adding a new compiled backend (step-by-step)
1) Add a CMake target
   - Add a target to your CMake configuration (top-level CMake or `pyalp` subdirectory). Name it with the prefix used by `setup.py` (for example `pyalp_mybackend` if you want the backend import name to be `pyalp_mybackend`).
   - Ensure the target produces a shared library file named so that it will be discoverable by the existing glob in `pyalp/setup.py` (the packaging code looks for `build/**/<target>*.(so|pyd)`).
   - If the backend needs additional compile flags or third-party deps, add those to the CMake target and to the cibuildwheel before-build step where platform-specific dependencies are installed.

2) Expose the pybind11 module name correctly
   - The module name that Python imports must match the filename stem: for a target `pyalp_mybackend` the shared object should become something like `pyalp_mybackend.cpython-311-x86_64-linux-gnu.so` and will be installed into the `pyalp` package as `pyalp/mybackend` importable as `pyalp.pyalp_mybackend` or accessed by the helper APIs.
   - `setup.py` maps module names to the extension name `pyalp.<module_name>`; if you introduce a module with a different naming scheme, update `pyalp/setup.py`'s discovery or add an explicit mapping.

3) Update CI build targets
   - The cibuildwheel `CIBW_BEFORE_BUILD` script exports a `BUILD_TARGETS` variable used by CMake to restrict which targets to build. Edit `.github/workflows/publish-to-testpypi.yml` under `CIBW_BEFORE_BUILD` to include your new target name in `BUILD_TARGETS`.
   - If your backend requires platform-specific dependency installation (e.g., libnuma, libomp) ensure those package installs are available in the before-build block.

4) Update packaging helpers if needed
   - If your module uses a new stem that the setup script won't detect, add the module name to the `supported` list in `pyalp/setup.py` or rely on the glob search.
   - If you want to bundle multiple backends under a different naming convention, update `find_all_prebuilt()` discovery logic and the code that constructs `Extension(f"pyalp.{modname}")` entries.

5) Add/adjust tests
   - Add small smoke tests (ideally under `tests/python/` or `tests/smoke/`) that run the new backend. Prefer running each backend in its own process where feasible to avoid pybind11 registration collisions.

6) Build and test locally (quick recipe)
   - Ensure system deps installed: cmake, ninja, a C++ toolchain and any library dependencies.
   - Create a per-ABI build dir and configure CMake as CI does. Example (for Python 3.11):
    ```bash
     mkdir -p build/cp311
     cmake -S . -B build/cp311 -G Ninja -DCMAKE_BUILD_TYPE=Release -DENABLE_PYALP=ON -DCMAKE_POSITION_INDEPENDENT_CODE=ON -DPython3_EXECUTABLE=$(which python3)
     cmake --build build/cp311 --target pyalp_ref pyalp_mybackend --parallel
     ```

   - Build a wheel locally from the `pyalp` package. From the repository root:
    ```bash
     export CMAKE_BUILD_DIR="$(pwd)/build/cp311"
     cd pyalp
     # Build a wheel using the package directory's setup.py
     python -m pip wheel . --no-deps -w ../wheelhouse

     # Install and test the wheel in a fresh venv
     python -m venv /tmp/venv_test
     source /tmp/venv_test/bin/activate
     python -m pip install --upgrade pip
     python -m pip install ../wheelhouse/alp-graphblas-*.whl
     ```

   - Note: `--no-deps` is optional when building locally; published wheels should contain runtime dependency metadata so that pip will pull `numpy` automatically.

Releases and publishing (how CI is wired)
- Creating a TestPyPI release (normal path):
  1. Bump the version in `pyalp/pyproject.toml` (recommended) and commit.
  2. Create a git tag of the form `pyalp.vX.Y.Z` and push the tag. The `publish-to-testpypi.yml` workflow is triggered on push tags matching `pyalp.v*`.
  3. The workflow builds wheels (cibuildwheel), uploads wheel artifacts as GitHub workflow artifacts, publishes to TestPyPI, and creates a GitHub Release with the wheel assets.

 - Promoting to PyPI (two-step gated publish):
   - The `publish-to-testpypi.yml` workflow automatically builds and deploys wheels to TestPyPI and then attempts to install and verify those wheels in a fresh virtual environment. Occasionally this verification can fail due to propagation delays between upload and availability; if that happens, re-run the workflow (or re-trigger the release) until the verification completes successfully.
   - The `promote-to-pypi.yml` workflow is triggered manually (`workflow_dispatch`) and it is enabld only with the `pyalp.v*` tag. It downloads the assets attached to the GitHub Release and uploads them to PyPI using the secret `PYPI_API_TOKEN`.
  - The promote job is configured to use the repository `production` environment. Access to the `PYPI_API_TOKEN` secret in that environment requires an approval step by repository administrators (see Settings → Environments → production).

Checklist before releasing
- Bump `pyalp/pyproject.toml` version.
- Ensure `pyalp/pyproject.toml` includes runtime dependencies (e.g., `numpy>=1.22`) so pip installs them automatically.
- Ensure `CIBW_BEFORE_BUILD` in `.github/workflows/publish-to-testpypi.yml` builds your new backend (`BUILD_TARGETS` updated).

----------------------
Local developer workflow (CMake-generated target)
------------------------------------------------

The project now exposes a CMake-generated `pyalp` target that builds all
enabled pyalp backends and packages wheel(s) using the same packaging logic
that CI uses. This is the recommended local path and replaces the previous
helper script.

Usage:

```bash
# Configure from repo root (LOCAL profile enables host-optimizations)
cmake -S . -B build/host -DALP_BUILD_PROFILE=LOCAL -DENABLE_PYALP=ON -G Ninja

# Build and package via the CMake target (this will place wheels in build/host/dist)
cmake --build build/host --target pyalp --parallel
```

After the target completes you will see a message pointing to the wheel(s).
You can either add the generated python directory to `PYTHONPATH` for quick
iteration:

```bash
export PYTHONPATH="$PYTHONPATH:$(pwd)/build/host/python"
```

Or install the wheel into a venv:

```bash
python -m venv /tmp/pyalp-venv
source /tmp/pyalp-venv/bin/activate
pip install build/host/dist/*.whl
```

If you need to reproduce CI-style portable wheels, configure with the
`DEPLOYMENT` profile instead:

```bash
cmake -S . -B build/cp311 -DALP_BUILD_PROFILE=DEPLOYMENT -DENABLE_PYALP=ON -G Ninja
cmake --build build/cp311 --target pyalp --parallel
```

Notes:
- Ensure system dependencies like `libnuma-dev` and `libomp` are installed when building backends that require them.
- The packaging step relies on `CMAKE_BUILD_DIR` to locate generated metadata and prebuilt `.so` files; the CMake target sets this environment appropriately when invoking `pip wheel`.

Troubleshooting / common pitfalls
- Missing metadata in wheels: Make sure CMake writes the generated `pyalp_metadata.py` into the per-ABI build dir (CI sets `CMAKE_BUILD_DIR` and `setup.py` copies `pyalp_metadata.py` -> `_metadata.py`). If your metadata template changed, update `pyalp/src/pyalp/_metadata.py.in`.
- Prebuilt `.so` not found: `pyalp/setup.py` discovers prebuilt shared objects under `build/**`. Ensure you used the same target name and that the produced filename contains the Python ABI tag (or set `PREBUILT_PYALP_SO` to the path).
- ABI contamination across wheels: CI uses per-ABI build directories (e.g. `build/cp311`) to avoid cross-ABI contamination. When testing locally, clean build dirs between ABI runs.
- pybind11 registration collisions: If you see type-registration errors when importing multiple different backends in the same process, prefer running backends in separate processes or ensure pybind11 wrappers use `py::module_local()` for types that may be defined in multiple modules.

Security notes
- The promotion workflow uses a `PYPI_API_TOKEN` stored as a secret (likely in the repository environment `production`). If you did not create this token yourself, check:
  - Repository Settings → Secrets and variables → Actions
  - Environments → production → Secrets
  - Organization-level secrets (if applicable)
- Rotate/revoke tokens if you discover an unexpected token.

Appendix — quick pointers to edit points
- Add CMake target: top-level CMake / `pyalp/src` CMakeLists.
- Ensure discovery in `pyalp/setup.py`: supported names in `find_all_prebuilt()` and the glob-based discovery.
- Include generated metadata: `pyalp/src/pyalp/_metadata.py.in` (CMake variables are substituted into this template).
- CI build targets: `.github/workflows/publish-to-testpypi.yml` (search for `BUILD_TARGETS` and `BACKEND_FLAGS` in `CIBW_BEFORE_BUILD`).
- Promote workflow: `.github/workflows/promote-to-pypi.yml` (uses `PYPI_API_TOKEN` and `environment: production`).


