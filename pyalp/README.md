
# pyalp (packaged)

This directory contains the Python package layout for the `pyalp` bindings
that expose parts of the ALP GraphBLAS project via pybind11.

Quick start
-----------

Create and activate a virtual environment, then install the package (example
using TestPyPI):

```bash
python -m venv venv
source venv/bin/activate
pip install --index-url https://test.pypi.org/simple/ --no-deps pyalp
```

Basic usage
-----------

```python
import pyalp
print('pyalp version:', pyalp.version())
print('build metadata:', pyalp.get_build_metadata())
print('algorithm readme (first 200 chars):')
print(pyalp.get_algorithm_metadata().get('readme','')[:200])
```

Runtime metadata
----------------

The package provides a small runtime metadata module generated at build time
from CMake. Useful keys in `pyalp.get_build_metadata()` include:

- `version` — pyalp package version
- `build_type` — CMake build type used (e.g., Release)
- `alp_version` — ALP repository version or tag used to build
- `alp_git_commit` / `alp_git_branch` — Git information captured by CI
- `license` — detected repository license (e.g. Apache-2.0)

`pyalp.get_algorithm_metadata()` contains available algorithm/backends and
also includes a `readme` key with the package README contents.

Packaging notes (for maintainers)
--------------------------------

- The CI uses a top-level CMake configure/build to produce the native shared
	object and a CMake-configured `_metadata.py`. The packaging `setup.py` then
	copies the built `.so` and `_metadata.py` into the wheel.
- The CI passes Git/version information into CMake so the generated metadata
	is populated even in detached/CI environments.

If you modify the metadata template, update `pyalp/src/pyalp/_metadata.py.in`.

License
-------

See the repository `LICENSE` at the project root; the packaging pipeline
attempts to detect and embed the license string in runtime metadata.
