"""
Parametrized smoke test that runs the conjugate-gradient example against all
available pyalp backend modules (pyalp_ref, pyalp_omp, pyalp_nonblocking).

This is adapted from tests/python/test.py but runs the same assertions for
each backend installed in the `pyalp` package. If a backend is not present in
the wheel, the test is skipped.
"""
import os
import shutil
import subprocess
import sys
import pytest
from pathlib import Path


BACKENDS = ["pyalp_ref", "pyalp_omp", "pyalp_nonblocking", "_pyalp"]


def backend_exists_in_package(backend: str) -> bool:
    # Check installed package dir for a backend shared object
    try:
        import pyalp
        p = Path(pyalp.__file__).parent
        patterns = [f"{backend}*.so", f"{backend}*.pyd"]
        for pat in patterns:
            if any(p.glob(pat)):
                return True
    except Exception:
        return False
    return False


@pytest.mark.parametrize("backend", BACKENDS)
def test_conjugate_gradient_backend_subprocess(backend):
    if not backend_exists_in_package(backend):
        pytest.skip(f"backend {backend} not present in installed package")

    # Run the smoke test in a fresh Python subprocess to avoid in-process
    # pybind11 type registration conflicts between multiple extension modules.
    python_exe = sys.executable
    runner = Path(__file__).with_name("backend_smoke_runner.py")
    if not runner.exists():
        pytest.skip("backend smoke runner script not found")
    proc = subprocess.run([python_exe, str(runner), backend], capture_output=True, text=True)
    if proc.returncode != 0:
        # Give helpful debug output
        print(proc.stdout)
        print(proc.stderr)
    assert proc.returncode == 0, f"backend {backend} failed with return code {proc.returncode}\nSTDOUT:\n{proc.stdout}\nSTDERR:\n{proc.stderr}"
