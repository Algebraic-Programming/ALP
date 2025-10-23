from setuptools import setup, Extension
from setuptools import find_packages
import sys
import os
import glob
import shutil
_have_pybind11 = False
try:
    # import lazily — only needed when we build from sources
    from pybind11.setup_helpers import Pybind11Extension, build_ext
    _have_pybind11 = True
except Exception:
    Pybind11Extension = None
    build_ext = None

here = os.path.abspath(os.path.dirname(__file__))

prebuilt_so = os.environ.get("PREBUILT_PYALP_SO") or os.environ.get("PYALP_PREBUILT_SO")
# Prefer a prebuilt extension compiled by CMake if present in the tree
if not prebuilt_so:
    candidates = []
    # Source tree location (if copied there)
    candidates.extend(glob.glob(os.path.join(here, 'src', 'pyalp', '_pyalp*.so')))
    candidates.extend(glob.glob(os.path.join(here, 'src', 'pyalp', '_pyalp*.pyd')))
    # Typical CMake build tree locations
    # When building from the repo root
    candidates.extend(glob.glob(os.path.join(here, '..', 'build_pyalp', 'pyalp', 'src', 'pyalp', '_pyalp*.so')))
    candidates.extend(glob.glob(os.path.join(here, '..', 'build_pyalp', 'pyalp', 'src', 'pyalp', '_pyalp*.pyd')))
    # When building from the pyalp package root (cibuildwheel container mounts pyalp/ at /project)
    candidates.extend(glob.glob(os.path.join(here, 'build_pyalp', 'src', 'pyalp', '_pyalp*.so')))
    candidates.extend(glob.glob(os.path.join(here, 'build_pyalp', 'src', 'pyalp', '_pyalp*.pyd')))
    # Also allow building from repo CMake targets that produce pyalp_ref / pyalp_omp
    candidates.extend(glob.glob(os.path.join(here, '..', 'build_pyalp', '**', 'pyalp_ref*.so'), recursive=True))
    candidates.extend(glob.glob(os.path.join(here, '..', 'build_pyalp', '**', 'pyalp_ref*.pyd'), recursive=True))
    candidates.extend(glob.glob(os.path.join(here, '..', 'build_pyalp', '**', 'pyalp_omp*.so'), recursive=True))
    candidates.extend(glob.glob(os.path.join(here, '..', 'build_pyalp', '**', 'pyalp_omp*.pyd'), recursive=True))
    if candidates:
        prebuilt_so = candidates[0]
package_data = {}
ext_modules = []
if prebuilt_so:
    # If a prebuilt shared object is supplied, copy it into the package directory
    if os.path.exists(prebuilt_so):
        basename = os.path.basename(prebuilt_so)
        # Normalize the filename to private module name _pyalp, preserving ABI/platform suffix
        # Accept both prebuilt names: _pyalp*.so and pyalp_ref*.so / pyalp_omp*.so
        name_root, ext = os.path.splitext(basename)
        # strip potential leading package/module part until first dot, then keep the suffix
        dot_index = basename.find('.')
        suffix = basename[dot_index:] if dot_index != -1 else ext
        dest_name = '_pyalp' + suffix
        dest_path = os.path.join(here, 'src', 'pyalp', dest_name)
        shutil.copyfile(prebuilt_so, dest_path)
        # ensure the copied .so is included in the wheel as package data
        package_data = {'pyalp': [dest_name]}
    else:
        raise FileNotFoundError(f"PREBUILT_PYALP_SO set but file not found: {prebuilt_so}")
else:
    if not _have_pybind11:
        raise RuntimeError("pybind11 is required to build the extension from sources. Install pybind11 or provide PREBUILT_PYALP_SO to bundle a prebuilt .so.")
    # At this point Pybind11Extension and build_ext must be available
    assert Pybind11Extension is not None
    ext_modules = [
        Pybind11Extension(
            "pyalp._pyalp",
            ["src/pyalp/bindings.cpp"],
            include_dirs=[
                os.path.join(here, "src"),
                os.path.join(here, "src", "pyalp"),
                os.path.join(here, "extern", "pybind11", "include"),
                os.path.normpath(os.path.join(here, "..", "include")),  # project GraphBLAS headers
            ],
            define_macros=[("PYALP_MODULE_NAME", "_pyalp")],
            cxx_std=14,
        )
    ]

setup_kwargs = {
    "name": "pyalp",
    "version": "0.0.0",
    "description": "pyalp package (C++ bindings)",
    "packages": find_packages(where="src"),
    "package_dir": {"": "src"},
    "ext_modules": ext_modules,
    "include_package_data": True,
    "package_data": package_data,
}

# Only supply cmdclass when build_ext is available (pybind11 installed).
if build_ext is not None:
    setup_kwargs["cmdclass"] = {"build_ext": build_ext}

setup(**setup_kwargs)
