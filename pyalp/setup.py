from setuptools import setup, Extension
from setuptools import find_packages
from setuptools.command.build_ext import build_ext as _build_ext
import sys
import os
import glob
import shutil
import sysconfig
bdist_wheel_cmd = None
try:
    # Used to mark wheel as non-pure when bundling a prebuilt .so
    from wheel.bdist_wheel import bdist_wheel as _bdist_wheel

    class bdist_wheel(_bdist_wheel):
        def finalize_options(self):
            super().finalize_options()
            # wheel contains a native shared object; mark as platform-specific
            self.root_is_pure = False

    bdist_wheel_cmd = bdist_wheel
except Exception:
    bdist_wheel_cmd = None
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
def find_prebuilt():
    candidates = []
    # Top-level CMake build tree locations (preferred flow)
    candidates.extend(glob.glob(os.path.join(here, '..', 'build', '**', 'pyalp_ref*.so'), recursive=True))
    candidates.extend(glob.glob(os.path.join(here, '..', 'build', '**', 'pyalp_ref*.pyd'), recursive=True))
    candidates.extend(glob.glob(os.path.join(here, '..', 'build', '**', '_pyalp*.so'), recursive=True))
    candidates.extend(glob.glob(os.path.join(here, '..', 'build', '**', '_pyalp*.pyd'), recursive=True))
    # Prefer the candidate matching the current Python tag
    py_tag = f"cpython-{sys.version_info[0]}{sys.version_info[1]}"
    matching = [c for c in candidates if py_tag in os.path.basename(c)] or candidates
    return matching[0] if matching else None

if not prebuilt_so:
    prebuilt_so = find_prebuilt()
package_data = {}
ext_modules = []

class build_ext_copy_prebuilt(_build_ext):
    """Custom build_ext that copies a prebuilt shared object into the build dir.

    This ensures the extension is installed into platlib and the wheel is valid
    for auditwheel repair.
    """

    def build_extension(self, ext):
        # Determine target path for the extension
        target_path = self.get_ext_fullpath(ext.name)
        os.makedirs(os.path.dirname(target_path), exist_ok=True)
        src = os.environ.get("PREBUILT_PYALP_SO") or os.environ.get("PYALP_PREBUILT_SO") or find_prebuilt()
        if not src or not os.path.exists(src):
            raise RuntimeError("Prebuilt pyalp shared object not found during build_ext")
        shutil.copyfile(src, target_path)

if prebuilt_so:
    if not os.path.exists(prebuilt_so):
        raise FileNotFoundError(f"PREBUILT_PYALP_SO set but file not found: {prebuilt_so}")
    # Declare a binary extension so files go to platlib; actual build just copies the prebuilt .so
    ext_modules = [Extension("pyalp._pyalp", sources=[])]
else:
    if not _have_pybind11:
        raise RuntimeError("pybind11 is required to build the extension from sources. Install pybind11 or provide PREBUILT_PYALP_SO to bundle a prebuilt .so.")
    assert Pybind11Extension is not None
    ext_modules = [
        Pybind11Extension(
            "pyalp._pyalp",
            ["src/pyalp/bindings.cpp"],
            include_dirs=[
                os.path.join(here, "src"),
                os.path.join(here, "src", "pyalp"),
                os.path.join(here, "extern", "pybind11", "include"),
                os.path.normpath(os.path.join(here, "..", "include")),
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
}

# Supply cmdclass entries for build_ext (copy-prebuilt or pybind11) and bdist_wheel
cmdclass = {}
if prebuilt_so:
    cmdclass["build_ext"] = build_ext_copy_prebuilt
elif build_ext is not None:
    cmdclass["build_ext"] = build_ext
if bdist_wheel_cmd is not None:
    cmdclass["bdist_wheel"] = bdist_wheel_cmd
if cmdclass:
    setup_kwargs["cmdclass"] = cmdclass

setup(**setup_kwargs)
