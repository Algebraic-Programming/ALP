"""
Root setup.py that builds the pyalp wheel from the repository root.

This script mirrors pyalp/setup.py so that cibuildwheel can copy the
entire repository (needed for the top-level CMake build) while pip
still builds the Python wheel using the pyalp package located under
pyalp/src.
"""

from setuptools import setup
from setuptools import find_packages
import os
import glob
import shutil

# Mark wheel as non-pure when bundling a prebuilt shared object
bdist_wheel_cmd = None
try:
	from wheel.bdist_wheel import bdist_wheel as _bdist_wheel

	class bdist_wheel(_bdist_wheel):
		def finalize_options(self):
			super().finalize_options()
			self.root_is_pure = False

	bdist_wheel_cmd = bdist_wheel
except Exception:
	bdist_wheel_cmd = None

here = os.path.abspath(os.path.dirname(__file__))
pkg_root = os.path.join(here, "pyalp")
pkg_src = os.path.join(pkg_root, "src")
pkg_mod_dir = os.path.join(pkg_src, "pyalp")

# Prefer a prebuilt extension compiled by CMake if present in the tree
prebuilt_so = os.environ.get("PREBUILT_PYALP_SO") or os.environ.get("PYALP_PREBUILT_SO")
if not prebuilt_so:
	candidates = []
	# Already-copied location inside the package (e.g., from a prior stage)
	candidates.extend(glob.glob(os.path.join(pkg_mod_dir, "_pyalp*.so")))
	candidates.extend(glob.glob(os.path.join(pkg_mod_dir, "_pyalp*.pyd")))
	# Top-level CMake build tree locations (the before_build step compiles here)
	candidates.extend(glob.glob(os.path.join(here, "build", "**", "pyalp_ref*.so"), recursive=True))
	candidates.extend(glob.glob(os.path.join(here, "build", "**", "pyalp_ref*.pyd"), recursive=True))
	candidates.extend(glob.glob(os.path.join(here, "build", "pyalp", "src", "pyalp", "_pyalp*.so")))
	candidates.extend(glob.glob(os.path.join(here, "build", "pyalp", "src", "pyalp", "_pyalp*.pyd")))
	if candidates:
		prebuilt_so = candidates[0]

package_data = {}
if prebuilt_so:
	if os.path.exists(prebuilt_so):
		basename = os.path.basename(prebuilt_so)
		# Normalize to module name _pyalp, preserving ABI/platform suffix
		dot_index = basename.find('.')
		suffix = basename[dot_index:] if dot_index != -1 else os.path.splitext(basename)[1]
		dest_name = "_pyalp" + suffix
		os.makedirs(pkg_mod_dir, exist_ok=True)
		shutil.copyfile(prebuilt_so, os.path.join(pkg_mod_dir, dest_name))
		package_data = {"pyalp": [dest_name]}
	else:
		raise FileNotFoundError(f"PREBUILT_PYALP_SO set but file not found: {prebuilt_so}")
else:
	# We do not attempt to compile from sources at the repository root.
	# The CI should have built the extension via CMake beforehand.
	raise RuntimeError(
		"No prebuilt pyalp shared object found. Ensure CMake built pyalp_ref before packaging."
	)

cmdclass = {}
if bdist_wheel_cmd is not None:
	cmdclass["bdist_wheel"] = bdist_wheel_cmd

kwargs = dict(
	name="pyalp",
	version="0.0.0",
	description="pyalp package (C++ bindings)",
	packages=find_packages(where=pkg_src),
	package_dir={"": "pyalp/src"},
	include_package_data=True,
	package_data=package_data,
)
if cmdclass:
	kwargs["cmdclass"] = cmdclass

setup(**kwargs)
