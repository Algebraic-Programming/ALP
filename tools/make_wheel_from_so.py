#!/usr/bin/env python3
"""Simple helper: build a wheel that packages a prebuilt .so into the pyalp package.

Usage: make_wheel_from_so.py <path-to-so> --out-dir <outdir>

The script will create <outdir>/<name>-<version>-<tags>.whl containing:
 - pyalp/__init__.py (minimal stub)
 - pyalp/<so_basename>
 - pyalp-<version>.dist-info/{METADATA,WHEEL,RECORD}

This is intentionally minimal and meant for CI-snapshots where the compiled .so
is produced by your CMake job and we only need to bundle it into a wheel.
"""

import argparse
import re
import sys
import zipfile
import sysconfig
from pathlib import Path

NAME = "pyalp"
VERSION = "0.0.0"


def infer_cp_tag_from_filename(name: str) -> str | None:
    # try to find cp311/cp312 style
    m = re.search(r"cp(\d{2,3})", name)
    if m:
        return f"cp{m.group(1)}"
    # try to find cpython-311 style
    m = re.search(r"cpython-(\d{3})", name)
    if m:
        return f"cp{m.group(1)}"
    return None


def make_wheel(so_path: Path, out_dir: Path) -> Path:
    if not so_path.exists():
        raise FileNotFoundError(f".so not found: {so_path}")
    so_name = so_path.name
    cp_tag = infer_cp_tag_from_filename(so_name)
    if not cp_tag:
        # Fallback to current interpreter if tag cannot be inferred
        cp_tag = f"cp{sys.version_info.major}{sys.version_info.minor}"
    py_tag = cp_tag
    abi_tag = cp_tag
    plat = sysconfig.get_platform().replace("-", "_").replace(".", "_")
    wheel_fname = f"{NAME}-{VERSION}-{py_tag}-{abi_tag}-{plat}.whl"
    out_dir.mkdir(parents=True, exist_ok=True)
    wheel_path = out_dir / wheel_fname

    init_py = (
        "try:\n"
        "    from . import _pyalp\n"
        "except Exception:\n"
        "    _pyalp = None\n"
        "__all__ = [\"_pyalp\"]\n"
    )

    dist_info = f"{NAME}-{VERSION}.dist-info"
    metadata = (
        "Metadata-Version: 2.1\n"
        f"Name: {NAME}\n"
        f"Version: {VERSION}\n"
        "Summary: pyalp packaged wheel (prebuilt .so)\n"
    )
    wheel_meta = (
        "Wheel-Version: 1.0\n"
        "Generator: make_wheel_from_so.py\n"
        "Root-Is-Purelib: false\n"
        f"Tag: {py_tag}-{abi_tag}-{plat}\n"
    )

    with zipfile.ZipFile(wheel_path, "w", compression=zipfile.ZIP_DEFLATED) as z:
        z.writestr(f"{NAME}/__init__.py", init_py)
        # Normalize the extension module name to _pyalp.so so the package can import it as pyalp._pyalp
        so_target_name = "_pyalp.so"
        z.write(so_path, f"{NAME}/{so_target_name}")
        z.writestr(f"{dist_info}/METADATA", metadata)
        z.writestr(f"{dist_info}/WHEEL", wheel_meta)
        # RECORD should list files; for minimal CI, leave entries empty (tools may complain but pip accepts)
        z.writestr(f"{dist_info}/RECORD", "")

    return wheel_path


def parse_args(argv):
    p = argparse.ArgumentParser(description="Make simple wheel from prebuilt .so")
    p.add_argument("so", help="Path to prebuilt .so file")
    p.add_argument("--out-dir", default="dist_wheel", help="Output directory")
    return p.parse_args(argv)


def main(argv):
    args = parse_args(argv)
    so_path = Path(args.so)
    out_dir = Path(args.out_dir)
    try:
        wheel = make_wheel(so_path, out_dir)
        print("Wheel written to", wheel)
    except Exception as e:
        print("ERROR:", e, file=sys.stderr)
        return 2
    return 0


if __name__ == "__main__":
    raise SystemExit(main(sys.argv[1:]))
