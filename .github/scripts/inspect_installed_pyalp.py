#!/usr/bin/env python3
"""Print basic information about the installed `pyalp` package.

This script is intended to be invoked from CI after installing the package
from TestPyPI. It prints the package file, available binary modules, and
the runtime build metadata exposed by the package.
"""
import pkgutil
import sys

try:
    import pyalp
except Exception:
    print('ERROR: failed to import pyalp', file=sys.stderr)
    raise

print('pyalp package:', getattr(pyalp, '__file__', None))
print('available modules in package:', [m.name for m in pkgutil.iter_modules(pyalp.__path__)])
try:
    print('build metadata:', pyalp.get_build_metadata())
except Exception as e:
    print('metadata error:', e)
print('listed backends via helper:', pyalp.list_backends())
