#!/usr/bin/env python3
"""Run the repo's backend smoke runner against an installed pyalp package.

This script is intended to be invoked from CI after installing pyalp from
TestPyPI. It accepts a single argument (backend name) and will skip if that
backend is not present in the installed package.
"""
import sys
import subprocess

try:
    import pyalp
except Exception:
    print('ERROR: failed to import pyalp', file=sys.stderr)
    raise


def main(argv):
    if len(argv) < 2:
        print('Usage: run_backend_smoke_installed.py <backend_name>', file=sys.stderr)
        return 2
    backend = argv[1]
    backends = pyalp.list_backends()
    print('discovered backends:', backends)
    if backend not in backends:
        print(f'backend {backend} not present in installed package, skipping')
        return 0

    rc = subprocess.call([sys.executable, 'tests/python/backend_smoke_runner.py', backend])
    if rc != 0:
        print(f'backend {backend} smoke runner failed with exit {rc}', file=sys.stderr)
    return rc


if __name__ == '__main__':
    sys.exit(main(sys.argv))
