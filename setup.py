"""This repository is not meant to be built as a Python package at the root.

Please build wheels from the 'pyalp' subdirectory.
"""

from setuptools import setup

raise SystemExit(
    "Use 'pip wheel ./pyalp' (cibuildwheel points to the 'pyalp' subdirectory)."
)
