#!/usr/bin/env bash
# Clone Eigen (header-only) into extern/eigen. Inherits repository license.
# We do NOT modify Eigen; its own license remains in its source tree.
set -euo pipefail

REPO_ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
TARGET_DIR="${REPO_ROOT}/extern/eigen"

if [[ -d "${TARGET_DIR}/Eigen" ]]; then
  echo "Eigen already present at ${TARGET_DIR}" >&2
  exit 0
fi

mkdir -p "${REPO_ROOT}/extern"
cd "${REPO_ROOT}/extern"

# Shallow clone for minimal footprint; adjust branch/tag as needed
EIGEN_URL_SSH="git@gitlab.com:libeigen/eigen.git"
EIGEN_URL_HTTPS="https://gitlab.com/libeigen/eigen.git"

echo "Attempting to clone Eigen (shallow) via SSH: ${EIGEN_URL_SSH}" >&2
if ! git clone --depth 1 "${EIGEN_URL_SSH}" eigen; then
  echo "SSH clone failed; trying HTTPS fallback: ${EIGEN_URL_HTTPS}" >&2
  if ! git clone --depth 1 "${EIGEN_URL_HTTPS}" eigen; then
    echo "Failed to clone Eigen via SSH and HTTPS." >&2
    exit 1
  fi
fi

echo "Eigen cloned to ${TARGET_DIR}" >&2
