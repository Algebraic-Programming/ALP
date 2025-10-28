#!/usr/bin/env bash
set -euo pipefail
# Wait until a given package/version appears on TestPyPI (JSON API), with retries.
# Usage: wait_for_testpypi_release.sh <package-name> <version> [max_attempts] [sleep_seconds]

pkg=${1:-}
ver=${2:-}
max_attempts=${3:-12}
sleep_secs=${4:-10}

if [ -z "$pkg" ] || [ -z "$ver" ]; then
  echo "Usage: $0 <package-name> <version> [max_attempts] [sleep_seconds]" >&2
  exit 2
fi

url="https://test.pypi.org/pypi/${pkg}/json"
echo "Waiting for ${pkg}==${ver} to appear on TestPyPI (polling ${url})"

attempt=1
while [ $attempt -le $max_attempts ]; do
  echo "Attempt ${attempt}/${max_attempts}..."
  # fetch JSON; tolerate transient network errors by not exiting on curl non-zero
  body=$(curl -sS --max-time 10 "${url}" || true)
  if [ -n "$body" ]; then
    found=$(printf "%s" "$body" | python3 - <<PY
import sys, json
try:
    j = json.load(sys.stdin)
    releases = j.get('releases', {})
    print('1' if '${ver}' in releases else '0')
except Exception:
    print('0')
PY
)
    if [ "$found" = "1" ]; then
      echo "Found ${pkg} ${ver} on TestPyPI"
      exit 0
    fi
  else
    echo "No response from TestPyPI (attempt ${attempt})"
  fi

  attempt=$((attempt + 1))
  if [ $attempt -le $max_attempts ]; then
    echo "Sleeping ${sleep_secs}s before retrying..."
    sleep ${sleep_secs}
  fi
done

echo "Timed out waiting for ${pkg}==${ver} on TestPyPI after ${max_attempts} attempts" >&2
exit 1
