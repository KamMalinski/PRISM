#!/usr/bin/env bash
set -euo pipefail

python3 "$(cd "$(dirname "$0")" && pwd)/bootstrap.py" --platform linux
