#!/bin/sh
set -e
cd "$(dirname "$0")"
exec .venv/bin/python run_dashboard.py "$@"
