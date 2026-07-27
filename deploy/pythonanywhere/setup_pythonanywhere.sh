#!/bin/bash
set -euo pipefail

PROJECT_DIR="$(cd "$(dirname "$0")/../.." && pwd)"
VENV_NAME="${1:-cctvzarr}"
VENV_DIR="$HOME/.virtualenvs/$VENV_NAME"

echo "project: $PROJECT_DIR"
echo "virtualenv: $VENV_DIR"

python3 -m venv "$VENV_DIR"
source "$VENV_DIR/bin/activate"
pip install --upgrade pip
pip install -e "$PROJECT_DIR[ui]" a2wsgi

cd "$PROJECT_DIR"
mkdir -p videos stores mock_gcs

echo
echo "Done. Next steps are in deploy/pythonanywhere/README.md"
