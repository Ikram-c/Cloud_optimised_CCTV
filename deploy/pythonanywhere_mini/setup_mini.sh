#!/bin/bash
set -euo pipefail

PROJECT_DIR="$(cd "$(dirname "$0")/../.." && pwd)"
VENV_DIR="$HOME/.virtualenvs/cctvmini"
PY="$VENV_DIR/bin/python"

echo "== cctv_zarr mini setup (PythonAnywhere free tier) =="

echo "-- tiny virtualenv (reuses preinstalled system packages)"
python3 -m venv --system-site-packages "$VENV_DIR"

echo "-- checking which packages the system already provides"
"$PY" -c "import numpy, cv2, yaml" 2>/dev/null \
  || "$PY" -m pip install --no-cache-dir \
       "numpy>=1.26" "opencv-python-headless>=4.9" "pyyaml>=6.0"
"$PY" -c "import fastapi, uvicorn" 2>/dev/null \
  || "$PY" -m pip install --no-cache-dir \
       "fastapi>=0.110" "uvicorn>=0.29"
"$PY" -c "import a2wsgi" 2>/dev/null \
  || "$PY" -m pip install --no-cache-dir "a2wsgi>=1.10"
"$PY" -m pip install --no-cache-dir --no-deps -e "$PROJECT_DIR"

cd "$PROJECT_DIR"
mkdir -p videos stores mock_gcs

echo "-- access code (the demo URL is public)"
"$PY" - <<'EOF'
import secrets
from pathlib import Path
import yaml

path = Path("config.mini.yaml")
raw = yaml.safe_load(path.read_text())
if not raw["security"]["auth_token"]:
    code = secrets.token_urlsafe(9)
    raw["security"]["auth_token"] = code
    path.write_text(yaml.safe_dump(raw, sort_keys=False))
    print(f"generated access code: {code}")
else:
    print(f"access code already set: {raw['security']['auth_token']}")
EOF

echo "-- generating and pre-processing the sample clip"
"$PY" scripts/make_sample_video.py --out videos/sample_footage.mp4
"$PY" -m cctv_zarr.cli --config config.mini.yaml \
  ingest videos/sample_footage.mp4 \
  --start-time "$(date -u +%Y-%m-%dT%H:%M:%S+00:00)"

echo "-- footprint"
du -sh "$VENV_DIR" stores videos mock_gcs 2>/dev/null || true

echo
echo "Done. Now wire the Web tab:"
echo "  virtualenv:  $VENV_DIR"
echo "  WSGI file:   point it at deploy/pythonanywhere_mini/wsgi_app.py"
echo "  (see deploy/pythonanywhere_mini/README.md)"
