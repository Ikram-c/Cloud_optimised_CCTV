"""PythonAnywhere ASGI entry point (experimental uvicorn sites).

Exposes the FastAPI panel as ``asgi_app:app`` for PythonAnywhere's
ASGI beta, created with the pa CLI (see the README in this folder).
"""

import os
import sys
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parent.parent.parent
CONFIG_PATH = PROJECT_ROOT / "config.pythonanywhere.yaml"

if str(PROJECT_ROOT / "src") not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT / "src"))
os.chdir(PROJECT_ROOT)

from cctv_zarr.webapp.server import create_app

app = create_app(CONFIG_PATH)
