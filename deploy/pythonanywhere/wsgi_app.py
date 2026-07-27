"""PythonAnywhere WSGI entry point (classic Web-tab deployment).

Wraps the FastAPI panel in a WSGI adapter (a2wsgi) so it runs under
PythonAnywhere's standard uWSGI stack on any account type. Point the
Web tab's WSGI configuration file at this module (see the README in
this folder).
"""

import os
import sys
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parent.parent.parent
CONFIG_PATH = PROJECT_ROOT / "config.pythonanywhere.yaml"

if str(PROJECT_ROOT / "src") not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT / "src"))
os.chdir(PROJECT_ROOT)

from a2wsgi import ASGIMiddleware

from cctv_zarr.webapp.server import create_app

application = ASGIMiddleware(create_app(CONFIG_PATH))
