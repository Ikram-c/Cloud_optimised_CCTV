"""PythonAnywhere WSGI entry point for the free-tier mini demo.

Identical to the standard deployment but bound to config.mini.yaml,
whose settings are sized so the virtualenv, the sample footage, and
real video processing all fit a free account's disk and CPU quota.
"""

import os
import sys
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parent.parent.parent
CONFIG_PATH = PROJECT_ROOT / "config.mini.yaml"

if str(PROJECT_ROOT / "src") not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT / "src"))
os.chdir(PROJECT_ROOT)

from a2wsgi import ASGIMiddleware

from cctv_zarr.webapp.server import create_app

application = ASGIMiddleware(create_app(CONFIG_PATH))
