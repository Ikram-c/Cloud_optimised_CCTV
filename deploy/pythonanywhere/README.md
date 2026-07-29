# Deploying cctv_zarr on PythonAnywhere

Two supported routes. Route A (WSGI) uses the normal Web tab and works
on any account, including free. Route B (ASGI) uses PythonAnywhere's
experimental uvicorn sites via the `pa` CLI.

## 0. Get the code onto PythonAnywhere

Open a Bash console and either clone your GitHub repo:

```bash
git clone https://github.com/<you>/cctv_zarr.git
```

or upload the zip via the Files tab and `unzip` it. Then:

```bash
cd ~/cctv_zarr
bash deploy/pythonanywhere/setup_pythonanywhere.sh
```

This creates the `cctvzarr` virtualenv, installs the package with the
panel dependencies plus the `a2wsgi` adapter, and creates the data
folders (videos, stores, mock_gcs) inside the project.

## Route A - classic Web app (WSGI, any account)

1. Web tab -> Add a new web app -> Manual configuration -> the Python
   version matching your virtualenv.
2. In the Virtualenv section enter: `/home/<you>/.virtualenvs/cctvzarr`
3. Open the WSGI configuration file link and replace its entire
   contents with:

```python
import sys
sys.path.insert(
    0, "/home/<you>/cctv_zarr/deploy/pythonanywhere",
)
from wsgi_app import application
```

4. Reload the web app. The panel is live at
   `https://<you>.pythonanywhere.com`.

## Route B - ASGI beta (uvicorn, pa CLI)

```bash
pip install --user --upgrade pythonanywhere
pa website create --domain <you>.pythonanywhere.com \
    --command '/home/<you>/.virtualenvs/cctvzarr/bin/uvicorn \
--app-dir /home/<you>/cctv_zarr/deploy/pythonanywhere \
--uds $DOMAIN_SOCKET asgi_app:app'
```

An API token must be set up first (Account -> API token). The feature
is beta: no static-file mappings and limited web UI management.

## Platform notes and limits

- `config.pythonanywhere.yaml` is used by both entry points: all
  paths are relative to the project folder, the cloud archive uses
  the offline mock (a local folder standing in for GCS). Edit it to change behaviour.
- Web workers are recycled by the platform, and free accounts have
  limited CPU seconds per day. The panel's background processing
  works, but for long batches prefer running the CLI in a console or
  a Scheduled/Always-on task:
  `~/.virtualenvs/cctvzarr/bin/cctv-zarr --config config.pythonanywhere.yaml ingest videos/cam01.mp4 --upload`
- Free accounts have a 512 MB disk quota - video processing fills it
  quickly. Keep test clips short or upgrade for real footage.
- `config.pythonanywhere.yaml` caps panel uploads at 50 MB per video
  (`ui.max_upload_mb`) so one visitor cannot fill the disk; the cap
  is shown in the drop zone and enforced on both ends. Raise it (or
  set 0 to disable) on a paid account.
- Concurrent demo visitors are expected: batches queue in a bounded
  waiting line (`ui.max_queued_jobs`, default 4) served by one
  worker, each browser tracks its own job, and panel state is saved
  per browser - so several people can click through the demo at
  once without stepping on each other. On free accounts the shared
  CPU allowance still applies; pre-ingest the showpiece stores and
  let visitors mostly browse, search, and view Results.
- Uploads from the panel arrive in 8 MB parts, well under the
  platform's request-size limit.
- The panel has no authentication. On PythonAnywhere it is public at
  your domain: enable the Web tab's password protection (Route A;
  paid feature on some plans) or treat the deployment as a demo with
  non-sensitive footage.
