# cctv_zarr

[![CI](https://github.com/Ikram-c/cctv_zarr/actions/workflows/ci.yml/badge.svg)](https://github.com/Ikram-c/cctv_zarr/actions/workflows/ci.yml)
[![Demo](https://github.com/Ikram-c/cctv_zarr/actions/workflows/demo-pages.yml/badge.svg)](https://ikram-c.github.io/cctv_zarr/)


CCTV-optimised, movement-aware OME-Zarr archival with chunk-level cloud
query. An independent sibling of `Cloud_Dashcam_POC`, rebuilt around a
fixed-camera surveillance workload: encoder-aligned chunking, optical-flow
per-object detection of movement in any direction, and an archive you
can query
so that only the chunks you asked for ever leave the cloud.

## Design

### Encoder-aligned chunking (H.264 / H.263)

Standard CCTV encoders emit a closed GOP — an IDR keyframe followed by
predicted frames — at a fixed cadence (typically every 1–2 s). A GOP is the
smallest unit you can cut, decode, or export independently. `cctv_zarr`
makes the **Zarr time-chunk exactly one GOP** (`gop.gop_frames`, default
30): every chunk boundary is a legal cut point, every chunk is one object
in the archive, and every chunk is independently fetchable and decodable.

Zarr requires uniform chunk sizes along an axis, so movement cannot
literally change a chunk's length — and it doesn't need to. Movement
*activates* chunking the way a real encoder inserts an
IDR on demand: each movement event is snapped outward to the GOP grid,
every covered chunk is flagged, and the chunk where the event begins is
recorded with `trigger: "motion"` (continuation chunks are ordinary
`"gop"` boundaries).

### Movement detection and per-object events (optical flow)

`flow.MotionTracker` runs dense Farneback optical flow on downscaled grey
frames and uses only the flow **magnitude** - direction is deliberately
ignored, so every kind of movement is monitored equally. Moving pixels
group into object-sized blobs (connected components over a per-pixel
speed threshold and a minimum-area gate), and blobs associate
frame-to-frame into per-object tracks by nearest-centroid matching.

A **movement event** is one object's or person's continuous motion: it
opens at their first moving frame and closes at their last. A track
survives a pause of up to `pause_buffer_s` (default **5 seconds**), so
someone who stops and then continues stays a single uninterrupted event;
only a longer pause or leaving the scene ends it. Pause frames inside an
event count as part of it; quiet frames after the last motion do not.
Tracks shorter than `min_track_frames` are treated as noise. Every event
is recorded in the manifest with its frame range and ISO start/end
timestamps, and the per-frame `movement_detected` array is derived from
the event spans.

### Store layout (OME-Zarr / NGFF 0.4, Zarr v2)

```
cam01.zarr/
  .zgroup, .zattrs        multiscales (axes t,c,z,y,x; t-scale = 1/fps)
                          + "cctv" namespace: the chunk manifest
  0/                      image, uint8, (t, c, z, y, x); t-chunk = 1 GOP
  movement_detected/      uint8 per frame — binary category:
                          0 = no movement, 1 = movement detected
  timestamps/             float64 POSIX seconds (UTC) per frame
```

Each manifest entry records the chunk's frame range, **ISO start/end
timestamps**, its binary `movement_detected` flag, and its trigger:

```json
{"index": 1, "start_frame": 10, "end_frame": 20,
 "start_time": "2026-07-20T14:00:00.400000+00:00",
 "end_time": "2026-07-20T14:00:00.800000+00:00",
 "movement_detected": 1, "trigger": "motion"}
```

The store is written by a small built-in Zarr v2 writer (numpy + zlib,
no zarr-python dependency), and is readable by zarr-python, ome-zarr-py,
napari, or any conforming client.

### Chunk-level cloud query

One file = one object in GCS, so a query maps to a minimal set of
downloads. `QueryClient.from_archive` fetches metadata only (a few KB);
`select(start=…, end=…, movement=True)` resolves the time window and/or
movement category against the manifest; `fetch()` downloads exactly the
matching GOP-chunk objects and returns the frames of just that section.

```python
from cctv_zarr import CloudArchive, MockStorageClient, QueryClient

client  = MockStorageClient("mock_gcs")          # or google.cloud.storage.Client()
archive = CloudArchive(client, "cctv-archive", "sites/gate3")
q   = QueryClient.from_archive(archive, "cam01.zarr", "query_cache")
sel = q.select(start="2026-07-20T14:03:00+00:00",
               end="2026-07-20T14:07:00+00:00", movement=True)
frames, stamps = q.fetch(sel)                    # only these chunks move
q.export_mp4(sel, "event.mp4")
```

## Install

```bash
uv sync                         # or: pip install -e .
uv sync --extra gcs             # real Google Cloud Storage client
uv sync --extra dev             # pytest
```

## CLI

```bash
# video file -> OME-Zarr store -> cloud archive
cctv-zarr --config config.yaml ingest cam01.mp4 \
    --start-time 2026-07-20T14:00:00+00:00 --upload

# list + fetch only the matching section from the archive
cctv-zarr --config config.yaml query cam01.zarr \
    --start 2026-07-20T14:00:01+00:00 --end 2026-07-20T14:00:04+00:00 \
    --movement-only --export-mp4 event.mp4

# a store on disk instead of the archive
cctv-zarr query stores/cam01.zarr --local --movement-only
```

## Compression, highlighted

Every ingest measures and records its own compression in the store
manifest (`cctv.compression`): the stored size, the raw uncompressed
size at source resolution, the source file size, the lossless codec
ratio (zlib at the configured level, vs raw frames at the stored
resolution), and the overall **footprint ratio** (resize + lossless
compression together). The panel keeps the number in view: the
Library header aggregates the whole archive ("Altogether: N MB of raw
frames stored in M MB - Xx smaller than raw"), every Library row and
every batch result carries its own "Xx smaller than raw", and a
dedicated **Results** tab shows uncompressed vs stored storage: a
hero figure with the aggregate ratio ("25.4x smaller - 618 MB stored
instead of 15.3 GB uncompressed") over paired horizontal bars per
recording, uncompressed against stored on a shared scale. The
figures are honest by construction - the baseline is raw frames, and
the source file size is stored alongside so the store is never
misrepresented as smaller than the already-H.264-encoded original.

For demos, `config.demo.yaml` (also mirrored into
`config.pythonanywhere.yaml`) trades fidelity for speed and footprint:
frames stored at 320 px wide with zlib level 6, motion analysis at
128x96, a 3000-frame per-video cap, and lighter previews - on
1080p source footage the footprint ratio lands around 25-40x.

## Network-loss redundancy

Every archive transfer retries with bounded exponential backoff, and
uploads are resumable: a journal beside each store records which objects
have landed, so a connection drop mid-upload loses nothing - the footage
stays intact on local disk and `cctv-zarr push` re-sends only the
missing objects (`cctv-zarr ingest --upload` reports pending counts
instead of failing). Interrupted queries resume too: chunk objects
already in the local cache are never re-fetched.

## Web panel

```bash
uv sync --extra ui
cctv-zarr-ui --config config.yaml     # open http://127.0.0.1:8432
```

Four destinations (Add footage, Library, Search, Results) in a fixed
49px tab bar that becomes a 260-320px sidebar on wide screens. Videos are
picked entirely by clicking: a "Choose files" button opens the native
file dialog (with a "Choose a folder" variant and drag-and-drop onto
the panel), picked files copy to the archive machine in bounded
chunked uploads with a progress bar, and processing starts
automatically when the copy finishes. A per-video size cap
(`ui.max_upload_mb`, off by default, 50 MB in the demo configs) is
shown in the drop zone, enforced in the browser before any copying
starts, and enforced again server-side.

The panel is multi-user by design, sized for shared demos: each
batch goes into a bounded waiting line (`ui.max_queued_jobs`)
served by one worker, so two people pressing "Add" at the same
moment both succeed - the second sees "Waiting in line - 1 task(s)
ahead" instead of an error, and nothing fights for the CPU. Every
browser polls its own job by id and auto-saves its own panel state
(a per-browser client id), so concurrent visitors never see each
other's progress bars or overwrite each other's saved searches. The same view also browses
the server filesystem from the configured video directory: folders open
in place, video rows are selectable (44px targets, size shown, an
"Ingested" badge on files that already have a store), and a batch runs
either the explicit selection or the whole folder - sequentially, with
a live progress bar, the current filename, and per-video results in
which one corrupt file fails alone without stopping the rest.
The Query view starts from a tap-to-choose
recording list - every ingested video, including panel uploads,
appears as a selectable 44px row with its movement summary, so
nothing is ever typed - and renders an
activity ring for the movement ratio, a single-series GOP-chunk timeline
(movement chunks in the series hue, quiet chunks neutral, 2px gaps), and
a 44px-row chunk list with per-chunk timestamps; tapping a chunk opens an
explicit-dismiss sheet with a frame-by-frame preview of just that
section, and right-click/long-press offers a context menu (view section,
export chunk). Built to platform interface guidelines: system typography
at 17px base with fluid headings, light and dark modes at >= 4.5:1 text
contrast, safe-area insets, 44px touch targets, pill toggles, contained
scroll views, spinners deferred one second, and implicit auto-save of
panel state (30 s interval, restored on load - no manual save button).

## Configuration

Every tunable lives in `config.yaml`: `gop` (encoder GOP length, frame
cap), `flow` (Farnebäck parameters, motion threshold, minimum object
size, the 5 s pause buffer, association radius, noise debounce),
`zarr` (resize, grayscale, compression), `archive` (bucket,
prefix, mock switch), `runtime` (paths, fps fallback, failure
budget), `ui` (host/port, previews, the per-video upload cap, and
the batch waiting-line depth for concurrent users).

## Deploying via GitHub

Three GitHub Actions workflows ship with the repo (`.github/workflows/`):

- **ci.yml** - on every push and pull request: the repo style gate
  (`scripts/check_style.py`: 79-column limit, no tabs, no comments
  outside docstrings, and static NASA Power of 10 checks on the
  library - no recursion, no while loops so every loop carries an
  explicit fixed bound, no function over 60 lines) plus the full
  offline test suite on Python 3.11 and 3.12.
- **release.yml** - on pushing a tag like `v0.1.0`: re-runs the tests,
  builds the wheel and sdist, attaches them to a GitHub Release, and
  builds/pushes a Docker image to GitHub Container Registry as
  `ghcr.io/<owner>/cctv_zarr:<version>` and `:latest`. Uses only the
  built-in `GITHUB_TOKEN` - no secrets to configure.
- **demo-pages.yml** - on every push to main: builds a static,
  backend-free demo of the web panel (canned data, full click-through)
  and publishes it to GitHub Pages. Enable Pages once in the repo
  settings (Settings -> Pages -> Source: GitHub Actions).

First-time setup:

```bash
cd cctv_zarr
git init && git add -A && git commit -m "cctv_zarr v0.1.0"
git remote add origin git@github.com:<you>/cctv_zarr.git
git push -u origin main
git tag v0.1.0 && git push origin v0.1.0
```

Run the released container anywhere (footage and stores live in a
mounted volume; the panel binds 0.0.0.0 inside the container - keep it
behind your own network controls):

```bash
docker run -p 8432:8432 -v $PWD/data:/data \
    ghcr.io/<you>/cctv_zarr:latest
```

Badge URLs in this README assume the `Ikram-c/cctv_zarr` repository
path; adjust them if the repo lives elsewhere.

## Testing

```bash
uv run pytest        # 43 tests, fully offline (synthetic scenes, mock GCS)
```

`tests/test_flow.py` proves a growing object triggers and lateral/static
scenes do not; `tests/test_query_archive.py` proves a movement query
downloads only the flagged chunk objects and nothing else.

## EU / GDPR posture

Compliance capabilities ship built in and default-on - see
`COMPLIANCE.md` for the full mapping to the review findings.
Retention is enforced chunk-by-chunk from manifest timestamps
(`retention:` config; `cctv-zarr prune`, journalled), with
separate clocks for movement and quiet footage and an option to
never store quiet chunks at all. Subject-access and erasure
tooling exist (`cctv-zarr access-request` with redaction masks,
`cctv-zarr erase` across store, archive, and cache). The panel
requires an access code when configured, refuses to leave loopback
without one, confines browsing/ingest to the video area, tombstones
instead of destroying stores on re-ingest, and access-logs every
query, preview, and export. Archive objects can be encrypted
client-side, bucket location is verified (Chapter V), and every
store carries governance metadata including a non-biometric-source
declaration backed by an acceptable-use clause in the LICENSE.
Deployment contexts matter: workplace and in-vehicle uses engage
national procedural gates (works councils, inspectorates, sectoral
rules) that are the operator's responsibility, and
`flow.record_events: false` is the minimising setting for those
contexts. None of this is legal advice.

## Deploying on Google Compute Engine (free trial)

`deploy/gce/` holds a complete deployment sized for a free-tier
e2-micro with a 30 GB disk: `setup_gce.sh` (swap, headless-OpenCV
venv via the `[gce]` extra, systemd service, daily retention prune
timer), `config.gce.yaml` (512 MB upload cap, tight retention
clocks, loopback + SSH-tunnel access), and a README with the full
disk budget - venv, the largest video, its store, the demo archive
copy, and exports all total under 8 GB of the 30 GB disk.

## Known limitations

- The GOP grid is configured, not parsed from the bitstream; set
  `gop.gop_frames` to your camera's actual GOP length so archive chunks
  truly align with encoder IDRs.
- One store per camera per recording session; multi-camera collections
  are a directory of stores under one archive prefix.
- Frames are stored as pixels (uint8, zlib), not as H.264 bitstream —
  chunk objects are larger than the source video. If storage cost
  dominates, keep the source MP4 alongside and treat the store as the
  queryable index + export path.
