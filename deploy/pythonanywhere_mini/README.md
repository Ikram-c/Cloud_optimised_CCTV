# cctv_zarr mini - a free-tier PythonAnywhere demo that fits

A miniature deployment sized so **both the virtualenv and real
video processing fit a free (Beginner) account**: 512 MB disk,
one web app, and a small daily CPU allowance. Two tricks make it
fit. The virtualenv is created with `--system-site-packages`, so
the heavy packages PythonAnywhere already ships (numpy, OpenCV)
are reused instead of reinstalled - the venv itself stays in the
tens of megabytes. And `config.mini.yaml` shrinks the processing
envelope: 240 px stored frames, 96x72 motion analysis, a
300-frame per-video cap, a 25 MB upload cap, 12-frame previews,
and tight retention clocks so nothing accumulates.

Measured on the bundled synthetic clip (299 frames, 320x240):
ingest completes in ~2 CPU-seconds with an 80 MB peak - a few
seconds of a free account's daily allowance - and produces a
~15 MB store with 6 movement sections and exactly one movement
event (the clip's mid-walk pause is bridged by the 5-second
buffer, so Search shows the pause-and-continue story).

## Deploy (about 10 minutes)

1. Sign up at pythonanywhere.com (free Beginner account).
2. **Files** tab: upload `cctv_zarr.zip` to your home directory.
3. **Consoles** tab: open a Bash console:

   ```bash
   unzip cctv_zarr.zip
   bash cctv_zarr/deploy/pythonanywhere_mini/setup_mini.sh
   ```

   The script builds the small venv, installs only what the
   system does not already provide, **generates an access code**
   (printed at the end - your URL is public, so the panel
   requires it), then generates the synthetic sample clip and
   pre-processes it so the Library, Search, and Results tabs have
   data before anyone visits.

4. **Web** tab: Add a new web app -> Manual configuration -> the
   Python version matching the console's `python3 -V`.
5. Virtualenv section: `/home/<you>/.virtualenvs/cctvmini`
6. Open the WSGI configuration file link and replace its whole
   contents with:

   ```python
   import sys
   sys.path.insert(
       0, "/home/<you>/cctv_zarr/deploy/pythonanywhere_mini",
   )
   from wsgi_app import application
   ```

7. Reload. The panel is live at
   `https://<you>.pythonanywhere.com` behind the access code.

## Budgets

Disk (512 MB quota):

| Item | Size |
|---|---|
| venv (`--system-site-packages`; worst case if pip must install OpenCV+numpy itself) | ~40 MB (~260 MB worst case) |
| Repo code | ~2 MB |
| Sample clip | ~0.2 MB |
| Its store | ~15 MB |
| One 25 MB uploaded clip + its store | ~45 MB |
| **Typical total** | **~100 MB** |

CPU (small daily allowance, then throttled to the tortoise
queue): the seeded sample costs a few seconds; each uploaded clip
(capped at 25 MB / 300 frames) costs roughly 10-40 s depending on
resolution. Budget one or two live ingests per day and let
visitors browse, search, and view Results - those endpoints read
kilobytes of metadata and cost effectively nothing.

## Demo flow

Open the URL, enter the access code, then: **Library** (the
sample store with its "smaller than raw" line) -> **Results**
(uncompressed-vs-stored bars) -> **Search** (tap the recording:
activity ring, one movement event spanning the pause, timeline,
tap a section for the frame preview). To show live processing,
drop one short clip on Add footage - under the 25 MB cap - and
watch the progress bar.

## Notes

- After the seeded ingest the sample source file moves to
  `videos/_ingested/` (the mini config quarantines sources), so
  the folder browser starts empty - that is the retention policy
  working, not a bug.
- The daily prune has no timer on PythonAnywhere; free accounts
  get one daily **Scheduled task** - point it at:
  `~/.virtualenvs/cctvmini/bin/python -m cctv_zarr.cli --config
  /home/<you>/cctv_zarr/config.mini.yaml prune`
- Free web apps sleep after three months unless you press "Run
  until 3 months from today" on the Web tab.
- Compression honesty: on the low-texture synthetic clip the
  footprint ratio is ~5x; on real 1080p footage the same settings
  land far higher. The Results tab's baseline note applies.
