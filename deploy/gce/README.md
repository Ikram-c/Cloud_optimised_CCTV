# Deploying cctv_zarr on a Google Compute Engine trial instance

Sized for the free tier / trial: an **e2-micro** (2 shared vCPU,
1 GB RAM) with a **30 GB** standard persistent disk in one of the
free-tier regions (us-west1, us-central1, us-east1). Everything -
virtualenv, the largest permitted video, its store, the demo
archive copy, exports, and quarantine - fits in under a quarter of
the disk, and the retention timer keeps it that way.

## 1. Create the instance

```bash
gcloud compute instances create cctv-demo \
    --machine-type=e2-micro \
    --zone=us-central1-a \
    --image-family=debian-12 --image-project=debian-cloud \
    --boot-disk-size=30GB --boot-disk-type=pd-standard
```

(On trial credit, `e2-small` [2 GB RAM] is more comfortable and
still costs pennies; nothing else changes.)

## 2. Install

```bash
gcloud compute scp cctv_zarr.zip cctv-demo:~ --zone=us-central1-a
gcloud compute ssh cctv-demo --zone=us-central1-a
# on the VM:
sudo apt-get install -y unzip && unzip cctv_zarr.zip
bash cctv_zarr/deploy/gce/setup_gce.sh
```

The script adds 2 GB of swap (pip needs it on 1 GB RAM), creates
the venv at `/opt/cctv_zarr/venv` using the `[gce]` extra -
**headless** OpenCV, so no GUI libraries and a smaller footprint -
copies `config.gce.yaml` into place, and installs two systemd
units: the panel (loopback, auto-restart, capped at 700 MB RAM)
and a daily `cctv-zarr prune` timer.

## 3. Reach the panel (secure by default)

The panel binds loopback only. From your own machine:

```bash
gcloud compute ssh cctv-demo --zone=us-central1-a \
    -- -N -L 8432:127.0.0.1:8432
```

then open <http://127.0.0.1:8432>. The SSH tunnel is encrypted end
to end and nothing is exposed to the internet - the right default
for footage of people (GDPR Art. 32). To serve it publicly
instead, set `security.auth_token`, `security.allow_remote: true`,
and put a TLS terminator (e.g. Caddy) in front; the panel refuses
to leave loopback without both.

## Disk budget (30 GB disk)

| Item | Size |
|---|---|
| Debian 12 base image | ~2.5 GB |
| Swap file | 2.0 GB |
| Virtualenv (`[gce]`: headless OpenCV, numpy, FastAPI, uvicorn, cryptography) | ~0.6 GB |
| Largest single video (`ui.max_upload_mb: 512`) | 0.5 GB |
| Its store at demo settings (320 px, zlib-6, 3,000-frame cap) | ~0.3 GB |
| Mock-archive copy when demoing upload/query | ~0.3 GB |
| Exports, query cache, quarantined sources (all on retention clocks) | ~1.5 GB |
| **Total worst case** | **~7.7 GB** |

That leaves >22 GB headroom - room for a dozen demo videos at
once. Three mechanisms keep it bounded permanently: the 512 MB
per-video upload cap (enforced in the browser and server-side),
`source_after_ingest: quarantine` plus short retention clocks
(7 days movement / 48 h quiet / 24 h exports and cache), and the
daily prune timer that enforces them (journalled, so deletion is
provable).

RAM budget: the 320 px pipeline peaks well under 300 MB (one GOP
of frames in memory at a time); the service cap is 700 MB, and
swap absorbs pip's install-time spike. Ingest of a 512 MB video
takes a few minutes on shared vCPU - the panel's queue and
progress bar handle that gracefully; pre-ingest before a live
demo, exactly as with any small host.

## Compression demo numbers

At `config.gce.yaml` settings, 1080p source footage lands around
25-40x smaller than raw frames in the Results tab; the store for
a 512 MB source file is typically 150-350 MB (the "extra space
from demoing the compression" line in the budget). The honest
baseline note applies: the ratio is against raw frames, and the
source file size is recorded in every store manifest.

## GDPR posture on this deployment

Mock archive keeps footage on the VM (no Chapter V transfer);
loopback + SSH tunnel avoids public exposure; retention, the
prune timer, upload caps, quarantined sources, and journalled
deletion ship enabled in `config.gce.yaml`. Fill in the
`governance:` block with the real controller before pointing any
real camera at it, and see COMPLIANCE.md at the repo root for the
full mapping.
