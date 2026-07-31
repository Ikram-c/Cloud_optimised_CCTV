#!/bin/bash
set -euo pipefail

APP_DIR="/opt/cctv_zarr"
REPO_DIR="$(cd "$(dirname "$0")/../.." && pwd)"
SWAP_FILE="/swapfile"

echo "== cctv_zarr GCE setup (e2-micro friendly) =="

if [ ! -f "$SWAP_FILE" ]; then
  echo "-- adding 2G swap (1 GB RAM instances need it for pip)"
  sudo fallocate -l 2G "$SWAP_FILE"
  sudo chmod 600 "$SWAP_FILE"
  sudo mkswap "$SWAP_FILE"
  sudo swapon "$SWAP_FILE"
  echo "$SWAP_FILE none swap sw 0 0" | sudo tee -a /etc/fstab
fi

echo "-- system packages"
sudo apt-get update -q
sudo apt-get install -y -q python3-venv python3-pip

echo "-- application layout"
sudo mkdir -p "$APP_DIR/data/videos" "$APP_DIR/data/stores" \
             "$APP_DIR/data/mock_gcs"
sudo chown -R "$USER":"$USER" "$APP_DIR"

echo "-- virtual environment (headless OpenCV, no GUI libs)"
python3 -m venv "$APP_DIR/venv"
"$APP_DIR/venv/bin/pip" install --upgrade pip
"$APP_DIR/venv/bin/pip" install --no-cache-dir "$REPO_DIR[gce]"

echo "-- config"
cp -n "$REPO_DIR/config.gce.yaml" "$APP_DIR/config.yaml" || true

echo "-- systemd units (panel + daily retention prune)"
sudo cp "$REPO_DIR/deploy/gce/cctv-zarr.service" \
        "$REPO_DIR/deploy/gce/cctv-zarr-prune.service" \
        "$REPO_DIR/deploy/gce/cctv-zarr-prune.timer" \
        /etc/systemd/system/
sudo sed -i "s|__USER__|$USER|g" /etc/systemd/system/cctv-zarr*.service
sudo systemctl daemon-reload
sudo systemctl enable --now cctv-zarr.service
sudo systemctl enable --now cctv-zarr-prune.timer

echo "-- disk budget after install"
df -h /
du -sh "$APP_DIR/venv"

echo
echo "Done. Reach the panel from your machine with:"
echo "  gcloud compute ssh \$(hostname) -- -N -L 8432:127.0.0.1:8432"
echo "then open http://127.0.0.1:8432"
