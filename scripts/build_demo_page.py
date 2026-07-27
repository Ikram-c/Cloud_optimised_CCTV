#!/usr/bin/env python3
"""Build a static GitHub Pages demo of the CCTV Archive panel.

Extracts the panel HTML from webapp/server.py textually (no runtime
dependencies beyond the standard library) and injects a canned-data
fetch stub, so the deployed page demonstrates the full click-through
flow - folder browsing, batch progress, movement search with the
activity ring, timeline, and per-object movement list - without any
backend.

Usage:
    python scripts/build_demo_page.py --out site
"""

import argparse
import sys
from pathlib import Path

SERVER_PATH = (
    Path(__file__).resolve().parent.parent
    / "src" / "cctv_zarr" / "webapp" / "server.py"
)
HTML_START = '_INDEX_HTML = """'
HTML_END = '"""\n'

DEMO_STUB = """
<script>
const DEMO_NOTE = "Static demo - data is canned, no backend attached.";
const CANNED = {
  "/api/status": {state: "done", total: 3, done: 3, current: "",
    items: [
    {video: "gate_cam_morning.mp4", store: "gate_cam_morning.zarr",
     seconds: 148.0, frames: 3700, chunks: 124,
     movement_chunks: 11, events: 3},
    {video: "car_park_east.mp4", store: "car_park_east.zarr",
     seconds: 60.0, frames: 1500, chunks: 50,
     movement_chunks: 0, events: 0},
    {video: "loading_bay.mp4", store: "loading_bay.zarr",
     seconds: 95.2, frames: 2380, chunks: 80,
     movement_chunks: 6, events: 2}],
    error: null},
  "/api/prefs": {prefs: {tab: "query", store: "gate_cam_morning.zarr"}},
  "/api/stores": {stores: [
    {name: "gate_cam_morning.zarr", chunks: 124, movement_chunks: 11,
     movement_ratio: 0.089, seconds: 148.0, size_mb: 301.5},
    {name: "car_park_east.zarr", chunks: 50, movement_chunks: 0,
     movement_ratio: 0, seconds: 60.0, size_mb: 122.9},
    {name: "loading_bay.zarr", chunks: 80, movement_chunks: 6,
     movement_ratio: 0.075, seconds: 95.2, size_mb: 194.0}]},
  "/api/browse": {path: "/data/videos", parent: "/data",
    folders: ["gate", "car_park"],
    videos: [
      {name: "gate_cam_morning.mp4", path: "/v/1.mp4",
       size_mb: 220.4, ingested: true},
      {name: "car_park_east.mp4", path: "/v/2.mp4",
       size_mb: 198.1, ingested: true},
      {name: "loading_bay.mp4", path: "/v/3.mp4",
       size_mb: 305.7, ingested: true}]},
};
function pad(n) { return String(n).padStart(2, "0"); }
function iso(minute, second) {
  return "2026-07-20T14:" + pad(minute) + ":" + pad(second) + "+00:00";
}
function chunks() {
  const out = [];
  for (let i = 0; i < 40; i++) {
    const mv = (i >= 6 && i <= 9) || (i >= 24 && i <= 28) ? 1 : 0;
    const s0 = i * 6;
    const s1 = s0 + 6;
    out.push({index: i, start_frame: i * 30, end_frame: (i + 1) * 30,
      start_time: iso(Math.floor(s0 / 60), s0 % 60),
      end_time: iso(Math.floor(s1 / 60), s1 % 60),
      movement_detected: mv,
      trigger: mv && (i === 6 || i === 24) ? "motion" : "gop"});
  }
  return out;
}
const EVENTS = [
  {start_time: iso(0, 36), end_time: iso(1, 0), seconds: 24.0},
  {start_time: iso(2, 24), end_time: iso(2, 54), seconds: 30.0},
];
window.fetch = function (path, options) {
  let body;
  if (path === "/api/query") {
    const all = chunks();
    const recs = all.filter(c => c.movement_detected === 1);
    body = {records: recs, all_chunks: all, events: EVENTS,
            matched: recs.length, total: all.length};
  } else if (path === "/api/ingest" || path === "/api/upload"
             || path === "/api/export" || path === "/api/frames") {
    return Promise.resolve({ok: false, json: () =>
      Promise.resolve({detail: DEMO_NOTE})});
  } else {
    body = CANNED[path] || {};
  }
  return Promise.resolve({ok: true,
    json: () => Promise.resolve(body)});
};
</script>
"""


def extract_panel_html() -> str:
    """Pull the panel HTML constant out of server.py textually.

    Returns:
        str: The panel document.

    Raises:
        SystemExit: If the constant cannot be located.
    """
    text = SERVER_PATH.read_text(encoding="utf-8")
    start = text.find(HTML_START)
    if start < 0:
        raise SystemExit("panel HTML not found in server.py")
    start += len(HTML_START)
    end = text.find(HTML_END, start)
    if end < 0:
        raise SystemExit("panel HTML terminator not found")
    return text[start:end]


def build(out_dir: Path) -> Path:
    """Write the demo site.

    Args:
        out_dir (Path): Output directory.

    Returns:
        Path: The written index.html.
    """
    html = extract_panel_html()
    marker = '<script>\n"use strict";'
    if marker not in html:
        raise SystemExit("panel script marker not found")
    html = html.replace(marker, DEMO_STUB + marker)
    html = html.replace("\\\\u", "\\u")
    out_dir.mkdir(parents=True, exist_ok=True)
    index = out_dir / "index.html"
    index.write_text(html, encoding="utf-8")
    return index


def main() -> int:
    """Entry point."""
    parser = argparse.ArgumentParser(description="Build the Pages demo")
    parser.add_argument("--out", type=Path, default=Path("site"))
    args = parser.parse_args()
    index = build(args.out)
    print(f"demo written: {index}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
