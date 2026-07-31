"""Web panel API tests. Fully offline; ingest exercised end-to-end."""

import base64
import time
from datetime import datetime, timedelta, timezone
from pathlib import Path

import pytest
import yaml

fastapi = pytest.importorskip("fastapi")
from fastapi.testclient import TestClient

from cctv_zarr.writer import CctvZarrWriter
from cctv_zarr.config import Settings
from cctv_zarr.webapp.server import create_app

from conftest_util import approach_frames, static_frames, write_video

CONFIG = Path(__file__).parent.parent / "config.yaml"
T0 = datetime(2026, 7, 20, 14, 0, tzinfo=timezone.utc)
DEADLINE_S = 30.0


def _write_config(tmp_path: Path) -> Path:
    """Write a test config rooted at tmp_path.

    Args:
        tmp_path (Path): Test-scoped directory.

    Returns:
        Path: The written config path.
    """
    raw = yaml.safe_load(CONFIG.read_text())
    raw["gop"]["gop_frames"] = 10
    raw["runtime"]["store_directory"] = str(tmp_path / "stores")
    raw["runtime"]["video_directory"] = str(tmp_path)
    raw["archive"]["local_root"] = str(tmp_path / "mock_gcs")
    tmp_path.mkdir(parents=True, exist_ok=True)
    path = tmp_path / "config.yaml"
    path.write_text(yaml.safe_dump(raw))
    return path


@pytest.fixture
def env(tmp_path):
    config_path = _write_config(tmp_path)
    settings = Settings.load(config_path)
    frames = static_frames(12) + approach_frames(16) + static_frames(12)
    video = write_video(tmp_path / "cam01.mp4", frames)
    store = Path(settings.runtime.store_directory) / "cam01.zarr"
    CctvZarrWriter(settings).ingest(video, store, start_time=T0)
    return {
        "client": TestClient(create_app(config_path)),
        "video": video,
        "tmp": tmp_path,
    }


def _wait_terminal(client) -> str:
    """Poll the status endpoint until the job settles.

    Args:
        client: The TestClient.

    Returns:
        str: Terminal state.
    """
    deadline = time.time() + DEADLINE_S
    state = "running"
    while time.time() < deadline:
        state = client.get("/api/status").json()["state"]
        if state in ("done", "failed"):
            break
        time.sleep(0.1)
    return state


class TestIndex:
    def test_serves_panel_markers(self, env):
        text = env["client"].get("/").text
        for marker in ("CCTV Archive", "Add footage", "Library",
                       "Search footage", "Choose files",
                       "Drop videos here",
                       "Only show movement", "tab-bar",
                       ">Results<", "Storage savings",
                       "view-storage", "storage-bars",
                       "Uncompressed", "Stored",
                       "upload-limit", "Waiting in line",
                       "/api/config", "query-store-list",
                       "tap to choose"):
            assert marker in text

    def test_no_technical_jargon_in_labels(self, env):
        text = env["client"].get("/").text
        for jargon in (">Ingest<", ">Stores<", ">Query<",
                       "Window start (ISO)", "Export MP4",
                       ">Chunk<", "movement chunks<"):
            assert jargon not in text

    def test_guideline_rules_present(self, env):
        text = env["client"].get("/").text
        for rule in ("-apple-system", "font-size: 17px",
                     "clamp(17px, 0.5vw + 15px, 19px)",
                     "clamp(84px, 12vw, 112px)",
                     "min-width: 1280px",
                     "--series-raw",
                     "min-height: 44px",
                     "env(safe-area-inset-bottom)",
                     "border-radius: 22.5%",
                     "minmax(260px, 320px) 1fr",
                     "overscroll-behavior: contain",
                     "stroke-dasharray: 100",
                     "prefers-color-scheme: dark",
                     "z-index: 50"):
            assert rule in text


class TestStatusAndStores:
    def test_initial_state_idle(self, env):
        body = env["client"].get("/api/status").json()
        assert body["state"] == "idle"

    def test_stores_lists_summary(self, env):
        body = env["client"].get("/api/stores").json()
        assert len(body["stores"]) == 1
        store = body["stores"][0]
        assert store["name"] == "cam01.zarr"
        assert store["chunks"] == 4
        assert store["movement_chunks"] >= 1
        assert 0.0 <= store["movement_ratio"] <= 1.0

    def test_stores_carry_compression(self, env):
        store = env["client"].get("/api/stores").json()["stores"][0]
        block = store["compression"]
        assert block["footprint_ratio"] > 1.0
        assert block["raw_mb"] > block["stored_mb"]

    def test_compression_highlight_served(self, env):
        text = env["client"].get("/").text
        assert "smaller than raw" in text
        assert "library-compression" in text


class TestBrowseEndpoint:
    def test_default_folder_lists_videos(self, env):
        body = env["client"].post("/api/browse", json={}).json()
        names = [v["name"] for v in body["videos"]]
        assert "cam01.mp4" in names

    def test_marks_already_ingested(self, env):
        body = env["client"].post("/api/browse", json={}).json()
        by_name = {v["name"]: v for v in body["videos"]}
        assert by_name["cam01.mp4"]["ingested"] is True

    def test_lists_subfolders(self, env):
        (env["tmp"] / "night").mkdir()
        body = env["client"].post("/api/browse", json={}).json()
        assert "night" in body["folders"]
        assert body["parent"]

    def test_missing_folder_is_400(self, env):
        res = env["client"].post(
            "/api/browse", json={"path": "/nowhere/at/all"},
        )
        assert res.status_code == 400


class TestUploadEndpoint:
    def _send_parts(self, client, name, payload, upload_id="u1"):
        half = len(payload) // 2
        parts = [payload[:half], payload[half:]]
        result = None
        for seq, chunk in enumerate(parts):
            result = client.post("/api/upload", json={
                "upload_id": upload_id,
                "name": name,
                "seq": seq,
                "last": seq == len(parts) - 1,
                "data": base64.b64encode(chunk).decode("ascii"),
            })
        return result

    def test_chunked_upload_roundtrip(self, env):
        payload = b"fake video bytes" * 100
        res = self._send_parts(env["client"], "picked.mp4", payload)
        assert res.status_code == 200
        body = res.json()
        assert body["done"] is True
        saved = Path(body["path"])
        assert saved.name == "picked.mp4"
        assert saved.read_bytes() == payload

    def test_collision_gets_new_name(self, env):
        payload = b"bytes" * 20
        first = self._send_parts(
            env["client"], "cam01.mp4", payload, upload_id="ua",
        ).json()
        assert Path(first["path"]).name == "cam01_1.mp4"

    def test_non_video_extension_is_400(self, env):
        res = env["client"].post("/api/upload", json={
            "upload_id": "ub", "name": "notes.txt", "seq": 0,
            "last": True, "data": "",
        })
        assert res.status_code == 400

    def test_traversal_name_is_400(self, env):
        res = env["client"].post("/api/upload", json={
            "upload_id": "uc", "name": "../evil.mp4", "seq": 0,
            "last": True, "data": "",
        })
        assert res.status_code == 400

    def test_out_of_order_part_is_400(self, env):
        res = env["client"].post("/api/upload", json={
            "upload_id": "ud", "name": "late.mp4", "seq": 3,
            "last": True, "data": "",
        })
        assert res.status_code == 400

    def test_bad_base64_is_400(self, env):
        res = env["client"].post("/api/upload", json={
            "upload_id": "ue", "name": "bad.mp4", "seq": 0,
            "last": True, "data": "not base64!!!",
        })
        assert res.status_code == 400

    def test_uploaded_video_ingests(self, env, tmp_path):
        frames = static_frames(10) + approach_frames(10)
        source = write_video(tmp_path / "source_clip.mp4", frames)
        payload = source.read_bytes()
        body = self._send_parts(
            env["client"], "clicked.mp4", payload, upload_id="uf",
        ).json()
        res = env["client"].post("/api/ingest", json={
            "videos": [body["path"]],
        })
        assert res.status_code == 200
        assert _wait_terminal(env["client"]) == "done"
        names = [
            s["name"] for s in
            env["client"].get("/api/stores").json()["stores"]
        ]
        assert "clicked.zarr" in names


class TestIngestEndpoint:
    def test_selected_videos_batch(self, env, tmp_path):
        frames = static_frames(10) + approach_frames(10)
        videos = [
            write_video(tmp_path / f"cam0{i}.mp4", frames)
            for i in (2, 3)
        ]
        res = env["client"].post("/api/ingest", json={
            "videos": [str(v) for v in videos],
        })
        assert res.status_code == 200
        assert res.json()["count"] == 2
        assert _wait_terminal(env["client"]) == "done"
        status = env["client"].get("/api/status").json()
        assert status["done"] == 2 and status["total"] == 2
        assert all("error" not in item for item in status["items"])
        names = [
            s["name"] for s in
            env["client"].get("/api/stores").json()["stores"]
        ]
        assert "cam02.zarr" in names and "cam03.zarr" in names

    def test_whole_folder_batch(self, env, tmp_path):
        folder = tmp_path / "batchdir"
        folder.mkdir()
        frames = static_frames(10) + approach_frames(10)
        for name in ("a.mp4", "b.mp4"):
            write_video(folder / name, frames)
        (folder / "notes.txt").write_text("not a video")
        res = env["client"].post(
            "/api/ingest", json={"folder": str(folder)},
        )
        assert res.status_code == 200
        assert res.json()["count"] == 2
        assert _wait_terminal(env["client"]) == "done"
        status = env["client"].get("/api/status").json()
        assert [item["video"] for item in status["items"]] == [
            "a.mp4", "b.mp4",
        ]

    def test_batch_tolerates_bad_item(self, env, tmp_path):
        frames = static_frames(10) + approach_frames(10)
        good = write_video(tmp_path / "good.mp4", frames)
        bad = tmp_path / "bad.mp4"
        bad.write_bytes(b"not a real video")
        res = env["client"].post("/api/ingest", json={
            "videos": [str(bad), str(good)],
        })
        assert res.status_code == 200
        assert _wait_terminal(env["client"]) == "done"
        items = env["client"].get("/api/status").json()["items"]
        assert "error" in items[0]
        assert "error" not in items[1]

    def test_missing_video_is_400(self, env):
        res = env["client"].post("/api/ingest", json={
            "videos": ["/nowhere/ghost.mp4"],
        })
        assert res.status_code == 400

    def test_empty_selection_is_400(self, env):
        res = env["client"].post("/api/ingest", json={})
        assert res.status_code == 400


class TestQueryEndpoint:
    def test_movement_only_filters(self, env):
        body = env["client"].post("/api/query", json={
            "store": "cam01.zarr", "movement_only": True,
        }).json()
        assert body["total"] == 4
        assert body["matched"] >= 1
        assert all(
            r["movement_detected"] == 1 for r in body["records"]
        )

    def test_time_window_filters(self, env):
        start = (T0 + timedelta(seconds=0.4)).isoformat()
        end = (T0 + timedelta(seconds=0.9)).isoformat()
        body = env["client"].post("/api/query", json={
            "store": "cam01.zarr", "start": start, "end": end,
        }).json()
        assert [r["index"] for r in body["records"]] == [1, 2]

    def test_events_returned_with_query(self, env):
        body = env["client"].post(
            "/api/query", json={"store": "cam01.zarr"},
        ).json()
        assert "events" in body
        assert len(body["events"]) >= 1
        for event in body["events"]:
            assert event["start_time"] < event["end_time"]
            assert event["seconds"] > 0

    def test_records_carry_timestamps_and_binary_flag(self, env):
        body = env["client"].post(
            "/api/query", json={"store": "cam01.zarr"},
        ).json()
        for record in body["records"]:
            assert record["movement_detected"] in (0, 1)
            assert record["start_time"] < record["end_time"]

    def test_unknown_store_is_404(self, env):
        res = env["client"].post(
            "/api/query", json={"store": "ghost.zarr"},
        )
        assert res.status_code == 404

    def test_traversal_store_name_is_400(self, env):
        res = env["client"].post(
            "/api/query", json={"store": "../evil"},
        )
        assert res.status_code == 400

    def test_bad_timestamp_is_400(self, env):
        res = env["client"].post("/api/query", json={
            "store": "cam01.zarr", "start": "not a time",
        })
        assert res.status_code == 400


class TestFramesAndExport:
    def test_frames_previews_one_chunk(self, env):
        body = env["client"].post("/api/frames", json={
            "store": "cam01.zarr", "chunk_index": 1,
        }).json()
        assert body["chunk"]["index"] == 1
        assert len(body["frames"]) >= 1
        assert all(len(f["jpeg"]) > 100 for f in body["frames"])

    def test_frames_unknown_chunk_is_404(self, env):
        res = env["client"].post("/api/frames", json={
            "store": "cam01.zarr", "chunk_index": 99,
        })
        assert res.status_code == 404

    def test_export_writes_clip(self, env):
        body = env["client"].post("/api/export", json={
            "store": "cam01.zarr", "movement_only": True,
        }).json()
        assert body["frames"] > 0
        assert Path(body["path"]).exists()

    def test_export_empty_selection_is_400(self, env):
        start = (T0 + timedelta(hours=6)).isoformat()
        res = env["client"].post("/api/export", json={
            "store": "cam01.zarr", "start": start,
        })
        assert res.status_code == 400


class TestPanelSecurity:
    def _secured(self, tmp_path):
        config_path = _write_config(tmp_path)
        raw = yaml.safe_load(config_path.read_text())
        raw["security"] = {"auth_token": "secret-code"}
        config_path.write_text(yaml.safe_dump(raw))
        return TestClient(create_app(config_path))

    def test_api_requires_token(self, tmp_path):
        client = self._secured(tmp_path)
        assert client.get("/api/stores").status_code == 401
        assert client.get("/api/status").status_code == 401
        res = client.post(
            "/api/browse", json={},
            headers={"Authorization": "Bearer wrong"},
        )
        assert res.status_code == 401

    def test_token_grants_access(self, tmp_path):
        client = self._secured(tmp_path)
        headers = {"Authorization": "Bearer secret-code"}
        assert client.get(
            "/api/stores", headers=headers,
        ).status_code == 200
        assert client.get("/").status_code == 200

    def test_no_token_config_stays_open(self, env):
        assert env["client"].get("/api/stores").status_code == 200

    def test_browse_confined_to_video_area(self, env):
        res = env["client"].post("/api/browse", json={"path": "/"})
        assert res.status_code == 400
        assert "outside" in res.json()["detail"]

    def test_ingest_confined_to_video_area(self, env, tmp_path):
        outside = tmp_path.parent / "outside.mp4"
        res = env["client"].post("/api/ingest", json={
            "videos": [str(outside)],
        })
        assert res.status_code == 400

    def test_reingest_tombstones_not_destroys(self, env):
        client = env["client"]
        video = env["video"]
        res = client.post("/api/ingest", json={
            "videos": [str(video)],
        })
        assert res.status_code == 200
        assert _wait_terminal(client) == "done"
        stores = Path(env["tmp"]) / "stores"
        replaced = stores / "_replaced" / "cam01.zarr"
        assert replaced.is_dir()
        assert (replaced / ".zattrs").is_file()
        log = (stores / "deletion_log.jsonl").read_text()
        assert "replace" in log

    def test_export_is_access_logged(self, env):
        res = env["client"].post("/api/export", json={
            "store": "cam01.zarr", "movement_only": True,
            "client": "auditme",
        })
        assert res.status_code == 200
        log_path = Path(env["tmp"]) / "stores" / "access_log.jsonl"
        text = log_path.read_text()
        assert '"action": "export"' in text
        assert "auditme" in text

    def test_replaced_stores_hidden_from_listing(self, env):
        client = env["client"]
        client.post("/api/ingest", json={
            "videos": [str(env["video"])],
        })
        assert _wait_terminal(client) == "done"
        names = [
            s["name"] for s in
            client.get("/api/stores").json()["stores"]
        ]
        assert names.count("cam01.zarr") == 1


class TestPrefs:
    def test_roundtrip(self, env):
        saved = env["client"].post("/api/prefs", json={
            "prefs": {"tab": "query", "movement_only": True},
        }).json()
        assert saved["saved"] is True
        body = env["client"].get("/api/prefs").json()
        assert body["prefs"]["tab"] == "query"
        assert body["prefs"]["movement_only"] is True

    def test_empty_when_unsaved(self, env, tmp_path):
        fresh = TestClient(create_app(_write_config(tmp_path / "b")))
        assert fresh.get("/api/prefs").json() == {"prefs": {}}

    def test_per_client_isolation(self, env):
        client = env["client"]
        client.post("/api/prefs", json={
            "prefs": {"tab": "query"}, "client": "alpha",
        })
        client.post("/api/prefs", json={
            "prefs": {"tab": "stores"}, "client": "beta",
        })
        alpha = client.get("/api/prefs?client=alpha").json()
        beta = client.get("/api/prefs?client=beta").json()
        assert alpha["prefs"]["tab"] == "query"
        assert beta["prefs"]["tab"] == "stores"
        assert client.get("/api/prefs").json() == {"prefs": {}}

    def test_bad_client_id_is_422(self, env):
        res = env["client"].post("/api/prefs", json={
            "prefs": {}, "client": "../evil",
        })
        assert res.status_code == 422


class TestDemoLimits:
    def test_config_reports_limits(self, env):
        body = env["client"].get("/api/config").json()
        assert body["max_upload_mb"] == 0
        assert body["max_queued_jobs"] >= 1

    def test_upload_over_config_limit_is_400(self, tmp_path):
        config_path = _write_config(tmp_path)
        raw = yaml.safe_load(config_path.read_text())
        raw["ui"]["max_upload_mb"] = 1
        config_path.write_text(yaml.safe_dump(raw))
        client = TestClient(create_app(config_path))
        payload = b"x" * 1_200_000
        res = client.post("/api/upload", json={
            "upload_id": "ux", "name": "big.mp4", "seq": 0,
            "last": True,
            "data": base64.b64encode(payload).decode("ascii"),
        })
        assert res.status_code == 400
        assert "1 MB limit" in res.json()["detail"]
        leftovers = [
            p for p in tmp_path.iterdir()
            if p.name.startswith(".upload_")
        ]
        assert leftovers == []

    def test_upload_under_config_limit_ok(self, tmp_path):
        config_path = _write_config(tmp_path)
        raw = yaml.safe_load(config_path.read_text())
        raw["ui"]["max_upload_mb"] = 1
        config_path.write_text(yaml.safe_dump(raw))
        client = TestClient(create_app(config_path))
        payload = b"x" * 200_000
        res = client.post("/api/upload", json={
            "upload_id": "uy", "name": "small.mp4", "seq": 0,
            "last": True,
            "data": base64.b64encode(payload).decode("ascii"),
        })
        assert res.status_code == 200
        assert res.json()["done"] is True


class TestJobQueue:
    def test_concurrent_batches_queue_not_409(self, env, tmp_path):
        frames = static_frames(10) + approach_frames(10)
        one = write_video(tmp_path / "q1.mp4", frames)
        two = write_video(tmp_path / "q2.mp4", frames)
        client = env["client"]
        first = client.post("/api/ingest", json={
            "videos": [str(one)],
        })
        second = client.post("/api/ingest", json={
            "videos": [str(two)],
        })
        assert first.status_code == 200
        assert second.status_code == 200
        id1 = first.json()["job_id"]
        id2 = second.json()["job_id"]
        assert id1 != id2
        for job_id in (id1, id2):
            deadline = time.time() + DEADLINE_S
            state = "queued"
            while time.time() < deadline:
                state = client.get(
                    f"/api/status?job={job_id}",
                ).json()["state"]
                if state in ("done", "failed"):
                    break
                time.sleep(0.1)
            assert state == "done"
        names = [
            s["name"] for s in
            client.get("/api/stores").json()["stores"]
        ]
        assert "q1.zarr" in names and "q2.zarr" in names

    def test_status_carries_job_id(self, env, tmp_path):
        frames = static_frames(10) + approach_frames(10)
        video = write_video(tmp_path / "q3.mp4", frames)
        res = env["client"].post("/api/ingest", json={
            "videos": [str(video)],
        })
        job_id = res.json()["job_id"]
        body = env["client"].get(f"/api/status?job={job_id}").json()
        assert body["job_id"] == job_id
        assert body["state"] in ("queued", "running", "done")
        assert _wait_terminal(env["client"]) == "done"

    def test_unknown_job_is_idle(self, env):
        body = env["client"].get("/api/status?job=nope").json()
        assert body["state"] == "idle"
