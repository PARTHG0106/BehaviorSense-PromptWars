"""Execute notebook 05's HTTP route code with real scheduling and fake GPU work."""

from __future__ import annotations

import ast
import concurrent.futures
import json
import pathlib
import shutil
import sys
import tempfile
import threading
import time
from types import SimpleNamespace

import pytest
from fastapi import FastAPI, File, Form, Header, HTTPException, UploadFile
from fastapi.testclient import TestClient

ROOT = pathlib.Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))

from behaviorsense.service.chunk_uploads import UploadStore, install_chunk_upload_routes
from behaviorsense.service.clip_workers import ClipJob, ClipQueueFull, ClipWorker, ClipWorkerPool


def until(predicate, timeout=5):
    deadline = time.monotonic() + timeout
    while not predicate():
        if time.monotonic() >= deadline:
            raise AssertionError("timed out waiting for the notebook worker state")
        time.sleep(0.01)


@pytest.fixture
def notebook(tmp_path, monkeypatch):
    document = json.loads((ROOT / "notebooks" / "05_serve_inference_online.ipynb").read_text(encoding="utf-8"))
    sources = [cell["source"] for cell in document["cells"] if cell["cell_type"] == "code"]
    sources = ["".join(source) if isinstance(source, list) else source for source in sources]
    video_source = next(source for source in sources if '@app.post("/video")' in source)
    blocking_source = next(source for source in sources if "def _blocking(" in source)
    wanted = {"video", "_start_video", "_complete_chunk_upload", "_video_stream"}
    tree = ast.parse(video_source)
    nodes = [node for node in tree.body
             if isinstance(node, (ast.FunctionDef, ast.AsyncFunctionDef)) and node.name in wanted]
    assert {node.name for node in nodes} == wanted
    blocking = next(node for node in ast.parse(blocking_source).body
                    if isinstance(node, ast.FunctionDef) and node.name == "_blocking")
    classifiers = {f"cuda:{index}": object() for index in range(2)}
    pool = ClipWorkerPool([ClipWorker(device, device, classifier)
                           for device, classifier in classifiers.items()], max_waiting=1)
    app = FastAPI()
    store = UploadStore(limit_mb=1, directory=tmp_path, start_janitor=False)
    paths, decoded, classified = [], [], []
    guard = threading.Lock()

    def make_temp(*args, **kwargs):
        path = tempfile.mkdtemp(*args, dir=tmp_path, **kwargs)
        paths.append(pathlib.Path(path))
        return path

    def stage(poses, t_start, t_pose, *, stages, classifier, reid_note):
        assert classifier is classifiers[poses["device"]]
        # The notebook deliberately removes pixels after extraction: all subsequent
        # work must use the materialized pose data and the assigned classifier.
        assert not pathlib.Path(poses["path"]).exists()
        with guard:
            classified.append((poses["source"], poses["device"]))
        yield "result", {"source": poses["source"], "device": poses["device"],
                         "stages_run": stages, "reid_note": reid_note}

    def extract(path, **kwargs):
        with guard:
            decoded.append((path, kwargs))
        return {"source": pathlib.Path(path).read_text(), "device": kwargs["device"], "path": path}

    namespace = {"app": app, "File": File, "Form": Form, "Header": Header,
                 "HTTPException": HTTPException, "UploadFile": UploadFile,
                 "_auth": lambda token: None, "STATE": {"ready": True, "error": None,
                     "clip_pool": pool, "sample_fps": 15.0},
                 "RTMO": "fake-rtmo.onnx", "OSNET": "fake-osnet.pth", "MAX_UPLOAD_MB": 1,
                 "MAX_FRAMES": 36000, "SAMPLE_FPS": 15.0, "json": json, "pathlib": pathlib,
                 "tempfile": SimpleNamespace(mkdtemp=make_temp), "shutil": shutil, "time": time,
                 "ClipJob": ClipJob, "ClipQueueFull": ClipQueueFull, "_rss_mb": lambda: 0,
                 "extract_isolated": extract, "_video_stages": stage}
    # The real route's housekeeping can run on machines with no torch installation.
    monkeypatch.setitem(sys.modules, "torch", SimpleNamespace(cuda=SimpleNamespace(is_available=lambda: False)))
    exec(compile(ast.Module(body=[blocking, *nodes], type_ignores=[]), "notebook05-routes", "exec"), namespace)
    install_chunk_upload_routes(app, store, namespace["_auth"], namespace["_complete_chunk_upload"])
    result = SimpleNamespace(app=app, store=store, pool=pool, namespace=namespace,
                             decoded=decoded, classified=classified, paths=paths, extract=extract)
    yield result
    until(lambda: pool.snapshot()["active_clips"] == 0 and pool.snapshot()["queued_clips"] == 0)
    store.close()


def chunk(client, source, *, enrol="", gallery=""):
    data = source.encode()
    response = client.post("/uploads", json={"filename": source + ".mp4", "size": len(data),
                                            "enrol": enrol, "gallery": gallery})
    assert response.status_code == 200
    created = response.json()
    url = "/uploads/" + created["upload_id"]
    headers = {"X-BS-Upload-Token": created["upload_token"]}
    assert client.put(url + "?offset=0", headers=headers, content=data).json()["offset"] == len(data)
    return url, headers


def events(response):
    assert response.status_code == 200
    return [json.loads(line) for line in response.text.splitlines()]


@pytest.mark.parametrize("transport", ["multipart", "chunks"])
def test_real_routes_assign_distinct_devices_and_classifiers(notebook, transport):
    rendezvous = threading.Barrier(2)
    gallery_text = '{"residents":{"Ada":{"centroid":[1,0,0]}}}'
    seen_gallery = []

    def extract(path, **kwargs):
        assert pathlib.Path(kwargs["gallery"]).read_text() == gallery_text
        seen_gallery.append(kwargs["gallery"])
        rendezvous.wait(timeout=5)
        return notebook.extract(path, **kwargs)

    notebook.namespace["extract_isolated"] = extract
    with TestClient(notebook.app) as client:
        prepared = [chunk(client, source, enrol="Ada", gallery=gallery_text)
                    for source in ("first", "second")] if transport == "chunks" else []

        def request(index):
            if transport == "chunks":
                url, headers = prepared[index]
                return client.post(url + "/complete?stages=12", headers=headers)
            source = ("first", "second")[index]
            return client.post("/video?stages=12", files={"file": (source + ".mp4", source.encode())},
                               data={"enrol": "Ada", "gallery": gallery_text})

        with concurrent.futures.ThreadPoolExecutor(max_workers=2) as executor:
            responses = list(executor.map(request, range(2)))
    payloads = [next(event["payload"] for event in events(response) if event["event"] == "result")
                for response in responses]
    assert {payload["device"] for payload in payloads} == {"cuda:0", "cuda:1"}
    assert {payload["source"] for payload in payloads} == {"first", "second"}
    assert all(payload["stages_run"] == "12" for payload in payloads)
    assert all(kwargs["device"] == kwargs["osnet_device"] for _, kwargs in notebook.decoded)
    assert all(kwargs["max_frames"] == 36000 and kwargs["target_fps"] == 15 for _, kwargs in notebook.decoded)
    assert len(notebook.classified) == 2
    until(lambda: not notebook.store.uploads)
    assert all(not pathlib.Path(path).exists() for path in seen_gallery)
    assert all(not path.exists() for path in notebook.paths)


def test_chunk_queue_full_releases_rejected_file_and_finishes_accepted_jobs(notebook):
    release = threading.Event()

    def extract(path, **kwargs):
        assert release.wait(5)
        return notebook.extract(path, **kwargs)

    notebook.namespace["extract_isolated"] = extract
    with TestClient(notebook.app) as client:
        prepared = [chunk(client, f"clip{index}") for index in range(4)]
        rejected_id = prepared[-1][0].rsplit("/", 1)[-1]
        rejected_path = notebook.store.uploads[rejected_id].path

        def complete(index):
            url, headers = prepared[index]
            return client.post(url + "/complete", headers=headers)

        with concurrent.futures.ThreadPoolExecutor(max_workers=3) as executor:
            active = [executor.submit(complete, index) for index in (0, 1)]
            try:
                until(lambda: notebook.pool.snapshot()["active_clips"] == 2)
                active.append(executor.submit(complete, 2))
                until(lambda: notebook.pool.snapshot()["queued_clips"] == 1)
                rejected = complete(3)
                assert rejected.status_code == 429
                assert rejected_id not in notebook.store.uploads
                assert not rejected_path.parent.exists()
            finally:
                release.set()
            for future in active:
                assert any(event["event"] == "result" for event in events(future.result(timeout=5)))
    until(lambda: not notebook.store.uploads)
    assert notebook.pool.snapshot()["active_clips"] == 0


@pytest.mark.parametrize("failure", ["loading", "decode"])
def test_chunk_failures_leave_no_file_or_pool_reservation(notebook, failure):
    if failure == "loading":
        notebook.namespace["STATE"]["ready"] = False
    else:
        def invalid_video(path, **kwargs):
            raise ValueError("invalid video fixture")
        notebook.namespace["extract_isolated"] = invalid_video
    with TestClient(notebook.app) as client:
        url, headers = chunk(client, "badclip")
        path = next(iter(notebook.store.uploads.values())).path
        response = client.post(url + "/complete", headers=headers)
    if failure == "loading":
        assert response.status_code == 503
    else:
        error = next(event for event in events(response) if event["event"] == "error")
        assert error["kind"] == "decode"
        assert "invalid video fixture" in error["detail"]
    until(lambda: not notebook.store.uploads)
    assert not path.parent.exists()
    assert notebook.pool.snapshot()["active_clips"] == 0
    assert notebook.pool.snapshot()["queued_clips"] == 0


if __name__ == "__main__":
    class RunnerSummary:
        def pytest_terminal_summary(self, terminalreporter):
            stats = terminalreporter.stats
            passed = len(stats.get("passed", []))
            total = passed + len(stats.get("failed", [])) + len(stats.get("error", []))
            print(f"{passed}/{total} passed; {len(stats.get('skipped', []))} skipped")

    sys.exit(pytest.main([__file__, "-q"], plugins=[RunnerSummary()]))
