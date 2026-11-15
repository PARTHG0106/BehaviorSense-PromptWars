"""Transport boundaries and cleanup, exercised without GPUs or remote services."""

from __future__ import annotations

import concurrent.futures
import hashlib
import http.client
import http.server
import importlib.util
import json
import sys
import threading
from pathlib import Path

import pytest

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))

from behaviorsense.service.chunk_uploads import (  # noqa: E402
    UploadError, UploadStore, install_chunk_upload_routes,
)
from behaviorsense.service.upload_limits import (  # noqa: E402
    UPLOAD_CHUNK_BYTES, max_upload_mb, upload_health,
)


@pytest.fixture
def store(tmp_path):
    result = UploadStore(limit_mb=1, directory=tmp_path, start_janitor=False)
    yield result
    result.close()


def add(store, data=b"hello", **metadata):
    created = store.create({"filename": "camera.mp4", "size": len(data), **metadata})
    writer = store.begin_chunk(created["upload_id"], created["upload_token"], 0, len(data))
    writer.write(data)
    writer.finish()
    return created


def test_config_and_100mb_boundary(monkeypatch, tmp_path):
    monkeypatch.delenv("BS_MAX_UPLOAD_MB", raising=False)
    assert max_upload_mb() == 100
    assert upload_health()["upload_chunk_bytes"] == 8 << 20
    configured = UploadStore(directory=tmp_path, start_janitor=False)
    try:
        accepted = configured.create({"size": 100 << 20, "filename": "limit.mp4"})
        assert configured.uploads[accepted["upload_id"]].path.stat().st_size == 0
        offset = 0
        block = b"x" * UPLOAD_CHUNK_BYTES
        while offset < 100 << 20:
            part = block[:min(UPLOAD_CHUNK_BYTES, (100 << 20) - offset)]
            writer = configured.begin_chunk(accepted["upload_id"], accepted["upload_token"],
                                            offset, len(part))
            writer.write(part)
            offset = writer.finish()["offset"]
        complete = configured.take(accepted["upload_id"], accepted["upload_token"])
        assert complete.path.stat().st_size == 100 << 20
        configured.release(complete)
        with pytest.raises(UploadError) as rejected:
            configured.create({"size": (100 << 20) + 1})
        assert rejected.value.status == 413
    finally:
        configured.close()
    for raw in ("0", "-1", "1.5", "broken", ""):
        monkeypatch.setenv("BS_MAX_UPLOAD_MB", raw)
        with pytest.raises(ValueError, match="BS_MAX_UPLOAD_MB"):
            max_upload_mb()


def test_exact_file_bytes_and_missing_chunks(store):
    data = b"z" * (1 << 20)
    created = store.create({"filename": "../../folder\\video.MOV", "size": len(data)})
    uid, token = created["upload_id"], created["upload_token"]
    job = store.uploads[uid]
    assert job.filename == "video.MOV"
    assert job.path.suffix == ".mov"
    assert job.path.parent.parent == store.directory
    first = store.begin_chunk(uid, token, 0, len(data) - 1)
    first.write(data[:-1])
    first.finish()
    with pytest.raises(UploadError, match="incomplete"):
        store.take(uid, token)
    with pytest.raises(UploadError, match="declared file size"):
        store.begin_chunk(uid, token, len(data) - 1, 2)
    last = store.begin_chunk(uid, token, len(data) - 1, 1)
    last.write(b"z")
    last.finish()
    taken = store.take(uid, token)
    assert taken.path.read_bytes() == data
    store.release(taken)
    assert not taken.path.parent.exists()
    assert not store.uploads


def test_chunk_ownership_offsets_and_rollback(store):
    created = store.create({"size": 8})
    uid, token = created["upload_id"], created["upload_token"]
    for bad in ("wrong", "秘密"):
        with pytest.raises(UploadError) as error:
            store.begin_chunk(uid, bad, 0, 4)
        assert error.value.status == 404
    writer = store.begin_chunk(uid, token, 0, 4)
    writer.write(b"ab")
    with pytest.raises(UploadError, match="already in use"):
        store.begin_chunk(uid, token, 0, 4)
    with pytest.raises(UploadError, match="incomplete chunk"):
        writer.finish()
    writer.abort()
    assert store.uploads[uid].path.stat().st_size == 0
    assert store.uploads[uid].offset == 0
    writer = store.begin_chunk(uid, token, 0, 4)
    writer.write(b"abcd")
    assert writer.finish()["offset"] == 4
    with pytest.raises(UploadError, match="incorrect offset"):
        store.begin_chunk(uid, token, 0, 4)
    with pytest.raises(UploadError) as oversized:
        store.begin_chunk(uid, token, 4, UPLOAD_CHUNK_BYTES + 1)
    assert oversized.value.status == 413


def test_reservations_expire_but_processing_stays_bounded(tmp_path):
    now = [0.0]
    store = UploadStore(directory=tmp_path, limit_mb=1, max_uploads=2, ttl_s=10,
                        clock=lambda: now[0], start_janitor=False)
    try:
        first = add(store)
        active = store.take(first["upload_id"], first["upload_token"])
        abandoned = store.create({"size": 1 << 20})
        abandoned_path = store.uploads[abandoned["upload_id"]].path
        with pytest.raises(UploadError) as full:
            store.create({"size": 1})
        assert full.value.status == 429
        now[0] = 11
        store.cleanup()
        assert active.path.exists()
        assert not abandoned_path.exists()
        assert len(store.uploads) == 1
        second = store.create({"size": 1})
        with pytest.raises(UploadError, match="already in use"):
            store.cancel(first["upload_id"], first["upload_token"])
        store.cancel(second["upload_id"], second["upload_token"])
        store.release(active)
        assert not store.uploads
    finally:
        store.close()


def test_concurrent_creation_never_overbooks(store):
    def attempt(_):
        try:
            return store.create({"size": 1 << 20})
        except UploadError as exc:
            assert exc.status == 429
            return None
    with concurrent.futures.ThreadPoolExecutor(max_workers=12) as pool:
        results = list(pool.map(attempt, range(20)))
    assert sum(result is not None for result in results) == 4
    assert sum(job.size for job in store.uploads.values()) == 4 << 20


def test_fastapi_auth_stream_and_worker_lifetime(store):
    from fastapi import FastAPI, HTTPException
    from fastapi.responses import StreamingResponse
    from fastapi.testclient import TestClient

    future = concurrent.futures.Future()
    seen = []
    app = FastAPI()

    def auth(token):
        if token != "session-secret":
            raise HTTPException(401, "not authorized")

    async def complete(upload, request):
        seen.append((upload.path.read_bytes(), upload.metadata, request.query_params["stages"]))
        response = StreamingResponse(iter([b'{"event":"accepted"}\n', b'{"event":"result"}\n']),
                                     media_type="application/x-ndjson")
        response.upload_done = future
        return response

    install_chunk_upload_routes(app, store, auth, complete)
    with TestClient(app) as client:
        assert client.post("/uploads", json={"size": 3}).status_code == 401
        headers = {"X-BS-Token": "session-secret"}
        created = client.post("/uploads", headers=headers,
                              json={"filename": "clip.mp4", "size": 3, "enrol": "Ada"}).json()
        url = "/uploads/" + created["upload_id"]
        headers["X-BS-Upload-Token"] = created["upload_token"]
        assert client.post(url + "/complete", headers=headers).status_code == 409
        assert client.put(url + "?offset=0", headers=headers, content=b"abc").json()["offset"] == 3
        assert client.put(url + "?offset=0", headers=headers, content=b"a").status_code == 409
        response = client.post(url + "/complete?stages=12", headers=headers)
        assert response.status_code == 200
        assert [json.loads(line)["event"] for line in response.text.splitlines()] == ["accepted", "result"]
        assert seen == [(b"abc", {"enrol": "Ada", "gallery": ""}, "12")]
        assert len(store.uploads) == 1  # HTTP stream ending does not delete a worker's input.
        assert client.delete(url, headers=headers).status_code == 409
        future.set_result(None)
        assert not store.uploads


def test_fastapi_rejection_and_response_cleanup(store):
    from fastapi import FastAPI, HTTPException
    from fastapi.responses import StreamingResponse
    from fastapi.testclient import TestClient

    app = FastAPI()

    async def complete(upload, request):
        if upload.metadata["enrol"] == "reject":
            raise HTTPException(429, "both GPUs occupied")
        return StreamingResponse(iter([b'{"event":"result"}\n']))

    install_chunk_upload_routes(app, store, lambda token: None, complete)
    with TestClient(app) as client:
        for enrol, code in (("reject", 429), ("", 200)):
            created = client.post("/uploads", json={"size": 1, "enrol": enrol}).json()
            headers = {"X-BS-Upload-Token": created["upload_token"]}
            url = "/uploads/" + created["upload_id"]
            assert client.put(url, headers=headers, content=b"x").status_code == 200
            assert client.post(url + "/complete", headers=headers).status_code == code
            assert not store.uploads


def test_asgi_disconnect_keeps_upload_until_worker_exits(store):
    """A broken socket cancels delivery but cannot delete a running decoder's input."""
    import asyncio
    from fastapi import FastAPI
    from starlette.requests import ClientDisconnect
    from behaviorsense.service.clip_workers import ClipJob, ClipWorker, ClipWorkerPool, clip_response

    created = add(store)
    upload = store.uploads[created["upload_id"]]
    pool = ClipWorkerPool([ClipWorker("cuda:0", "cuda:0", None)])
    started, release, cleaned = threading.Event(), threading.Event(), threading.Event()
    jobs = []
    app = FastAPI()

    def process(worker):
        started.set()
        assert release.wait(5), "test did not release the simulated decoder"
        assert upload.path.read_bytes() == b"hello"
        yield '{"event":"result"}\n'

    async def complete(job_upload, request):
        job = ClipJob(pool, pool.submit(), process, lambda: None, max_frames=36000)
        jobs.append(job)
        response = clip_response(job)
        job.start()
        return response

    install_chunk_upload_routes(app, store, lambda token: None, complete)

    async def disconnect():
        scope = {"type": "http", "asgi": {"version": "3.0", "spec_version": "2.4"},
                 "http_version": "1.1", "method": "POST", "scheme": "http", "root_path": "",
                 "path": f'/uploads/{created["upload_id"]}/complete', "query_string": b"",
                 "headers": [(b"content-length", b"0"),
                             (b"x-bs-upload-token", created["upload_token"].encode())],
                 "server": ("test", 80), "client": ("test", 1234)}

        async def receive():
            return {"type": "http.request", "body": b"", "more_body": False}

        async def send(message):
            if message["type"] == "http.response.body":
                assert await asyncio.to_thread(started.wait, 2)
                raise OSError("simulated client disconnect")

        with pytest.raises(ClientDisconnect):
            await app(scope, receive, send)

    try:
        asyncio.run(disconnect())
        assert jobs[0].cancelled.is_set()
        assert not jobs[0].done.done()
        assert upload.path.exists()
        assert len(store.uploads) == 1
        assert pool.snapshot()["active_clips"] == 1
        jobs[0].done.add_done_callback(lambda _: cleaned.set())
    finally:
        release.set()
        if jobs:
            jobs[0].done.result(timeout=5)
            assert cleaned.wait(5)
    assert not store.uploads
    assert not upload.path.exists()
    assert pool.snapshot()["active_clips"] == 0


def load_web(name):
    spec = importlib.util.spec_from_file_location("test_" + name, ROOT / "web" / (name + ".py"))
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


def test_local_forwarding_chunks_and_legacy_multipart(tmp_path):
    dev = load_web("dev_backend")
    bridge = load_web("local_backend")
    gpu_store = UploadStore(limit_mb=10, directory=tmp_path, start_janitor=False)
    browser_store = UploadStore(limit_mb=10, directory=tmp_path, start_janitor=False)
    received, chunk_lengths = [], []

    class GPU(dev.Handler):
        uploads = gpu_store

        def do_GET(self):
            self._send({**upload_health(10), "ready": True})

        def do_PUT(self):
            chunk_lengths.append(int(self.headers["Content-Length"]))
            super().do_PUT()

        def _complete_upload(self, upload):
            received.append((hashlib.sha256(upload.path.read_bytes()).hexdigest(), upload.metadata))
            self._stream([{"event": "accepted"}, {"event": "result", "payload": {"ok": True}}])

    server = http.server.ThreadingHTTPServer(("127.0.0.1", 0), GPU)
    thread = threading.Thread(target=server.serve_forever, daemon=True)
    thread.start()
    try:
        data = b"x" * (9 << 20)
        created = browser_store.create({"filename": "sample.webm", "size": len(data), "enrol": "Ada"})
        for offset in range(0, len(data), UPLOAD_CHUNK_BYTES):
            part = data[offset:offset + UPLOAD_CHUNK_BYTES]
            writer = browser_store.begin_chunk(created["upload_id"], created["upload_token"], offset, len(part))
            writer.write(part)
            writer.finish()
        upload = browser_store.take(created["upload_id"], created["upload_token"])
        upstream = bridge.Upstream(f"http://127.0.0.1:{server.server_port}")
        events = list(upstream.video_file_stages_12(upload, gallery_json='{"residents":{}}'))
        assert chunk_lengths == [8 << 20, 1 << 20]
        assert events[-1] == {"event": "result", "payload": {"ok": True}}
        assert sum(event["event"] == "heartbeat" for event in events) == 2
        assert received == [(hashlib.sha256(data).hexdigest(), {"enrol": "Ada", "gallery": '{"residents":{}}'})]
        assert not gpu_store.uploads
        browser_store.release(upload)

        # The legacy limit counts file bytes and accepts multipart overhead at the exact cap.
        GPU.uploads = UploadStore(limit_mb=1, directory=tmp_path, start_janitor=False)
        boundary = "legacy-boundary"
        head = (f'--{boundary}\r\nContent-Disposition: form-data; name="file"; '
                'filename="limit.mp4"\r\nContent-Type: video/mp4\r\n\r\n').encode()
        tail = f"\r\n--{boundary}--\r\n".encode()
        for file_size, expected in ((1 << 20, 200), ((1 << 20) + 1, 413)):
            connection = http.client.HTTPConnection("127.0.0.1", server.server_port, timeout=5)
            body = head + b"a" * file_size + tail
            connection.request("POST", "/video", body,
                               {"Content-Type": f"multipart/form-data; boundary={boundary}"})
            response = connection.getresponse()
            assert response.status == expected
            response.read()
            connection.close()
        # An impossible advertised body is rejected without reading any body bytes.
        connection = http.client.HTTPConnection("127.0.0.1", server.server_port, timeout=5)
        connection.request("POST", "/video", b"", {"Content-Length": str(3 << 20)})
        response = connection.getresponse()
        assert response.status == 413
        response.read()
        connection.close()
        GPU.uploads.close()
    finally:
        server.shutdown()
        server.server_close()
        gpu_store.close()
        browser_store.close()


def test_gallery_concurrent_updates_are_atomic(tmp_path):
    bridge = load_web("local_backend")
    gallery = bridge.GalleryStore(tmp_path / "gallery.json")
    embeddings = [[1.0] * 16]
    with concurrent.futures.ThreadPoolExecutor(max_workers=4) as pool:
        assert list(pool.map(lambda index: gallery.enrol("Ada", embeddings), range(12))) == [1] * 12
    saved = json.loads(gallery.path.read_text())
    assert saved["residents"]["Ada"]["n_updates"] == 12
    assert not list(tmp_path.glob(".bs-gallery-*"))


if __name__ == "__main__":
    class RunnerSummary:
        def pytest_terminal_summary(self, terminalreporter):
            stats = terminalreporter.stats
            passed = len(stats.get("passed", []))
            total = passed + len(stats.get("failed", [])) + len(stats.get("error", []))
            print(f"{passed}/{total} passed; {len(stats.get('skipped', []))} skipped")

    sys.exit(pytest.main([__file__, "-q"], plugins=[RunnerSummary()]))
