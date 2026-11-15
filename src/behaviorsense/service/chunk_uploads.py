"""Bounded, disk-backed upload assembly for requests crossing Cloudflare tunnels.

Each upload reserves its declared size before any bytes arrive. At most four uploads
(including analyses in progress) exist at once, so incomplete clients cannot allocate
unbounded temporary storage. Pending uploads expire; analyses keep their reservation
until their response finishes. Random upload IDs and separate bearer secrets keep one
client from appending to another client's file even on a public, tokenless demo.
"""

from __future__ import annotations

import atexit
import hmac
import json
import re
import secrets
import shutil
import tempfile
import threading
import time
from dataclasses import dataclass, field
from pathlib import Path
from urllib.parse import parse_qs, urlsplit

from .upload_limits import MAX_METADATA_BYTES, UPLOAD_CHUNK_BYTES, max_upload_mb


class UploadError(ValueError):
    def __init__(self, status: int, detail: str):
        super().__init__(detail)
        self.status = status
        self.detail = detail


@dataclass
class Upload:
    upload_id: str
    token: str
    path: Path
    filename: str
    size: int
    metadata: dict
    offset: int = 0
    touched: float = field(default_factory=time.monotonic)
    processing: bool = False
    receiving: bool = False


class ChunkWriter:
    def __init__(self, store, upload: Upload, length: int):
        self.store, self.upload, self.length = store, upload, length
        self.written, self.closed = 0, False
        self.file = upload.path.open("r+b")
        self.file.seek(upload.offset)

    def write(self, data: bytes) -> None:
        if self.closed or self.written + len(data) > self.length:
            raise UploadError(413, "chunk exceeds declared Content-Length")
        self.file.write(data)
        self.written += len(data)

    def finish(self) -> dict:
        if self.written != self.length:
            raise UploadError(422, "incomplete chunk; retry from the previous offset")
        self.file.close()
        with self.store.lock:
            self.upload.offset += self.written
            self.upload.receiving = False
            self.upload.touched = self.store.clock()
        self.closed = True
        return {"upload_id": self.upload.upload_id, "offset": self.upload.offset}

    def abort(self) -> None:
        if self.closed:
            return
        try:
            self.file.seek(self.upload.offset)
            self.file.truncate()
        finally:
            self.file.close()
            with self.store.lock:
                self.upload.receiving = False
                self.upload.touched = self.store.clock()
            self.closed = True


class UploadStore:
    def __init__(self, *, limit_mb: int | None = None, directory: Path | None = None,
                 max_uploads: int = 4, ttl_s: float = 1800, clock=time.monotonic,
                 start_janitor: bool = True):
        self.limit_mb = max_upload_mb() if limit_mb is None else limit_mb
        if self.limit_mb < 1 or max_uploads < 1 or ttl_s <= 0:
            raise ValueError("upload limits and expiry must be positive")
        self.max_bytes = self.limit_mb << 20
        self.max_uploads, self.ttl_s, self.clock = max_uploads, ttl_s, clock
        self.directory = Path(tempfile.mkdtemp(prefix="bs-chunks-", dir=directory))
        self.lock = threading.RLock()
        self.uploads: dict[str, Upload] = {}
        self.stop = threading.Event()
        if start_janitor:
            threading.Thread(target=self._janitor, daemon=True).start()
        atexit.register(self.close)

    def _janitor(self) -> None:
        while not self.stop.wait(min(60, self.ttl_s)):
            self.cleanup()

    def cleanup(self) -> None:
        with self.lock:
            for job in list(self.uploads.values()):
                if (not job.processing and not job.receiving
                        and self.clock() - job.touched >= self.ttl_s):
                    self._remove(job)

    def close(self) -> None:
        self.stop.set()
        with self.lock:
            shutil.rmtree(self.directory, ignore_errors=True)
            self.uploads.clear()
        atexit.unregister(self.close)

    def _remove(self, job: Upload) -> None:
        shutil.rmtree(job.path.parent, ignore_errors=True)
        self.uploads.pop(job.upload_id, None)

    def create(self, payload: object) -> dict:
        if not isinstance(payload, dict):
            raise UploadError(422, "upload metadata must be a JSON object")
        size = payload.get("size")
        if isinstance(size, bool) or not isinstance(size, int) or size <= 0:
            raise UploadError(422, "size must be a positive integer byte count")
        if size > self.max_bytes:
            raise UploadError(413, f"upload exceeds {self.limit_mb} MB")
        name = payload.get("filename", "clip.mp4")
        if not isinstance(name, str) or not name or len(name) > 512:
            raise UploadError(422, "filename must contain 1 to 512 characters")
        name = name.replace("\\", "/").rsplit("/", 1)[-1]
        suffix = Path(name).suffix.lower()
        if not re.fullmatch(r"\.[a-z0-9]{1,10}", suffix):
            suffix = ".mp4"
        metadata = {key: payload.get(key, "") for key in ("enrol", "gallery")}
        if any(not isinstance(value, str) for value in metadata.values()):
            raise UploadError(422, "enrol and gallery must be strings")
        if len(metadata["enrol"]) > 256 or len(json.dumps(metadata).encode()) > MAX_METADATA_BYTES:
            raise UploadError(413, "upload metadata is too large")
        with self.lock:
            self.cleanup()
            if len(self.uploads) >= self.max_uploads:
                raise UploadError(429, "upload storage is busy; finish or cancel an existing upload")
            upload_id, token = secrets.token_hex(16), secrets.token_urlsafe(32)
            parent = self.directory / upload_id
            parent.mkdir()
            path = parent / ("clip" + suffix)
            path.touch(exist_ok=False)
            self.uploads[upload_id] = Upload(upload_id, token, path, name, size, metadata,
                                             touched=self.clock())
            return {"upload_id": upload_id, "upload_token": token, "offset": 0,
                    "chunk_bytes": UPLOAD_CHUNK_BYTES}

    def _get(self, upload_id: str, token: str) -> Upload:
        self.cleanup()
        job = self.uploads.get(upload_id)
        if job is None or not hmac.compare_digest(job.token.encode(), token.encode()):
            raise UploadError(404, "upload not found or expired")
        return job

    def begin_chunk(self, upload_id: str, token: str, offset: int, length: int) -> ChunkWriter:
        if length <= 0:
            raise UploadError(422, "empty chunk")
        if length > UPLOAD_CHUNK_BYTES:
            raise UploadError(413, "chunk exceeds 8 MB")
        with self.lock:
            job = self._get(upload_id, token)
            if job.processing or job.receiving:
                raise UploadError(409, "upload is already in use")
            if offset != job.offset:
                raise UploadError(409, f"incorrect offset; expected {job.offset}")
            if job.offset + length > job.size:
                raise UploadError(413, "chunk exceeds the declared file size")
            writer = ChunkWriter(self, job, length)
            job.receiving = True
            return writer

    def take(self, upload_id: str, token: str) -> Upload:
        with self.lock:
            job = self._get(upload_id, token)
            if job.receiving or job.processing:
                raise UploadError(409, "upload is already in use")
            if job.offset != job.size:
                raise UploadError(409, f"upload incomplete; received {job.offset} of {job.size} bytes")
            job.processing = True
            return job

    def release(self, job: Upload) -> None:
        with self.lock:
            if self.uploads.get(job.upload_id) is job:
                self._remove(job)

    def cancel(self, upload_id: str, token: str) -> None:
        with self.lock:
            job = self._get(upload_id, token)
            if job.processing or job.receiving:
                raise UploadError(409, "upload is already in use")
            self._remove(job)


def content_length(headers, limit: int) -> int:
    if headers.get("Transfer-Encoding"):
        raise UploadError(411, "Content-Length is required")
    try:
        length = int(headers.get("Content-Length", ""))
    except (ValueError, TypeError) as exc:
        raise UploadError(411, "Content-Length is required") from exc
    if length < 0:
        raise UploadError(422, "invalid Content-Length")
    if length > limit:
        raise UploadError(413, "request body is too large")
    return length


def chunk_route(path: str) -> tuple[str, str, int]:
    parsed = urlsplit(path)
    if parsed.path == "/uploads":
        return "create", "", 0
    match = re.fullmatch(r"/uploads/([a-f0-9]{32})(/complete)?", parsed.path)
    if not match:
        raise UploadError(404, "upload route not found")
    if match.group(2):
        return "complete", match.group(1), 0
    try:
        offset = int(parse_qs(parsed.query).get("offset", ["0"])[0])
    except ValueError as exc:
        raise UploadError(422, "offset must be an integer") from exc
    return "chunk", match.group(1), offset


def receive_legacy_multipart(handler, store: UploadStore) -> Upload:
    """Compatibility reader for existing /video clients, bounded before body allocation.

    New clients stream chunks to disk. Legacy multipart clients have one bounded request
    in memory while parsing; servers additionally limit their concurrent legacy requests.
    File bytes, rather than multipart overhead, determine the advertised upload limit.
    """
    from email.parser import BytesParser
    from email.policy import default

    length = content_length(handler.headers, store.max_bytes + MAX_METADATA_BYTES)
    if not length:
        raise UploadError(422, "empty upload")
    content_type = handler.headers.get("Content-Type", "")
    if not content_type.lower().startswith("multipart/form-data"):
        raise UploadError(422, "expected a multipart file upload")
    handler.connection.settimeout(60)
    raw = handler.rfile.read(length)
    if len(raw) != length:
        raise UploadError(422, "incomplete upload")
    message = BytesParser(policy=default).parsebytes(
        ("Content-Type: " + content_type + "\r\nMIME-Version: 1.0\r\n\r\n").encode() + raw)
    if not message.is_multipart() or message.defects:
        raise UploadError(422, "invalid multipart upload")
    metadata, video_data, filename = {}, None, "clip.mp4"
    for part in message.iter_parts():
        name = part.get_param("name", header="Content-Disposition")
        data = part.get_payload(decode=True) or b""
        if name == "file":
            if video_data is not None:
                raise UploadError(422, "exactly one video file is required")
            video_data, filename = data, part.get_filename() or filename
        elif name in ("enrol", "gallery"):
            if len(data) > MAX_METADATA_BYTES:
                raise UploadError(413, "upload metadata is too large")
            try:
                metadata[name] = data.decode("utf-8", errors="strict")
            except UnicodeError as exc:
                raise UploadError(422, "upload metadata must be UTF-8") from exc
    if video_data is None:
        raise UploadError(422, "file field is required")
    created = store.create({"filename": filename, "size": len(video_data), **metadata})
    upload_id, token = created["upload_id"], created["upload_token"]
    try:
        for offset in range(0, len(video_data), UPLOAD_CHUNK_BYTES):
            part = memoryview(video_data)[offset:offset + UPLOAD_CHUNK_BYTES]
            writer = store.begin_chunk(upload_id, token, offset, len(part))
            try:
                writer.write(part)
                writer.finish()
            finally:
                writer.abort()
        return store.take(upload_id, token)
    except BaseException:
        store.cancel(upload_id, token)
        raise


def serve_chunk_request(handler, store: UploadStore, complete, send_json) -> None:
    """Adapt the protocol to BaseHTTPRequestHandler; complete(job) writes its response."""
    writer = None
    try:
        route, upload_id, offset = chunk_route(handler.path)
        token = handler.headers.get("X-BS-Upload-Token", "")
        method = handler.command
        handler.connection.settimeout(60)
        if route == "create" and method == "POST":
            length = content_length(handler.headers, MAX_METADATA_BYTES)
            raw = handler.rfile.read(length)
            if len(raw) != length:
                raise UploadError(422, "incomplete metadata")
            try:
                payload = json.loads(raw)
            except (ValueError, UnicodeError) as exc:
                raise UploadError(422, "invalid upload metadata JSON") from exc
            send_json(store.create(payload), 201)
        elif route == "chunk" and method == "PUT":
            length = content_length(handler.headers, UPLOAD_CHUNK_BYTES)
            writer = store.begin_chunk(upload_id, token, offset, length)
            remaining = length
            while remaining:
                raw = handler.rfile.read(min(1 << 20, remaining))
                if not raw:
                    raise UploadError(422, "incomplete chunk")
                writer.write(raw)
                remaining -= len(raw)
            send_json(writer.finish())
        elif route == "chunk" and method == "DELETE":
            if handler.headers.get("Transfer-Encoding") or handler.headers.get("Content-Length", "0") != "0":
                raise UploadError(422, "cancellation request must have no body")
            store.cancel(upload_id, token)
            send_json({"cancelled": True})
        elif route == "complete" and method == "POST":
            if handler.headers.get("Transfer-Encoding") or handler.headers.get("Content-Length", "0") != "0":
                raise UploadError(422, "completion request must have no body")
            job = store.take(upload_id, token)
            try:
                complete(job)
            finally:
                store.release(job)
        else:
            raise UploadError(405, "method not allowed")
    except UploadError as exc:
        # Rejected bodies must not become another request on this persistent connection.
        handler.close_connection = True
        send_json({"detail": exc.detail}, exc.status)
    except TimeoutError:
        handler.close_connection = True
        send_json({"detail": "upload request timed out"}, 408)
    finally:
        if writer is not None:
            writer.abort()


def install_chunk_upload_routes(app, store: UploadStore, auth, complete) -> None:
    """Install FastAPI routes. async complete(upload, request) returns the video response.

    The callback may read or move upload.path. Its response background task runs before
    the store releases the reservation; processing uploads never expire mid-analysis.
    """
    import asyncio
    from fastapi import HTTPException, Request
    from starlette.background import BackgroundTask

    async def endpoint(request):
        auth(request.headers.get("X-BS-Token", ""))
        writer = None
        try:
            route, upload_id, offset = chunk_route(str(request.url))
            token = request.headers.get("X-BS-Upload-Token", "")
            if route == "create" and request.method == "POST":
                length = content_length(request.headers, MAX_METADATA_BYTES)
                raw = bytearray()
                async with asyncio.timeout(180):
                    async for part in request.stream():
                        if len(raw) + len(part) > length:
                            raise UploadError(413, "metadata exceeds declared Content-Length")
                        raw.extend(part)
                if len(raw) != length:
                    raise UploadError(422, "incomplete metadata")
                try:
                    payload = json.loads(raw)
                except (ValueError, UnicodeError) as exc:
                    raise UploadError(422, "invalid upload metadata JSON") from exc
                return store.create(payload)
            if route == "chunk" and request.method == "PUT":
                length = content_length(request.headers, UPLOAD_CHUNK_BYTES)
                writer = store.begin_chunk(upload_id, token, offset, length)
                async with asyncio.timeout(180):
                    async for part in request.stream():
                        writer.write(part)
                return writer.finish()
            if route == "chunk" and request.method == "DELETE":
                if request.headers.get("Transfer-Encoding") or request.headers.get("Content-Length", "0") != "0":
                    raise UploadError(422, "cancellation request must have no body")
                store.cancel(upload_id, token)
                return {"cancelled": True}
            if route == "complete" and request.method == "POST":
                if request.headers.get("Transfer-Encoding") or request.headers.get("Content-Length", "0") != "0":
                    raise UploadError(422, "completion request must have no body")
                job = store.take(upload_id, token)
                try:
                    response = await complete(job, request)
                except BaseException:
                    store.release(job)
                    raise
                previous = response.background
                upload_done = getattr(response, "upload_done", None)
                if upload_done is not None:
                    # A cancelled HTTP stream can leave a decode child finishing its
                    # current work. The notebook signals when that worker actually exits.
                    upload_done.add_done_callback(lambda _future: store.release(job))
                    return response

                async def finished():
                    try:
                        if previous is not None:
                            await previous()
                    finally:
                        store.release(job)

                response.background = BackgroundTask(finished)
                if hasattr(response, "body_iterator"):
                    previous_iterator = response.body_iterator

                    async def guarded_stream():
                        try:
                            async for item in previous_iterator:
                                yield item
                        finally:
                            store.release(job)

                    response.body_iterator = guarded_stream()
                return response
            raise UploadError(405, "method not allowed")
        except UploadError as exc:
            raise HTTPException(exc.status, exc.detail) from exc
        except TimeoutError as exc:
            raise HTTPException(408, "upload request timed out") from exc
        finally:
            if writer is not None:
                writer.abort()

    # Avoid a module-level FastAPI dependency in local/dev servers. Supplying the concrete
    # annotation also works when this module uses postponed annotations.
    endpoint.__annotations__["request"] = Request
    app.add_api_route("/uploads", endpoint, methods=["POST"])
    app.add_api_route("/uploads/{upload_id}", endpoint, methods=["PUT", "DELETE"])
    app.add_api_route("/uploads/{upload_id}/complete", endpoint, methods=["POST"])
