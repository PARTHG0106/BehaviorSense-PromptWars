"""Bounded clip admission and one physical GPU per active clip.

The processing thread owns its slot. An HTTP disconnect requests cancellation but
does not release that slot while a decode subprocess is still using the GPU.
"""
from __future__ import annotations

from collections import deque
from concurrent.futures import Future
from dataclasses import dataclass
import json
import os
import queue
import threading
import time
from typing import Any


def max_parallel_clips():
    try:
        value = int(os.environ.get("BS_MAX_PARALLEL_CLIPS", "2"))
    except ValueError as exc:
        raise ValueError("BS_MAX_PARALLEL_CLIPS must be 1 or 2") from exc
    if value not in (1, 2):
        raise ValueError("BS_MAX_PARALLEL_CLIPS must be 1 or 2")
    return value


@dataclass(frozen=True)
class ClipWorker:
    device: str
    torch_device: str
    classifier: Any


class ClipQueueFull(RuntimeError):
    pass


class ClipWorkerPool:
    def __init__(self, workers, *, max_waiting=2):
        workers = tuple(workers)
        if not workers or len({worker.device for worker in workers}) != len(workers):
            raise ValueError("Workers must have distinct devices")
        if max_waiting < 0:
            raise ValueError("max_waiting must be nonnegative")
        self.capacity = len(workers)
        self.max_waiting = max_waiting
        self._free = deque(workers)
        self._waiting = deque()
        self._tickets = {}
        self._condition = threading.Condition()

    def submit(self):
        with self._condition:
            if len(self._tickets) >= self.capacity + self.max_waiting:
                raise ClipQueueFull("Both processing slots and the waiting queue are full; retry when a clip finishes.")
            ticket = object()
            self._tickets[ticket] = None
            self._waiting.append(ticket)
            self._condition.notify_all()
            return ticket

    def acquire(self, ticket, timeout=0.1):
        deadline = time.monotonic() + timeout
        with self._condition:
            while True:
                if ticket not in self._tickets:
                    raise ValueError("Clip reservation has already been released")
                assigned = self._tickets[ticket]
                if assigned is not None:
                    return assigned
                if self._free and self._waiting[0] is ticket:
                    self._waiting.popleft()
                    worker = self._free.popleft()
                    self._tickets[ticket] = worker
                    self._condition.notify_all()
                    return worker
                remaining = deadline - time.monotonic()
                if remaining <= 0:
                    return None
                self._condition.wait(remaining)

    def release(self, ticket):
        with self._condition:
            if ticket not in self._tickets:
                return
            worker = self._tickets.pop(ticket)
            if worker is None:
                self._waiting.remove(ticket)
            else:
                self._free.append(worker)
            self._condition.notify_all()

    def snapshot(self):
        with self._condition:
            return {"max_parallel_clips": self.capacity,
                    "active_clips": sum(worker is not None for worker in self._tickets.values()),
                    "queued_clips": len(self._waiting), "max_queued_clips": self.max_waiting}


class ClipJob:
    """Stream a worker's NDJSON output while retaining ownership through cancellation."""
    def __init__(self, pool, ticket, process, cleanup, *, max_frames):
        self.pool, self.ticket = pool, ticket
        self.process, self.cleanup = process, cleanup
        self.max_frames = max_frames
        self.done = Future()
        self.cancelled = threading.Event()
        self._output = queue.Queue(maxsize=8)
        self._thread = threading.Thread(target=self._run, daemon=True, name="behaviorsense-clip")

    def start(self):
        try:
            self._thread.start()
        except BaseException:
            self.pool.release(self.ticket)
            self.cleanup()
            self.done.set_result(None)
            raise

    def cancel(self):
        self.cancelled.set()

    def _emit(self, value):
        while not self.cancelled.is_set():
            try:
                self._output.put(value, timeout=0.1)
                return True
            except queue.Full:
                continue
        return False

    def _event(self, value):
        return self._emit(json.dumps(value) + "\n")

    def _run(self):
        iterator = None
        try:
            self._event({"event": "accepted", "max_frames": self.max_frames,
                         "max_parallel_clips": self.pool.capacity})
            worker = None
            last_beat = time.monotonic()
            announced_wait = False
            while not self.cancelled.is_set():
                worker = self.pool.acquire(self.ticket)
                if worker is not None:
                    break
                if not announced_wait or time.monotonic() - last_beat >= 5:
                    self._event({"event": "progress" if not announced_wait else "heartbeat", "step": "queued"})
                    announced_wait, last_beat = True, time.monotonic()
            if self.cancelled.is_set():
                return
            self._event({"event": "progress", "step": "processing", "device": worker.device,
                         "max_parallel_clips": self.pool.capacity})
            iterator = iter(self.process(worker))
            while not self.cancelled.is_set():
                try:
                    value = next(iterator)
                except StopIteration:
                    break
                if not self._emit(value):
                    break
        except Exception as exc:
            self._event({"event": "error", "kind": "server", "detail": f"{type(exc).__name__}: {exc}"})
        finally:
            # Closing the pipeline's heartbeat iterator waits for its current
            # ThreadPoolExecutor task; the GPU cannot be reassigned before then.
            try:
                if iterator is not None and hasattr(iterator, "close"):
                    iterator.close()
            finally:
                try:
                    self.cleanup()
                finally:
                    self.pool.release(self.ticket)
                    self.done.set_result(None)

    def stream(self):
        try:
            while not self.done.done() or not self._output.empty():
                try:
                    yield self._output.get(timeout=0.1)
                except queue.Empty:
                    continue
        finally:
            self.cancel()


_REPORT_LOCK = threading.RLock()


def report_serialized(reporter, state):
    """Keep a shared reporter's request statistics attached to the correct report."""
    with _REPORT_LOCK:
        return reporter.report(state)


def clip_response(job):
    """Cancel on every ASGI exit, including a send failure before background tasks."""
    from starlette.background import BackgroundTask
    from starlette.responses import StreamingResponse

    class ClipResponse(StreamingResponse):
        async def __call__(self, scope, receive, send):
            try:
                await super().__call__(scope, receive, send)
            finally:
                job.cancel()

    response = ClipResponse(job.stream(), media_type="application/x-ndjson",
        headers={"Cache-Control": "no-store", "X-Accel-Buffering": "no"},
        background=BackgroundTask(job.cancel))
    response.upload_done = job.done
    return response
