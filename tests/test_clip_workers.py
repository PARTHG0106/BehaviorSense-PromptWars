"""CPU concurrency checks for clip ownership, admission, and cancellation."""

from __future__ import annotations

import asyncio
from concurrent.futures import Future, ThreadPoolExecutor
import json
import os
from pathlib import Path
import queue
import sys
import threading
import types
import unittest
from unittest.mock import Mock, patch

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))

from behaviorsense.service.clip_workers import (  # noqa: E402
    ClipJob, ClipQueueFull, ClipWorker, ClipWorkerPool, clip_response,
    max_parallel_clips, report_serialized,
)


def worker(index):
    classifier = Mock()
    classifier.predict.return_value = f"classifier-{index}"
    return ClipWorker(f"cuda:{index}", f"cuda:{index}", classifier)


class ClipWorkerPoolTests(unittest.TestCase):
    def test_admission_is_bounded_and_waiters_acquire_in_submission_order(self):
        workers = [worker(0), worker(1)]
        pool = ClipWorkerPool(workers, max_waiting=2)
        first, second = pool.submit(), pool.submit()
        self.assertIs(pool.acquire(first, timeout=0), workers[0])
        self.assertIs(pool.acquire(second, timeout=0), workers[1])
        third, fourth = pool.submit(), pool.submit()
        with self.assertRaises(ClipQueueFull):
            pool.submit()
        self.assertEqual(pool.snapshot(), {"max_parallel_clips": 2, "active_clips": 2,
                                          "queued_clips": 2, "max_queued_clips": 2})
        pool.release(first)
        self.assertIsNone(pool.acquire(fourth, timeout=0), "later requests must not jump the queue")
        self.assertIs(pool.acquire(third, timeout=0), workers[0])
        pool.release(second)
        self.assertIs(pool.acquire(fourth, timeout=0), workers[1])
        for ticket in (first, second, third, fourth):
            pool.release(ticket)
        self.assertEqual(pool.snapshot()["active_clips"], 0)
        self.assertEqual(pool.snapshot()["queued_clips"], 0)

    def test_single_gpu_only_admits_one_active_clip_and_releases_cancelled_waiter(self):
        gpu = worker(0)
        pool = ClipWorkerPool([gpu], max_waiting=2)
        first, second, third = pool.submit(), pool.submit(), pool.submit()
        self.assertIs(pool.acquire(first, timeout=0), gpu)
        self.assertIsNone(pool.acquire(second, timeout=0))
        self.assertIsNone(pool.acquire(third, timeout=0))
        pool.release(second)
        self.assertEqual(pool.snapshot()["queued_clips"], 1)
        pool.release(first)
        self.assertIs(pool.acquire(third, timeout=0), gpu)
        pool.release(third)
        with self.assertRaisesRegex(ValueError, "released"):
            pool.acquire(second)

    def test_invalid_devices_and_parallel_setting_fail_loudly(self):
        with self.assertRaises(ValueError):
            ClipWorkerPool([])
        with self.assertRaises(ValueError):
            ClipWorkerPool([worker(0), worker(0)])
        with self.assertRaises(ValueError):
            ClipWorkerPool([worker(0)], max_waiting=-1)
        with patch.dict(os.environ):
            os.environ.pop("BS_MAX_PARALLEL_CLIPS", None)
            self.assertEqual(max_parallel_clips(), 2)
            for raw in ("1", "2"):
                os.environ["BS_MAX_PARALLEL_CLIPS"] = raw
                self.assertEqual(max_parallel_clips(), int(raw))
            for raw in ("", "0", "3", "1.5", "many"):
                os.environ["BS_MAX_PARALLEL_CLIPS"] = raw
                with self.assertRaisesRegex(ValueError, "BS_MAX_PARALLEL_CLIPS"):
                    max_parallel_clips()


class ClipJobTests(unittest.TestCase):
    def test_two_clips_run_together_using_their_own_gpu_and_classifier(self):
        workers = [worker(0), worker(1)]
        pool = ClipWorkerPool(workers)
        entered = queue.Queue()
        finish = threading.Event()
        cleanups = [Mock(), Mock()]

        def process(assigned):
            entered.put(assigned)
            if not finish.wait(5):
                raise TimeoutError("test did not release the processing jobs")
            yield json.dumps({"event": "result", "device": assigned.device,
                              "classifier": assigned.classifier.predict()}) + "\n"

        jobs = [ClipJob(pool, pool.submit(), process, cleanup, max_frames=36000)
                for cleanup in cleanups]
        try:
            for job in jobs:
                job.start()
            active = [entered.get(timeout=3), entered.get(timeout=3)]
            self.assertEqual({item.device for item in active}, {"cuda:0", "cuda:1"})
            self.assertEqual({id(item.classifier) for item in active},
                             {id(item.classifier) for item in workers})
            self.assertEqual(pool.snapshot()["active_clips"], 2)
            self.assertTrue(all(not job.done.done() for job in jobs))
        finally:
            finish.set()
            for job in jobs:
                job.done.result(timeout=3)
        results = [[json.loads(line) for line in job.stream()] for job in jobs]
        payloads = [next(event for event in stream if event["event"] == "result") for stream in results]
        self.assertEqual({(item["device"], item["classifier"]) for item in payloads},
                         {("cuda:0", "classifier-0"), ("cuda:1", "classifier-1")})
        for cleanup in cleanups:
            cleanup.assert_called_once()
        self.assertEqual(pool.snapshot()["active_clips"], 0)

    def test_disconnect_keeps_slot_and_files_until_blocking_decode_and_iterator_close_finish(self):
        pool = ClipWorkerPool([worker(0)], max_waiting=1)
        decoding = threading.Event()
        finish_decode = threading.Event()
        cleaned = threading.Event()
        order = []

        def decode():
            decoding.set()
            if not finish_decode.wait(5):
                raise TimeoutError("test did not release the decoder")
            order.append("decode finished")

        def process(assigned):
            try:
                # Like the pipeline heartbeat wrapper, closing this iterator must
                # wait for the executor's active work before relinquishing the GPU.
                with ThreadPoolExecutor(max_workers=1) as executor:
                    future = executor.submit(decode)
                    while not future.done():
                        yield '{"event":"heartbeat"}\n'
                    future.result()
            finally:
                order.append("iterator closed")

        def cleanup():
            order.append("cleanup")
            self.assertEqual(pool.snapshot()["active_clips"], 1)
            cleaned.set()

        job = ClipJob(pool, pool.submit(), process, cleanup, max_frames=36000)
        next_ticket = None
        job.start()
        stream = job.stream()
        try:
            self.assertEqual(json.loads(next(stream))["event"], "accepted")
            self.assertTrue(decoding.wait(3))
            stream.close()  # The HTTP streaming consumer disconnects.
            self.assertTrue(job.cancelled.is_set())
            next_ticket = pool.submit()
            self.assertIsNone(pool.acquire(next_ticket, timeout=0.15))
            self.assertFalse(job.done.done())
            self.assertFalse(cleaned.is_set())
            self.assertEqual(pool.snapshot()["active_clips"], 1)
        finally:
            finish_decode.set()
            job.done.result(timeout=3)
            if next_ticket is not None:
                self.assertIsNotNone(pool.acquire(next_ticket, timeout=0.2))
                pool.release(next_ticket)
        self.assertEqual(order, ["decode finished", "iterator closed", "cleanup"])
        self.assertTrue(cleaned.is_set())

    def test_cancelling_a_queued_job_never_starts_processing(self):
        pool = ClipWorkerPool([worker(0)], max_waiting=1)
        active = pool.submit()
        pool.acquire(active)
        process, cleanup = Mock(), Mock()
        job = ClipJob(pool, pool.submit(), process, cleanup, max_frames=36000)
        try:
            job.start()
            stream = job.stream()
            self.assertEqual(json.loads(next(stream))["event"], "accepted")
            stream.close()
            job.done.result(timeout=3)
            process.assert_not_called()
            cleanup.assert_called_once()
            self.assertEqual(pool.snapshot()["active_clips"], 1)
            self.assertEqual(pool.snapshot()["queued_clips"], 0)
        finally:
            pool.release(active)

    def test_processing_failure_emits_error_and_releases_resources(self):
        pool = ClipWorkerPool([worker(0)])
        cleanup = Mock()
        process = Mock(side_effect=RuntimeError("bad decoder"))
        job = ClipJob(pool, pool.submit(), process, cleanup, max_frames=36000)
        job.start()
        events = [json.loads(line) for line in job.stream()]
        job.done.result(timeout=3)
        errors = [item for item in events if item["event"] == "error"]
        self.assertEqual(len(errors), 1)
        self.assertIn("bad decoder", errors[0]["detail"])
        cleanup.assert_called_once()
        self.assertEqual(pool.snapshot()["active_clips"], 0)

    def test_thread_start_failure_releases_reservation_and_cleans_upload(self):
        pool = ClipWorkerPool([worker(0)])
        cleanup = Mock()
        job = ClipJob(pool, pool.submit(), Mock(), cleanup, max_frames=36000)
        with patch.object(job._thread, "start", side_effect=RuntimeError("no thread")):
            with self.assertRaisesRegex(RuntimeError, "no thread"):
                job.start()
        cleanup.assert_called_once()
        self.assertTrue(job.done.done())
        self.assertEqual(pool.snapshot()["queued_clips"], 0)

    def test_asgi_send_failure_always_cancels_job(self):
        job = types.SimpleNamespace(stream=lambda: iter(['{"event":"accepted"}\n']),
                                    cancel=Mock(), done=Future())
        response = clip_response(job)

        async def send(message):
            raise OSError("client disconnected while sending headers")

        async def receive():
            await asyncio.Event().wait()

        with self.assertRaises(Exception):
            asyncio.run(response({"type": "http", "asgi": {"spec_version": "2.4"}}, receive, send))
        self.assertTrue(job.cancel.called)
        self.assertIs(response.upload_done, job.done)
        self.assertFalse(job.done.done(), "disconnect must not mark active work complete")


class ReporterSerializationTests(unittest.TestCase):
    def test_shared_reporter_cannot_overlap_or_mix_request_statistics(self):
        first_started, second_attempt, second_started = (threading.Event() for _ in range(3))
        finish_first = threading.Event()
        state = {"active": 0, "max_active": 0, "stats_owner": None}

        def report(name):
            state["active"] += 1
            state["max_active"] = max(state["max_active"], state["active"])
            state["stats_owner"] = name
            try:
                if name == "first":
                    first_started.set()
                    if not finish_first.wait(5):
                        raise TimeoutError("test did not release first report")
                else:
                    second_started.set()
                return {"name": name, "stats_owner": state["stats_owner"]}
            finally:
                state["active"] -= 1

        reporter = types.SimpleNamespace(report=report)

        def second():
            second_attempt.set()
            return report_serialized(reporter, "second")

        with ThreadPoolExecutor(max_workers=2) as executor:
            first = executor.submit(report_serialized, reporter, "first")
            other = None
            try:
                self.assertTrue(first_started.wait(3))
                other = executor.submit(second)
                self.assertTrue(second_attempt.wait(3))
                self.assertFalse(second_started.wait(0.15))
            finally:
                finish_first.set()
            self.assertEqual(first.result(timeout=3), {"name": "first", "stats_owner": "first"})
            self.assertIsNotNone(other)
            self.assertEqual(other.result(timeout=3), {"name": "second", "stats_owner": "second"})
        self.assertEqual(state["max_active"], 1)


if __name__ == "__main__":
    suite = unittest.defaultTestLoader.loadTestsFromModule(sys.modules[__name__])
    result = unittest.TextTestRunner(verbosity=1).run(suite)
    total = result.testsRun - len(result.skipped)
    passed = total - len(result.failures) - len(result.errors)
    print(f"{passed}/{total} passed; {len(result.skipped)} skipped")
    sys.exit(0 if result.wasSuccessful() else 1)
