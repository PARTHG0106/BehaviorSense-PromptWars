"""Durable coordination for the two prespecified, independent seed groups."""
from __future__ import annotations

from pathlib import Path
import re


def validate_group_result(stage, result, plan):
    """Reject untrusted output before it can become a resume or selection source."""
    expected = {
        "job_id": stage["active"], "stage": "train_group", "model": stage["model"],
        "seeds": [0, 1, 2], "source_sha256": plan["source_sha256"],
        "seed_runner_sha256": plan["seed_runner_sha256"],
        "feature_runner_sha256": plan["feature_runner_sha256"],
    }
    for key, value in expected.items():
        if result.get(key) != value:
            raise ValueError(f"Training group result {key} mismatch")
    if result.get("status") not in ("complete", "resumable", "needs_review"):
        raise ValueError("Training group has no safe terminal result")
    candidates = result.get("candidates", [])
    if (not isinstance(candidates, list) or len(candidates) != 3
            or any(not isinstance(item, dict) for item in candidates)
            or any(type(item.get("seed")) is not int for item in candidates)
            or {item["seed"] for item in candidates} != {0, 1, 2}):
        raise ValueError("Training group requires exactly three unique seeds")
    states = []
    for item in candidates:
        if item.get("model") != stage["model"]:
            raise ValueError("Training group candidate model mismatch")
        state = item.get("status")
        if state not in ("complete", "resumable", "pending", "needs_review"):
            raise ValueError("Training group candidate has an invalid status")
        states.append(state)
        if state == "complete":
            manifest = item.get("manifest", {})
            if manifest.get("convergence", {}).get("converged") is not True:
                raise ValueError("Completed training candidate has not converged")
            if (item.get("checkpoint_relative_path") !=
                    f"candidates/seed{item['seed']}/best_multilabel.pt"):
                raise ValueError("Completed candidate checkpoint path is not canonical")
            if not re.fullmatch(r"[0-9a-f]{64}", str(item.get("checkpoint_sha256", ""))):
                raise ValueError("Completed candidate checkpoint hash is invalid")
        elif state in ("resumable", "needs_review"):
            if (item.get("resume_checkpoint_relative_path") !=
                    f"candidates/seed{item['seed']}/last_multilabel.pt"):
                raise ValueError("Resume candidate checkpoint path is not canonical")
            if not re.fullmatch(r"[0-9a-f]{64}", str(item.get("resume_checkpoint_sha256", ""))):
                raise ValueError("Resume candidate checkpoint hash is invalid")
    if result["status"] == "complete" and any(state != "complete" for state in states):
        raise ValueError("Completed training group contains an unfinished candidate")
    if result["status"] == "resumable" and (
            "needs_review" in states or all(state == "complete" for state in states)):
        raise ValueError("Resumable training group has inconsistent candidate states")
    if result["status"] == "needs_review" and "needs_review" not in states:
        raise ValueError("Training group review status has no candidate requiring review")


def _groups(plan):
    stages = plan["stages"]
    start = plan["next_stage"]
    while start > 0 and stages[start - 1]["stage"] == "train_group":
        start -= 1
    stop = start
    while stop < len(stages) and stages[stop]["stage"] == "train_group":
        stop += 1
    groups = stages[start:stop]
    if (len(groups) != 2 or {stage.get("model") for stage in groups} !=
            {"skeleton", "context-fusion"} or any(stage.get("seeds") != [0, 1, 2]
                                                  for stage in groups)
            or len({stage["key"] for stage in groups}) != 2):
        raise ValueError("Parallel training requires two adjacent model groups with seeds 0, 1, 2")
    return start, stop, groups


def advance_parallel(api, directory, plan):
    """Advance both groups fairly, journaling every remote transition separately.

    Submit vacant siblings first. A failed or slow first group must not prevent
    the other from starting, publishing its result, or resuming safely.
    """
    from launch_multilabel_kaggle import (OutputPendingError, api_document,
                                         prepare_submission, read_json,
                                         retryable_error, write_json)

    directory = Path(directory)
    start, stop, groups = _groups(plan)
    changed = False
    transient = None
    newly_submitted = set()

    def persist():
        active = [stage["active"] for stage in groups if stage.get("active")]
        if len(active) != len(set(active)):
            raise ValueError("Parallel groups share an active notebook identity")
        plan["active_jobs"] = active
        plan["active"] = active[0] if active else None
        plan["parallel_errors"] = {stage["key"]: stage["error"] for stage in groups
                                   if stage.get("error")}
        plan["next_stage"] = next((start + i for i, stage in enumerate(groups)
                                   if not stage.get("complete")), stop)
        write_json(directory / "plan.json", plan)

    def fail(stage, message, *, clear_active=True):
        nonlocal changed
        stage["error"] = str(message)
        if stage.get("active"):
            attempt = next((item for item in reversed(stage["attempts"])
                            if item["ref"] == stage["active"]), None)
            if attempt is not None:
                attempt["failure"] = str(message)
        if clear_active:
            stage["active"] = None
        changed = True
        persist()

    # Recover compatibility aliases from authoritative per-stage journals.
    persist()
    for stage in groups:
        if stage.get("complete") or stage.get("active") or stage.get("error"):
            continue
        if len(stage["attempts"]) >= 16:
            fail(stage, "Training group exceeded 16 resume sessions")
            continue
        try:
            ref, submission = prepare_submission(directory, plan, stage)
        except Exception as exc:
            fail(stage, f"Cannot prepare training group: {type(exc).__name__}: {exc}")
            continue
        stage["active"] = ref
        attempt = {"ref": ref, "submitted": False}
        stage["attempts"].append(attempt)
        newly_submitted.add(ref)
        plan["status"] = "submitting"
        persist()  # A lost response must never cause a duplicate submission.
        changed = True
        try:
            response = api_document(api.kernels_push(str(submission), timeout="43200",
                                                      acc="NvidiaRtxPro6000"))
        except Exception as exc:
            attempt["submission_error"] = f"{type(exc).__name__}: {exc}"
            if retryable_error(exc):
                transient = transient or exc
                persist()
            else:
                # Preserve an uncertain remote identity, but require review.
                fail(stage, attempt["submission_error"], clear_active=False)
            continue
        attempt["response"] = response
        if response.get("error"):
            attempt["rejection"] = response
            stage.setdefault("rejected_submissions", []).append(stage["attempts"].pop())
            fail(stage, f"Kaggle submission rejected: {response['error']}")
            continue
        actual_ref = str(response.get("ref", "")).removeprefix("/code/")
        if actual_ref != ref:
            attempt["actual_ref"] = actual_ref
            fail(stage, f"Kaggle changed the requested notebook identity to {actual_ref}",
                 clear_active=False)
            continue
        invalid = [name for name in ("invalidDatasetSources", "invalidCompetitionSources",
                                    "invalidKernelSources", "invalidModelSources")
                   if response.get(name)]
        attempt["submitted"] = True
        if invalid:
            fail(stage, f"Kaggle rejected required input sources: {', '.join(invalid)}",
                 clear_active=False)
        else:
            persist()
            print(f"Submitted private stage https://www.kaggle.com/code/{ref}", flush=True)

    # Newly submitted notebooks get one pass for Kaggle's status to appear.
    for stage in groups:
        ref = stage.get("active")
        if not ref or ref in newly_submitted:
            continue
        attempt = next((item for item in reversed(stage["attempts"])
                        if item["ref"] == ref), None)
        if attempt is None:
            fail(stage, "Active training group has no journaled submission")
            continue
        try:
            response = api_document(api.kernels_status(ref))
            stage.pop("retry_error", None)
            stage["remote_status"] = response
            attempt["remote_status"] = response
            state = response["status"]
            if state in ("QUEUED", "RUNNING", "INITIALIZING"):
                persist()
                continue
            if state != "COMPLETE":
                fail(stage, f"{ref}: {state}: {response.get('failureMessage')}")
                continue
            out = directory / ref.split("/")[1] / "output"
            out.mkdir(parents=True, exist_ok=True)
            api.kernels_output(ref, str(out), file_pattern=r"(^|/)stage_result\.json$",
                               force=True, quiet=True)
            matches = list(out.rglob("stage_result.json"))
            if not matches:
                raise OutputPendingError("Completed training group has not published stage_result.json yet")
            if len(matches) != 1:
                raise ValueError("Training group output has ambiguous stage_result.json")
            result = read_json(matches[0])
            validate_group_result(stage, result, plan)
            attempt["result"] = result
            stage["result"] = result
            stage["active"] = None
            changed = True
            if stage.get("error"):
                # A bad submission response cannot be repaired by job completion.
                persist()
                continue
            if result["status"] == "complete":
                stage["complete"] = True
            elif result["status"] == "needs_review":
                fail(stage, "Training group requires review before further compute")
            elif len(stage["attempts"]) >= 16:
                fail(stage, "Training group exceeded 16 resume sessions")
            persist()
            print(f"{stage['key']}: {result['status']}", flush=True)
        except Exception as exc:
            if retryable_error(exc):
                transient = transient or exc
                stage["retry_error"] = f"{type(exc).__name__}: {exc}"
                persist()
            else:
                fail(stage, f"{type(exc).__name__}: {exc}")

    if all(stage.get("complete") for stage in groups):
        plan["status"] = "ready"
        persist()
        return True
    active = any(stage.get("active") for stage in groups)
    can_resume = any(not stage.get("complete") and not stage.get("error")
                     and not stage.get("active") for stage in groups)
    if not active and not can_resume:
        plan["status"] = "needs_attention"
        persist()
        errors = "; ".join(f"{stage['key']}: {stage['error']}" for stage in groups
                           if stage.get("error"))
        raise RuntimeError(f"Parallel training requires review: {errors}")
    plan["status"] = "running" if active else "ready"
    persist()
    if transient is not None:
        raise transient
    return changed
