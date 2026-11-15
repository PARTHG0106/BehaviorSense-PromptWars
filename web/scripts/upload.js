/* Two independent clip runs, with the selected clip shown in the existing report panel. */
import { state, uploadVideo, translateReport } from "bs/api";
import { SkeletonOverlay } from "bs/overlay";
import { renderStages } from "bs/stages";
import { mountCaregiverReport } from "bs/caregiver";
import { videoReport } from "bs/report-data";
import { getLanguage, onLanguageChange, t } from "bs/i18n";
import { createClipBatch, uploadLimits } from "./clip-batch.js";

const node = (id) => document.getElementById(id);

export function mountUpload() {
  const panel = node("upload");
  if (!panel) return null;
  const input = node("clip-input"), drop = node("clip-drop");
  const stage = node("clip-stage"), video = node("clip-video"), canvas = node("clip-canvas");
  const status = node("clip-status"), meta = node("clip-meta");
  const pipeline = node("clip-pipeline"), checks = node("clip-checks");
  const list = node("clip-jobs");
  const overlay = new SkeletonOverlay(canvas, video);
  const caregiverEl = node("caregiver-video");
  const caregiver = mountCaregiverReport(caregiverEl, { translate: translateReport });
  let active = null, displayedPayload = null, blobUrl = null;
  let currentStatus = { key: "uploadConnect", params: {} };
  const batch = createClipBatch({ upload: uploadVideo, onUpdate(job) {
    renderJobs();
    if (job === active) renderActive();
    updateControls();
  } });

  function say(key, tone = "", params = {}, detail = "") {
    currentStatus = { key, tone, params, detail };
    status.dataset.tone = tone;
    status.textContent = t(key, params);
    status.lang = getLanguage();
    status.title = detail;
  }
  function renderLimits() {
    node("clip-limits").textContent = t("uploadLimits", uploadLimits(state.health));
    node("clip-limits").lang = getLanguage();
    list.setAttribute("aria-label", t("clipJobsLabel"));
  }
  function updateControls() {
    input.disabled = batch.busy;
    node("clip-enrol").disabled = batch.busy;
    panel.setAttribute("aria-busy", String(batch.busy));
    drop.setAttribute("aria-disabled", String(batch.busy));
  }
  function renderJobs() {
    list.replaceChildren();
    list.hidden = batch.jobs.length === 0;
    for (const job of batch.jobs) {
      const button = document.createElement("button");
      button.type = "button";
      button.className = "clip-job";
      button.setAttribute("aria-pressed", String(job === active));
      button.textContent = `${job.file.name} · ${t(`clipPhase_${job.phase}`)}`;
      button.title = t(job.status.key, job.status.params);
      button.addEventListener("click", () => selectJob(job));
      list.append(button);
    }
  }
  function selectJob(job) {
    active = job;
    displayedPayload = null;
    overlay.stop();
    overlay.load({ tracks: [], fps: 20 });
    canvas.getContext("2d").clearRect(0, 0, canvas.width, canvas.height);
    if (blobUrl) URL.revokeObjectURL(blobUrl);
    blobUrl = URL.createObjectURL(job.file);
    video.src = blobUrl;
    stage.hidden = false;
    caregiverEl.hidden = true;
    caregiver.update({ claims: [], totalClaims: 0 });
    renderJobs();
    renderActive();
  }
  function renderActive() {
    if (!active) return;
    const { key, tone, params, detail } = active.status;
    say(key, tone, params, detail);
    if (!active.payload) {
      const span = active.stages.find((item) => item.agent === 1)?.payload || {};
      if (active.stages.length) {
        renderStages({ fps: span.fps || 20, frames_kept: span.frames_kept || 1, stages: active.stages },
          { pipelineEl: pipeline, checksEl: null });
      } else {
        pipeline.replaceChildren();
      }
      checks.replaceChildren();
      meta.textContent = active.file.name;
      return;
    }
    const data = active.payload;
    renderStages(data, { pipelineEl: pipeline, checksEl: checks });
    if (displayedPayload !== data) {
      displayedPayload = data;
      overlay.load(data); overlay.start();
      video.play().catch(() => {});
      caregiver.update(videoReport(data));
    }
    caregiverEl.hidden = false;
    const people = data.n_people ?? (data.tracks || []).length;
    const timing = data.timing || {};
    const subjectTracks = data.stages?.find((item) => item.agent === 2)?.payload?.subject_tracks || [];
    meta.textContent = [active.file.name, `${data.frames_kept} frames at ${data.fps} Hz`,
      `${data.width}×${data.height}`, `${people} ${people === 1 ? "track" : "tracks"}`
        + (subjectTracks.length > 1 ? ` (${subjectTracks.length} merged into one subject)` : ""),
      `pose ${timing.pose_s ?? "—"}s · classify ${timing.classify_s ?? "—"}s · behaviour ${timing.behaviour_s ?? "—"}s · report ${timing.report_s ?? "—"}s`,
      data.reid ? "re-id on" : "re-id off, roles unidentified", `τ=${data.tau}`,
    ].join(" · ") + (data.truncated ? ` · truncated at ${data.frames_kept} frames` : "");
    const mismatch = typeof overlay.sizeDisagreement === "function" ? overlay.sizeDisagreement() : null;
    if (mismatch) say("uploadOverlayMismatch", "bad", {}, mismatch);
  }
  async function analyse(files) {
    if (batch.busy || !files?.length) return;
    if (state.mode !== "live" || state.health?.video === false) {
      say(state.mode === "live" ? "uploadUnavailable" : "uploadConnect", "bad");
      input.value = "";
      return;
    }
    const error = batch.select(files, { health: state.health, enrol: (node("clip-enrol")?.value || "").trim() });
    if (error) { say(error.key, "bad", error.params); input.value = ""; return; }
    selectJob(batch.jobs[0]);
    const run = batch.run(state.health);
    updateControls();
    const runError = await run;
    if (runError) say(runError.key, "bad", runError.params);
    input.value = "";
    updateControls();
  }
  input.addEventListener("change", () => analyse(input.files));
  for (const name of ["dragenter", "dragover"]) drop.addEventListener(name, (event) => {
    event.preventDefault(); if (!batch.busy) drop.dataset.over = "1";
  });
  for (const name of ["dragleave", "drop"]) drop.addEventListener(name, (event) => {
    event.preventDefault(); delete drop.dataset.over;
    if (name === "drop") analyse(event.dataTransfer?.files);
  });
  video.addEventListener("loadedmetadata", () => overlay.resize());
  video.addEventListener("seeked", () => overlay.draw());
  window.addEventListener("resize", () => overlay.resize());
  onLanguageChange(() => {
    const { key, tone, params, detail } = currentStatus;
    say(key, tone, params, detail); renderLimits(); renderJobs();
  });
  renderLimits();
  return { sync(s) {
    const ok = s.mode === "live" && s.health?.video !== false;
    panel.dataset.ready = ok ? "1" : "0";
    renderLimits();
    if (!batch.busy && !active) say(ok ? "dropClip" : s.mode === "live" ? "uploadUnavailable" : "uploadConnect", ok ? "" : "bad");
  } };
}
