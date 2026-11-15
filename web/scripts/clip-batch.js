/* Each upload owns its progress and result. GPU scheduling belongs to the backend. */
import { videoReport } from "./report-data.js";

export function uploadLimits(health = {}) {
  const positive = (value, fallback) => typeof value === "number" && Number.isFinite(value) && value > 0 ? value : fallback;
  const maxUploadMb = positive(health?.max_upload_mb, 100);
  const maxFrames = Math.floor(positive(health?.max_frames, 36000));
  const sampleFps = positive(health?.sample_fps, 20);
  return { maxUploadMb, maxFrames, sampleFps, minutes: Math.round(maxFrames / sampleFps / 60 * 10) / 10 };
}

export function validateClips(files, health) {
  const selected = Array.from(files || []);
  if (selected.length > 2) return { key: "uploadTooMany", params: { count: 2 } };
  const { maxUploadMb } = uploadLimits(health);
  for (const file of selected) {
    if (file.size > maxUploadMb * 1024 * 1024) return { key: "uploadTooLarge", params: { maxUploadMb } };
    if (!file.size || (file.type ? !file.type.startsWith("video/") : !/\.(mp4|webm|mov|avi|mkv|m4v|mpeg|mpg|ogv)$/i.test(file.name))) {
      return { key: "uploadNotVideo", params: {} };
    }
  }
  return null;
}

export function createClipBatch({ upload, onUpdate = () => {} }) {
  let jobs = [], busy = false;
  const notify = (job) => onUpdate(job);
  const message = (job, key, params = {}, tone = "", detail = "") => {
    job.status = { key, params, tone, detail };
  };
  async function runJob(job) {
    job.phase = "running";
    job.stages = [];
    job.payload = null;
    message(job, "uploadRunning", { name: job.file.name }, "wait");
    notify(job);
    try {
      job.payload = await upload(job.file, {
        enrol: job.enrol,
        onEvent(event) {
          if (event.event === "progress" && event.step === "uploading") {
            const percent = Math.round(event.uploaded_bytes / event.total_bytes * 100);
            message(job, "uploadSending", { percent }, "wait");
            notify(job);
          } else if ((event.event === "progress" && event.step === "queued") || (event.event === "accepted" && event.queued)) {
            job.phase = "queued";
            message(job, "uploadQueued", {}, "wait");
            notify(job);
          } else if (event.event === "progress" && ["started", "processing", "worker_ready"].includes(event.step)) {
            job.phase = "running";
            message(job, "uploadRunning", { name: job.file.name }, "wait");
            notify(job);
          }
        },
        onStage(stage) {
          job.phase = "running";
          const index = job.stages.findIndex((item) => item.agent === stage.agent);
          if (index < 0) job.stages.push(stage); else job.stages[index] = stage;
          if (stage.agent < 4) message(job, `uploadStage${stage.agent}`, {}, "wait");
          notify(job);
        },
      });
      job.stages = job.payload.stages || job.stages;
      job.phase = "done";
      const failed = job.stages.find((stage) => stage.status === "failed");
      const report = videoReport(job.payload);
      if (failed) message(job, "uploadStageFailed", { agent: failed.agent }, "bad");
      else message(job, report.totalClaims === 0 ? "uploadNoClaims" : report.withheldClaims ? "uploadWithheld" : "uploadDone",
        { count: report.withheldClaims || report.claims.length });
    } catch (error) {
      job.phase = "failed";
      message(job, "uploadFailed", {}, "bad", error.message);
    }
    notify(job);
  }
  return {
    get jobs() { return jobs; },
    get busy() { return busy; },
    select(files, { health, enrol = "" } = {}) {
      if (busy) return { key: "uploadBusy", params: {} };
      const error = validateClips(files, health);
      if (error) return error;
      jobs = Array.from(files || [], (file, index) => ({ id: index, file, enrol,
        phase: "ready", stages: [], payload: null, status: { key: "uploadReady", params: {} } }));
      return null;
    },
    async run(health) {
      if (busy || !jobs.length) return;
      const error = validateClips(jobs.map((job) => job.file), health);
      if (error) return error;
      busy = true;
      try { await Promise.all(jobs.map(runJob)); }
      finally { busy = false; notify(null); }
    },
  };
}
