import test from 'node:test';
import assert from 'node:assert/strict';
import { createClipBatch, uploadLimits, validateClips } from '../web/scripts/clip-batch.js';
import { t } from '../web/scripts/i18n.js';

const clip = (name, size = 10) => ({ name, size, type: 'video/mp4' });
const result = (name) => ({ name, checks: [], stages: [] });

test('connected backend limits replace defaults and accept the exact byte limit', () => {
  assert.deepEqual(uploadLimits(null), { maxUploadMb: 100, maxFrames: 36000, sampleFps: 20, minutes: 30 });
  const health = { max_upload_mb: 60, max_frames: 12000, sample_fps: 15 };
  assert.equal(uploadLimits(health).minutes, 13.3);
  assert.equal(validateClips([clip('a.mp4', 60 * 1024 * 1024)], health), null);
  assert.deepEqual(validateClips([clip('a.mp4', 60 * 1024 * 1024 + 1)], health),
    { key: 'uploadTooLarge', params: { maxUploadMb: 60 } });
  assert.equal(validateClips([clip('a'), clip('b'), clip('c')], health).key, 'uploadTooMany');
  assert.equal(validateClips([{ name: 'empty.mp4', size: 0 }], health).key, 'uploadNotVideo');
  assert.deepEqual(uploadLimits({ max_upload_mb: -2, max_frames: NaN, sample_fps: Infinity }), uploadLimits());
});

test('both clips start together and keep stages and out-of-order results separate', async () => {
  const pending = new Map();
  const batch = createClipBatch({ upload(file, callbacks) {
    return new Promise((resolve, reject) => pending.set(file.name, { resolve, reject, callbacks }));
  } });
  assert.equal(batch.select([clip('first.mp4'), clip('second.mp4')], { enrol: 'Resident' }), null);
  const running = batch.run();
  assert.deepEqual([...pending.keys()], ['first.mp4', 'second.mp4']);
  assert.equal(batch.busy, true);
  assert.equal(batch.select([clip('replace.mp4')]).key, 'uploadBusy');
  pending.get('first.mp4').callbacks.onEvent({ event: 'progress', step: 'queued' });
  pending.get('second.mp4').callbacks.onStage({ agent: 1, status: 'done', payload: { frames_kept: 25 } });
  assert.equal(batch.jobs[0].phase, 'queued');
  assert.equal(batch.jobs[0].stages.length, 0);
  assert.equal(batch.jobs[1].stages[0].payload.frames_kept, 25);
  assert.equal(pending.get('second.mp4').callbacks.enrol, 'Resident');
  pending.get('second.mp4').resolve(result('second'));
  await new Promise((resolve) => setImmediate(resolve));
  assert.equal(batch.jobs[1].payload.name, 'second');
  assert.equal(batch.jobs[0].payload, null);
  pending.get('first.mp4').callbacks.onEvent({ event: 'progress', step: 'processing' });
  assert.equal(batch.jobs[0].phase, 'running');
  pending.get('first.mp4').resolve(result('first'));
  await running;
  assert.equal(batch.busy, false);
  assert.deepEqual(batch.jobs.map((job) => job.payload.name), ['first', 'second']);
});

test('one upload failure does not cancel or replace the other result', async () => {
  const batch = createClipBatch({ upload: async (file) => {
    if (file.name === 'bad.mp4') throw new Error('Upload rejected');
    return result('good');
  } });
  batch.select([clip('bad.mp4'), clip('good.mp4')]);
  await batch.run();
  assert.equal(batch.jobs[0].phase, 'failed');
  assert.equal(batch.jobs[0].status.detail, 'Upload rejected');
  assert.equal(batch.jobs[1].payload.name, 'good');
  assert.equal(batch.jobs[1].phase, 'done');
});

test('limits are checked again when starting after a backend switch', async () => {
  let requests = 0;
  const batch = createClipBatch({ upload: async () => { requests++; } });
  batch.select([clip('large.mp4', 90 * 1024 * 1024)]);
  const error = await batch.run({ max_upload_mb: 60 });
  assert.equal(error.key, 'uploadTooLarge');
  assert.equal(requests, 0);
  assert.equal(batch.busy, false);
});

test('all languages show the connected limits without leftover placeholders', () => {
  const params = uploadLimits({ max_upload_mb: 75, max_frames: 18000, sample_fps: 10 });
  for (const language of ['en', 'hi', 'mr']) {
    const text = t('uploadLimits', params, language);
    assert.match(text, /75/); assert.match(text, /18000/); assert.match(text, /30/);
    assert.doesNotMatch(text, /\{\w+\}/);
    assert.match(t('uploadTooLarge', params, language), /75/);
    if (language !== 'en') assert.match(text, /[\u0900-\u097f]/u);
  }
});
