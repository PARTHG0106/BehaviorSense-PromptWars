import test from 'node:test';
import assert from 'node:assert/strict';
import { readFileSync } from 'node:fs';

const source = readFileSync(new URL('../web/scripts/api.js', import.meta.url), 'utf8')
  .replace('"bs/translation-queue"', JSON.stringify(new URL('../web/scripts/translation-queue.js', import.meta.url).href))
  .replace('"bs/i18n"', JSON.stringify(new URL('../web/scripts/i18n.js', import.meta.url).href));
const { state, uploadVideo } = await import(`data:text/javascript;base64,${Buffer.from(source).toString('base64')}`);
const json = (data, status = 200) => new Response(JSON.stringify(data), { status, headers: { 'Content-Type': 'application/json' } });
const stream = (name) => new Response([
  JSON.stringify({ event: 'accepted' }),
  JSON.stringify({ event: 'stage', stage: { agent: 1, status: 'done' } }),
  JSON.stringify({ event: 'result', payload: { name } }), '',
].join('\n'), { headers: { 'Content-Type': 'application/x-ndjson' } });

test('large clips send bounded chunks and preserve enrolment and stream callbacks', async (context) => {
  state.base = 'https://test.invalid'; state.token = 'service-token';
  state.health = { upload_protocol: 'chunks-v1', upload_chunk_bytes: 8 * 1024 * 1024 };
  const requests = [], stages = [], events = [];
  context.mock.method(globalThis, 'fetch', async (url, options) => {
    requests.push({ url, ...options });
    assert.equal(options.headers['X-BS-Token'], 'service-token');
    if (url.endsWith('/uploads')) {
      assert.deepEqual(JSON.parse(options.body), { filename: 'large.mp4', size: 100 * 1024 * 1024, enrol: 'Resident' });
      return json({ upload_id: 'one', upload_token: 'upload-token', chunk_bytes: 8 * 1024 * 1024, offset: 0 });
    }
    assert.equal(options.headers['X-BS-Upload-Token'], 'upload-token');
    if (options.method === 'PUT') {
      assert.ok(options.body.size <= 8 * 1024 * 1024);
      return json({ offset: Number(new URL(url).searchParams.get('offset')) + options.body.size });
    }
    assert.equal(options.headers['X-BS-Enrol'], 'Resident');
    return stream('large');
  });
  const file = { name: 'large.mp4', size: 100 * 1024 * 1024, slice: (start, end) => ({ size: end - start }) };
  const payload = await uploadVideo(file, { enrol: 'Resident', onStage: (stage) => stages.push(stage), onEvent: (event) => events.push(event) });
  assert.equal(payload.name, 'large');
  assert.equal(requests.filter((request) => request.method === 'PUT').length, 13);
  assert.equal(stages.length, 1);
  assert.equal(events.filter((event) => event.step === 'uploading').at(-1).uploaded_bytes, file.size);
  assert.match(requests.at(-1).url, /\/complete\?stages=1234$/);
});

test('a rejected chunk is cleaned up without starting inference', async (context) => {
  state.health = { upload_protocol: 'chunks-v1' };
  const methods = [];
  context.mock.method(globalThis, 'fetch', async (url, options) => {
    methods.push(options.method);
    if (url.endsWith('/uploads')) return json({ upload_id: 'two', upload_token: 'token', chunk_bytes: 8 });
    if (options.method === 'PUT') return json({ detail: 'Chunk rejected' }, 413);
    return json({ deleted: true });
  });
  const file = new File(['a short file'], 'clip.mp4', { type: 'video/mp4' });
  await assert.rejects(uploadVideo(file), /Chunk rejected/);
  assert.deepEqual(methods, ['POST', 'PUT', 'DELETE']);
});

test('older backends retain multipart compatibility', async (context) => {
  state.health = { max_upload_mb: 60 };
  let calls = 0;
  context.mock.method(globalThis, 'fetch', async (url, options) => {
    calls++; assert.equal(url, 'https://test.invalid/video');
    assert.ok(options.body instanceof FormData);
    assert.equal(options.body.get('file').name, 'clip.mp4');
    return stream('legacy');
  });
  assert.equal((await uploadVideo(new File(['clip'], 'clip.mp4', { type: 'video/mp4' }))).name, 'legacy');
  assert.equal(calls, 1);
});
