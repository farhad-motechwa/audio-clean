// Execute the actual page script with deterministic DOM, fetch and timer stubs.
const assert = require('node:assert/strict');
const fs = require('node:fs');
const vm = require('node:vm');
const path = require('node:path');
const script = fs.readFileSync(path.join(__dirname, '../static/index.html'), 'utf8').match(/<script>([\s\S]*?)<\/script>/)[1];
function harness() {
  const nodes = new Map(), timers = new Map(), calls = [];
  let timerId = 0;
  function el(id) {
    if (!nodes.has(id)) nodes.set(id, {
      style: {}, events: {}, hidden: true, disabled: true, checked: true,
      value: id === 'strength' ? '90' : 'wav', paused: true, currentTime: 0,
      addEventListener(t, f) { this.events[t] = f; }, setAttribute() {},
      pause() { this.paused = true; }, play() { this.paused = false; return Promise.resolve(); },
      getContext() { return { clearRect() {}, fillRect() {} }; }, clientWidth: 600, clientHeight: 120,
    });
    return nodes.get(id);
  }
  const h = { el, timers, calls, response: { ok: true, json: async () => ({ id: 'first', status: 'queued' }) } };
  const context = {
    document: { getElementById: el, querySelector: () => ({ value: 'deepfilter' }), documentElement: {} },
    window: { addEventListener() {}, devicePixelRatio: 1 },
    getComputedStyle: () => ({ getPropertyValue: () => '' }), requestAnimationFrame() {},
    setInterval(f, ms) { const id = ++timerId; timers.set(id, { f, ms }); return id; },
    clearInterval(id) { timers.delete(id); }, FormData: class { append() {} },
    async fetch(url) { calls.push(url); if (h.response instanceof Error) throw h.response; return h.response; }, console,
  };
  vm.createContext(context); vm.runInContext(script, context);
  h.choose = name => vm.runInContext(`choose({name:${JSON.stringify(name)},size:1024})`, context);
  h.poll = () => [...timers.values()].find(t => t.ms === 1000).f();
  h.submit = () => el('go').events.click();
  return h;
}
async function started() { const h = harness(); h.choose('first.wav'); await h.submit(); return h; }
const checks = [];
function test(name, fn) { checks.push([name, fn]); }
test('active upload locks file selection and repeated submission', async () => {
  const h = await started(); h.choose('second.wav'); await h.submit();
  assert.equal(h.el('go').disabled, true); assert.equal(h.el('file').disabled, true);
  assert.match(h.el('chosen').textContent, /^first.wav/); assert.equal(h.calls.length, 1);
});
test('network failure exposes retry and stops polling without duplicate upload', async () => {
  const h = await started(); h.response = new Error('connection lost'); await h.poll();
  assert.equal(h.timers.size, 0); assert.equal(h.el('retry').hidden, false);
  assert.equal(h.el('go').disabled, true); assert.match(h.el('status-text').textContent, /Retry/);
  h.el('retry').events.click();
  h.response = { ok: true, json: async () => ({ id: 'first', status: 'processing', started: 1, estimate: 10 }) };
  await h.poll(); assert.equal(h.calls.filter(u => u === '/api/jobs').length, 1);
  assert.equal(h.calls.at(-1), '/api/jobs/first'); assert.equal(h.el('retry').hidden, true);
});
test('404 permits a new submission and displays the error', async () => {
  const h = await started(); h.response = { ok: false, status: 404 }; await h.poll();
  assert.equal(h.el('go').disabled, false); assert.equal(h.el('file').disabled, false);
  assert.equal(h.el('retry').hidden, true); assert.equal(h.timers.size, 0);
  assert.match(h.el('status-text').textContent, /Job not found/);
});
test('HTTP 500 and malformed status offer recovery', async () => {
  for (const response of [{ ok: false, status: 500 }, { ok: true, json: async () => ({}) }]) {
    const h = await started(); h.response = response; await h.poll();
    assert.equal(h.el('retry').hidden, false); assert.equal(h.timers.size, 0);
  }
});
test('engine failure unlocks the interface', async () => {
  const h = await started(); h.response = { ok: true, json: async () => ({ status: 'error', error: 'engine failed' }) };
  await h.poll(); assert.equal(h.el('go').disabled, false); assert.equal(h.timers.size, 0);
  assert.match(h.el('status-text').textContent, /engine failed/);
});
test('successful result restores controls and uses the tracked job URL', async () => {
  const h = await started();
  h.response = { ok: true, json: async () => ({ status: 'done', started: 1, finished: 3 }) };
  await h.poll(); assert.equal(h.el('go').disabled, false); assert.equal(h.timers.size, 0);
  assert.equal(h.el('download').href, '/api/jobs/first/result?download=true');
});
test('upload failure restores controls', async () => {
  const h = harness(); h.choose('first.wav'); h.response = new Error('offline'); await h.submit();
  assert.equal(h.el('go').disabled, false); assert.equal(h.el('file').disabled, false);
});
test('overlapping status requests are suppressed', async () => {
  const h = await started(); let resolve;
  h.response = { ok: true, json: () => new Promise(r => { resolve = r; }) };
  const first = h.poll(); await Promise.resolve(); await h.poll();
  assert.equal(h.calls.length, 2); resolve({ status: 'queued' }); await first;
});
(async () => {
  for (const [name, fn] of checks) { await fn(); console.log(`PASS ${name}`); }
  console.log(`${checks.length} frontend regression checks passed`);
})().catch(e => { console.error(e); process.exitCode = 1; });
