// Puts NVIDIA's GPC prior in the page brain's seat. Big part in a background worker (a new set of picks whenever it
// finishes one), small part (decoder, 3 M numbers) here every control tick on his current body.
export async function attachGPC(brain, ort, say) {
  const here = new URL('./', import.meta.url).href, iso = self.crossOriginIsolated, cores = navigator.hardwareConcurrency || 2;
  // the big part gets 2 cores by default (4 starved the game's own physics and drawing: user 23:30, poor frame rate); ?cores=1..4 to try others
  const ask = +(new URLSearchParams(location.search).get('cores') || 0);
  const threads = iso ? Math.max(1, Math.min(ask || 2, cores - 1, 4)) : 1;
  let fps = 0, fN = 0, fT = performance.now(); const raf = () => { fN++; const n = performance.now(); if (n - fT >= 1000) { fps = fN * 1000 / (n - fT); fN = 0; fT = n; } requestAnimationFrame(raf); }; requestAnimationFrame(raf);
  const w = new Worker(here + 'worker.js?v=5', { type: 'module' }), tracker = brain.session, st = { codes: null, busy: false, ms: 0, picks: 0, ticks: 0, sinceT: performance.now(), since: 0, rate: 0, err: null, loaded: {}, ready: false, threads };
  const TOTAL = 194.3e6;
  w.onmessage = e => {
    const m = e.data;
    if (m.load) { st.loaded[m.load] = m.got; const g = Object.values(st.loaded).reduce((a, b) => a + b, 0); say(`NVIDIA brain: downloading ${(g / 1e6).toFixed(0)} of ${(TOTAL / 1e6).toFixed(0)} MB (once; kept by the phone after)`); }
    else if (m.stage) say('NVIDIA brain: starting up…');
    else if (m.ready) { st.ready = true; say('NVIDIA brain: ready'); }
    else if (m.codes) { st.codes = m.codes; st.busy = false; st.ms += (m.ms - st.ms) * (st.picks ? 0.2 : 1); st.picks++; }
    else if (m.error) { st.err = m.error; st.busy = false; say('NVIDIA brain failed: ' + m.error.slice(0, 120)); }
  };
  w.postMessage({ init: true, threads });
  const dec = await ort.InferenceSession.create(new Uint8Array(await (await fetch(here + 'live/dec.onnx')).arrayBuffer()), { executionProviders: ['wasm'], graphOptimizationLevel: 'all' });
  brain.meta.getup = true;   // never the tracker's "gave up, go limp": this brain keeps control on the floor too
  let last = 0;
  brain.session = { run: async feeds => {
    const o = feeds.obs.data.subarray(0, 343);
    if (st.ready && !st.busy) { st.busy = true; w.postMessage({ obs: new Float32Array(o) }); }
    if (!st.codes) return tracker.run(feeds);   // until the first picks arrive he stands with the page's own tracker
    st.ticks++;
    const now = performance.now(); if (now - last > 1000) { st.rate = st.picks - st.since; st.since = st.picks; last = now; say(`${fps.toFixed(0)} fps · NVIDIA brain live · ${st.rate} picks a second · ${st.ms.toFixed(0)} ms each · ${threads} core${threads > 1 ? 's' : ''}${iso ? '' : ' (several-core mode refused)'}`); }
    return dec.run({ obs: new ort.Tensor('float32', new Float32Array(o), [1, 343]), codes: new ort.Tensor('float32', st.codes, [1, 40]) });
  } };
  return st;
}

// The latest test page's driver seat (kin.html uses drive.update / reset / pins / noFlat / brain): here the NVIDIA
// brain alone moves his joints. No recordings, no springs, no director. The page's human joint limits stay on.
import { humanCap } from '../sim/springs.js';
const STRONG = /[?&]strong/.test(location.search);   // ?strong = NVIDIA's training strength instead of a person's
export class GpcDrive {
  constructor(W, kin, brain) {
    this.W = W; this.kin = kin; this.brain = brain; this.G = kin.W; this.pins = null; this.noFlat = true; this.stats = {}; this.mode = 'gpc';
    kin.ext = W; brain.attach(); brain.told = true; this.cap();
    this.self = { get pos() { return brain.pos; }, get rot() { return brain.rot; }, get vel() { return brain.vel; }, get ang() { return brain.ang; } };
  }
  reset() {
    const W = this.W, d = W.d, g = this.G.d;
    for (let k = 0; k < 73; k++) d.qpos[k] = g.qpos[k];
    d.qvel.fill(0); d.xfrc_applied.fill(0); d.qfrc_applied.fill(0);
    W.mj.mj_forward(W.m, d); W.tControl = d.time; this.brain.attach(); this.brain.told = true; this.cap();
  }
  // each muscle no stronger than a person's (the page's own table, sim/springs.js)
  cap() { if (STRONG) return; const m = this.W.m; this.W.act.forEach((a, i) => { const c = humanCap(a.name); m.actuator_forcerange[2 * i] = -c; m.actuator_forcerange[2 * i + 1] = c; m.actuator_forcelimited[i] = 1; }); }
  async update(dt, fingerPush) {
    const W = this.W, d = W.d, B = this.brain, n = Math.min(40, Math.round(dt * 480));
    for (let i = 0; i < n; i++) {
      d.xfrc_applied.fill(0); if (fingerPush) fingerPush();
      if (B.due()) { B.down = false; B.lost = 0; await B.think(this.self); if (B.down) { B.down = false; B.setTone(1); } }
      W.step(B.control);
    }
  }
}
