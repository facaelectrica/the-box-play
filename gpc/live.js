// Puts NVIDIA's GPC prior in the page brain's seat. Big part in a background worker (a new set of picks whenever it
// finishes one), small part (decoder, 3 M numbers) here every control tick on his current body.
export async function attachGPC(brain, ort, say) {
  const here = new URL('./', import.meta.url).href, iso = self.crossOriginIsolated, cores = navigator.hardwareConcurrency || 2;
  // the big part's kernel on 3 cores by default when the browser allows shared memory (?cores=1..4 to try others)
  const ask = +(new URLSearchParams(location.search).get('cores') || 0);
  let threads = iso ? (ask ? Math.max(1, Math.min(ask, 4)) : Math.max(1, Math.min(3, cores - 1))) : 1;
  let fps = 0, fN = 0, fT = performance.now(); const raf = () => { fN++; const n = performance.now(); if (n - fT >= 1000) { fps = fN * 1000 / (n - fT); fN = 0; fT = n; } requestAnimationFrame(raf); }; requestAnimationFrame(raf);
  const w = new Worker(here + 'worker.js?v=10', { type: 'module' }), tracker = brain.session, st = { codes: null, busy: false, ms: 0, picks: 0, ticks: 0, sinceT: performance.now(), since: 0, rate: 0, err: null, loaded: {}, ready: false, threads };
  const TOTAL = 195.9e6;
  w.onmessage = e => {
    const m = e.data;
    if (m.load) { st.loaded[m.load] = m.got; const g = Object.values(st.loaded).reduce((a, b) => a + b, 0); say(`NVIDIA brain: downloading ${(g / 1e6).toFixed(0)} of ${(TOTAL / 1e6).toFixed(0)} MB (once; kept by the phone after)`); }
    else if (m.stage) say('NVIDIA brain: starting up…');
    else if (m.ready) { st.ready = true; threads = m.threads || 1; st.threads = threads; st.loadS = +((performance.now() - st.t0) / 1000).toFixed(1); say('NVIDIA brain: ready'); }
    else if (m.codes) { st.codes = m.codes; st.busy = false; st.fresh = st.askMode === st.mode; st.ms += (m.ms - st.ms) * (st.picks ? 0.2 : 1); st.picks++;
      const now = performance.now(); st.hist.push([+((now - st.t0) / 1000).toFixed(1), Math.round(m.ms), Math.round(now - st.sent)]); if (st.hist.length > 400) st.hist.splice(0, st.hist.length - 400); st.got = now; }
    else if (m.error) { st.err = m.error; st.busy = false; say('NVIDIA brain failed: ' + m.error.slice(0, 120)); }
  };
  st.hist = []; st.t0 = performance.now(); st.fpsHist = [];
  // for snapshots: what the phone was doing (user 23:45: more data instead of screenshots)
  st.report = () => ({ engine: 'gemv.wasm', engineCores: st.threads, mode: st.mode, head: st.head && +st.head.toFixed(2), hips: st.hips && +st.hips.toFixed(2), switchesNote: '[seconds since load, down|up, head height]', switches: st.switches, picks: st.picks, ticks: st.ticks, upS: +((performance.now() - st.t0) / 1000).toFixed(0), loadS: st.loadS, msNow: Math.round(st.ms), picksPerS: st.rate, fps: Math.round(fps),
    pickAgeMs: st.got ? Math.round(performance.now() - st.got) : null, err: st.err, strong: /[?&]strong/.test(location.search),
    cores: navigator.hardwareConcurrency, memGB: navigator.deviceMemory, isolated: !!self.crossOriginIsolated, ua: navigator.userAgent, screen: [screen.width, screen.height, devicePixelRatio], hidden: document.hidden,
    fpsEach5s: st.fpsHist.slice(-120), picksNote: '[seconds since load, ms to compute, ms from asking to having]', picksLast: st.hist.slice(-400) });
  setInterval(() => { st.fpsHist.push(Math.round(fps)); if (st.fpsHist.length > 240) st.fpsHist.shift(); }, 5000);
  w.postMessage({ init: true, threads });
  const load = async n => new Uint8Array(await (await fetch(here + n)).arrayBuffer()), so = { executionProviders: ['wasm'], graphOptimizationLevel: 'all' };
  const dec = await ort.InferenceSession.create(await load('live/dec.onnx'), so), enc = await ort.InferenceSession.create(await load('live/enc.onnx'), so);
  brain.meta.getup = true;   // never the tracker's "gave up, go limp"
  // Two ways to choose his 40 codes each tick; the small part (dec) always turns them into joint commands on his body now.
  //   idle     the translator (enc): the codes that lead to the standing pose, fresh every tick. He stands, takes shoves.
  //   getup    he is down: the big brain's picks (background, a few a second, nudged toward standing) until he is up.
  // The switch goes by his body (head and hips height). ?dance = the big brain alone, as in live 9.
  const Q = new URLSearchParams(location.search), DANCE = Q.has('dance'), BETA = +(Q.get('nudge') ?? 0.3);
  st.mode = DANCE ? 'dance' : 'idle'; st.switches = []; let lowN = 0, upN = 0, last = 0; st.fresh = false;
  brain.session = { run: async feeds => {
    const all = feeds.obs.data, o = new Float32Array(all.subarray(0, 343)), head = o[0] + o[18], hips = o[0];
    st.ticks++; st.head = head; st.hips = hips;
    const want = DANCE ? null : (await enc.run({ block: new ort.Tensor('float32', new Float32Array(all.subarray(343, 895)), [1, 552]) })).codes.data;
    if (!DANCE) {
      lowN = head < 1.05 ? lowN + 1 : 0; upN = head > 1.3 && hips > 0.8 ? upN + 1 : 0;
      if (st.mode === 'idle' && lowN >= 9) { st.mode = 'getup'; st.fresh = false; st.switches.push([+((performance.now() - st.t0) / 1000).toFixed(1), 'down', +head.toFixed(2)]); }
      else if (st.mode === 'getup' && upN >= 6) { st.mode = 'idle'; st.switches.push([+((performance.now() - st.t0) / 1000).toFixed(1), 'up', +head.toFixed(2)]); }
      if (st.switches.length > 60) st.switches.shift();
    }
    if (st.mode !== 'idle' && st.ready && !st.busy) {
      st.busy = true; st.sent = performance.now(); st.askMode = st.mode;
      w.postMessage({ obs: o, want: want ? Float32Array.from(want, v => v + 4) : null, beta: DANCE ? 0 : BETA });
    }
    const now = performance.now(); if (now - last > 1000) { st.rate = st.picks - st.since; st.since = st.picks; last = now;
      say(`${fps.toFixed(0)} fps · ${st.mode === 'idle' ? 'idle (small part every frame)' : st.mode === 'getup' ? `getting up · big brain ${st.rate} picks a second, ${st.ms.toFixed(0)} ms each` : `big brain alone · ${st.rate} picks a second, ${st.ms.toFixed(0)} ms each`}${st.ready ? '' : ' · big brain still loading'} · ${threads} core${threads > 1 ? 's' : ''}`); }
    let codes = want;
    if (st.mode !== 'idle') { if (st.fresh) codes = st.codes; else if (DANCE) return tracker.run(feeds); }   // until this episode's first pick: the translator (dance: the old tracker)
    return dec.run({ obs: new ort.Tensor('float32', o, [1, 343]), codes: new ort.Tensor('float32', Float32Array.from(codes), [1, 40]) });
  } };
  return st;
}

// The latest test page's driver seat (kin.html uses drive.update / reset / pins / noFlat / brain): here the NVIDIA
// brain alone moves his joints. No recordings, no springs, no director. The page's human joint limits stay on.
import { humanCap } from '../sim/springs.js';
const STRONG = /[?&]strong/.test(location.search), DANCE2 = /[?&]dance/.test(location.search), DROP = /[?&]drop/.test(location.search);   // ?strong = NVIDIA's training strength instead of a person's
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
    if (DROP && !this.dropped && d.time > 3) {   // test hook (?drop): 3 s in, lay him on the floor, to watch the get-up
      this.dropped = true; d.qpos[2] = -W.box.h / 2 + 0.22; d.qpos[3] = Math.SQRT1_2; d.qpos[4] = Math.SQRT1_2; d.qpos[5] = 0; d.qpos[6] = 0; d.qvel.fill(0); W.mj.mj_forward(W.m, d); }
    for (let i = 0; i < n; i++) {
      d.xfrc_applied.fill(0); if (fingerPush) fingerPush();
      if (B.due()) { B.down = false; B.lost = 0; await B.think(DANCE2 ? this.self : undefined); if (B.down) { B.down = false; B.setTone(1); } }
      W.step(B.control);
    }
  }
}
