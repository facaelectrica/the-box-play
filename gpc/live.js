// Puts NVIDIA's GPC prior in the page brain's seat. Big part in a background worker (a new set of picks whenever it
// finishes one), small part (decoder, 3 M numbers) here every control tick on his current body.
import { qrot, qmul, NB } from '../sim/brain.js';
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
  st.mode = DANCE ? 'dance' : 'idle'; st.switches = []; let last = 0; st.fresh = false;
  // ---------- the director: a list of orders (written in gpc/editor.html) ----------
  // Order: on, start, stop, strength, steps. Start: all its conditions true for `for` s; stop likewise (none: when the
  // start is no longer true). The first order in the list (switched on) whose start holds is given; while it is given,
  // only an order above it can take over. No order given: the chooser decides, free. Checked 30 times a second.
  // Strength: control = the translator drives him toward the step's pose every tick (chooser off); guide = the chooser
  // drives him, its choices tilted toward the step's pose. Steps: poses in order; the next step comes when he is close
  // to this one or after its time; the last step holds.
  const IDLE = { name: 'Idle', on: true, strength: 'control', start: { all: [{ s: 'head', c: '>', v: 1.30 }, { s: 'hips', c: '>', v: 0.80 }], for: 0.2 }, stop: { all: [{ s: 'head', c: '<', v: 1.05 }], for: 0.3 },
    steps: [{ pose: { RightArm_y: -1.25, LeftArm_y: 1.25, RightArm_z: -0.2, LeftArm_z: 0.2 }, time: 1 }] };
  IDLE.do = { type: 'pose' }; st.orders = [IDLE]; st.active = DANCE ? null : IDLE; st.step = 0; st.stepT = 0; st.force = null; const tm = new Map();
  const v1 = O => { const o = O.orders.find(x => x.name === 'Idle'), w = o.when || {};   // the editor's first file format
    return { version: 2, orders: [{ name: 'Idle', start: { all: [{ s: 'head', c: '>', v: w.upHeadAbove ?? 1.3 }, { s: 'hips', c: '>', v: w.upHipsAbove ?? 0.8 }], for: w.upFor ?? 0.2 }, stop: { all: [{ s: 'head', c: '<', v: w.downHeadBelow ?? 1.05 }], for: w.downFor ?? 0.3 }, pose: o.do.pose }] }; };
  const norm = o => { if (!o.do) o.do = { type: 'pose' }; if (!o.steps) o.steps = [{ pose: o.pose || {}, time: 1 }]; delete o.pose; if (o.on === undefined) o.on = true; if (!o.strength) o.strength = 'control'; for (const st_ of o.steps) st_._stand = null; return o; };
  st.apply = O => { if (!O || !O.orders) return false; if (!O.version || O.version < 2) O = v1(O);
    st.orders = O.orders.map(norm); tm.clear();
    if (st.rec) { brain.react = null; st.rec = null; }
    st.active = DANCE ? null : (st.orders.find(o => st.active && o.name === st.active.name) || null); st.step = 0; st.stepT = 0; standFor(st.active, 0); st.ordersFile = O; return true; };
  st.setStepPose = (o, k, pose) => { const sp = o.steps[k]; sp.pose = pose; sp._stand = null; if (st.active === o && st.step === k) standFor(o, k); };
  st.setPose = (o, pose) => st.setStepPose(o, 0, pose);
  function standFor(o, k) { if (!o) return; const sp = o.steps[Math.min(k, o.steps.length - 1)]; brain.stand = sp._stand || (sp._stand = makeStand(sp.pose || {})); }
  // a pose = the body's orientation (root_w/x/y/z, upright if absent) + joint angles. Placed with its lowest point on
  // the floor (standing: the soles; lying: whatever touches), on his spot, turned the way he faces (brain.updateRef).
  function makeStand(pose) {
    const W = brain.W, mj = W.mj, m = W.m, d2 = new mj.MjData(m), qa = brain.qadr;
    const r = pose.root_w !== undefined ? [pose.root_w, pose.root_x, pose.root_y, pose.root_z] : [1, 0, 0, 0], rl = Math.hypot(...r) || 1;
    for (let k = 0; k < 4; k++) d2.qpos[qa + 3 + k] = r[k] / rl;
    for (const a of W.act) d2.qpos[a.qadr] = pose[a.name] || 0;
    mj.mj_kinematics(m, d2);
    const P = new Float64Array(NB * 3), R = new Float64Array(NB * 4);
    for (let b = 0; b < NB; b++) { const id = brain.bodyId[b]; for (let k = 0; k < 3; k++) P[3 * b + k] = d2.xpos[3 * id + k];
      R[4 * b] = d2.xquat[4 * id + 1]; R[4 * b + 1] = d2.xquat[4 * id + 2]; R[4 * b + 2] = d2.xquat[4 * id + 3]; R[4 * b + 3] = d2.xquat[4 * id]; }
    let lo = 1e9; const X = d2.geom_xmat, sz = m.geom_size;
    for (let g = 0; g < m.ngeom; g++) { if (!W.isFig[m.geom_bodyid[g]]) continue; const t = m.geom_type[g]; let dz;
      if (t === 6) { dz = 0; for (let k = 0; k < 3; k++) dz += Math.abs(X[9 * g + 6 + k]) * sz[3 * g + k]; } else if (t === 3) dz = Math.abs(X[9 * g + 8]) * sz[3 * g + 1] + sz[3 * g]; else dz = sz[3 * g];
      lo = Math.min(lo, d2.geom_xpos[3 * g + 2] - dz); }
    const tx = [0, 0, 0]; qrot(R, 0, 1, 0, 0, tx, 0); const h = Math.atan2(tx[1], tx[0]), q = [0, 0, Math.sin(-h / 2), Math.cos(-h / 2)], t = [0, 0, 0];
    const S = { pos: new Float64Array(NB * 3), rot: new Float64Array(NB * 4), height: P[2] - lo };
    for (let b = 0; b < NB; b++) { qrot(q, 0, P[3 * b] - P[0], P[3 * b + 1] - P[1], P[3 * b + 2] - P[2], t, 0); S.pos.set(t, 3 * b); qmul(q, 0, R, 4 * b, S.rot, 4 * b); }
    d2.delete(); return S;
  }
  // his body as it is now, as a pose (for "new order from this moment"): orientation without heading + joint angles
  st.poseNow = () => { const d = brain.W.d, qa = brain.qadr, out = {}; for (const a of brain.W.act) out[a.name] = d.qpos[a.qadr];
    const w = d.qpos[qa + 3], x = d.qpos[qa + 4], y = d.qpos[qa + 5], z = d.qpos[qa + 6], R = [x, y, z, w], tx = [0, 0, 0]; qrot(R, 0, 1, 0, 0, tx, 0);
    const h = Math.atan2(tx[1], tx[0]), q = [0, 0, Math.sin(-h / 2), Math.cos(-h / 2)], o = [0, 0, 0, 1]; qmul(q, 0, R, 0, o, 0);
    out.root_w = o[3]; out.root_x = o[0]; out.root_y = o[1]; out.root_z = o[2]; return out; };
  // how far his joints are from a step's pose (mean over the hinges, radians)
  function poseGap(pose) { const d = brain.W.d; let s = 0, n = 0; for (const a of brain.W.act) { s += Math.abs(d.qpos[a.qadr] - (pose[a.name] || 0)); n++; } return s / n; }
  // ---------- Do: Recording (a real person's movement from the library, performed by the brain under physics) ----------
  // do = { type: 'recording', file } or { type: 'recording', category } (the recording whose first frame is closest to
  // his body now). The recording is anchored at his hips and heading; when it ends and the order still holds, the
  // director picks again (a new try).
  const LIB = { list: null, bufs: {} };
  st.library = async () => { if (!LIB.list) { try { const k = await (await fetch(here + '../body/clips/kin/kin.json')).json(), L = [];
      for (const [c, v] of Object.entries(k.getups || {})) for (const x of v) L.push({ file: x.file, cat: 'getup-' + c, label: x.file.replace(/^kin\//, '').replace(/\.bin$/, '') });
      for (const x of k.reacts || []) L.push({ file: x.file, cat: 'recovery-' + (x.kind || ''), label: x.file.replace(/^kin\//, '').replace(/\.bin$/, '') + (x.src ? ' · ' + x.src : '') });
      for (const x of k.idles || []) L.push({ file: x.file, cat: 'idle', label: (x.label || x.file) });
      for (const x of k.rests || []) L.push({ file: x.file, cat: 'rest-' + (x.kind || ''), label: x.file.replace(/^kin\//, '').replace(/\.bin$/, '') });
      LIB.list = L; } catch (e) { LIB.list = []; } } return LIB.list; };
  // each recording's first-frame shape, read once (only the first frame is fetched)
  st.starts = async L => { await Promise.all(L.filter(x => !x.s0).map(async x => { try { const r = await fetch(here + '../body/clips/' + x.file, { headers: { Range: 'bytes=0-1503' } }), b = new Float32Array((await r.arrayBuffer()).slice(0, 1504)); x.s0 = shapeOf(b, 3); } catch (_) { x.s0 = [0, 0, 1]; } })); };
  st.clip = async f => LIB.bufs[f] || (LIB.bufs[f] = new Float32Array(await (await fetch(here + '../body/clips/' + f)).arrayBuffer()));
  const hdg = (q, i) => { const t = [0, 0, 0]; qrot(q, i, 1, 0, 0, t, 0); return Math.atan2(t[1], t[0]); };
  // a body shape that does not depend on where he faces: the hips' three axes, how much each points up
  const shapeOf = (q, i) => { const a = [0, 0, 0], b = [0, 0, 0], c = [0, 0, 0]; qrot(q, i, 1, 0, 0, a, 0); qrot(q, i, 0, 1, 0, b, 0); qrot(q, i, 0, 0, 1, c, 0); return [a[2], b[2], c[2]]; };
  async function startRecording(o) {
    const D = o.do || {}; let file = D.file;
    if (!file && D.category) {
      const L = (await st.library()).filter(x => x.cat === D.category || x.cat.startsWith(D.category + '-') || (D.category === 'getup' && x.cat.startsWith('getup-'))), me = shapeOf(brain.rot, 0), d = brain.W.d;
      let best = null; await st.starts(L);
      const near = L.map(x => ({ x, e: Math.hypot(me[0] - x.s0[0], me[1] - x.s0[1], me[2] - x.s0[2]) })).sort((a, b) => a.e - b.e).slice(0, 6);
      for (const { x, e: e0 } of near) { const b = await st.clip(x.file); let e = e0, n = 0, h = 0;
        for (const a of brain.W.act) { h += Math.abs(d.qpos[a.qadr] - b[299 + a.qadr]); n++; } e += 0.3 * h / n; if (!best || e < best.e) best = { e, file: x.file }; }
      file = best && best.file;
    }
    if (!file || st.active !== o) return;
    const buf = await st.clip(file); if (st.active !== o) return;
    const n = Math.floor(buf.length / 376), P = brain.pos;
    brain.react = { buf, n, end: n - 1, fps: 30, mid: [buf[0], buf[1]], h0: hdg(buf, 3), t: 0, xy: [P[0], P[1]], yaw: hdg(brain.rot, 0) };
    if (brain.blendStart) brain.blendStart();
    st.rec = { file, n, o }; st.fresh = false;
  }
  // the signs: heights above the floor (m), lean of his trunk from upright (deg), speed of his hips (m/s), box tilt (deg)
  const BI = { head: 6, chest: 3, hips: 0, rightHand: 10, leftHand: 14, rightFoot: 17, leftFoot: 21 };
  st.signs = {};
  function readSigns() {
    const S = st.signs, P = brain.pos, g = brain.ground;
    for (const k in BI) S[k] = P[3 * BI[k] + 2] - g;
    const R = brain.rot, q = 4 * 3; S.lean = Math.acos(Math.max(-1, Math.min(1, 1 - 2 * (R[q] * R[q] + R[q + 1] * R[q + 1])))) * 180 / Math.PI;
    const V = brain.vel; S.speed = Math.hypot(V[0], V[1], V[2]);
    const G = brain.W.m.opt.gravity, gn = Math.hypot(G[0], G[1], G[2]) || 1; S.tilt = Math.acos(Math.max(-1, Math.min(1, -G[2] / gn))) * 180 / Math.PI;
    { const R2 = brain.rot, q2 = 4 * 3, f = [0, 0, 0], l = [0, 0, 0]; qrot(R2, q2, 0, -1, 0, f, 0); qrot(R2, q2, 1, 0, 0, l, 0);
      S.lyingOn = S.lean < 50 ? 'upright' : f[2] > 0.5 ? 'back' : f[2] < -0.5 ? 'front' : l[2] > 0 ? 'right side' : 'left side'; }
    S.tiltLR = Math.asin(Math.max(-1, Math.min(1, -G[0] / gn))) * 180 / Math.PI; S.tiltFB = Math.asin(Math.max(-1, Math.min(1, G[1] / gn))) * 180 / Math.PI;
  }
  const holds = c => c && c.all && c.all.length ? c.all.every(x => { const v = st.signs[x.s]; return v !== undefined && (x.c === '=' ? v === x.v : x.c === '<' ? v < x.v : v > x.v); }) : false;
  function held(key, ok, secs) { const n = ok ? (tm.get(key) || 0) + 1 : 0; tm.set(key, n); return n >= Math.max(1, Math.round((secs || 0) * 30)); }
  function direct() {
    readSigns(); const t = +((performance.now() - st.t0) / 1000).toFixed(1);
    const startOK = st.orders.map((o, k) => o.on !== false && held('s' + k, holds(o.start), o.start && o.start.for));
    let next = st.active && st.active.on !== false ? st.active : null, ai = st.orders.indexOf(st.active);
    if (next) { const o = next, stop = o.stop && o.stop.all && o.stop.all.length ? holds(o.stop) : !holds(o.start);
      if (held('x' + ai, stop, o.stop ? o.stop.for : 0.3)) next = null; }
    const k = startOK.findIndex(Boolean); if (k >= 0 && (next === null || k < st.orders.indexOf(next))) next = st.orders[k];
    if (st.force) next = st.orders.find(o => o.name === st.force) || next;
    if (next !== st.active) { st.switches.push([t, next ? next.name : 'chooser', +st.signs.head.toFixed(2)]); if (st.switches.length > 60) st.switches.shift();
      st.active = next; st.step = 0; st.stepT = 0; standFor(next, 0); st.fresh = false; for (const key of [...tm.keys()]) if (key[0] === 'x') tm.delete(key);
      if (st.rec) { brain.react = null; st.rec = null; } if (next && next.do && next.do.type === 'recording') { st.rec = { pending: true }; startRecording(next); } }
    if (st.active && st.active.do && st.active.do.type === 'recording' && st.rec && !st.rec.pending && !brain.react) { st.rec = { pending: true }; startRecording(st.active); }   // ended, order still holds: another try
    // steps: on to the next one when he is close to this pose or its time is up
    const o = st.active;
    if (o && !(o.do && o.do.type === 'recording') && o.steps.length > 1 && st.step < o.steps.length - 1) { st.stepT += 1 / 30; const sp = o.steps[st.step];
      if (poseGap(sp.pose) < (sp.close ?? 0.12) || st.stepT > (sp.time ?? 1)) { st.step++; st.stepT = 0; standFor(o, st.step); } }
    const mode = !o ? 'getup' : o.strength === 'guide' ? 'guide' : 'idle'; if (mode !== st.mode) { st.mode = mode; st.fresh = false; }
  }
  st.mode = DANCE ? 'dance' : 'idle'; standFor(st.active, 0);
  try { const saved = JSON.parse(localStorage.getItem('gpc.orders') || 'null'); if (saved) st.apply(saved); else fetch(here + 'orders.json?t=' + Date.now()).then(r => r.ok ? r.json() : null).then(O => O && st.apply(O)).catch(() => {}); } catch (_) {}
  brain.session = { run: async feeds => {
    const all = feeds.obs.data, o = new Float32Array(all.subarray(0, 343));
    st.ticks++;
    const want = DANCE ? null : (await enc.run({ block: new ort.Tensor('float32', new Float32Array(all.subarray(343, 895)), [1, 552]) })).codes.data;
    if (!DANCE) { direct(); st.head = st.signs.head; st.hips = st.signs.hips; }
    if (st.mode !== 'idle' && st.ready && !st.busy) {
      st.busy = true; st.sent = performance.now(); st.askMode = st.mode;
      w.postMessage({ obs: o, want: want ? Float32Array.from(want, v => v + 4) : null, beta: DANCE || st.mode !== 'guide' ? 0 : (st.active && st.active.guide) || 1 });   // the chooser alone is free; guided, its choices lean toward the step's pose
    }
    const now = performance.now(); if (now - last > 1000) { st.rate = st.picks - st.since; st.since = st.picks; last = now;
      say(`${fps.toFixed(0)} fps · ${st.mode === 'idle' || st.mode === 'guide' ? `order: ${st.active ? st.active.name : '?'}` : st.mode === 'getup' ? `getting up · big brain ${st.rate} picks a second, ${st.ms.toFixed(0)} ms each` : `big brain alone · ${st.rate} picks a second, ${st.ms.toFixed(0)} ms each`}${st.ready ? '' : ' · big brain still loading'} · ${threads} core${threads > 1 ? 's' : ''}`); }
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
