// The Box - recordings page: his physical body driven by a trained balance brain (user Nov 14 17:47: iterate until he
// behaves like a human). Why: springs alone could not keep him up (0% help: down in 2 s), and any help strong enough
// to keep him up held 10-100% of his weight from outside (a puppet). Balancing a physical body on its own feet is what
// the tracker network does (NVIDIA's SOMA tracker and our trained brains, same input/output as the game).
//
// Here the network does not get the game's reference: it gets the recordings engine's pose (sim/kin.js running on a
// ghost copy of the world: pose-matched idles, his own upper body kept through recoveries, lean into slopes,
// recoveries picked by the capture point, get-ups). This is the DReCon layout: the recordings decide WHAT he does,
// the network makes his physical body DO it, with joint muscles only. No outside forces (only your finger).
import { qmul, qrot } from './brain.js';

const NB = 23;
import { Pins } from './pins.js?v=31';
import { ARMS, ARM_W, LEGS, solve6 } from './kin.js?v=31';
const clamp = (v, a, b) => Math.max(a, Math.min(b, v));
function hullPts(P) {
  const p = P.slice().sort((a, b) => a[0] - b[0] || a[1] - b[1]); if (p.length < 3) return p;
  const cr = (o, a, b) => (a[0] - o[0]) * (b[1] - o[1]) - (a[1] - o[1]) * (b[0] - o[0]), lo = [], up = [];
  for (const q of p) { while (lo.length >= 2 && cr(lo[lo.length - 2], lo[lo.length - 1], q) <= 0) lo.pop(); lo.push(q); }
  for (const q of p.reverse()) { while (up.length >= 2 && cr(up[up.length - 2], up[up.length - 1], q) <= 0) up.pop(); up.push(q); }
  return lo.slice(0, -1).concat(up.slice(0, -1));
}
function outsidePt(H, x) {   // how far x is outside the outline H (0 inside), and the direction out
  let inside = H.length >= 3, best = 1e9, dir = [0, 0];
  for (let i = 0; i < H.length; i++) {
    const a = H[i], b = H[(i + 1) % H.length], e = [b[0] - a[0], b[1] - a[1]], L2 = e[0] * e[0] + e[1] * e[1] || 1e-12;
    if (H.length >= 3 && e[0] * (x[1] - a[1]) - e[1] * (x[0] - a[0]) < 0) inside = false;
    const t = clamp(((x[0] - a[0]) * e[0] + (x[1] - a[1]) * e[1]) / L2, 0, 1), c = [a[0] + e[0] * t, a[1] + e[1] * t];
    const dd = Math.hypot(x[0] - c[0], x[1] - c[1]); if (dd < best) { best = dd; dir = [x[0] - c[0], x[1] - c[1]]; }
  }
  return inside ? { d: 0, dir: [0, 0] } : { d: best, dir };
}
export class BrainDrive {
  // springs (sim/springs.js), optional: the network was never taught to get up from the floor (it gives up past
  // 0.5 m), so during a get-up the springs and their help carry him through the get-up recording instead
  constructor(W, kin, brain, springs = null) {
    this.W = W; this.kin = kin; this.brain = brain; this.G = kin.W; this.springs = springs;
    kin.ext = W;   // the engine's falls follow this body
    this.ref = { pos: new Float64Array(NB * 3), rot: new Float64Array(NB * 4), vel: new Float64Array(NB * 3), ang: new Float64Array(NB * 3) };
    this.prevRef = null; this.lostT = 0; this.sup = null; this.stats = {}; this.getupBy = 'kin';
    // supporting hands and feet pinned in place (needs the welds addPins puts in the body file)
    const P = new Pins(W, kin); if (P.ok) this.pins = P;
    if (springs) springs.armHook = tg => { this.getupArms(tg); if (!this.getupRise(tg)) this.getupLegs(tg); this.getupLean(tg); };
  }
  reset() {
    const W = this.W, d = W.d, g = this.G.d;
    for (let k = 0; k < 73; k++) d.qpos[k] = g.qpos[k];
    d.qvel.fill(0); d.xfrc_applied.fill(0); d.qfrc_applied.fill(0);
    W.mj.mj_forward(W.m, d); W.tControl = d.time;
    this.brain.attach(); this.brain.told = true; this.prevRef = null; this.lostT = 0; this.mode = 'follow';
    if (this.pins) this.pins.reset();
  }
  // the ghost's 23 parts, in the brain's gravity frame (brain.G, set by readState), with speeds from tick to tick
  buildRef(dt) {
    const B = this.brain, gd = this.G.d, R = this.ref, t = [0, 0, 0], q = [0, 0, 0, 1];
    const P = new Float64Array(NB * 3), Q = new Float64Array(NB * 4);
    for (let b = 0; b < NB; b++) {
      const id = B.bodyId[b];
      qrot(B.G, 0, gd.xpos[3 * id], gd.xpos[3 * id + 1], gd.xpos[3 * id + 2], t, 0); P.set(t, 3 * b);
      q[0] = gd.xquat[4 * id + 1]; q[1] = gd.xquat[4 * id + 2]; q[2] = gd.xquat[4 * id + 3]; q[3] = gd.xquat[4 * id];
      qmul(B.G, 0, q, 0, Q, 4 * b);
    }
    const pr = this.prevRef;
    for (let b = 0; b < NB; b++) {
      if (pr && dt > 0) {
        for (let k = 0; k < 3; k++) { const v = (P[3 * b + k] - pr.P[3 * b + k]) / dt; R.vel[3 * b + k] += (v - R.vel[3 * b + k]) * 0.6; }
        // angular velocity: q_now * conj(q_prev) as a rotation vector / dt
        const c = [-pr.Q[4 * b], -pr.Q[4 * b + 1], -pr.Q[4 * b + 2], pr.Q[4 * b + 3]], r = [0, 0, 0, 1];
        qmul(Q, 4 * b, c, 0, r, 0); if (r[3] < 0) for (let k = 0; k < 4; k++) r[k] = -r[k];
        const s = Math.hypot(r[0], r[1], r[2]), th = 2 * Math.atan2(s, r[3]), f = s > 1e-9 ? th / s / dt : 2 / dt;
        for (let k = 0; k < 3; k++) R.ang[3 * b + k] += (r[k] * f - R.ang[3 * b + k]) * 0.6;
      } else { for (let k = 0; k < 3; k++) { R.vel[3 * b + k] = 0; R.ang[3 * b + k] = 0; } }
    }
    R.pos.set(P); R.rot.set(Q);
    this.prevRef = { P, Q };
    // where he looks (the page's gaze(): at you, now and then at the floor, toward where he's falling): the reference's
    // neck and head turn to it (Brain.applyLook), his real neck follows
    if (this.brain.lookAt) this.brain.applyLook(R);
    return R;
  }
  // the ghost stays where his real body is: its spot follows his hips (slowly while he stands, faster while he steps),
  // so a push that carries him away is not "undone" by the network walking him back to the ghost
  followBody(dt) {
    const K = this.kin, d = this.W.d, A = K.anchor;
    if (!(K.state === 'stand' || K.state === 'react')) return;
    const me = K.toLocal([d.qpos[0], d.qpos[1], d.qpos[2]]), gh = K.toLocal([this.G.d.qpos[0], this.G.d.qpos[1], this.G.d.qpos[2]]);
    const k = 1 - Math.exp(-dt * (K.state === 'react' ? 2 : 0.8));
    A.x += (me[0] - gh[0]) * k; A.y += (me[1] - gh[1]) * k;
  }
  // (falls worker) muscle tone while falling, by body part, as a fraction of the network's own gains: a falling person
  // keeps the head up and the trunk together (neck 0.9, spine 0.6), braces with the arms (0.55) and lets the legs give
  // (0.4); once on the floor it eases over 2 s to a resting tone that still holds his shape (0.7 / 0.45 / 0.35 / 0.3).
  // (Before: everything at 35% -> 25%, earlier 12% -> 4%: the neck folded, the body poured into corners.)
  groupTone(t) {
    const B = this.brain, m = this.W.m, u = Math.min(1, Math.max(0, (t - 0.8) / 2));
    const T = (a, b) => a + (b - a) * u;
    for (let i = 0; i < B.meta.kp.length; i++) {
      const a = B.actIdx[i], n = this.W.act[a].name;
      const f = /Neck|Head/.test(n) ? T(0.9, 0.7) : /Spine|Chest/.test(n) ? T(0.75, 0.6) : /Shoulder|Arm|Hand/.test(n) ? Math.max(T(0.55, 0.35), (this.headWallT > 0 ? 1 + ((this.hwTone ?? 1.3) - 1) * (this.hwPushK || 0) : 0.85) * (this.guardW || 0)) : T(0.4, 0.3);
      const kp = B.meta.kp[i] * f, kd = B.meta.kd[i] * Math.sqrt(f);
      m.actuator_gainprm[10 * a] = kp; m.actuator_biasprm[10 * a + 1] = -kp; m.actuator_biasprm[10 * a + 2] = -kd;
    }
  }
  // (combined build) hands on the walls: the network was never trained with walls; following a reaching pose with
  // the hands hitting the wood, its arms bounced around. While he braces, his arm joints (collarbones to wrists)
  // follow the recording's arms directly (their own muscles, NVIDIA's gains), holding the arms' weight; the network
  // keeps everything else. Faded in and out with the brace.
  // arms up around the head, elbows bent, forearms over it, chin tucked (also switched to mid-fall, see update)
  headBrace() {
    const W = this.W, d = W.d, set = (n, v) => { const i = W.act.findIndex(a => a.name === n); if (i >= 0) this.brace[i] = v; };
    this.hold = W.act.map(a => d.qpos[a.qadr]);   // from where his arms are now
    set('RightArm_y', 0.9); set('LeftArm_y', -0.9); set('RightArm_z', 0.9); set('LeftArm_z', -0.9);
    set('RightForeArm_z', 2.2); set('LeftForeArm_z', -2.2); set('RightHand_x', 0.3); set('LeftHand_x', 0.3);
    set('Neck1_x', 0.35); set('Neck2_x', 0.15); set('Head_x', 0.1);
    this.headFirst = true; this.headT = this.fallT || 0;
  }
  curlBall() {
    const W = this.W, set = (n, v) => { const i = W.act.findIndex(a => a.name === n); if (i >= 0) this.brace[i] = v; };
    this.headBrace();
    for (const side of ['Right', 'Left']) { set(side + 'Leg_x', -1.7); set(side + 'Shin_x', 2.1); set(side + 'Foot_x', 0.3); }
    set('Spine1_x', 0.35); set('Spine2_x', 0.3); set('Chest_x', 0.25); set('Neck1_x', 0.45); set('Neck2_x', 0.2); set('Head_x', 0.15);
    this.curled = true;
  }
  // (user 19:12: "the head is very important. He always wants to protect the head so he doesn't hit it directly")
  // seconds until his head reaches a wall, the floor or the ceiling at its present speed (9 = no threat)
  headThreat() {
    const W = this.W, d = W.d, hb = W.bodyId.Head, p = [d.xpos[3 * hb], d.xpos[3 * hb + 1], d.xpos[3 * hb + 2]];
    if (!this.headPrev) { this.headPrev = p; return 9; }
    const v = [0, 1, 2].map(i => (p[i] - this.headPrev[i]) * 30); this.headPrev = p;   // called at 30 Hz
    const sp = Math.hypot(...v); if (sp < 0.8) return 9;
    const b = W.box, planes = [[0, -b.w / 2], [0, b.w / 2], [1, 0], [1, b.D], [2, -b.h / 2], [2, b.h / 2]];
    let t = 9;
    for (const [ax, at] of planes) { if (W.open && ax !== 2) continue; const gap = at - p[ax], vv = v[ax]; if (gap * vv > 0) t = Math.min(t, (Math.abs(gap) - 0.12) / Math.abs(vv)); }
    return t;
  }
  headDownNow() {
    const W = this.W, d = W.d, hb = W.bodyId.Head, hp = W.bodyId.Hips, g0 = W.m.opt.gravity, gn0 = Math.hypot(g0[0], g0[1], g0[2]) || 9.81;
    const ux = d.xpos[3 * hb] - d.xpos[3 * hp], uy = d.xpos[3 * hb + 1] - d.xpos[3 * hp + 1], uz = d.xpos[3 * hb + 2] - d.xpos[3 * hp + 2], ul = Math.hypot(ux, uy, uz) || 1;
    return (ux * g0[0] + uy * g0[1] + uz * g0[2]) / ul / gn0;
  }
  // (head worker, user 19:12 "He always wants to protect the head so he doesn't hit it directly") where his head is
  // about to hit: its path over the next 0.7 s (its speed now, plus part of gravity: he is not in free fall) against
  // the walls, the floor and the ceiling; the first surface it would reach within a head's radius. null = no threat.
  headImpact() {
    const W = this.W, d = W.d, hb = W.bodyId.Head, p = [d.xpos[3 * hb], d.xpos[3 * hb + 1], d.xpos[3 * hb + 2]];
    const pv = this.hgPrev; this.hgPrev = p; if (!pv) return null;
    const v = [0, 1, 2].map(i => (p[i] - pv[i]) * 30), g = W.m.opt.gravity, b = W.box;   // called at 30 Hz
    // only while his head is really on its way (at rest, "part of gravity" alone would find the floor under it every time
    // and keep his arms up and chin down through the whole lying-still part, into the get-up)
    if (Math.hypot(v[0], v[1], v[2]) < 0.7) return null;
    const planes = [[0, -b.w / 2, 1, 'left'], [0, b.w / 2, -1, 'right'], [1, 0, 1, 'glass'], [1, b.D, -1, 'back'], [2, -b.h / 2, 1, 'floor'], [2, b.h / 2, -1, 'ceil']];
    // in the air (nothing of him touches the box: the phone flipped, he drops) gravity has all of him: full g
    let touch = false; const m = W.m;
    for (let i = 0; i < d.ncon && !touch; i++) { const c = d.contact.get(i), b1 = m.geom_bodyid[c.geom1], b2 = m.geom_bodyid[c.geom2]; c.delete(); if (W.isFig[b1] !== W.isFig[b2]) touch = true; }
    const ga = touch ? 0.3 : 0.5, gv = touch ? 0.6 : 1.0;
    for (let t = 0; t <= 0.7; t += 1 / 60) {
      const q = [0, 1, 2].map(i => p[i] + v[i] * t + ga * g[i] * t * t);
      for (const [ax, at, sg, nm] of planes) {
        if (W.open && nm !== 'floor') continue;
        if (sg * (q[ax] - at) > 0.13) continue;
        const vin = -sg * (v[ax] + gv * g[ax] * t);   // speed into that surface when it gets there
        if (vin < 0.5) continue;
        const n = [0, 0, 0]; n[ax] = sg; const P = q.slice(); P[ax] = at;
        return { t, P, n, nm, vin };
      }
    }
    return null;
  }
  // both hands go to the surface where his head would hit, a hand's width either side of that point, ahead of the
  // head (resolved-rate arm IK over the upper arm and forearm hinges, within the joint limits); chin tucked, or the
  // face turned aside when it is the face that would hit. Returns the [actuator, target] pairs to blend in.
  // both hands to a surface point I (P on it, n out of it into the box), a hand's width (latW) either side of it, off
  // it by off: resolved-rate arm IK over the upper arm and forearm hinges, within the joint limits (the head guard's,
  // shared with the fall catch). about: the body part whose sides the hands go to. Returns [actuator, target] pairs.
  armsTo(I, latW, off, about) {
    const W = this.W, d = W.d, m = W.m, mj = W.mj;
    if (!this.hgArms) this.hgArms = ['Right', 'Left'].map(side => ({
      side, sh: W.bodyId[side + 'Arm'], hand: W.bodyId[side + 'Hand'],
      js: ['Arm_x', 'Arm_y', 'Arm_z', 'ForeArm_x', 'ForeArm_y', 'ForeArm_z'].map(n => { const nm = side + n, j = mj.mj_name2id(m, mj.mjtObj.mjOBJ_JOINT.value, nm); return { j, a: W.act.findIndex(a => a.name === nm) }; }).filter(o => o.j >= 0 && o.a >= 0) }));
    const pos = b => [d.xpos[3 * b], d.xpos[3 * b + 1], d.xpos[3 * b + 2]], sub = (a, b) => [a[0] - b[0], a[1] - b[1], a[2] - b[2]];
    const dot = (a, b) => a[0] * b[0] + a[1] * b[1] + a[2] * b[2], cross = (a, b) => [a[1] * b[2] - a[2] * b[1], a[2] * b[0] - a[0] * b[2], a[0] * b[1] - a[1] * b[0]];
    const out = [], hp = pos(about);
    for (const A of this.hgArms) {
      const S = pos(A.sh), hd = pos(A.hand);
      let lat = sub(S, hp); const ln = dot(lat, I.n); lat = lat.map((x, i) => x - ln * I.n[i]); const ll = Math.hypot(...lat) || 1;
      let T = I.P.map((x, i) => x + I.n[i] * off + lat[i] / ll * latW);
      const ST = sub(T, S), sl = Math.hypot(...ST); if (sl > 0.56) T = S.map((x, i) => x + ST[i] * 0.56 / sl);
      let e = sub(T, hd); const el = Math.hypot(...e); if (el > 0.2) e = e.map(x => x * 0.2 / el);
      // Jacobian of the wrist position over this arm's hinges: axis x (wrist - joint anchor)
      const J = A.js.map(o => cross([d.xaxis[3 * o.j], d.xaxis[3 * o.j + 1], d.xaxis[3 * o.j + 2]], sub(hd, [d.xanchor[3 * o.j], d.xanchor[3 * o.j + 1], d.xanchor[3 * o.j + 2]])));
      // damped least squares: dq = J^T (J J^T + l^2 I)^-1 e
      const M = [[0, 0, 0], [0, 0, 0], [0, 0, 0]], l2 = 0.02;
      for (let r = 0; r < 3; r++) for (let c = 0; c < 3; c++) { let s = r === c ? l2 : 0; for (const col of J) s += col[r] * col[c]; M[r][c] = s; }
      const det = M[0][0] * (M[1][1] * M[2][2] - M[1][2] * M[2][1]) - M[0][1] * (M[1][0] * M[2][2] - M[1][2] * M[2][0]) + M[0][2] * (M[1][0] * M[2][1] - M[1][1] * M[2][0]);
      if (Math.abs(det) < 1e-12) continue;
      const inv = [[M[1][1] * M[2][2] - M[1][2] * M[2][1], M[0][2] * M[2][1] - M[0][1] * M[2][2], M[0][1] * M[1][2] - M[0][2] * M[1][1]],
        [M[1][2] * M[2][0] - M[1][0] * M[2][2], M[0][0] * M[2][2] - M[0][2] * M[2][0], M[0][2] * M[1][0] - M[0][0] * M[1][2]],
        [M[1][0] * M[2][1] - M[1][1] * M[2][0], M[0][1] * M[2][0] - M[0][0] * M[2][1], M[0][0] * M[1][1] - M[0][1] * M[1][0]]];
      const y = [0, 1, 2].map(r => (inv[r][0] * e[0] + inv[r][1] * e[1] + inv[r][2] * e[2]) / det);
      A.js.forEach((o, c) => {
        let q = d.qpos[W.act[o.a].qadr] + dot(J[c], y);
        if (m.jnt_limited[o.j]) q = Math.max(m.jnt_range[2 * o.j], Math.min(m.jnt_range[2 * o.j + 1], q));
        out.push([o.a, q]);
      });
    }
    return out;
  }
  // (user 23:11: "when I rotate the phone ... what's becoming the floor now should be the surface where he puts the
  // hands for balance. But instead, the right hand is like floating in the air") where his chest is going: its path over
  // the next 0.8 s (its speed now plus part of gravity) against the walls, floor and ceiling; slow or still, the
  // surface straight below his chest along gravity (the one becoming the floor). Its point and outward normal.
  bodyImpact() {
    const W = this.W, d = W.d, cb = W.bodyId.Chest, p = [d.xpos[3 * cb], d.xpos[3 * cb + 1], d.xpos[3 * cb + 2]];
    const pv = this.cgPrev; this.cgPrev = p;
    const v = pv ? [0, 1, 2].map(i => (p[i] - pv[i]) * 30) : [0, 0, 0], g = W.m.opt.gravity, b = W.box, gn = Math.hypot(g[0], g[1], g[2]) || 9.81;
    const planes = [[0, -b.w / 2, 1, 'left'], [0, b.w / 2, -1, 'right'], [1, 0, 1, 'glass'], [1, b.D, -1, 'back'], [2, -b.h / 2, 1, 'floor'], [2, b.h / 2, -1, 'ceil']];
    const hit = (q, reach) => { for (const [ax, at, sg, nm] of planes) { if (W.open && nm !== 'floor') continue; if (sg * (q[ax] - at) > reach) continue; const n = [0, 0, 0]; n[ax] = sg; const P = q.slice(); P[ax] = at; return { P, n, nm }; } return null; };
    if (Math.hypot(...v) > 0.5) for (let t = 0; t <= 0.8; t += 1 / 30) {
      const q = [0, 1, 2].map(i => p[i] + v[i] * t + 0.3 * g[i] * t * t), h = hit(q, 0.3); if (h) return h;
    }
    // along gravity: the nearest surface below the chest, within arm's reach of it (0.9 m)
    let best = null;
    for (const [ax, at, sg, nm] of planes) {
      if (W.open && nm !== 'floor') continue;
      const gd = g[ax] / gn; if (-sg * gd < 0.35) continue;          // that surface is not "below" him
      const dist = (p[ax] - at) * sg / (-sg * gd); if (dist < 0 || dist > 0.9) continue;
      if (!best || dist < best.dist) { const n = [0, 0, 0]; n[ax] = sg; const P = p.map((x, i) => x + g[i] / gn * dist); P[ax] = at; best = { P, n, nm, dist }; }
    }
    return best;
  }
  // his head within 20 cm of a box face other than the one he stands/lies on: that face (point under his head, its
  // normal into the box)
  headWall() {
    const W = this.W, d = W.d, K = this.kin, hb = W.bodyId.Head, p = [d.xpos[3 * hb], d.xpos[3 * hb + 1], d.xpos[3 * hb + 2]];
    if (W.open || !W.faces) return null;
    let best = null;
    for (const f of W.faces()) {
      if (f.name === K.G.name) continue;
      // only a wall he is leaning onto (gravity pulls him toward it: it is at least ~25 deg "below" him); a wall beside
      // him on a level floor needs no holding off (pushing on it shoved him across the box, 1 run in 4)
      const g = W.m.opt.gravity, gn = Math.hypot(g[0], g[1], g[2]) || 9.81;
      if (-(g[0] * f.n[0] + g[1] * f.n[1] + g[2] * f.n[2]) / gn < (this.hwLean ?? 0.42)) continue;   // (sin 25 deg: below that he gets up / catches himself; at 15 deg the push kept
      // him from catching himself on the glass when the floor levelled: 0 of 4 get-ups)
      const dist = (p[0] - f.p[0]) * f.n[0] + (p[1] - f.p[1]) * f.n[1] + (p[2] - f.p[2]) * f.n[2];
      if (dist > (this.headWallT > 0 ? 0.32 : (this.hwNear ?? 0.2)) || (best && dist > best.dist)) continue;   // (held off: stays on till 32 cm, no bouncing back)
      best = { P: p.map((x, i) => x - f.n[i] * dist), n: f.n.slice(), nm: f.name, dist };
    }
    return best;
  }
  headGuard() {
    const H = this.headImpact();
    if (H) this.hgLast = { ...H, at: this.fallT }; else if (this.hgLast && this.fallT - this.hgLast.at > 0.35) this.hgLast = null;
    const I = this.hgLast; if (!I) return null;
    const W = this.W, K = this.kin, out = this.armsTo(I, this.hgLat ?? 0.13, this.hgOff ?? 0, W.bodyId.Head);
    const sub = (a, b) => [a[0] - b[0], a[1] - b[1], a[2] - b[2]], dot = (a, b) => a[0] * b[0] + a[1] * b[1] + a[2] * b[2];
    // the neck: chin to the chest; when his face is what would hit (the surface is in front of him), turned aside
    const set = (n, v) => { const i = W.act.findIndex(a => a.name === n); if (i >= 0) out.push([i, v]); };
    const { f } = K.facing(), o0 = K.toWorld([0, 0, 0]), f1 = K.toWorld([f[0], f[1], 0]), fw = sub(f1, o0);
    if (dot(fw, I.n) < -0.5) { set('Neck1_y', 0.35); set('Neck1_x', -0.05); set('Head_x', 0); }
    else { set('Neck1_x', 0.4); set('Neck2_x', 0.15); set('Head_x', 0.1); }
    return out;
  }
  // (user 18:50: "the most important part is that the feet stay planted ... keep balance and keep the feet planted")
  // a leg whose foot the recording has planted, and whose real foot is on the floor, follows the recording's leg angles
  // (blend plantW): the network alone dragged planted feet 6-7 cm/s on slow tilts while the recording's stayed put
  plantLegs() {
    const W = this.W, B = this.brain, K = this.kin, d = W.d, k = this.plantW ?? 0;
    if (!(k > 0) || !(K.state === 'stand' || K.state === 'react')) return;
    if (!this.legJ) this.legJ = ['Left', 'Right'].map(side => W.act.map((a, j) => a.name.startsWith(side) && /(Leg|Shin|Foot)_/.test(a.name) ? j : -1).filter(j => j >= 0));
    ['Left', 'Right'].forEach((side, s) => {
      const L = K.locks && K.locks[side]; if (!L || !L.on || L.w < 1) return;
      const b = W.bodyId[side + 'Foot'], h = K.toLocal([d.xpos[3 * b], d.xpos[3 * b + 1], d.xpos[3 * b + 2]])[2];
      if (h > 0.14) return;   // his real foot is in the air: the network's step
      for (const j of this.legJ[s]) { const a = W.act[j]; B.targets[j] += (this.G.d.qpos[a.qadr] - B.targets[j]) * k; }
    });
  }
  // (user 18:56: "Humans always plant the base of the feet, especially when they are out of balance. You would never
  // tiptoe ... he's trying to reach the floor with the foot, and so he tiptoes") a foot near the floor whose recorded
  // foot is planted gets the ankle angles that put its sole flat: the recording's foot orientation (flat on the floor),
  // expressed in his real shin's frame; the toes as in the recording. Faded in from 12 cm to 4 cm above the floor.
  flatFeet() {
    if (this.noFlat) return;
    const W = this.W, B = this.brain, K = this.kin, d = W.d, m = W.m, gd = this.G.d;
    if (!(K.state === 'stand' || K.state === 'react')) return;
    if (!this.footJ) this.footJ = ['Left', 'Right'].map(side => ({
      side, foot: W.bodyId[side + 'Foot'], shin: W.bodyId[side + 'Shin'], toe: W.bodyId[side + 'ToeBase'],
      fj: ['x', 'y', 'z'].map(a => W.act.findIndex(q => q.name === side + 'Foot_' + a)),
      tj: ['x', 'y', 'z'].map(a => W.act.findIndex(q => q.name === side + 'ToeBase_' + a)) }));
    const wq = (A, b) => [A[4 * b], A[4 * b + 1], A[4 * b + 2], A[4 * b + 3]];
    const mul = (a, b) => [a[0] * b[0] - a[1] * b[1] - a[2] * b[2] - a[3] * b[3], a[0] * b[1] + a[1] * b[0] + a[2] * b[3] - a[3] * b[2], a[0] * b[2] - a[1] * b[3] + a[2] * b[0] + a[3] * b[1], a[0] * b[3] + a[1] * b[2] - a[2] * b[1] + a[3] * b[0]];
    const cj = a => [a[0], -a[1], -a[2], -a[3]];
    this.flatW = [0, 0];
    this.footJ.forEach((F, k) => {
      const L = K.locks && K.locks[F.side]; if (!L || !L.on) return;
      // how high his real foot is: the lower of ankle-to-floor and toe-to-floor, minus their resting heights
      const za = K.toLocal([d.xpos[3 * F.foot], d.xpos[3 * F.foot + 1], d.xpos[3 * F.foot + 2]])[2] - 0.08, zt = K.toLocal([d.xpos[3 * F.toe], d.xpos[3 * F.toe + 1], d.xpos[3 * F.toe + 2]])[2] - 0.03;
      // only when he is ON HIS TOES (heel lifted: ankle more than 6 cm above the toe joint; flat ~5 cm) near the floor:
      // a flat or rocking foot is his ankle balancing (the network's main tool), and pinning it flat made him fall
      const heel = (za + 0.08) - (zt + 0.03), tip = Math.max(0, Math.min(1, (heel - 0.06) / 0.03));
      const h = Math.min(za, zt), w = Math.max(0, Math.min(1, (0.12 - h) / 0.08)) * L.w * tip;
      if (w <= 0) return;
      this.flatW[k] = w;
      // wanted foot orientation in the world = the recording's (flat, planted); hinge rotation = (shin * body_quat)^-1 * wanted
      // only the tilt of his foot is corrected (its turn on the floor stays his): the sole's "up" axis in the foot's own
      // frame is taken from the recording's planted foot; his foot is turned the shortest way so that axis points up
      const gq = wq(gd.xquat, F.foot), pq = wq(d.xquat, F.foot), shin = wq(d.xquat, F.shin), bq = [m.body_quat[4 * F.foot], m.body_quat[4 * F.foot + 1], m.body_quat[4 * F.foot + 2], m.body_quat[4 * F.foot + 3]];
      const upW = K.toWorld([0, 0, 1]), o = K.toWorld([0, 0, 0]), upG = [upW[0] - o[0], upW[1] - o[1], upW[2] - o[2]];
      const rot = (q, v) => { const r = mul(mul(q, [0, v[0], v[1], v[2]]), cj(q)); return [r[1], r[2], r[3]]; };
      const uLoc = rot(cj(gq), upG), cur = rot(pq, uLoc);
      const dt0 = cur[0] * upG[0] + cur[1] * upG[1] + cur[2] * upG[2], ax = [cur[1] * upG[2] - cur[2] * upG[1], cur[2] * upG[0] - cur[0] * upG[2], cur[0] * upG[1] - cur[1] * upG[0]];
      let qc0 = [1 + dt0, ax[0], ax[1], ax[2]]; const qn0 = Math.hypot(...qc0) || 1; qc0 = qc0.map(v => v / qn0);
      const want = mul(qc0, pq);
      const r = mul(cj(mul(shin, bq)), want);
      // x then y then z hinges (R = Rx Ry Rz)
      const [qw, qx, qy, qz] = r, R00 = 1 - 2 * (qy * qy + qz * qz), R01 = 2 * (qx * qy - qw * qz), R02 = 2 * (qx * qz + qw * qy), R12 = 2 * (qy * qz - qw * qx), R22 = 1 - 2 * (qx * qx + qy * qy);
      const ang = [Math.atan2(-R12, R22), Math.asin(Math.max(-1, Math.min(1, R02))), Math.atan2(-R01, R00)];
      F.fj.forEach((j, i) => { if (j >= 0) B.targets[j] += (ang[i] - B.targets[j]) * w; });
      F.tj.forEach(j => { if (j >= 0) B.targets[j] += (gd.qpos[W.act[j].qadr] - B.targets[j]) * w; });
    });
  }
  armsToWalls() {
    const W = this.W, B = this.brain, K = this.kin, w = K.braceW || 0;
    if (!this.armJ) this.armJ = W.act.map((a, j) => /(Arm|ForeArm|Hand)_/.test(a.name) && !/Shoulder/.test(a.name) ? j : -1).filter(j => j >= 0);   // collarbones stay the network's
    this.armW = w * w * (3 - 2 * w);
    if (this.armW <= 0) return;
    for (const j of this.armJ) { const a = W.act[j]; B.targets[j] += (this.G.d.qpos[a.qadr] - B.targets[j]) * this.armW; }
  }
  // (user 20:49: "the hands are still not flat against the wall") copying the recording's arm ANGLES puts his hand
  // where the recording's hand would be only if his trunk is where the recording's is - it rarely is (he leans
  // differently), so his hands braced in mid-air. Instead: his own arm, from his own shoulders, reaches the wall
  // target (wrist on the wood, palm flat, fingers up): damped least squares on a scratch copy of his body, 10 rounds,
  // collarbones left alone; the result becomes the arm muscles' targets.
  // his own arms reach: each wrist to tgt[side] with the palm facing along NIN[side] (fingers free), elbows ~18 cm under
  // the shoulders if `elbow`; damped least squares on the scratch copy D of his body (from his real pose), 10 rounds,
  // collarbones left alone (shared by the wall bracing and the get-ups)
  solveArms(sides, tgt, NIN, up, elbow = true) {
    const K = this.kin, W = this.W, m = W.m, mj = W.mj, nv = m.nv;
    const D = this.ikD || (this.ikD = new mj.MjData(m));
    if (!this.jp2) { this.jp2 = new mj.DoubleBuffer(3 * nv); this.jr2 = new mj.DoubleBuffer(3 * nv); this.jp3 = new mj.DoubleBuffer(3 * nv); this.jr3 = new mj.DoubleBuffer(3 * nv); }
    D.qpos.set(W.d.qpos);
    // where his hips face (the body's -y), along the ground: for the elbow-back wish below
    let fw = null;
    { const hb = W.bodyId.Hips, X = W.d.xmat; let f = [-X[9 * hb + 1], -X[9 * hb + 4], -X[9 * hb + 7]]; const fu = f[0] * up[0] + f[1] * up[1] + f[2] * up[2];
      f = f.map((v, i) => v - fu * up[i]); const fl = Math.hypot(...f); if (fl > 0.3) fw = f.map(v => v / fl); }
    // (user 21:40: "the shoulders are doing rotations that are not possible") start from the most similar arm pose a
    // real person made (armdb: ~30k poses from the BONES library: wrist and palm direction seen from the chest), then
    // refine; the solver alone found palms flat through upper-arm twists of 120-145 deg
    if (this.armDB) {
      mj.mj_kinematics(m, D);
      const cb = W.bodyId.Chest, R = Array.from(D.xmat.slice(9 * cb, 9 * cb + 9)), pc = [D.xpos[3 * cb], D.xpos[3 * cb + 1], D.xpos[3 * cb + 2]];
      const toC = v => [R[0] * v[0] + R[3] * v[1] + R[6] * v[2], R[1] * v[0] + R[4] * v[1] + R[7] * v[2], R[2] * v[0] + R[5] * v[1] + R[8] * v[2]];
      for (const side of sides) {
        const si = side === 'Right' ? 0 : 1, w = toC([tgt[side][0] - pc[0], tgt[side][1] - pc[1], tgt[side][2] - pc[2]]), n = toC(NIN[side]), A = this.armDB;
        let best = -1, bd = 1e9;
        for (let r = 0; r < A.length; r += 19) {
          if (A[r] !== si) continue;
          const e = (A[r + 1] - w[0]) ** 2 + (A[r + 2] - w[1]) ** 2 + (A[r + 3] - w[2]) ** 2 + 0.03 * ((A[r + 4] - n[0]) ** 2 + (A[r + 5] - n[1]) ** 2 + (A[r + 6] - n[2]) ** 2);
          if (e < bd) { bd = e; best = r; }
        }
        if (best >= 0) { const dof = ARMS[side].dof; for (let c = 3; c < 12; c++) D.qpos[dof + 1 + c] = A[best + 7 + c]; }
      }
    }
    const rw = 0.35, qm = (a, b) => [a[0] * b[0] - a[1] * b[1] - a[2] * b[2] - a[3] * b[3], a[0] * b[1] + a[1] * b[0] + a[2] * b[3] - a[3] * b[2], a[0] * b[2] - a[1] * b[3] + a[2] * b[0] + a[3] * b[1], a[0] * b[3] + a[1] * b[2] - a[2] * b[1] + a[3] * b[0]];
    const qr = (q, v) => { const r = qm(qm(q, [0, v[0], v[1], v[2]]), [q[0], -q[1], -q[2], -q[3]]); return [r[1], r[2], r[3]]; };
    const rv = q => { if (q[0] < 0) q = q.map(v => -v); const s = Math.hypot(q[1], q[2], q[3]); if (s < 1e-9) return [0, 0, 0]; const a = 2 * Math.atan2(s, q[0]); return [q[1] / s * a, q[2] / s * a, q[3] / s * a]; };
    for (let it = 0; it < 10; it++) {
      mj.mj_kinematics(m, D); mj.mj_comPos(m, D);
      for (const side of sides) {
        const A = ARMS[side], b = W.bodyId[A.hand], T = tgt[side];
        const cq = [D.xquat[4 * b], D.xquat[4 * b + 1], D.xquat[4 * b + 2], D.xquat[4 * b + 3]];
        // the palm only has to face the wall (its normal = into the wood); which way the fingers point is left to
        // the arm (asking for "fingers up" too pinned the left arm against its wrist and elbow limits, 50 cm short)
        const pn = qr(cq, [0, 0, -1]), nin = NIN[side];
        const cx = [pn[1] * nin[2] - pn[2] * nin[1], pn[2] * nin[0] - pn[0] * nin[2], pn[0] * nin[1] - pn[1] * nin[0]], sn = Math.hypot(...cx), cs = pn[0] * nin[0] + pn[1] * nin[1] + pn[2] * nin[2];
        const ang = Math.atan2(sn, cs), rot = sn > 1e-9 ? cx.map(v => v / sn * ang) : [0, 0, 0];
        const e = [T[0] - D.xpos[3 * b], T[1] - D.xpos[3 * b + 1], T[2] - D.xpos[3 * b + 2], ...rot.map(v => v * rw)];
        mj.mj_jacBody(m, D, this.jp2, this.jr2, b);
        const JP = this.jp2.GetView(), JR = this.jr2.GetView(), N = ARM_W.length, J = [];
        for (let r = 0; r < 6; r++) { const row = []; for (let c = 0; c < N; c++) row.push(r < 3 ? JP[r * nv + A.dof + c] : JR[(r - 3) * nv + A.dof + c] * rw); J.push(row); }
        // (user 21:11: "we rarely raise the elbows. The elbows always try to aim down. It's more comfortable for the
        // shoulders") a 7th, softer wish: the elbow ~18 cm below the shoulder (along up), the forearm rising to the wall
        const eb = W.bodyId[side + 'ForeArm'], sb = W.bodyId[side + 'Arm'];
        const ez = (D.xpos[3 * eb] - D.xpos[3 * sb]) * up[0] + (D.xpos[3 * eb + 1] - D.xpos[3 * sb + 1]) * up[1] + (D.xpos[3 * eb + 2] - D.xpos[3 * sb + 2]) * up[2];
        mj.mj_jacBody(m, D, this.jp3, this.jr3, eb);
        const JE = this.jp3.GetView(), ew = 0.5, rowE = [];
        for (let c = 0; c < N; c++) rowE.push((up[0] * JE[A.dof + c] + up[1] * JE[nv + A.dof + c] + up[2] * JE[2 * nv + A.dof + c]) * ew);
        if (elbow) { J.push(rowE); e.push(Math.max(-0.1, Math.min(0.1, (-0.18 - ez))) * ew); }
        // (snapshot 23:14: "the right arm elbow is too forward ... The elbow not only should be down, but also facing
        // closer to the back") an 8th wish: the elbow at least 4 cm behind the shoulder (along where his hips face), like
        // a person pushing on a wall beside him - one-sided, it only pulls an elbow that has come forward
        if (elbow && fw && !globalThis.NOELBACK) {
          const ef = (D.xpos[3 * eb] - D.xpos[3 * sb]) * fw[0] + (D.xpos[3 * eb + 1] - D.xpos[3 * sb + 1]) * fw[1] + (D.xpos[3 * eb + 2] - D.xpos[3 * sb + 2]) * fw[2];
          if (ef > -0.04) { const rowF = []; for (let c = 0; c < N; c++) rowF.push((fw[0] * JE[A.dof + c] + fw[1] * JE[nv + A.dof + c] + fw[2] * JE[2 * nv + A.dof + c]) * ew); J.push(rowF); e.push(Math.max(-0.1, -0.04 - ef) * ew); }
        }
        const R = J.length, M = []; for (let i = 0; i < R; i++) { M.push([]); for (let j = 0; j < R; j++) { let s = 0; for (let c = 0; c < N; c++) s += J[i][c] * ARM_W[c] * J[j][c]; M[i].push(s + (i === j ? 0.003 : 0)); } }
        const y = solve6(M, e);
        for (let c = 0; c < N; c++) {
          let s = 0; for (let i = 0; i < R; i++) s += J[i][c] * y[i];
          const qa = A.dof + c + 1, jid = m.dof_jntid[A.dof + c];
          let q = D.qpos[qa] + Math.max(-0.15, Math.min(0.15, ARM_W[c] * s));
          if (m.jnt_limited[jid]) {
            const lo = m.jnt_range[2 * jid], hi = m.jnt_range[2 * jid + 1], mg = 0.17;   // keep 10 deg off the stops: a
            q = Math.max(lo, Math.min(hi, q));                                          // shoulder pressed on its stops
            if (c >= 3 && c < 6) { if (q < lo + mg) q += (lo + mg - q) * 0.5; else if (q > hi - mg) q -= (q - hi + mg) * 0.5; }   // looks wrenched
          }
          D.qpos[qa] = q;
        }
      }
    }
    void K; return D;
  }
  // get-ups (user 21:16: "they use their arms as levers to get up"): where the recording has a hand on the floor (or
  // puts it there within 0.3 s), his own arm reaches the floor there - the recording's hand seen from the recording's
  // shoulder, placed the same way from HIS shoulder, wrist 3 cm over the floor, palm down; a pinned hand stays put
  // (pressing a little). The recording's arm angles alone left his hands in the air (his trunk is never quite where
  // the recording's is). tg: the springs' joint targets (by actuator).
  getupArms(tg) {
    const K = this.kin, W = this.W, d = W.d, G = this.G.d, cl = K.anim && K.anim.clip; if (!cl || this.noGetupArms) return;
    const o = K.toWorld([0, 0, 0]), u1 = K.toWorld([0, 0, 1]), up = [u1[0] - o[0], u1[1] - o[1], u1[2] - o[2]];
    const pin = this.pins && this.pins.on || {}, sides = [], tgt = {}, NIN = {};
    const P3 = (dd, b) => [dd.xpos[3 * b], dd.xpos[3 * b + 1], dd.xpos[3 * b + 2]];
    for (const side of ['Left', 'Right']) {
      const k = side === 'Left' ? 2 : 3, Pn = pin[side + 'Hand'];
      if (Pn && Pn.ground) { tgt[side] = Pn.p.map((v, i) => v - up[i] * 0.004); }
      else if (cl.contact(K.anim.x, k) > 0.5 || cl.contact(K.anim.x + 12, k) > 0.5 || cl.contact(K.anim.x + 24, k) > 0.5) {   // (user 22:54: arms to the floor BEFORE the torso needs them: 0.8 s ahead)
        const hb = W.bodyId[side + 'Hand'], sb = W.bodyId[side + 'Arm'];
        const rel = P3(G, hb).map((v, i) => v - P3(G, sb)[i]), rn = rel[0] * up[0] + rel[1] * up[1] + rel[2] * up[2];
        const T = P3(d, sb).map((v, i) => v + rel[i] - rn * up[i]), h = (T[0] - o[0]) * up[0] + (T[1] - o[1]) * up[1] + (T[2] - o[2]) * up[2];
        tgt[side] = T.map((v, i) => v - (h - 0.03) * up[i]);
      } else continue;
      sides.push(side); NIN[side] = up.map(v => -v);
    }
    if (!sides.length) return;
    const D = this.solveArms(sides, tgt, NIN, up, false);
    W.act.forEach((a, j) => { const sd = sides.find(s => a.name.startsWith(s + 'Arm_') || a.name.startsWith(s + 'ForeArm_') || a.name.startsWith(s + 'Hand_')); if (sd) tg[j] = D.qpos[a.qadr]; });
  }
  // get-ups, the legs (user 23:03: "the feet should be planted on the floor ... He's trying to get up, but he's raising
  // his legs"; 23:06: "His feet never get planted. The knees are bending too much"): while the recording sits or lies
  // (its hips under 55 cm) and isn't kneeling on that leg, his own leg puts the foot flat on the floor in front of HIS
  // hips - the recording's foot direction from its hips, at a comfortable 35-60 cm (knee ~70-110 deg, not folded shut),
  // hip twist small (no knees popping out). The recordings lift and wave their legs as they sit up (a fainting actor,
  // a roll): a person pushing himself up keeps his feet down and pushes with them.
  getupLegs(tg) {
    const K = this.kin, W = this.W, m = W.m, mj = W.mj, nv = m.nv, d = W.d, G = this.G.d; if (this.noGetupLegs) return;
    const o = K.toWorld([0, 0, 0]), u1 = K.toWorld([0, 0, 1]), up = [u1[0] - o[0], u1[1] - o[1], u1[2] - o[2]];
    const P3 = (dd, b) => [dd.xpos[3 * b], dd.xpos[3 * b + 1], dd.xpos[3 * b + 2]], dot = (a, b) => a[0] * b[0] + a[1] * b[1] + a[2] * b[2];
    const hb = W.bodyId.Hips, gH = P3(G, hb), hz = dot(gH.map((v, i) => v - o[i]), up);
    this.legPlant = null;
    if (hz > 0.55) { this.legW = 0; this.tuckW = 0; return; }
    const myH = P3(d, hb), myHz = dot(myH.map((v, i) => v - o[i]), up), under = myH.map((v, i) => v - myHz * up[i]);
    const sides = [], tgt = {}, tuck = [];
    for (const side of ['Left', 'Right']) {
      const kb = W.bodyId[side + 'Shin'], fb = W.bodyId[side + 'Foot'];
      // the recording kneels on it: its own leg, but (user 23:22: "the feet are always stretched. A normal person would
      // rotate the feet ... so they would land on the floor and use that to get up") the toes tucked under: ankle at
      // a right angle to the shin, toes bent up, so the ball of the foot pushes on the floor (not the top of the foot)
      if (dot(P3(G, kb).map((v, i) => v - o[i]), up) < 0.16) { tuck.push(side); continue; }
      const rel = P3(G, fb).map((v, i) => v - gH[i]), rn = dot(rel, up), hor = rel.map((v, i) => v - rn * up[i]);
      let L = Math.hypot(...hor), dir = L > 1e-3 ? hor.map(v => v / L) : null;
      if (!dir) continue;
      // (Version 25: at 35-60 cm his weight sat 38 cm behind his feet and he could not rise) close to the hips, knees up
      L = clamp(L, 0.22, 0.4);
      // (the recording's foot sideways from the hips is kept: hip width, not crossed under him)
      tgt[side] = under.map((v, i) => v + dir[i] * L + up[i] * 0.05);
      sides.push(side);
    }
    this.tuckW = tuck.length ? Math.min(1, (this.tuckW || 0) + 1 / 30 / 0.4) : 0;
    W.act.forEach((a, j) => { for (const sd of tuck) {
      if (a.name === sd + 'Foot_x') tg[j] += (Math.min(tg[j], -0.05) - tg[j]) * this.tuckW;
      if (a.name === sd + 'ToeBase_x') tg[j] += (-0.85 - tg[j]) * this.tuckW; } });
    if (!sides.length) { this.legW = 0; return; }
    const D = this.ikL || (this.ikL = new mj.MjData(m));
    if (!this.jpL) { this.jpL = new mj.DoubleBuffer(3 * nv); this.jrL = new mj.DoubleBuffer(3 * nv); }
    D.qpos.set(d.qpos);
    const LW = [1, 1, 0.3, 1, 0, 0, 0.8, 0.8, 0.2], rw = 0.3;
    for (let it = 0; it < 12; it++) {
      mj.mj_kinematics(m, D); mj.mj_comPos(m, D);
      for (const side of sides) {
        const Lg = LEGS[side], b = W.bodyId[Lg.foot], T = tgt[side];
        // sole flat: the foot's rest "up" (local z) along the ground's up
        const R = D.xmat.slice(9 * b, 9 * b + 9), fz = [R[2], R[5], R[8]];
        const cx = [fz[1] * up[2] - fz[2] * up[1], fz[2] * up[0] - fz[0] * up[2], fz[0] * up[1] - fz[1] * up[0]];
        const e = [T[0] - D.xpos[3 * b], T[1] - D.xpos[3 * b + 1], T[2] - D.xpos[3 * b + 2], cx[0] * rw, cx[1] * rw, cx[2] * rw];
        mj.mj_jacBody(m, D, this.jpL, this.jrL, b);
        const JP = this.jpL.GetView(), JR = this.jrL.GetView(), N = 9, J = [];
        for (let r = 0; r < 6; r++) { const row = []; for (let c = 0; c < N; c++) row.push(r < 3 ? JP[r * nv + Lg.dof + c] : JR[(r - 3) * nv + Lg.dof + c] * rw); J.push(row); }
        const M = []; for (let i = 0; i < 6; i++) { M.push([]); for (let j = 0; j < 6; j++) { let s = 0; for (let c = 0; c < N; c++) s += J[i][c] * LW[c] * J[j][c]; M[i].push(s + (i === j ? 0.003 : 0)); } }
        const y = solve6(M, e);
        for (let c = 0; c < N; c++) { let s = 0; for (let i = 0; i < 6; i++) s += J[i][c] * y[i]; D.qpos[Lg.dof + 1 + c] += clamp(LW[c] * s, -0.2, 0.2); }
        // within the joints' ranges, hip twist and spread kept small, knee 25-120 deg
        for (let c = 0; c < N; c++) { const jid = m.dof_jntid[Lg.dof + c], qa = Lg.dof + 1 + c; D.qpos[qa] = clamp(D.qpos[qa], m.jnt_range[2 * jid], m.jnt_range[2 * jid + 1]); }
        D.qpos[Lg.dof + 3] = clamp(D.qpos[Lg.dof + 3], -0.35, 0.35); D.qpos[Lg.dof + 2] = clamp(D.qpos[Lg.dof + 2], -0.5, 0.5);
        D.qpos[Lg.dof + 4] = clamp(D.qpos[Lg.dof + 4], 0.45, 2.4);
      }
    }
    // eased in over the first 0.3 s of each planting (no snap)
    this.legW = Math.min(1, (this.legW || 0) + 1 / 30 / 0.3);
    W.act.forEach((a, j) => { const sd = sides.find(s => a.name.startsWith(s + 'Leg_') || a.name.startsWith(s + 'Shin_') || a.name.startsWith(s + 'Foot_')); if (sd) tg[j] += (D.qpos[a.qadr] - tg[j]) * this.legW; });
    this.legPlant = { sides, tgt };
  }
  // (Version 25; render of a steep-fall get-up: once the recording's hips passed 55 cm, its leg angles took over while
  // his hips were still at 20 cm - knees splayed out, a leg out sideways, a frog squat for 3 s) rising with both feet
  // flat on the floor: his own legs, solved on a copy of HIS body whose hips are put up to 7 cm higher (toward the
  // recording's height), with each foot kept where it stands, flat, knees forward (hip twist and spread small). The
  // legs extend toward that and, feet on the floor, push him up - standing up from a squat. Returns true when it ran.
  getupRise(tg) {
    const K = this.kin, W = this.W, m = W.m, mj = W.mj, nv = m.nv, d = W.d, G = this.G.d; if (this.noGetupRise) return false;
    const o = K.toWorld([0, 0, 0]), u1 = K.toWorld([0, 0, 1]), up = [u1[0] - o[0], u1[1] - o[1], u1[2] - o[2]];
    const P3 = (dd, b) => [dd.xpos[3 * b], dd.xpos[3 * b + 1], dd.xpos[3 * b + 2]], dot = (a, b) => a[0] * b[0] + a[1] * b[1] + a[2] * b[2];
    const ht = p => dot(p.map((v, i) => v - o[i]), up), hb = W.bodyId.Hips;
    const gz = ht(P3(G, hb)), mz = ht(P3(d, hb));
    const feet = {};
    for (const sd of ['Left', 'Right']) { const a = P3(d, W.bodyId[sd + 'Foot']), t = P3(d, W.bodyId[sd + 'ToeBase']); if (globalThis.RISEDBG) console.log('RISE', d.time.toFixed(2), sd, ht(a).toFixed(3), ht(t).toFixed(3), 'gz', gz.toFixed(2), 'mz', mz.toFixed(2), 'lean', (this.leanB || 0).toFixed(2)); if (ht(a) > 0.11 || ht(t) > 0.07) { this.riseW = 0; return false; } feet[sd] = a; }
    if (gz < mz + 0.03 || mz > 0.93) { this.riseW = 0; return false; }
    // his weight over his feet first (getupLean): while it's well behind, he leans, the rise waits
    if ((this.leanB || 0) > 0.06) { this.riseW = 0; return false; }
    const D = this.ikL || (this.ikL = new mj.MjData(m));
    if (!this.jpL) { this.jpL = new mj.DoubleBuffer(3 * nv); this.jrL = new mj.DoubleBuffer(3 * nv); }
    D.qpos.set(d.qpos);
    const dz = Math.min(0.07, gz - mz); for (let i = 0; i < 3; i++) D.qpos[i] += up[i] * dz;
    const LW = [1, 0.6, 0.2, 1, 0, 0, 1, 0.6, 0.2], rw = 0.3, sides = ['Left', 'Right'];
    for (let it = 0; it < 12; it++) {
      mj.mj_kinematics(m, D);
      for (const side of sides) {
        const Lg = LEGS[side], b = W.bodyId[Lg.foot], T = feet[side];
        const R = D.xmat.slice(9 * b, 9 * b + 9), fz = [R[2], R[5], R[8]];
        const cx = [fz[1] * up[2] - fz[2] * up[1], fz[2] * up[0] - fz[0] * up[2], fz[0] * up[1] - fz[1] * up[0]];
        const e = [T[0] - D.xpos[3 * b], T[1] - D.xpos[3 * b + 1], T[2] - D.xpos[3 * b + 2], cx[0] * rw, cx[1] * rw, cx[2] * rw];
        mj.mj_jacBody(m, D, this.jpL, this.jrL, b);
        const JP = this.jpL.GetView(), JR = this.jrL.GetView(), N = 9, J = [];
        for (let r = 0; r < 6; r++) { const row = []; for (let c = 0; c < N; c++) row.push(r < 3 ? JP[r * nv + Lg.dof + c] : JR[(r - 3) * nv + Lg.dof + c] * rw); J.push(row); }
        const M = []; for (let i = 0; i < 6; i++) { M.push([]); for (let j = 0; j < 6; j++) { let s2 = 0; for (let c = 0; c < N; c++) s2 += J[i][c] * LW[c] * J[j][c]; M[i].push(s2 + (i === j ? 0.003 : 0)); } }
        const y = solve6(M, e);
        for (let c = 0; c < N; c++) { let s2 = 0; for (let i = 0; i < 6; i++) s2 += J[i][c] * y[i]; D.qpos[Lg.dof + 1 + c] += clamp(LW[c] * s2, -0.2, 0.2); }
        for (let c = 0; c < N; c++) { const jid = m.dof_jntid[Lg.dof + c], qa = Lg.dof + 1 + c; D.qpos[qa] = clamp(D.qpos[qa], m.jnt_range[2 * jid], m.jnt_range[2 * jid + 1]); }
        D.qpos[Lg.dof + 3] = clamp(D.qpos[Lg.dof + 3], -0.25, 0.25); D.qpos[Lg.dof + 2] = clamp(D.qpos[Lg.dof + 2], -0.35, 0.35);
      }
    }
    this.riseW = Math.min(1, (this.riseW || 0) + 1 / 30 / 0.25);
    W.act.forEach((a, j) => { if (/^(Left|Right)(Leg|Shin|Foot)_/.test(a.name)) tg[j] += (D.qpos[a.qadr] - tg[j]) * this.riseW; });
    this.legPlant = { sides, tgt: feet, rise: dz };
    return true;
  }
  // (Version 25; user 23:17: "a real person would actually bend their legs and then use them to raise themselves up")
  // rising off the floor, his weight has to be over his feet first: sat against a wall (back-into-wall test) he stayed
  // braced in a wall-sit for 6 s while the recording stood up. While the recording rises and both his feet are down,
  // if his centre of mass is behind the middle of his feet (along the way his feet point), he bends forward at the
  // hips and trunk (more the further behind, up to ~45 deg), and the recording waits for him (kin.getupLeanWait).
  getupLean(tg) {
    const K = this.kin, W = this.W, d = W.d, G = this.G.d; K.getupLeanWait = false; if (this.noGetupLean) return;
    const o = K.toWorld([0, 0, 0]), u1 = K.toWorld([0, 0, 1]), up = [u1[0] - o[0], u1[1] - o[1], u1[2] - o[2]];
    const P3 = (dd, b) => [dd.xpos[3 * b], dd.xpos[3 * b + 1], dd.xpos[3 * b + 2]], dot = (a, b) => a[0] * b[0] + a[1] * b[1] + a[2] * b[2];
    const ht = p => dot(p.map((v, i) => v - o[i]), up), hb = W.bodyId.Hips;
    const gz = ht(P3(G, hb)), mz = ht(P3(d, hb));
    if (!(gz > mz + 0.04 || (gz > 0.4 && mz < 0.75))) { this.leanW = Math.max(0, (this.leanW || 0) - 1 / 30 / 0.5); }
    else {
      const A = [], F = [];
      for (const sd of ['Left', 'Right']) { const a = P3(d, W.bodyId[sd + 'Foot']), t = P3(d, W.bodyId[sd + 'ToeBase']); if (ht(a) > 0.16) { this.leanB = 0; K.getupLeanWait = false; this.leanW = Math.max(0, (this.leanW || 0) - 1 / 30 / 0.5); return; } A.push(a); F.push(t.map((v, i) => v - a[i])); }
      // the soles' outline (heel to the ball of the foot, foot width) and his centre of mass, seen along gravity; how
      // far outside it toward his back (his chest's back, flattened)
      const flat = p => { const r = p.map((v, i) => v - o[i]), h = dot(r, up); return r.map((v, i) => v - h * up[i]); };
      const ex = Math.abs(up[0]) < 0.9 ? [1, 0, 0] : [0, 1, 0], e1 = flat(o.map((v, i) => v + ex[i])), l1 = Math.hypot(...e1), a1 = e1.map(v => v / l1);
      const a2 = [up[1] * a1[2] - up[2] * a1[1], up[2] * a1[0] - up[0] * a1[2], up[0] * a1[1] - up[1] * a1[0]], to2 = p => { const q = flat(p); return [dot(q, a1), dot(q, a2)]; };
      const Pp = [];
      A.forEach((a, k) => { const t = a.map((v, i) => v + F[k][i]), A2 = to2(a), T2 = to2(t); let u = [T2[0] - A2[0], T2[1] - A2[1]]; const L = Math.hypot(...u) || 1; u = u.map(v => v / L); const n = [-u[1], u[0]];
        for (const [al, sg] of [[-0.05, 1], [-0.05, -1], [L + 0.03, 1], [L + 0.03, -1]]) Pp.push([A2[0] + u[0] * al + n[0] * 0.04 * sg, A2[1] + u[1] * al + n[1] * 0.04 * sg]); });
      const H = hullPts(Pp), c = to2([d.subtree_com[3 * hb], d.subtree_com[3 * hb + 1], d.subtree_com[3 * hb + 2]]), out = outsidePt(H, c);
      const cb = W.bodyId.Chest, R9 = d.xmat.slice(9 * cb, 9 * cb + 9), back = to2(o.map((v, i) => v + [R9[1], R9[4], R9[7]][i]));   // chest's +y = his back
      const bl = Math.hypot(...back) || 1, behind = out.d > 0 ? out.d * Math.max(0, (out.dir[0] * back[0] + out.dir[1] * back[1]) / (Math.hypot(...out.dir) * bl || 1)) : -0.01;
      this.leanB = behind;
      const want = clamp((behind + 0.02) / 0.2, 0, 1);
      this.leanW = (this.leanW || 0) + (want - (this.leanW || 0)) * (1 - Math.exp(-1 / 30 / 0.25));
      K.getupLeanWait = behind > 0.05;
    }
    const w = this.leanW || 0; if (w <= 0.01) return;
    // hip flexion = Leg_x negative; trunk forward = Spine/Chest _x positive
    W.act.forEach((a, j) => {
      if (/^(Left|Right)Leg_x$/.test(a.name)) tg[j] = Math.min(tg[j], tg[j] - 0.55 * w);
      else if (/^(Spine1|Spine2|Chest)_x$/.test(a.name)) tg[j] = Math.max(tg[j], tg[j] + 0.12 * w);
    });
  }
  // (user 09:07: "a lot of foot sliding when getting up. It plays the correct motion but the feet aren't planted") a foot
  // or hand that comes down on the floor during a get-up stays where it landed (position and turn), his leg or arm
  // reaching to it, until the recording lifts it again (its own foot/hand rises off the floor) or it can't be reached
  // (then it is put down again where it is). The recording, played on his body, slid them: its hips are placed and
  // turned to fit him and the box, and the recorded person's feet don't land where his do.
  getupLocks() {
    const K = this.kin, W = this.W, m = W.m, mj = W.mj, nv = m.nv, d = W.d, G = this.G.d;
    const o = K.toWorld([0, 0, 0]), u1 = K.toWorld([0, 0, 1]), up = [u1[0] - o[0], u1[1] - o[1], u1[2] - o[2]];
    const dot = (a, b) => a[0] * b[0] + a[1] * b[1] + a[2] * b[2], P3 = (dd, b) => [dd.xpos[3 * b], dd.xpos[3 * b + 1], dd.xpos[3 * b + 2]];
    const ht = p => dot(p.map((v, i) => v - o[i]), up);
    const Q = (dd, b) => [dd.xquat[4 * b], dd.xquat[4 * b + 1], dd.xquat[4 * b + 2], dd.xquat[4 * b + 3]];
    const qmu = (a, b) => [a[0] * b[0] - a[1] * b[1] - a[2] * b[2] - a[3] * b[3], a[0] * b[1] + a[1] * b[0] + a[2] * b[3] - a[3] * b[2], a[0] * b[2] - a[1] * b[3] + a[2] * b[0] + a[3] * b[1], a[0] * b[3] + a[1] * b[2] - a[2] * b[1] + a[3] * b[0]];
    const rvec = q => { if (q[0] < 0) q = q.map(v => -v); const sn = Math.hypot(q[1], q[2], q[3]); if (sn < 1e-9) return [0, 0, 0]; const a = 2 * Math.atan2(sn, q[0]); return [q[1] / sn * a, q[2] / sn * a, q[3] / sn * a]; };
    const L = this.gLocks || (this.gLocks = {}); this.lockUp = up;
    mj.mj_kinematics(m, d);
    // down: a foot's ankle within 4 cm of where it sits on the floor (or its toes on it); a hand's wrist within 6 cm
    const isDown = (dd, n) => { const b = W.bodyId[n];
      if (/Foot/.test(n)) { const t = W.bodyId[n.replace('Foot', 'ToeBase')]; return ht(P3(dd, b)) < 0.09 || ht(P3(dd, t)) < 0.045; }
      return ht(P3(dd, b)) < 0.075; };
    const want = {};
    for (const n of ['LeftFoot', 'RightFoot', 'LeftHand', 'RightHand']) {
      const recDown = isDown(G, n), b = W.bodyId[n];
      // (V30) the feet's spots come from the recording: where the recorded person's foot plants, frozen there; his leg
      // reaches it (locked where his own foot happened to be - often still 5-8 cm up, coming down from the blend - it crept)
      if (!L[n] && recDown && (/Foot/.test(n) || isDown(d, n))) L[n] = /Foot/.test(n) ? { p: P3(G, b), q: Q(G, b), t: 0 } : { p: P3(d, b), q: Q(d, b), t: 0 };
      if (L[n]) { L[n].t += 1 / 30; want[n] = L[n]; }
    }
    if (!Object.keys(want).length) return;
    const D = this.ikG || (this.ikG = new mj.MjData(m));
    if (!this.jpG) { this.jpG = new mj.DoubleBuffer(3 * nv); this.jrG = new mj.DoubleBuffer(3 * nv); }
    D.qpos.set(d.qpos);
    const chain = n => /Foot/.test(n) ? { dof: LEGS[n.startsWith('Left') ? 'Left' : 'Right'].dof, N: 9, LW: [1, 1, 0.3, 1, 0, 0, 1, 1, 0.5] }
      : { dof: ARMS[n.startsWith('Left') ? 'Left' : 'Right'].dof, N: 12, LW: ARM_W };
    const rw = globalThis.GL_RW ?? (ht([d.qpos[0], d.qpos[1], d.qpos[2]]) > 0.8 ? 0.08 : 0.25);   // standing up: the spot matters more than a perfectly flat sole
    // the planted feet are the fixed points (a person's body moves over their feet, not the feet under it): first his
    // whole body moves along the floor so the planted feet can be reached where they are (the recording's placement
    // moves with it), then his legs and arms reach them
    const feetIn = Object.keys(want).filter(n => /Foot/.test(n));
    if (feetIn.length && !globalThis.GL_NOSHIFT) for (let pass = 0; pass < 3; pass++) {
      this.limbIK(D, want, chain, rw, 10);
      mj.mj_kinematics(m, D);
      const r = [0, 0, 0]; for (const n of feetIn) { const b = W.bodyId[n]; for (let k = 0; k < 3; k++) r[k] += (want[n].p[k] - D.xpos[3 * b + k]) / feetIn.length; }
      // along the floor fully; up/down only downward (his hips higher than his legs reach: he stays a little lower)
      const rn = dot(r, up); for (let k = 0; k < 3; k++) r[k] -= Math.max(0, rn) * up[k];
      if (Math.hypot(...r) < 0.003) break;
      for (let k = 0; k < 3; k++) { r[k] *= 0.8; D.qpos[k] += r[k]; d.qpos[k] += r[k]; }
      this.stats.lockShift = (this.stats.lockShift || 0) + Math.hypot(...r);
    }
    this.limbIK(D, want, chain, rw, globalThis.GL_IT ?? 14);
    mj.mj_kinematics(m, D);
    for (const n in want) {
      const b = W.bodyId[n], lk = want[n], C = chain(n), err = Math.hypot(lk.p[0] - D.xpos[3 * b], lk.p[1] - D.xpos[3 * b + 1], lk.p[2] - D.xpos[3 * b + 2]);
      // out of reach: the spot creeps toward where the limb can be (a slow slide, not a jump); far out, it lets go
      if (err > 0.05) { const bp = [D.xpos[3 * b], D.xpos[3 * b + 1], D.xpos[3 * b + 2]]; for (let k = 0; k < 3; k++) lk.p[k] += (bp[k] - lk.p[k]) * 0.25; }
      for (let c = 0; c < C.N; c++) { const qa = C.dof + 1 + c; d.qpos[qa] = D.qpos[qa]; }
    }
    this.stats.locks = (this.stats.locks || 0) + Object.keys(want).length;
  }
  // damped least squares on a scratch body D: each part in `want` to its locked spot and turn (chains from chain(n))
  limbIK(D, want, chain, rw, iters) {
    const W = this.W, m = W.m, mj = W.mj, nv = m.nv;
    const qmu = (a, b) => [a[0] * b[0] - a[1] * b[1] - a[2] * b[2] - a[3] * b[3], a[0] * b[1] + a[1] * b[0] + a[2] * b[3] - a[3] * b[2], a[0] * b[2] - a[1] * b[3] + a[2] * b[0] + a[3] * b[1], a[0] * b[3] + a[1] * b[2] - a[2] * b[1] + a[3] * b[0]];
    const rvec = q => { if (q[0] < 0) q = q.map(v => -v); const sn = Math.hypot(q[1], q[2], q[3]); if (sn < 1e-9) return [0, 0, 0]; const a = 2 * Math.atan2(sn, q[0]); return [q[1] / sn * a, q[2] / sn * a, q[3] / sn * a]; };
    for (let it = 0; it < iters; it++) {
      mj.mj_kinematics(m, D);
      for (const n in want) {
        const lk = want[n], b = W.bodyId[n], C = chain(n);
        const r = rvec(qmu(lk.q, [D.xquat[4 * b], -D.xquat[4 * b + 1], -D.xquat[4 * b + 2], -D.xquat[4 * b + 3]]));
        // (a planted foot may swivel on the floor: its turn about "up" is free, it only stays flat)
        if (this.lockUp && /Foot/.test(n)) { const u = this.lockUp, ru = r[0] * u[0] + r[1] * u[1] + r[2] * u[2]; for (let k = 0; k < 3; k++) r[k] -= ru * u[k]; }
        const e = [lk.p[0] - D.xpos[3 * b], lk.p[1] - D.xpos[3 * b + 1], lk.p[2] - D.xpos[3 * b + 2], r[0] * rw, r[1] * rw, r[2] * rw];
        mj.mj_jacBody(m, D, this.jpG, this.jrG, b);
        const JP = this.jpG.GetView(), JR = this.jrG.GetView(), J = [];
        const U = this.lockUp && /Foot/.test(n) ? this.lockUp : null;
        for (let rr = 0; rr < 6; rr++) { const row = []; for (let c = 0; c < C.N; c++) { let v = rr < 3 ? JP[rr * nv + C.dof + c] : JR[(rr - 3) * nv + C.dof + c] * rw;
          if (U && rr >= 3) { const ju = (JR[C.dof + c] * U[0] + JR[nv + C.dof + c] * U[1] + JR[2 * nv + C.dof + c] * U[2]) * rw; v -= ju * U[rr - 3]; } row.push(v); } J.push(row); }
        const M = []; for (let i = 0; i < 6; i++) { M.push([]); for (let j = 0; j < 6; j++) { let s2 = 0; for (let c = 0; c < C.N; c++) s2 += J[i][c] * C.LW[c] * J[j][c]; M[i].push(s2 + (i === j ? 0.002 : 0)); } }
        const y = solve6(M, e);
        for (let c = 0; c < C.N; c++) { let s2 = 0; for (let i = 0; i < 6; i++) s2 += J[i][c] * y[i]; const qa = C.dof + 1 + c, jid = m.dof_jntid[C.dof + c];
          D.qpos[qa] = clamp(D.qpos[qa] + clamp(C.LW[c] * s2, -0.2, 0.2), m.jnt_range[2 * jid], m.jnt_range[2 * jid + 1]); }
      }
    }
  }
  physArms() {
    const K = this.kin, W = this.W, m = W.m, mj = W.mj, nv = m.nv;
    if (!(this.armW > 0) || !K.hands) { this.armSmOn = false; K.armOverride = null; return; }
    // a wall spot behind his shoulder (his body has turned): that arm doesn't reach for it (snapshot 22:37)
    K.armSkip = K.armSkip || { Left: 0, Right: 0 };
    for (const sd of ['Left', 'Right']) K.armSkip[sd] = Math.max(0, K.armSkip[sd] - 1 / 30);
    const sides = Object.keys(K.hands).filter(sd => {
      const awk = this.pins && this.pins.awkward && !this.pins.on[sd + 'Hand'] && this.pins.awkward(sd, K.hands[sd].p);
      if (awk) K.armSkip[sd] = 1.5;   // that arm leaves the wall alone for 1.5 s (the recording's arm meanwhile)
      return !awk;
    });
    if (!sides.length) { this.armSmOn = false; return; }
    const D = this.ikD || (this.ikD = new mj.MjData(m));
    if (!this.jp2) { this.jp2 = new mj.DoubleBuffer(3 * nv); this.jr2 = new mj.DoubleBuffer(3 * nv); this.jp3 = new mj.DoubleBuffer(3 * nv); this.jr3 = new mj.DoubleBuffer(3 * nv); }
    const u0 = K.toWorld([0, 0, 0]), u1 = K.toWorld([0, 0, 1]), up = [u1[0] - u0[0], u1[1] - u0[1], u1[2] - u0[2]];
    D.qpos.set(W.d.qpos);
    // where each wrist goes: pinned -> where the pin holds it; else the recording's spot, but just touching the wood
    // (the recording's spot is 1.5 cm INTO the wall, for the recording's own lean: his real hands pressed and shoved him)
    const pin = this.pins && this.pins.on || {}, tgt = {};
    for (const side of sides) { const H = K.hands[side], P = pin[side + 'Hand'];
      tgt[side] = P && P.out ? P.p.map((v, i) => v - P.out[i] * 0.004) : H.wall ? H.p.map((v, i) => v + H.wall.out[i] * 0.037) : H.p; }
    const rw = 0.35, qm = (a, b) => [a[0] * b[0] - a[1] * b[1] - a[2] * b[2] - a[3] * b[3], a[0] * b[1] + a[1] * b[0] + a[2] * b[3] - a[3] * b[2], a[0] * b[2] - a[1] * b[3] + a[2] * b[0] + a[3] * b[1], a[0] * b[3] + a[1] * b[2] - a[2] * b[1] + a[3] * b[0]];
    const qr = (q, v) => { const r = qm(qm(q, [0, v[0], v[1], v[2]]), [q[0], -q[1], -q[2], -q[3]]); return [r[1], r[2], r[3]]; };
    const rv = q => { if (q[0] < 0) q = q.map(v => -v); const s = Math.hypot(q[1], q[2], q[3]); if (s < 1e-9) return [0, 0, 0]; const a = 2 * Math.atan2(s, q[0]); return [q[1] / s * a, q[2] / s * a, q[3] / s * a]; };
    const NIN = {}; for (const side of sides) { const H = K.hands[side]; NIN[side] = H.wall ? H.wall.out.map(v => -v) : qr(H.q, [0, 0, -1]); }
    this.solveArms(sides, tgt, NIN, up, true);
    // (snapshot 23:02: "his left hand is far from the wall, but it's not doing anything ... either his hand is on the
    // wall or the hand is down") an arm that isn't reaching a wall goes back to the recording's own (relaxed) arm. It
    // used to keep his real arm's current angles as its target, with its weight held up: it froze half-raised.
    for (const j of this.armJ) { const a = W.act[j]; if (!sides.some(s => a.name.startsWith(s))) D.qpos[a.qadr] = this.G.d.qpos[a.qadr]; }
    // how far his real hand is from the wall target right now (tests)
    { const S = this.stats.arm || (this.stats.arm = { n: 0, pos: 0, ang: 0 }); const d = W.d;
      for (const side of sides) { const b = W.bodyId[ARMS[side].hand], H = K.hands[side];
        S.pos += Math.hypot(H.p[0] - d.xpos[3 * b], H.p[1] - d.xpos[3 * b + 1], H.p[2] - d.xpos[3 * b + 2]) * 100;
        const pn = qr([d.xquat[4 * b], d.xquat[4 * b + 1], d.xquat[4 * b + 2], d.xquat[4 * b + 3]], [0, 0, -1]), o = H.wall ? H.wall.out : [0, 0, 0];
        { const eb = W.bodyId[side + 'ForeArm'], sb = W.bodyId[side + 'Arm']; S.elb = (S.elb || 0) + ((d.xpos[3 * eb] - d.xpos[3 * sb]) * up[0] + (d.xpos[3 * eb + 1] - d.xpos[3 * sb + 1]) * up[1] + (d.xpos[3 * eb + 2] - d.xpos[3 * sb + 2]) * up[2]) * 100;
          const hb = W.bodyId.Hips, X = d.xmat; let f = [-X[9 * hb + 1], -X[9 * hb + 4], -X[9 * hb + 7]]; const fu = f[0] * up[0] + f[1] * up[1] + f[2] * up[2]; f = f.map((v, i) => v - fu * up[i]); const fl = Math.hypot(...f) || 1;
          const efw = ((d.xpos[3 * eb] - d.xpos[3 * sb]) * f[0] + (d.xpos[3 * eb + 1] - d.xpos[3 * sb + 1]) * f[1] + (d.xpos[3 * eb + 2] - d.xpos[3 * sb + 2]) * f[2]) / fl * 100; S.ef = (S.ef || 0) + efw; S.efMax = Math.max(S.efMax ?? -99, efw); }
        S.tw = Math.max(S.tw || 0, Math.abs(d.qpos[ARMS[side].dof + 4]) * 57.3); S.twSum = (S.twSum || 0) + Math.abs(d.qpos[ARMS[side].dof + 4]) * 57.3;
        S.ang += Math.acos(Math.max(-1, Math.min(1, -(pn[0] * o[0] + pn[1] * o[1] + pn[2] * o[2])))) * 57.3; S.n++; } }
    if (globalThis.ARMDBG) { mj.mj_kinematics(m, D); const d = W.d; for (const side of sides) { const b = W.bodyId[ARMS[side].hand], H = K.hands[side], sh = W.bodyId[side + 'Arm'];
      const ik = Math.hypot(H.p[0] - D.xpos[3 * b], H.p[1] - D.xpos[3 * b + 1], H.p[2] - D.xpos[3 * b + 2]), reach = Math.hypot(H.p[0] - d.xpos[3 * sh], H.p[1] - d.xpos[3 * sh + 1], H.p[2] - d.xpos[3 * sh + 2]);
      let jt = 0; for (let c = 3; c < 12; c++) jt = Math.max(jt, Math.abs(D.qpos[ARMS[side].dof + c + 1] - d.qpos[ARMS[side].dof + c + 1]));
      if (side === 'Left' && globalThis.ARMDBG === 2) { const A = ARMS[side]; const lim = []; for (let c = 3; c < 12; c++) { const jid = m.dof_jntid[A.dof + c], q = D.qpos[A.dof + c + 1]; lim.push(W.mj.mj_id2name(m, W.mj.mjtObj.mjOBJ_JOINT.value, jid) + ' ' + (q * 57.3).toFixed(0) + ' [' + (m.jnt_range[2 * jid] * 57.3).toFixed(0) + ',' + (m.jnt_range[2 * jid + 1] * 57.3).toFixed(0) + ']'); } console.log('LIM', lim.join(' | ')); }
      console.log('ARM', d.time.toFixed(2), side, 'ik resid', (ik * 100).toFixed(0), 'cm reach', (reach * 100).toFixed(0), 'cm  joint gap', (jt * 57.3).toFixed(0), 'deg armW', this.armW.toFixed(2)); } }
    if (this.noPhysArms) return;
    // smoothed (0.2 s): his body sways, the solution with it; following it tick by tick shook the arms and, through
    // them, his whole body (feet slid 26 cm/s on a slow tilt)
    if (!this.armSm) this.armSm = new Float64Array(m.nq);
    const k = this.armSmOn ? 1 - Math.exp(-(1 / 30) / 0.2) : 1; this.armSmOn = true;
    for (const j of this.armJ) { const a = W.act[j], qa = a.qadr; this.armSm[qa] += (D.qpos[qa] - this.armSm[qa]) * k; this.brain.targets[j] += (this.armSm[qa] - this.brain.targets[j]) * this.armW; }
    if (this.armRef) K.armOverride = this.armJ.map(j => [W.act[j].qadr, this.armSm[W.act[j].qadr]]);
  }
  armHold() {
    const W = this.W, d = W.d;
    if (!this.armJ) return;
    for (const j of this.armJ) { const dof = W.act[j].qadr - 1; d.qfrc_applied[dof] = (this.armW || 0) > 0 && this.kin.state !== 'fall' ? d.qfrc_bias[dof] * this.armW : 0; }
  }
  height(dd, b) { return this.kin.toLocal([dd.xpos[3 * b], dd.xpos[3 * b + 1], dd.xpos[3 * b + 2]])[2]; }
  async update(dt, fingerPush) {
    const W = this.W, d = W.d, m = W.m, K = this.kin, B = this.brain, PH = 1 / 480;
    if (this.pins && !this.noPins) this.pins.update(dt);
    else if (this.pins) this.pins.reset();
    if (this.springs && K.state === 'getup' && this.getupBy === 'brain') {
      // the network moves his joints through the get-up recording; the springs' hips help (outside force) lifts
      if (this.mode !== 'getup') { this.mode = 'getup'; B.attach(); B.told = true; this.prevRef = null; this.springs.helpK = 0; this.springs.gPrev = null; }
      const S = this.springs, gr = S.ghostRoot(dt), n = Math.min(40, Math.round(dt / PH)), M = W.mass || 70;
      S.helpK = Math.min(1, S.helpK + dt / 0.6);
      const w0 = 2 * Math.PI * 1.5, kp = M * w0 * w0, kd = 2 * 0.8 * M * w0, kr = 10 * w0 * w0, kdr = 2 * 0.8 * 10 * w0, H = S.help * 1.6 * S.helpK;
      let hs = 0;
      for (let i = 0; i < n; i++) {
        d.xfrc_applied.fill(0); if (fingerPush) fingerPush();
        if (B.due()) { B.readState(); B.down = false; await B.think(this.buildRef(1 / B.hz)); if (B.down) { B.down = false; B.setTone(1); } }
        S.hipsHelp(gr, H, kp, kd, kr, kdr); hs += S.lastHelp || 0;
        W.step(B.control);
      }
      this.stats.help = hs / Math.max(1, n); this.stats.err = B.err || 0; return;
    }
    // (Version 25; user 23:32: "They need to be able to naturally get up") get-ups are the real person's recording
    // played on his body, as games do: entered at the moment of the recording that looks most like him lying there,
    // blended from his fallen pose over 0.4 s (the recordings engine also eases in), his whole body following it; the
    // physics and the balance network take over again once he stands (the recording's last pose, still). Driving his
    // physical body through the recordings with muscles failed 3 times in 4 (stuck in a squat, a wall-sit, a frog), and
    // made poses no person makes (his trunk held up on one hand).
    if (K.state === 'getup' && this.getupBy === 'kin') {
      if (this.mode !== 'getup') { this.mode = 'getup'; this.kinFrom = Float64Array.from(d.qpos); this.kinT = 0; this.gLocks = {}; d.xfrc_applied.fill(0); d.qfrc_applied.fill(0); if (this.springs) this.springs.restoreStrength(); }
      this.kinT += dt;
      // (Version 28) your finger pressing on him while he gets up: the recording can't feel it; a firm push (over a
      // third of his weight, for a quarter second) puts him down again, the physics takes him
      if (fingerPush) {
        d.xfrc_applied.fill(0); fingerPush(); let fx = 0, fy = 0, fz = 0; const x = d.xfrc_applied;
        for (let b = 0; b < m.nbody; b++) { fx += x[6 * b]; fy += x[6 * b + 1]; fz += x[6 * b + 2]; }
        d.xfrc_applied.fill(0);
        this.pushT = Math.hypot(fx, fy, fz) > 0.33 * (W.mass || 70) * 9.81 ? (this.pushT || 0) + dt : 0;
        if (this.pushT > 0.25) { this.pushT = 0; K.startFall('pushed during the get-up'); return; }
      }
      // (user 09:09: "a clear snap when it starts playing a get up animation") 0.8 s from his fallen pose (was 0.4)
      const g = this.G.d.qpos, F = this.kinFrom, a0 = Math.min(1, this.kinT / (this.kinBlend ?? 0.8)), a = a0 * a0 * (3 - 2 * a0);
      for (let i = 0; i < 3; i++) d.qpos[i] = F[i] + (g[i] - F[i]) * a;
      let q0 = [F[3], F[4], F[5], F[6]], q1 = [g[3], g[4], g[5], g[6]]; if (q0[0] * q1[0] + q0[1] * q1[1] + q0[2] * q1[2] + q0[3] * q1[3] < 0) q1 = q1.map(v => -v);
      const qq = q0.map((v, i) => v + (q1[i] - v) * a), qn = Math.hypot(...qq) || 1; for (let i = 0; i < 4; i++) d.qpos[3 + i] = qq[i] / qn;
      for (let i = 7; i < m.nq; i++) { let e = g[i] - F[i]; e = Math.atan2(Math.sin(e), Math.cos(e)); d.qpos[i] = F[i] + e * a; }
      W.mj.mj_kinematics(m, d);
      // his legs: feet planted flat while he sits, toes tucked under while he kneels (getupLegs; user 23:03, 23:22)
      // (user 09:09 "a clear snap") the planted legs / tucked toes switch on and off and their solutions jump (a side
      // starts or stops kneeling, the recording's hips pass 55 cm): his legs follow those targets smoothly (0.12 s),
      // starting from the legs he had when he went down
      if (!this.noKinLegs) { const tg = W.act.map(ac => d.qpos[ac.qadr]); this.getupLegs(tg);
        if (!this.legSm || this.kinT <= dt + 1e-6) { this.legSm = new Float64Array(m.nq); this.legSm.set(this.kinFrom); }
        const k = 1 - Math.exp(-dt / (this.legSmT ?? 0.12));
        W.act.forEach((ac, j) => { if (/^(Left|Right)(Leg|Shin|Foot|ToeBase)_/.test(ac.name)) { const qa = ac.qadr; let e = tg[j] - this.legSm[qa]; e = Math.atan2(Math.sin(e), Math.cos(e)); this.legSm[qa] += e * k; d.qpos[qa] = this.legSm[qa]; } }); }
      d.qvel.fill(0); W.mj.mj_forward(m, d);
      // the blend from his fallen pose and his planted legs can still put a limb into a wall or the floor: his whole
      // body moves out of it (the recording itself is kept out by kin.keepLimbsIn)
      for (let it = 0; it < 3; it++) {
        const push = [0, 0, 0]; let any = false;
        for (let i = 0; i < d.ncon; i++) {
          const c = d.contact.get(i), g1 = c.geom1, g2 = c.geom2, dist = c.dist, n = [c.frame[0], c.frame[1], c.frame[2]]; c.delete();
          if (dist > -0.005) continue; const f1 = W.isFig[m.geom_bodyid[g1]], f2 = W.isFig[m.geom_bodyid[g2]]; if (f1 === f2) continue;
          const dir = f2 ? n : n.map(v => -v); for (let k = 0; k < 3; k++) { const e = -dist * dir[k]; if (Math.abs(e) > Math.abs(push[k])) { push[k] = e; any = true; } }
        }
        if (!any) break;
        for (let k = 0; k < 3; k++) d.qpos[k] += push[k]; W.mj.mj_forward(m, d);
      }
      if (globalThis.SLDBG) { W.mj.mj_kinematics(m, d); const b = W.bodyId.LeftFoot, a = K.anim; const cf = a.clip.body(Math.round(a.x), 21); const A = K.anchor; this._sl = { t: d.time, preLock: [d.xpos[3 * b], d.xpos[3 * b + 1]], ghost: [this.G.d.xpos[3 * b], this.G.d.xpos[3 * b + 1]], clipF: [cf[0], cf[1]], anc: [A.x, A.y, A.yaw], x: a.x, offT: K.offT }; }
      if (!this.noGetupLocks) { this.getupLocks(); W.mj.mj_forward(m, d); }
      // (V30, rulebook: walls beat planted feet) a last pass after the feet: anything of him in a wall or the floor moves
      // him out; the spots his feet/hands are held to and the recording's placement move with him (otherwise the next
      // frame pulls him back in and his feet creep)
      for (let it = 0; it < 3 && !this.noGetupLocks; it++) {
        const push = [0, 0, 0]; let any = false;
        for (let i = 0; i < d.ncon; i++) {
          const c = d.contact.get(i), g1 = c.geom1, g2 = c.geom2, dist = c.dist, n = [c.frame[0], c.frame[1], c.frame[2]]; c.delete();
          if (dist > -0.005) continue; const f1 = W.isFig[m.geom_bodyid[g1]], f2 = W.isFig[m.geom_bodyid[g2]]; if (f1 === f2) continue;
          const dir = f2 ? n : n.map(v => -v); for (let k = 0; k < 3; k++) { const e = -dist * dir[k]; if (Math.abs(e) > Math.abs(push[k])) { push[k] = e; any = true; } }
        }
        if (!any) break;
        for (let k = 0; k < 3; k++) d.qpos[k] += push[k]; W.mj.mj_forward(m, d);
        if (K.anchor) { const a0 = K.toLocal([0, 0, 0]), c0 = K.toLocal(push); K.anchor.x += c0[0] - a0[0]; K.anchor.y += c0[1] - a0[1]; }
        if (this.gLocks) for (const nn in this.gLocks) { const P = this.gLocks[nn].p; for (let k = 0; k < 3; k++) P[k] += push[k]; }
      }
      if (globalThis.SLDBG && this._sl) { const b = W.bodyId.LeftFoot, S = this._sl, P = this._slP; if (P) { const f = v => (v * 100).toFixed(1); console.log('SL', S.t.toFixed(3), 'x', S.x.toFixed(1), 'clip', f(S.clipF[0] - P.clipF[0]), f(S.clipF[1] - P.clipF[1]), 'anc', f(S.anc[0] - P.anc[0]), f(S.anc[1] - P.anc[1]), ((S.anc[2] - P.anc[2]) * 57.3).toFixed(1), 'ghost', f(S.ghost[0] - P.ghost[0]), f(S.ghost[1] - P.ghost[1]), 'preLock', f(S.preLock[0] - P.preLock[0]), f(S.preLock[1] - P.preLock[1]), 'final', f(d.xpos[3 * b] - P.final[0]), f(d.xpos[3 * b + 1] - P.final[1])); } S.final = [d.xpos[3 * b], d.xpos[3 * b + 1]]; this._slP = S; }
      d.time += dt;
      if (this.pins) this.pins.reset();
      K.getupLag = 0; K.getupHandWait = false; K.getupLeanWait = false;
      this.stats.err = 0; this.stats.help = 0; return;
    }
    if (this.springs && K.state === 'getup') {
      if (this.mode !== 'getup') { this.mode = 'getup'; const S = this.springs; S.mode = 'follow'; S.helpK = 0; S.gPrev = null; S.lostT = 0; S.setGains(S.stiff, Math.sqrt(S.stiff)); }
      this.springs.update(dt, fingerPush); this.stats.err = 0; this.stats.help = this.springs.stats.help; return;
    }
    this.stats.help = 0;
    if (this.mode === 'getup') { this.mode = 'follow'; d.qfrc_applied.fill(0); if (this.springs) this.springs.restoreStrength(); B.attach(); B.told = true; this.prevRef = null; this.lostT = 0; }
    this.followBody(dt);
    const falling = K.state === 'fall';
    // falling (user 18:23 "why does he get all floppy?"): his muscles stay at ~1/3, easing to 1/4 (was 12% -> 4%),
    // holding the pose he had when he went down, so he keeps his shape (with the joint limits) instead of folding
    if (falling && this.mode !== 'fall') {
      this.mode = 'fall'; B.down = true; this.fallT = 0; this.curled = false; this.headFirst = false; this.guardW = 0; this.hgLast = null; this.hgPrev = null; this.hgOut = null; this.stillT = 0; this.unbrace = 0; this.cgPrev = null; this.stillPrev = null;
      // (falls worker) what his muscles hold while he goes down: his pose at that moment, turned within 0.35 s into a
      // protective pose that depends on which way he is falling (his centre of mass's motion relative to where he
      // faces): forward -> both hands forward and down to catch himself, head up; backward -> hands back and down,
      // chin tucked, back rounded, knees and hips bend to sit down; sideways -> the arm on that side out and down, the
      // other across the front, knees bend. Signs from this body's forward kinematics: Arm_y + raises the arm (from
      // arms-down -1.25 / +1.25); RightArm_z + / LeftArm_z - swing the arm forward; RightForeArm_z + / LeftForeArm_z -
      // bend the elbow; spine/neck x + bends forward; Leg_x - flexes the hip; Shin_x + bends the knee.
      // (Before: both arms swung BACK and the elbows were pushed straight - signs were wrong.)
      this.hold = W.act.map(a => d.qpos[a.qadr]); this.brace = this.hold.slice();
      const set = (n, v) => { const i = W.act.findIndex(a => a.name === n); if (i >= 0) this.brace[i] = v; };
      const atLeast = (n, v) => { const i = W.act.findIndex(a => a.name === n); if (i >= 0) this.brace[i] = v > 0 ? Math.max(this.hold[i], v) : Math.min(this.hold[i], v); };
      const K = this.kin, cv = K.dirLocal([d.qvel[0], d.qvel[1], 0]), { f, l } = K.facing();
      const vf = cv[0] * f[0] + cv[1] * f[1], vl = cv[0] * l[0] + cv[1] * l[1];
      // too slow to tell (sliding down a wall): the way gravity pulls along the floor
      let dir = Math.atan2(vl, vf) * 180 / Math.PI;
      if (Math.hypot(vf, vl) < 0.15) { const g = W.m.opt.gravity, gl = K.dirLocal([g[0], g[1], g[2]]); dir = Math.atan2(gl[0] * l[0] + gl[1] * l[1], gl[0] * f[0] + gl[1] * f[1]) * 180 / Math.PI; }
      this.fallDir = dir;
      atLeast('RightShin_x', 0.45); atLeast('LeftShin_x', 0.45);
      // (user 18:59: "if I flip the phone upside down and he's about to bang his head ... he would protect it with his
      // arms") head first: his hips->head direction points along gravity (the box turned over, or he's tipping
      // head-down): both arms come up around the head, elbows bent, forearms over it, chin tucked
      const hb = W.bodyId.Head, hp = W.bodyId.Hips, g0 = W.m.opt.gravity, gn0 = Math.hypot(g0[0], g0[1], g0[2]) || 9.81;
      const ux = d.xpos[3 * hb] - d.xpos[3 * hp], uy = d.xpos[3 * hb + 1] - d.xpos[3 * hp + 1], uz = d.xpos[3 * hb + 2] - d.xpos[3 * hp + 2], ul = Math.hypot(ux, uy, uz) || 1;
      const headDown = (ux * g0[0] + uy * g0[1] + uz * g0[2]) / ul / gn0;   // 1 = head straight down
      this.headFirst = headDown > 0.25;
      // (user 19:06: "If he's suddenly falling because someone flipped the phone, he should curl as a ball ... the knees
      // near the head and brace yourself") the box turned over: knees pulled to the chest, back rounded, head tucked
      // between the arms
      if (/turned over/.test(this.kin.fallWhy || '')) this.curlBall();
      else if (this.headFirst) this.headBrace();
      else if (Math.abs(dir) < 55) {          // forward
        set('RightArm_y', -0.55); set('RightArm_z', 1.0); set('LeftArm_y', 0.55); set('LeftArm_z', -1.0);
        set('RightForeArm_z', 0.35); set('LeftForeArm_z', -0.35); set('Neck1_x', -0.15); set('Head_x', -0.1);
        atLeast('RightLeg_x', -0.3); atLeast('LeftLeg_x', -0.3);
      } else if (Math.abs(dir) > 125) {         // backward
        // (user 19:09: "If he's about to fall on his back, he would stretch his arms backwards so that his hands hit
        // the wall before his back") a wall behind him within 0.9 m: arms stretched back toward it, elbows almost
        // straight, palms back; no wall: hands back and down toward the floor
        const hipL = K.toLocal([d.xpos[3 * W.bodyId.Hips], d.xpos[3 * W.bodyId.Hips + 1], d.xpos[3 * W.bodyId.Hips + 2]]), bx = W.box;
        const back = [-f[0], -f[1]]; let wallD = 9;
        for (const [ax, lim] of [[0, bx.w / 2], [1, bx.D / 2]]) for (const sg of [1, -1]) if (back[ax] * sg > 0.3) wallD = Math.min(wallD, (sg * lim - hipL[ax]) / back[ax]);
        if (!W.open && wallD < 0.9) {
          set('RightArm_y', -0.8); set('RightArm_z', -1.1); set('LeftArm_y', 0.8); set('LeftArm_z', 1.1);
          set('RightForeArm_z', 0.15); set('LeftForeArm_z', -0.15); set('RightHand_x', -0.4); set('LeftHand_x', -0.4);
          this.reachBack = true;
        } else {
        set('RightArm_y', -0.95); set('RightArm_z', -0.55); set('LeftArm_y', 0.95); set('LeftArm_z', 0.55);
        set('RightForeArm_z', 0.3); set('LeftForeArm_z', -0.3);
        }
        set('Neck1_x', 0.4); set('Neck2_x', 0.1); set('Head_x', 0.1); set('Spine1_x', 0.2); set('Spine2_x', 0.1);
        atLeast('RightLeg_x', -0.7); atLeast('LeftLeg_x', -0.7); atLeast('RightShin_x', 0.8); atLeast('LeftShin_x', 0.8);
      } else {                                  // sideways: dir > 0 = toward his left
        const L = dir > 0, out = L ? 'Left' : 'Right', other = L ? 'Right' : 'Left', so = L ? 1 : -1;
        set(out + 'Arm_y', 0.6 * so); set(out + 'Arm_z', -0.35 * so); set(out + 'ForeArm_z', -0.3 * so);
        set(other + 'Arm_y', 1.0 * -so); set(other + 'Arm_z', 0.8 * so); set(other + 'ForeArm_z', 0.9 * so);
        set('Neck1_y', 0.15 * -so);
      }
      // (user 23:11: "his back looks broken again") the brace never asks the spine to twist or bend sideways: those stay
      // where a person's are at rest (0), and every target stays inside the middle 80% of its joint's range (contacts
      // and soft fall muscles had left Spine1/Spine2/Chest side-bend and twist all at their stops at once)
      W.act.forEach((a, j) => {
        if (/^(Spine1|Spine2|Chest)_[yz]$/.test(a.name)) this.brace[j] = 0;
        const jid = a.joint, lo = W.m.jnt_range[2 * jid], hi = W.m.jnt_range[2 * jid + 1];
        if (W.m.jnt_limited[jid] && hi > lo) { const mid = (lo + hi) / 2, half = (hi - lo) / 2 * 0.8; this.brace[j] = Math.max(mid - half, Math.min(mid + half, this.brace[j])); }
      });
      // resting arms (once still, after a head brace): down along his sides, elbows a little bent; neck neutral
      this.rest = this.brace.slice();
      { const set = (n, v) => { const i = W.act.findIndex(a => a.name === n); if (i >= 0) this.rest[i] = v; };
        set('RightArm_y', -1.1); set('LeftArm_y', 1.1); set('RightArm_z', 0.2); set('LeftArm_z', -0.2); set('RightArm_x', 0); set('LeftArm_x', 0);
        set('RightForeArm_z', 0.4); set('LeftForeArm_z', -0.4); set('Neck1_x', 0.1); set('Neck2_x', 0); set('Head_x', 0); set('Neck1_y', 0); }
      this.groupTone(0);
    }
    if (!falling && this.mode === 'fall') { this.mode = 'follow'; B.attach(); B.told = true; this.prevRef = null; }   // get-up: the network takes over again
    if (falling) { this.fallT += dt; this.groupTone(this.fallT); }
    const n = Math.min(40, Math.round(dt / PH));
    for (let i = 0; i < n; i++) {
      d.xfrc_applied.fill(0);
      if (fingerPush) fingerPush();
      if (B.due()) {
        if (!falling) { B.readState(); B.down = false; await B.think(this.buildRef(1 / B.hz)); if (B.down) { B.down = false; B.setTone(1); } this.plantLegs(); this.flatFeet(); this.armsToWalls(); this.physArms(); }
        else {   // falling: hold the shape (see above) instead of the network
          B.tNext = d.time + 1 / B.hz - 1e-9;
          // turning head-down during the fall (the phone flipped while he was still upright): protect the head now
          if (!this.headFirst && !this.curled && this.fallT < 2.5 && (this.headDownNow() > 0.25 || this.headThreat() < 0.35)) this.headBrace();
          const k = Math.min(1, (this.fallT - (this.headFirst ? this.headT || 0 : 0)) / (this.headFirst ? 0.25 : 0.35));
          W.act.forEach((a, j) => { B.targets[j] = this.hold[j] + (this.brace[j] - this.hold[j]) * k; });
          // (head worker) arms to where his head would hit, ahead of it (see headGuard); faded in within 0.1 s
          let hg = this.noGuard ? null : this.headGuard();
          // (user 23:09: "he often does this cross-arm pose when he's laying on the floor") once he lies still (his
          // chest under 0.25 m/s for 0.4 s) the head is safe: the arms leave it and the hands go to the floor beside
          // his chest, ready to push; (user 23:11) with no head hit coming, the hands reach for the surface he is
          // going toward (or the one becoming the floor under him) to catch and hold himself
          const cbd = W.bodyId.Chest, cp = [d.xpos[3 * cbd], d.xpos[3 * cbd + 1], d.xpos[3 * cbd + 2]], sp0 = this.stillPrev, cv6 = sp0 ? Math.hypot(cp[0] - sp0[0], cp[1] - sp0[1], cp[2] - sp0[2]) * B.hz : 9; this.stillPrev = cp;
          this.stillT = cv6 < 0.25 ? (this.stillT || 0) + 1 / B.hz : 0;
          const settled = this.stillT > 0.4 && this.fallT > 0.6;
          if (settled) hg = null;
          // (user 09:10: "she would have her hands against the wall, so she doesn't bang the face on the wall ... Humans
          // always protect their head") his head near a wall that isn't the ground (lying or leaning on it, or sliding
          // into it slowly): the hands go on that wall beside his face, a little into it, so the arms hold his head off
          // the wood (the guard used to end once he was still: his face rested on the wall); the head turned aside
          const hw = !hg && !this.noHeadWall && (settled || this.headWallT > 0) ? this.headWall() : null;   // once he's stopped (the falling guard does the rest)
          if (hw) { this.hwPushK = clamp(((this.hwGap ?? 0.17) - hw.dist) / 0.08, 0, 1); hg = this.armsTo(hw, this.hwLat ?? 0.2, -(this.hwPush ?? 0.06) * this.hwPushK, W.bodyId.Head); const set = (n, v) => { const i = W.act.findIndex(a => a.name === n); if (i >= 0) hg.push([i, v]); }; set('Neck1_y', 0.4); set('Neck1_x', 0.05); set('Head_x', 0); }   // (pushing only while his face is close - head centre within 17 cm, ~7 cm of air - then just resting the hands there)
          this.headWallT = hw ? (this.headWallT || 0) + 1 / B.hz : 0;
          let cg = null;
          if (!hg && !this.noCatch && !(this.curled && this.fallT < 1.5)) { const I = this.bodyImpact(); if (I) cg = this.armsTo(I, 0.17, 0.06, W.bodyId.Chest); }
          const want = hg || cg;
          this.guardW = Math.max(0, Math.min(1, (this.guardW || 0) + (want ? 1 / 3 : -1 / 12)));
          if (want) this.hgOut = want;
          // the crossed-arms brace (headBrace/curlBall) eases off once he is still
          if (settled && (this.headFirst || this.curled)) this.unbrace = Math.min(1, (this.unbrace || 0) + 1 / B.hz / 0.6);
          if (this.unbrace > 0) W.act.forEach((a, j) => { if (/Arm|Hand|Neck|Head/.test(a.name)) B.targets[j] += (this.rest[j] - B.targets[j]) * this.unbrace; });
          if (this.guardW > 0 && this.hgOut) for (const [j, q] of this.hgOut) B.targets[j] += (q - B.targets[j]) * this.guardW * (hg || settled ? 1 : 0.8);   // (user 09:05: lying there "she should put the hands on the floor to get up") still: fully
        }
      }
      this.armHold();
      W.step(B.control);
    }
    // losing it: far from the recording's pose for a while (the network was trained to give up past 0.5 m)
    // the engine sees his real pose (recoveries and returns to idle start from it; see Kin.setPhys)
    if (!falling) K.setPhys(d.qpos, d, dt);
    // a recovery recording waits for him: when his body lags the recording, it plays slower (down to 60%)
    if (K.state === 'react' && K.anim) K.anim.speed = Math.max(0.6, Math.min(1, 1 - ((B.err || 0) - 0.15) * 1.6));
    // round 5: far from the recording's pose is not the same as falling. He falls when his body is really going down
    // (hips under 60 cm, or head 40 cm below the recording's) for 0.4 s; if he is still up but the recording has
    // drifted away from him (err > 35 cm for 0.3 s: a step on a slope, the return to idle after it), the recording
    // starts again from his pose instead of pulling him over.
    if (!falling && K.state !== 'getup') {
      const hh = this.height(d, W.bodyId.Head), gh = this.height(this.G.d, W.bodyId.Head), hip = this.height(d, W.bodyId.Hips);
      const down = hip < 0.6 || gh - hh > 0.4 || (B.err || 0) > 0.9;
      this.lostT = down ? this.lostT + dt : Math.max(0, this.lostT - dt);
      if (this.lostT > 0.4) { this.lostT = 0; K.startFall('his body went down'); }
      else if (!down) {
        // during a recovery step the recording slows down for him instead (above); only a big gap restarts it there
        const lim = K.state === 'react' ? 0.55 : 0.35, tl = K.state === 'react' ? 0.4 : 0.3;
        // (combined build) the drift is judged on his hips, head and feet only: with his hands on the walls the
        // network follows the arms loosely, and arm error alone kept restarting the recording (6 restarts in a slow tilt)
        const gd = this.G.d, ids = ['Hips', 'Head', 'LeftFoot', 'RightFoot'].map(n => W.bodyId[n]);
        let core = 0; for (const b of ids) core = Math.max(core, Math.hypot(d.xpos[3 * b] - gd.xpos[3 * b], d.xpos[3 * b + 1] - gd.xpos[3 * b + 1], d.xpos[3 * b + 2] - gd.xpos[3 * b + 2]));
        this.driftT = core > lim ? (this.driftT || 0) + dt : 0;
        if (this.driftT > tl) { this.driftT = 0; K.resync(d.qpos); this.prevRef = null; }
      }
    }
    if (!falling && K.state === 'getup') {   // (no springs given) the network tries the get-up itself
      this.lostT = (B.err || 0) > 0.6 ? this.lostT + dt : Math.max(0, this.lostT - dt);
      if (this.lostT > 0.8) { this.lostT = 0; K.startFall('the get-up failed'); }
    }
    this.stats.err = B.err || 0;
  }
}

// (user 18:22: "you could never break his bones and his shape") each hinge of his physical body may only turn as far as
// real people's joints turn in all our recordings (idles, push recoveries, get-ups: body/clips/kin/limits.json = 0.2-99.8
// percentile of 11,090 frames, widened by 12 deg), knees never more than 5 deg past straight. Without this every hinge
// could turn +-180 deg: in falls he folded flat at the hips and his knees bent backwards.
export function applyHumanLimits(W, L) {
  const m = W.m, mj = W.mj; let n = 0;
  for (let j = 1; j < m.njnt; j++) {
    const name = mj.mj_id2name(m, mj.mjtObj.mjOBJ_JOINT.value, j), r = L[name];
    if (!r) continue;
    let [lo, hi] = r.map(v => v * Math.PI / 180);
    if (/Shin_x$/.test(name)) lo = Math.max(lo, -5 * Math.PI / 180);
    m.jnt_limited[j] = 1; m.jnt_range[2 * j] = lo; m.jnt_range[2 * j + 1] = hi; n++;
    // (falls worker) firmer stops: MuJoCo's default limit spring (0.02 s) let joints go 7-12 deg past their limit in falls
    m.jnt_solref[2 * j] = 0.008; m.jnt_solref[2 * j + 1] = 1;
  }
  return n;
}
