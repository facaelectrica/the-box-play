// The Box - "Recordings" mode (kin.html, Nov 14, user 16:21-16:28: "start with option one and see what we get").
//
// What moves him here: real people's recordings, played straight onto the body, and real physics only when he falls.
//  - standing: the idle recordings (BONES-SEED), one after another, eased into each other
//  - the room tilts: a balance model (a pendulum on the floor, the centre of pressure kept inside his feet) leans him
//    into the slope over his ankles; feet that the recording has on the floor stay planted (leg IK)
//  - he can't hold it with his ankles any more (the capture point leaves his feet, from a quick tilt or your finger):
//    a real person's recovery from a push in that direction (Inria push-recovery trials, CC BY 4.0)
//  - too steep, or the box turns over: he becomes a ragdoll in MuJoCo (gravity, walls, his joints soft) and falls
//  - once he lies still on a floor he can stand on: the get-up recording that starts the way he lies (back, front,
//    side), turned and moved to where he is, eased in from his pose; then back to standing
// Between recordings (and from the ragdoll into a get-up) the change is eased away over a moment (inertialization:
// the difference between the old pose and the new one, positions and speeds, decays like a damped spring), so
// nothing snaps.
//
// Frames: MuJoCo world (x right, y into the box, z up). "Ground" = the face of the box he stands on (W.faces()):
// ground-local z is that face's normal, origin on the face. Clips: qpos 73 per frame (root pos 3, quat w x y z, 66
// hinges x/y/z per joint), feet on z = 0, 30 fps, in the game's clip format (body/clips/*.bin, 376 floats/frame).

const F = 376, QO = 299, CO = 372, FPS = 30;

// ---------- quaternions, w x y z (MuJoCo's order) ----------
const qm = (a, b) => [a[0] * b[0] - a[1] * b[1] - a[2] * b[2] - a[3] * b[3], a[0] * b[1] + a[1] * b[0] + a[2] * b[3] - a[3] * b[2],
  a[0] * b[2] - a[1] * b[3] + a[2] * b[0] + a[3] * b[1], a[0] * b[3] + a[1] * b[2] - a[2] * b[1] + a[3] * b[0]];
const qc = a => [a[0], -a[1], -a[2], -a[3]];
function qv(q, v) {
  const w = q[0], x = q[1], y = q[2], z = q[3];
  const tx = 2 * (y * v[2] - z * v[1]), ty = 2 * (z * v[0] - x * v[2]), tz = 2 * (x * v[1] - y * v[0]);
  return [v[0] + w * tx + (y * tz - z * ty), v[1] + w * ty + (z * tx - x * tz), v[2] + w * tz + (x * ty - y * tx)];
}
const qn = q => { const l = Math.hypot(q[0], q[1], q[2], q[3]) || 1; return [q[0] / l, q[1] / l, q[2] / l, q[3] / l]; };
function rv2q(r) {
  const t = Math.hypot(r[0], r[1], r[2]);
  if (t < 1e-9) return qn([1, r[0] / 2, r[1] / 2, r[2] / 2]);
  const s = Math.sin(t / 2) / t; return [Math.cos(t / 2), r[0] * s, r[1] * s, r[2] * s];
}
function q2rv(q) {
  let w = q[0], x = q[1], y = q[2], z = q[3]; if (w < 0) { w = -w; x = -x; y = -y; z = -z; }
  const s = Math.hypot(x, y, z); if (s < 1e-9) return [2 * x, 2 * y, 2 * z];
  const t = 2 * Math.atan2(s, w); return [x / s * t, y / s * t, z / s * t];
}
const qz = a => [Math.cos(a / 2), 0, 0, Math.sin(a / 2)];
const wrap = a => a - 2 * Math.PI * Math.round(a / (2 * Math.PI));
const clamp = (v, a, b) => Math.max(a, Math.min(b, v));
const xyzw2wxyz = q => [q[3], q[0], q[1], q[2]];
// a joint's three hinges (about x, then y, then z: R = Rx Ry Rz) <-> a quaternion
const eulQ = (x, y, z) => qm(qm([Math.cos(x / 2), Math.sin(x / 2), 0, 0], [Math.cos(y / 2), 0, Math.sin(y / 2), 0]), [Math.cos(z / 2), 0, 0, Math.sin(z / 2)]);
function qEul(q, out, o) {
  const [w, x, y, z] = q;
  const R00 = 1 - 2 * (y * y + z * z), R01 = 2 * (x * y - w * z), R02 = 2 * (x * z + w * y), R12 = 2 * (y * z - w * x), R22 = 1 - 2 * (x * x + y * y);
  out[o] = Math.atan2(-R12, R22); out[o + 1] = Math.asin(clamp(R02, -1, 1)); out[o + 2] = Math.atan2(-R01, R00);
}
// the upper body's joints (qpos addresses of their x hinges): spine, chest, neck, head, collarbones, arms, forearms, hands
const UPPER = [7, 10, 13, 16, 19, 22, 25, 28, 31, 34, 37, 40, 43, 46];
// facing (yaw) of a root quaternion: the body's forward is its -y
function yawOf(q) { const f = qv(q, [0, -1, 0]); return Math.atan2(f[1], f[0]); }

// ---------- a recording ----------
export class Clip {
  constructor(buf, meta = {}) { this.buf = buf; this.n = Math.floor(buf.length / F); Object.assign(this, meta); }
  q(i) { return this.buf.subarray(i * F + QO, i * F + QO + 73); }
  contact(x, k) { const i = clamp(Math.round(x), 0, this.n - 1); return this.buf[i * F + CO + k]; }
  body(i, b) { return this.buf.subarray(i * F + 13 * b, i * F + 13 * b + 3); }
  // the pose at fractional frame x (clamped): positions and hinges blended, the root turn along the short way
  sample(x, out) {
    x = clamp(x, 0, this.n - 1);
    const i = Math.min(this.n - 2, Math.floor(x)), w = x - i, a = this.q(i), b = this.q(i + 1);
    for (let k = 0; k < 3; k++) out[k] = a[k] + (b[k] - a[k]) * w;
    const s = a[3] * b[3] + a[4] * b[4] + a[5] * b[5] + a[6] * b[6] < 0 ? -1 : 1;
    const q = qn([a[3] * (1 - w) + s * b[3] * w, a[4] * (1 - w) + s * b[4] * w, a[5] * (1 - w) + s * b[5] * w, a[6] * (1 - w) + s * b[6] * w]);
    out[3] = q[0]; out[4] = q[1]; out[5] = q[2]; out[6] = q[3];
    for (let k = 7; k < 73; k++) out[k] = a[k] + wrap(b[k] - a[k]) * w;
    return out;
  }
  // which way the spine runs along the floor at frame i (hips -> head), for lying poses
  spineYaw(i) { const h = this.body(i, 0), t = this.body(i, 6); return Math.atan2(t[1] - h[1], t[0] - h[0]); }
}

// pose difference b - a in "velocity space" (72: root pos 3, root turn 3 as a rotation vector, 66 hinges)
function poseDiff(a, b, out) {
  for (let k = 0; k < 3; k++) out[k] = b[k] - a[k];
  const r = q2rv(qm([b[3], b[4], b[5], b[6]], qc([a[3], a[4], a[5], a[6]])));
  out[3] = r[0]; out[4] = r[1]; out[5] = r[2];
  for (let k = 7; k < 73; k++) out[k - 1] = wrap(b[k] - a[k]);
  return out;
}
// pose p plus offset o (velocity space)
function poseAdd(p, o, out) {
  for (let k = 0; k < 3; k++) out[k] = p[k] + o[k];
  const q = qn(qm(rv2q([o[3], o[4], o[5]]), [p[3], p[4], p[5], p[6]]));
  out[3] = q[0]; out[4] = q[1]; out[5] = q[2]; out[6] = q[3];
  for (let k = 7; k < 73; k++) out[k] = p[k] + o[k - 1];
  return out;
}

// the legs: hinge dofs (qpos address = dof + 1) and how freely the IK may use each (knee twist/side bend stiff)
const LEGS = {
  Left: { foot: 'LeftFoot', dof: 60, k: 0 }, Right: { foot: 'RightFoot', dof: 48, k: 1 },
};
const LEG_W = [1, 1, 0.5, /* knee: bend, side, twist */ 1, 0.04, 0.04, /* ankle */ 1, 1, 0.4];

export class Kin {
  constructor(W, mj, lib, opt = {}) {
    this.W = W; this.mj = mj; this.lib = lib;
    const m = W.m;
    this.nv = m.nv;
    this.jp = new mj.DoubleBuffer(3 * m.nv); this.jr = new mj.DoubleBuffer(3 * m.nv);
    this.dq = new mj.DoubleBuffer(m.nv);
    this.hc = opt.comHeight ?? 0.95;          // pendulum length: his centre of mass above the ankles
    // support area around the middle of his feet where he can put his weight (m): toes, heels, the sides
    this.sup = { fwd: 0.10, back: -0.05, side: 0.085 };
    this.stepMargin = opt.stepMargin ?? 0.025; // capture point this far outside it -> a recovery step
    this.fallTilt = (opt.fallTilt ?? 32) * Math.PI / 180;
    this.blendHL = 0.22;                      // easing half-life between recordings (s)
    // gains of his joints while he falls (fraction of the servos' own), and the protective pose's arms
    this.limp = opt.limp ?? [0.35, 0.08, 1.2];   // from, to, over seconds
    this.state = 'stand'; this.t = 0; this.log = [];
    this.cur = new Float64Array(73); this.prev = new Float64Array(73); this.vel = new Float64Array(72);
    this.base = new Float64Array(73); this.tmp = new Float64Array(73); this.tmp2 = new Float64Array(73);
    this.off0 = new Float64Array(72); this.offV = new Float64Array(72); this.offT = 1e9; this.offHL = this.blendHL;
    this.anchor = { x: 0, y: 0, yaw: 0 };
    this.c = [0, 0]; this.cv = [0, 0]; this.xi = [0, 0];
    this.locks = { Left: null, Right: null };
    this.face = null; this.G = { q: [1, 0, 0, 0], p: [0, 0, 0], n: [0, 0, 1], room: 9, name: 'floor' };
    this.gains = W.act.map(a => ({ kp: a.kp, kd: a.kd }));
    this.pushF = [0, 0, 0];
    this.reset();
  }

  // ---------- ground frame ----------
  pickFace() {
    const W = this.W, g = W.m.opt.gravity, gn = Math.hypot(g[0], g[1], g[2]) || 1, U = [-g[0] / gn, -g[1] / gn, -g[2] / gn];
    const faces = W.faces ? W.faces() : [{ name: 'floor', n: [0, 0, 1], p: [0, W.box.D / 2, -W.box.h / 2], room: 9, q: [0, 0, 0, 1] }];
    const dot = f => f.n[0] * U[0] + f.n[1] * U[1] + f.n[2] * U[2];
    let best = faces[0]; for (const f of faces) if (dot(f) > dot(best)) best = f;
    const cur = faces.find(f => f.name === this.face);
    const Fc = cur && dot(cur) > dot(best) - 0.1 ? cur : best;
    this.face = Fc.name;
    // world -> ground: the face's turn (bringing its normal to +z), origin on the face, under the box's middle
    this.G = { q: xyzw2wxyz(Fc.q), p: Fc.p, n: Fc.n, room: Fc.room, name: Fc.name };
    return this.G;
  }
  toLocal(v) { const p = this.G.p; return qv(this.G.q, [v[0] - p[0], v[1] - p[1], v[2] - p[2]]); }
  dirLocal(v) { return qv(this.G.q, v); }
  toWorld(v) { const r = qv(qc(this.G.q), v), p = this.G.p; return [r[0] + p[0], r[1] + p[1], r[2] + p[2]]; }
  localPose(qw, out) {   // world qpos -> ground-local pose
    const p = this.toLocal([qw[0], qw[1], qw[2]]), q = qn(qm(this.G.q, [qw[3], qw[4], qw[5], qw[6]]));
    out[0] = p[0]; out[1] = p[1]; out[2] = p[2]; out[3] = q[0]; out[4] = q[1]; out[5] = q[2]; out[6] = q[3];
    for (let k = 7; k < 73; k++) out[k] = qw[k];
    return out;
  }
  writeWorld(pl) {       // ground-local pose -> d.qpos
    const d = this.W.d, p = this.toWorld([pl[0], pl[1], pl[2]]), q = qn(qm(qc(this.G.q), [pl[3], pl[4], pl[5], pl[6]]));
    d.qpos[0] = p[0]; d.qpos[1] = p[1]; d.qpos[2] = p[2]; d.qpos[3] = q[0]; d.qpos[4] = q[1]; d.qpos[5] = q[2]; d.qpos[6] = q[3];
    for (let k = 7; k < 73; k++) d.qpos[k] = wrap(pl[k]);
  }

  // ---------- playing recordings ----------
  // start clip at frame x0; anchor its root (or its spine, for lying starts) onto where he is now; ease from his
  // current pose. dir: +1 forward, pingpong for idles.
  play(clip, x0, mode, opt = {}) {
    const from = this.cur.slice();
    this.anim = { clip, x: x0, dir: 1, mode, speed: 1 };
    // recoveries (user 17:22: "the shoulders and neck change between the animation they were from without proper
    // retargeting"): the Inria fits hold the upper body in their own neutral (collarbones and chest at 0, elbows
    // straight), the idles don't. So a recovery keeps HIS upper body as it is at that moment, and adds only the
    // recording's own movement of each upper joint from its first frame (trunk bend, arm swing).
    if (opt.keepUpper && !this.noKeepUpper) {
      const ref = clip.sample(x0, new Float64Array(73));
      this.anim.upper = UPPER.map(a => ({ a, mine: eulQ(from[a], from[a + 1], from[a + 2]), refInv: qc(eulQ(ref[a], ref[a + 1], ref[a + 2])) }));
    }
    // anchor: the clip's frame-x0 root (xy, facing) onto his current root (xy, facing), on the ground
    const q0 = clip.sample(x0, this.tmp);
    let yawClip, yawNow;
    if (opt.spine) {
      yawClip = clip.spineYaw(Math.round(x0));
      yawNow = this.spineYawNow();
    } else {
      yawClip = yawOf([q0[3], q0[4], q0[5], q0[6]]);
      yawNow = yawOf([from[3], from[4], from[5], from[6]]);
    }
    const yaw = opt.yaw ?? wrap(yawNow - yawClip), c = Math.cos(yaw), s = Math.sin(yaw);
    const ax = (opt.x ?? from[0]) - (c * q0[0] - s * q0[1]), ay = (opt.y ?? from[1]) - (s * q0[0] + c * q0[1]);
    this.anchor = { x: ax, y: ay, yaw };
    // ease: offset = where he is - where the clip starts (and the same for speeds)
    this.basePose(this.base);
    poseDiff(this.base, from, this.off0);
    const bv = this.baseVel();
    for (let k = 0; k < 72; k++) this.offV[k] = (opt.still ? 0 : this.vel[k]) - bv[k];
    this.offV[3] = this.offV[4] = this.offV[5] = 0;
    this.offT = 0; this.offHL = opt.hl ?? this.blendHL;
    this.locks.Left = this.locks.Right = null;
  }
  anchored(q, out) {
    const A = this.anchor, c = Math.cos(A.yaw), s = Math.sin(A.yaw);
    out[0] = A.x + c * q[0] - s * q[1]; out[1] = A.y + s * q[0] + c * q[1]; out[2] = q[2];
    const r = qn(qm(qz(A.yaw), [q[3], q[4], q[5], q[6]]));
    out[3] = r[0]; out[4] = r[1]; out[5] = r[2]; out[6] = r[3];
    for (let k = 7; k < 73; k++) out[k] = q[k];
    return out;
  }
  basePose(out, x) {
    const a = this.anim, q = a.clip.sample(x ?? a.x, this.tmp2);
    if (a.upper) for (const u of a.upper) qEul(qn(qm(u.mine, qm(u.refInv, eulQ(q[u.a], q[u.a + 1], q[u.a + 2])))), q, u.a);
    return this.anchored(q, out);
  }
  baseVel() {   // the clip's own speed now (velocity space, per second)
    const a = this.anim, x = a.x, x2 = clamp(x + a.dir, 0, a.clip.n - 1), v = new Float64Array(72);
    if (x2 === x) return v;
    const p1 = this.basePose(new Float64Array(73), x), p2 = this.basePose(new Float64Array(73), x2);
    poseDiff(p1, p2, v); const k = FPS / Math.abs(x2 - x); for (let i = 0; i < 72; i++) v[i] *= k;
    return v;
  }
  offsetNow(out) {   // the easing offset decaying like a critically damped spring
    const w = 1.678 / this.offHL, t = this.offT, e = Math.exp(-w * t);
    for (let k = 0; k < 72; k++) out[k] = (this.off0[k] + (this.offV[k] + w * this.off0[k]) * t) * e;
    return out;
  }
  advance(dt) {
    const a = this.anim; a.x += a.dir * a.speed * dt * FPS;
    if (a.mode === 'pingpong') {
      if (a.x >= a.clip.n - 1) { a.x = 2 * (a.clip.n - 1) - a.x; a.dir = -1; }
      if (a.x <= 0) { a.x = -a.x; a.dir = 1; }
      return false;
    }
    if (a.x >= a.clip.n - 1) { a.x = a.clip.n - 1; return true; }   // finished
    return false;
  }
  spineYawNow() {
    const d = this.W.d, h = this.W.bodyId.Hips, t = this.W.bodyId.Head;
    const a = this.toLocal([d.xpos[3 * h], d.xpos[3 * h + 1], d.xpos[3 * h + 2]]), b = this.toLocal([d.xpos[3 * t], d.xpos[3 * t + 1], d.xpos[3 * t + 2]]);
    return Math.atan2(b[1] - a[1], b[0] - a[0]);
  }

  // ---------- states ----------
  reset() {
    const W = this.W, mj = this.mj;
    this.restoreGains();
    this.face = null; this.W.setGravity && 0;
    this.G = { q: [1, 0, 0, 0], p: [0, W.box.D / 2, -W.box.h / 2], n: [0, 0, 1], room: W.box.h, name: 'floor' };
    this.face = 'floor';
    const idle = this.lib.idles[0];
    this.idleK = 0; this.idleT = 2 + Math.random() * 2;
    // face the glass: world -y (toward you) = local -y on the floor
    const q0 = idle.clip.q(0);
    const yaw = wrap(-Math.PI / 2 - yawOf([q0[3], q0[4], q0[5], q0[6]]));
    this.cur.set(q0); this.cur[0] = 0; this.cur[1] = 0;   // placeholder for play()'s "from"
    this.vel.fill(0);
    this.play(idle.clip, 0, 'pingpong', { yaw, x: 0, y: 0, still: true });
    this.offT = 1e9;   // no easing at a reset
    this.c = [0, 0]; this.cv = [0, 0];
    this.state = 'stand'; this.t = 0; this.steps = []; this.tilted = 0;
    this.compose(0);
    W.d.qvel.fill(0); mj.mj_forward(W.m, W.d);
    this.event('reset');
  }
  event(s) { this.log.push({ t: +this.t.toFixed(2), s }); if (this.log.length > 60) this.log.shift(); this.last = s; }

  update(dt) {
    dt = Math.min(dt, 0.05);
    this.t += dt;
    if (this.state === 'fall') return this.updateFall(dt);
    this.pickFaceStanding();
    const g = this.W.m.opt.gravity, gl = qv(this.G.q, [g[0], g[1], g[2]]), G = Math.hypot(g[0], g[1], g[2]) || 9.81;
    const gn = -gl[2], tilt = Math.acos(clamp(gn / G, -1, 1));
    this.tilt = tilt;
    if (this.state === 'stand' || this.state === 'react') {
      // the box turned onto another face, or the floor too steep for long: he falls
      if (this.faceChanged) { this.faceChanged = false; return this.startFall('the box turned over'); }
      this.tilted = tilt > this.fallTilt ? this.tilted + dt : 0;
      if (this.tilted > 0.25) return this.startFall('floor too steep (' + Math.round(tilt * 57.3) + ' deg)');
    }
    if (this.state === 'stand') this.updateStand(dt, gl, gn);
    else if (this.state === 'react') this.updateReact(dt, gl, gn);
    else if (this.state === 'getup') this.updateGetup(dt);
    this.compose(dt);
  }
  pickFaceStanding() {
    const was = this.face; this.pickFace();
    if (was && this.face !== was && (this.state === 'stand' || this.state === 'react')) this.faceChanged = true;
    if (this.state === 'getup' && was && this.face !== was) this.faceChanged = true;
  }

  // the balance model: centre of mass c (ground plane, relative to the middle of his feet) on a pendulum of length hc;
  // the centre of pressure p stays inside his feet; capture point xi = where he'd have to put his weight to stop
  balance(dt, gl, gn, ctl = true) {
    const w2 = Math.max(1, gn) / this.hc, w = Math.sqrt(w2), gt = [gl[0], gl[1]];
    const { f, l } = this.facing();
    const m = this.W.mass || 70, Fp = this.pushF;
    const fp = [Fp[0] / m, Fp[1] / m];
    const n = Math.max(1, Math.ceil(dt / 0.004)), h = dt / n;
    let out = 0, dir = 0;
    for (let i = 0; i < n; i++) {
      const xi = [this.c[0] + this.cv[0] / w + gt[0] / w2, this.c[1] + this.cv[1] / w + gt[1] / w2];
      const xf = xi[0] * f[0] + xi[1] * f[1], xl = xi[0] * l[0] + xi[1] * l[1];
      const K = 2;
      const pf = clamp((1 + K) * xf, this.sup.back, this.sup.fwd), pl = clamp((1 + K) * xl, -this.sup.side, this.sup.side);
      const p = ctl ? [pf * f[0] + pl * l[0], pf * f[1] + pl * l[1]] : [0, 0];
      for (let k = 0; k < 2; k++) { const a = w2 * (this.c[k] - p[k]) + gt[k] + fp[k]; this.cv[k] += a * h; this.c[k] += this.cv[k] * h; }
      const ef = Math.max(xf - this.sup.fwd, this.sup.back - xf, 0), el = Math.max(Math.abs(xl) - this.sup.side, 0);
      out = Math.hypot(ef, el); dir = Math.atan2(xl, xf);
      this.xi = xi;
    }
    // keep the lean in a range a body can do (40 cm at the hips' height ~ 23 deg)
    const cl = Math.hypot(this.c[0], this.c[1]); if (cl > 0.4) { this.c[0] *= 0.4 / cl; this.c[1] *= 0.4 / cl; }
    return { out, dir: dir * 180 / Math.PI };
  }
  facing() {   // his forward and left along the ground, from the pose
    const yaw = yawOf([this.cur[3], this.cur[4], this.cur[5], this.cur[6]]);
    return { f: [Math.cos(yaw), Math.sin(yaw)], l: [-Math.sin(yaw), Math.cos(yaw)], yaw };
  }
  updateStand(dt, gl, gn) {
    const b = this.balance(dt, gl, gn);
    const xr = this.xiPrev ? Math.hypot(this.xi[0] - this.xiPrev[0], this.xi[1] - this.xiPrev[1]) / Math.max(dt, 1e-3) : 0;
    this.xiPrev = this.xi.slice(); this.xiRate = xr;
    if (b.out > this.stepMargin) return this.startReact(b.dir, b.out, xr);
    // calm: the next idle recording now and then
    if ((this.idleT -= dt) <= 0 && Math.hypot(this.cv[0], this.cv[1]) < 0.05) {
      this.idleK = (this.idleK + 1) % this.lib.idles.length; this.idleT = 25 + Math.random() * 20;
      this.play(this.lib.idles[this.idleK].clip, 0, 'pingpong', { hl: 0.45 });
      this.event('idle ' + this.lib.idles[this.idleK].label);
    }
    this.advance(dt);
  }
  startReact(ang, out, rate = 0) {
    // a recording of a person who lost balance this way: a step if it's thrown out fast (or far), else absorbed in place
    const kind = out > 0.06 || rate > 0.35 ? 'step' : 'inplace';
    this.xiPrev = null;
    let want = Math.round(ang / 45) * 45;
    const R = this.lib.reacts;
    if (Math.abs(ang) > 112) {   // backwards: no push recordings that way; the backward step recordings, if any
      const back = R.filter(c => c.ang === 180);
      if (!back.length) { this.c = this.c.map(v => v * 0.98); return; }
      want = 180;
    }
    want = clamp(want, -90, want === 180 ? 180 : 90);
    let pool = R.filter(c => c.ang === want && c.kind === kind); if (!pool.length) pool = R.filter(c => c.ang === want);
    if (!pool.length) return;
    const c = pool[(this.reactN = (this.reactN || 0) + 1) % pool.length];
    this.play(c.clip, (c.push_frame || 3) + 4, 'once', { hl: 0.18, keepUpper: true });
    this.state = 'react'; this.steps.push(this.t); this.steps = this.steps.filter(t => this.t - t < 5);
    this.event(`${kind} ${Math.round(ang)} deg (${c.src || c.file})`);
  }
  updateReact(dt, gl, gn) {
    // the recording carries him; his lean drifts to what the slope asks for (the step put his feet back under him)
    const w2 = Math.max(1, gn) / this.hc, ceq = [-gl[0] / w2, -gl[1] / w2], k = 1 - Math.exp(-dt / 0.35);
    for (let i = 0; i < 2; i++) { this.c[i] += (ceq[i] - this.c[i]) * k; this.cv[i] = 0; }
    if (this.advance(dt)) {
      // three recoveries in 5 s and still going: he can't keep it up
      if (this.steps.length >= 4) return this.startFall('too many recovery steps');
      this.state = 'stand';
      this.play(this.lib.idles[this.idleK].clip, 0, 'pingpong', { hl: 0.4 });
      this.idleT = 8 + Math.random() * 10;
      this.event('back to idle');
    }
  }

  // ---------- composing the pose: recording + easing + lean + planted feet ----------
  compose(dt) {
    const W = this.W, mj = this.mj, d = W.d, m = W.m;
    this.prev.set(this.cur);
    this.basePose(this.base);
    this.offT += dt;
    const o = this.offsetNow(new Float64Array(72));
    poseAdd(this.base, o, this.tmp);
    // the pose before the lean, in the world: where his feet would be
    this.writeWorld(this.tmp);
    mj.mj_kinematics(m, d);
    const kin = this.state === 'stand' || this.state === 'react';
    // planted feet: the recording says the foot is on the floor -> it stays where it landed
    const footNow = {};
    for (const side of ['Left', 'Right']) {
      const L = LEGS[side], b = W.bodyId[L.foot], c = this.anim.clip.contact(this.anim.x, L.k) > 0.5 && kin;
      const p = [d.xpos[3 * b], d.xpos[3 * b + 1], d.xpos[3 * b + 2]], q = [d.xquat[4 * b], d.xquat[4 * b + 1], d.xquat[4 * b + 2], d.xquat[4 * b + 3]];
      footNow[side] = { p, q };
      let lk = this.locks[side];
      if (c && !lk) lk = this.locks[side] = { p: p.slice(), q: q.slice(), w: 0, on: true };
      if (lk) {
        if (!c) lk.on = false;
        lk.w = clamp(lk.w + (lk.on ? 1 : -1) * dt / 0.12, 0, 1);
        if (!lk.on && lk.w <= 0) { this.locks[side] = null; continue; }
        // a planted foot far from where the recording has it (after a turn or a long ease) slides over slowly
        const e = [p[0] - lk.p[0], p[1] - lk.p[1], p[2] - lk.p[2]], el = Math.hypot(...e);
        if (el > 0.05 && this.offT > 2.5 * this.offHL) { const k = Math.min(1, 0.25 * dt / el); for (let i = 0; i < 3; i++) lk.p[i] += e[i] * k; }
      }
    }
    // lean: turn the whole body over the middle of his feet so his centre of mass goes to c
    if (kin) {
      const pl = this.tmp, pv = this.toLocal([(footNow.Left.p[0] + footNow.Right.p[0]) / 2, (footNow.Left.p[1] + footNow.Right.p[1]) / 2, (footNow.Left.p[2] + footNow.Right.p[2]) / 2]);
      const th = [this.c[0] / this.hc, this.c[1] / this.hc], a = Math.hypot(th[0], th[1]);
      if (a > 1e-6) {
        const r = qz(0), ax = [-th[1] / a, th[0] / a, 0], R = rv2q([ax[0] * a, ax[1] * a, 0]);
        const rel = [pl[0] - pv[0], pl[1] - pv[1], pl[2] - pv[2]], nr = qv(R, rel);
        pl[0] = pv[0] + nr[0]; pl[1] = pv[1] + nr[1]; pl[2] = pv[2] + nr[2];
        const q = qn(qm(R, [pl[3], pl[4], pl[5], pl[6]])); pl[3] = q[0]; pl[4] = q[1]; pl[5] = q[2]; pl[6] = q[3];
        void r;
      }
      this.keepInBox(pl);
      this.writeWorld(pl);
      this.legIK();
      this.localPose(d.qpos, this.cur);
    } else {
      this.keepInBox(this.tmp);
      this.cur.set(this.tmp); this.writeWorld(this.cur);
    }
    // its speed (velocity space), for easing into the next recording
    if (dt > 0) { poseDiff(this.prev, this.cur, this.vel); for (let k = 0; k < 72; k++) this.vel[k] /= dt; }
    d.qvel.fill(0);
    mj.mj_forward(m, d);
  }
  keepInBox(pl) {
    if (this.W.open || this.G.name !== 'floor' && this.G.name !== 'ceil') return;
    const b = this.W.box, mx = b.w / 2 - 0.22, my = b.D / 2 - 0.18;   // ground origin is the face's middle
    const nx = clamp(pl[0], -mx, mx), ny = clamp(pl[1], -my, my);
    if (nx !== pl[0] || ny !== pl[1]) { this.anchor.x += nx - pl[0]; this.anchor.y += ny - pl[1]; pl[0] = nx; pl[1] = ny; }
  }
  // the legs reach the planted feet (damped least squares on each leg's 9 hinges, MuJoCo's Jacobians)
  legIK() {
    const W = this.W, mj = this.mj, m = W.m, d = W.d, nv = this.nv;
    const act = ['Left', 'Right'].filter(s => this.locks[s] && this.locks[s].w > 0);
    if (!act.length) return;
    for (let it = 0; it < 6; it++) {
      mj.mj_kinematics(m, d); mj.mj_comPos(m, d);
      let worst = 0;
      for (const side of act) {
        const L = LEGS[side], lk = this.locks[side], b = W.bodyId[L.foot];
        const p = [d.xpos[3 * b], d.xpos[3 * b + 1], d.xpos[3 * b + 2]], q = [d.xquat[4 * b], d.xquat[4 * b + 1], d.xquat[4 * b + 2], d.xquat[4 * b + 3]];
        const w = lk.w * lk.w * (3 - 2 * lk.w);
        const ep = [(lk.p[0] - p[0]) * w, (lk.p[1] - p[1]) * w, (lk.p[2] - p[2]) * w];
        const er = q2rv(qm(lk.q, qc(q))).map(v => v * w * 0.4);
        worst = Math.max(worst, Math.hypot(...ep));
        mj.mj_jacBody(m, d, this.jp, this.jr, b);
        const JP = this.jp.GetView(), JR = this.jr.GetView();
        // J (6 x 9), weighted
        const J = []; for (let r = 0; r < 6; r++) { const row = []; for (let c = 0; c < 9; c++) { const dof = L.dof + c; row.push((r < 3 ? JP[r * nv + dof] : JR[(r - 3) * nv + dof] * 0.4)); } J.push(row); }
        const e = [...ep, ...er];
        // dq = Wd J^T (J Wd J^T + l^2 I)^-1 e
        const A = []; for (let i = 0; i < 6; i++) { A.push([]); for (let j = 0; j < 6; j++) { let s = 0; for (let c = 0; c < 9; c++) s += J[i][c] * LEG_W[c] * J[j][c]; A[i].push(s + (i === j ? 0.01 : 0)); } }
        const y = solve6(A, e);
        for (let c = 0; c < 9; c++) { let s = 0; for (let i = 0; i < 6; i++) s += J[i][c] * y[i]; d.qpos[L.dof + c + 1] += clamp(LEG_W[c] * s, -0.12, 0.12); }
        lk.err = Math.hypot(lk.p[0] - p[0], lk.p[1] - p[1], lk.p[2] - p[2]);
      }
      if (worst < 0.002) break;
    }
    // a planted foot the leg can't reach lets go (it lands again where the recording puts it)
    for (const side of act) { const lk = this.locks[side]; if (lk.err > 0.08) lk.on = false; }
  }

  // ---------- falling: MuJoCo takes over ----------
  startFall(why) {
    const W = this.W, mj = this.mj, m = W.m, d = W.d;
    if (this.ext) {   // a real body (sim/springs.js) falls; the ghost only watches it
      this.state = 'fall'; this.fallT = 0; this.restT = 0; this.restoreGains();
      return this.event('fall: ' + why);
    }
    // the speeds he has now (the recording + lean), as the ragdoll's start
    const q1 = new Float64Array(73), q2 = new Float64Array(73);
    const save = this.cur.slice();
    this.writeWorld(this.prev); q1.set(d.qpos); this.writeWorld(save); q2.set(d.qpos);
    const dt = 1 / 60;
    mj.mj_differentiatePos(m, this.dq, dt, q1, q2);
    const v = this.dq.GetView(); for (let k = 0; k < m.nv; k++) d.qvel[k] = clamp(v[k], -8, 8);
    mj.mj_forward(m, d);
    this.state = 'fall'; this.fallT = 0; this.restT = 0;
    this.target = W.act.map(a => d.qpos[a.qadr]);
    // protective: arms come forward and a bit out, elbows soften (hinge angles of this body; 0 = T-pose)
    this.brace = this.target.slice();
    const set = (n, v) => { const i = W.act.findIndex(a => a.name === n); if (i >= 0) this.brace[i] = v; };
    set('RightArm_y', -0.7); set('RightArm_z', -0.9); set('LeftArm_y', 0.7); set('LeftArm_z', 0.9);
    set('RightForeArm_z', -0.5); set('LeftForeArm_z', 0.5);
    W.tControl = d.time;
    this.event('fall: ' + why);
  }
  setGains(s, sd) {
    const m = this.W.m;
    this.W.act.forEach((a, i) => { const g = this.gains[i]; m.actuator_gainprm[10 * i] = g.kp * s; m.actuator_biasprm[10 * i + 1] = -g.kp * s; m.actuator_biasprm[10 * i + 2] = -g.kd * sd; });
  }
  restoreGains() { if (this.gains) this.setGains(1, 1); }
  updateFall(dt) {
    const W = this.W, d = W.d, m = W.m, PH = 1 / 480;
    this.fallT += dt;
    if (this.ext) {   // the ghost takes the real body's state
      const e = this.ext.d; for (let k = 0; k < 73; k++) d.qpos[k] = e.qpos[k]; for (let k = 0; k < m.nv; k++) d.qvel[k] = e.qvel[k];
      this.mj.mj_forward(m, d);
    } else {
    const [s0, s1, T] = this.limp, u = clamp(this.fallT / T, 0, 1), s = s0 + (s1 - s0) * u;
    this.setGains(s, Math.sqrt(s));
    const kb = clamp(this.fallT / 0.35, 0, 1);
    const tg = this.target.map((v, i) => v + (this.brace[i] - v) * kb * 0.7);
    const ctl = () => tg; ctl.hz = 60;
    const n = Math.min(40, Math.round(dt / PH));
    for (let i = 0; i < n; i++) W.step(ctl);
    }
    this.pickFace();
    // lying still?
    const hipsV = Math.hypot(d.qvel[0], d.qvel[1], d.qvel[2]);
    let jv = 0; for (let k = 6; k < m.nv; k++) jv = Math.max(jv, Math.abs(d.qvel[k]));
    this.restT = hipsV < 0.12 && jv < 1.5 ? this.restT + dt : 0;
    this.localPose(d.qpos, this.cur); this.vel.fill(0);
    const g = W.m.opt.gravity, gl = qv(this.G.q, [g[0], g[1], g[2]]), G = Math.hypot(g[0], g[1], g[2]) || 9.81;
    const tilt = Math.acos(clamp(-gl[2] / G, -1, 1));
    if (this.fallT > 1.0 && this.restT > 0.6 && tilt < this.fallTilt * 0.75 && this.G.room >= 1.9) this.startGetup();
  }

  // ---------- getting up ----------
  startGetup() {
    const W = this.W, d = W.d;
    this.restoreGains();
    const hipsZ = this.cur[2];
    if (hipsZ > 0.75) {   // he never went down (caught against a wall): straight back to standing
      this.state = 'stand'; this.c = [0, 0]; this.cv = [0, 0];
      this.play(this.lib.idles[this.idleK].clip, 0, 'pingpong', { hl: 0.4, still: true });
      return this.event('caught himself: standing');
    }
    // how he lies: his chest's front (the body's -y) against the ground's up
    const ch = W.bodyId.Chest, R = [d.xquat[4 * ch], d.xquat[4 * ch + 1], d.xquat[4 * ch + 2], d.xquat[4 * ch + 3]];
    const fr = qv(this.G.q, qv(R, [0, -1, 0])), lf = qv(this.G.q, qv(R, [1, 0, 0]));
    const G = this.lib.getups;
    let kind = fr[2] > 0.45 ? 'back' : fr[2] < -0.45 ? 'front' : 'side';
    let pool = G[kind] || [];
    if (kind === 'side') {   // the side recordings are all on one side: use them only if he lies on that side
      pool = pool.filter(c => Math.sign(c.lieSide || 0) === Math.sign(lf[2]));
      if (!pool.length) { kind = fr[2] > 0 ? 'back' : 'front'; pool = G[kind]; }
    }
    const c = pool[(this.getN = (this.getN || 0) + 1) % pool.length];
    this.state = 'getup';
    this.play(c.clip, 0, 'once', { spine: true, hl: 0.35, still: true });
    this.event('get up from ' + kind + ' (' + c.file + ')');
  }
  updateGetup(dt) {
    if (this.faceChanged) { this.faceChanged = false; return this.startFall('the box turned over'); }
    if (this.advance(dt)) {
      this.state = 'stand'; this.c = [0, 0]; this.cv = [0, 0]; this.steps = [];
      this.play(this.lib.idles[this.idleK].clip, 0, 'pingpong', { hl: 0.5 });
      this.idleT = 10 + Math.random() * 10;
      this.event('standing again');
    }
  }
}

// small dense solve (Gaussian elimination with partial pivoting)
function solve6(A, b) {
  const n = b.length, M = A.map((r, i) => [...r, b[i]]);
  for (let c = 0; c < n; c++) {
    let p = c; for (let r = c + 1; r < n; r++) if (Math.abs(M[r][c]) > Math.abs(M[p][c])) p = r;
    [M[c], M[p]] = [M[p], M[c]];
    const v = M[c][c] || 1e-12;
    for (let r = c + 1; r < n; r++) { const f = M[r][c] / v; for (let k = c; k <= n; k++) M[r][k] -= f * M[c][k]; }
  }
  const x = new Array(n).fill(0);
  for (let r = n - 1; r >= 0; r--) { let s = M[r][n]; for (let k = r + 1; k < n; k++) s -= M[r][k] * x[k]; x[r] = s / (M[r][r] || 1e-12); }
  return x;
}

// the library: body/clips/kin/kin.json lists the recordings (paths relative to body/clips/)
export async function loadKinLibrary(base = 'body/clips/', fetchBuf = u => fetch(u).then(r => { if (!r.ok) throw new Error(u + ' ' + r.status); return r.arrayBuffer(); })) {
  const J = JSON.parse(new TextDecoder().decode(await fetchBuf(base + 'kin/kin.json')));
  const load = async e => ({ ...e, clip: new Clip(new Float32Array(await fetchBuf(base + e.file)), e) });
  const lib = { idles: await Promise.all(J.idles.map(load)), reacts: await Promise.all(J.reacts.map(load)), getups: {} };
  for (const k of Object.keys(J.getups)) lib.getups[k] = await Promise.all(J.getups[k].map(load));
  lib.reacts.forEach(r => { r.ang = r.ang; });
  return lib;
}
