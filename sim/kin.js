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

// ---------- pose matching (user 17:22: "the transition between poses doesn't make sense") ----------
// A pose's fingerprint: feet, hands and head relative to his hips, turned to where he faces, plus hips height. Used to
// enter a recording at the frame that already looks like him, instead of at its start.
const BODY23 = ['Hips', 'Spine1', 'Spine2', 'Chest', 'Neck1', 'Neck2', 'Head', 'RightShoulder', 'RightArm', 'RightForeArm', 'RightHand', 'LeftShoulder', 'LeftArm', 'LeftForeArm', 'LeftHand', 'RightLeg', 'RightShin', 'RightFoot', 'RightToeBase', 'LeftLeg', 'LeftShin', 'LeftFoot', 'LeftToeBase'];
const FEAT_BODIES = [[21, 2.0], [17, 2.0], [14, 0.5], [10, 0.5], [6, 1.0]];   // left foot, right foot, hands, head
function featOf(pos, rootQ, out) {   // pos(b) -> [x,y,z]; rootQ w x y z
  const yaw = yawOf(rootQ), c = Math.cos(-yaw), s = Math.sin(-yaw), h = pos(0);
  let k = 0;
  for (const [b, w] of FEAT_BODIES) {
    const p = pos(b), dx = p[0] - h[0], dy = p[1] - h[1];
    out[k++] = (c * dx - s * dy) * w; out[k++] = (s * dx + c * dy) * w; out[k++] = (p[2] - h[2]) * w;
  }
  out[k++] = h[2] * 2;
  return out;
}
Clip.prototype.features = function () {
  if (this._f) return this._f;
  this._f = [];
  for (let i = 0; i < this.n; i++) {
    const B = this.buf, o = i * F, q = [B[o + 6], B[o + 3], B[o + 4], B[o + 5]];   // hips quat xyzw -> wxyz
    this._f.push(featOf(b => [B[o + 13 * b], B[o + 13 * b + 1], B[o + 13 * b + 2]], q, new Float64Array(16)));
  }
  return this._f;
};
const featDist = (a, b) => { let s = 0; for (let i = 0; i < a.length; i++) s += (a[i] - b[i]) ** 2; return Math.sqrt(s / 6); };

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
// the arms for bracing on the walls: collarbone (a little), shoulder, elbow (ForeArm_z bends it; x/y barely)
const ARMS = { Right: { hand: 'RightHand', dof: 24, sign: -1 }, Left: { hand: 'LeftHand', dof: 36, sign: 1 } };
const ARM_W = [0, 0, 0, 1, 1, 1, 0.05, 0.05, 1, 0.6, 0.6, 0.6];   // collarbone (not moved: user 18:48 "how can the shoulder be coming out of its joint?"), shoulder, elbow, wrist
// bracing (tunable): start when the floor tilts more than `tilt` rad or turns faster than `rate` rad/s; reach in `reach`
// s, keep `hold` s after it calms, let go in `release` s; palm centre `fwd` m forward of and `down` m under the
// shoulder, `gap` m off the wood; wrist `palm` m behind the palm centre; walls within `maxReach` m of the shoulder
const BR = { tilt: 0.1, rate: 0.26, reach: 0.6, hold: 1.5, release: 0.9, fwd: 0.12, down: 0.22, gap: -0.015, palm: 0.07, maxReach: 0.62, rotW: 0.3, lean: 0.5, base: 1 };
export const BRACE = BR;
// rotation matrix columns (x, y, z axes in the world) -> quaternion w x y z
function matQ(x, y, z) {
  const m00 = x[0], m11 = y[1], m22 = z[2], tr = m00 + m11 + m22;
  let q;
  if (tr > 0) { const S = Math.sqrt(tr + 1) * 2; q = [0.25 * S, (y[2] - z[1]) / S, (z[0] - x[2]) / S, (x[1] - y[0]) / S]; }
  else if (m00 > m11 && m00 > m22) { const S = Math.sqrt(1 + m00 - m11 - m22) * 2; q = [(y[2] - z[1]) / S, 0.25 * S, (y[0] + x[1]) / S, (z[0] + x[2]) / S]; }
  else if (m11 > m22) { const S = Math.sqrt(1 + m11 - m00 - m22) * 2; q = [(z[0] - x[2]) / S, (y[0] + x[1]) / S, 0.25 * S, (z[1] + y[2]) / S]; }
  else { const S = Math.sqrt(1 + m22 - m00 - m11) * 2; q = [(x[1] - y[0]) / S, (z[0] + x[2]) / S, (z[1] + y[2]) / S, 0.25 * S]; }
  return qn(q);
}

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
    const usePre = this.havePre && !opt.fromCur && !opt.fromPose;
    const from = opt.fromPose ? opt.fromPose.slice() : usePre ? this.pre.slice() : this.cur.slice();
    const fromVel = opt.fromPose ? (opt.fromVel || new Float64Array(72)) : usePre ? this.preVel : this.vel;
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
    for (let k = 0; k < 72; k++) this.offV[k] = (opt.still ? 0 : fromVel[k]) - bv[k];
    this.offV[3] = this.offV[4] = this.offV[5] = 0;
    this.offT = 0; this.offHL = opt.hl ?? this.blendHL;
    this.locks.Left = this.locks.Right = null;
  }
  turnAnchor(a) {   // turn the recording's placement by a about his hips
    const A = this.anchor, h = [this.cur[0], this.cur[1]], c = Math.cos(a), s = Math.sin(a);
    const rx = A.x - h[0], ry = A.y - h[1];
    A.x = h[0] + c * rx - s * ry; A.y = h[1] + s * rx + c * ry; A.yaw = wrap(A.yaw + a);
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

  curFeatures() {
    const d = this.W.d, ids = this.W.bodyNames.map(n => this.W.bodyId[n]);
    const loc = b => this.toLocal([d.xpos[3 * ids[b]], d.xpos[3 * ids[b] + 1], d.xpos[3 * ids[b] + 2]]);
    return featOf(loc, [this.cur[3], this.cur[4], this.cur[5], this.cur[6]], new Float64Array(16));
  }
  // the idle frame (any idle, or any but `not`) that looks most like him now
  bestIdle(not = -1, feat = null) {
    const f = feat || this.curFeatures(); let best = null;
    this.lib.idles.forEach((I, k) => {
      if (k === not) return;
      const FF = I.clip.features();
      for (let i = 0; i < FF.length; i += 2) { const dd = featDist(f, FF[i]); if (!best || dd < best.d) best = { k, x: i, d: dd }; }
    });
    return best;
  }
  toIdle(hl, why, not = -1, fromBody = false) {
    const b = this.bestIdle(not, fromBody && this.physOk ? this.physFeat : null);
    this.idleK = b.k;
    const from = fromBody && this.physOk ? this.physUnleaned() : null;
    this.play(this.lib.idles[b.k].clip, b.x, 'pingpong', { hl, fromPose: from || undefined, fromVel: from ? this.physVel : undefined });
    this.event(`${why}: ${this.lib.idles[b.k].label} at ${(b.x / FPS).toFixed(1)} s (pose gap ${(b.d * 100).toFixed(0)} cm)`);
    return b;
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
    this.havePre = false;
    this.play(idle.clip, 0, 'pingpong', { yaw, x: 0, y: 0, still: true, fromCur: true });
    this.offT = 1e9;   // no easing at a reset
    this.c = [0, 0]; this.cv = [0, 0];
    this.state = 'stand'; this.t = 0; this.resetT = 0; this.steps = []; this.tilted = 0;
    this.braceW = 0; this.hands = null; this.calmT = 0; this.upPrev = null; this.upRate = 0; this.wanted = false;
    this.compose(0);
    W.d.qvel.fill(0); mj.mj_forward(W.m, W.d);
    this.event('reset');
  }
  event(s) { this.log.push({ t: +this.t.toFixed(2), s }); if (this.log.length > 60) this.log.shift(); this.last = s; }

  update(dt) {
    dt = Math.max(0, Math.min(dt, 0.05));   // (a negative frame time from the page's clock reset fed the brace)
    this.t += dt;
    if (this.state === 'fall') return this.updateFall(dt);
    this.pickFaceStanding();
    const g = this.W.m.opt.gravity, gl = qv(this.G.q, [g[0], g[1], g[2]]), G = Math.hypot(g[0], g[1], g[2]) || 9.81;
    const gn = -gl[2], tilt = Math.acos(clamp(gn / G, -1, 1));
    this.tilt = tilt;
    if (this.state === 'stand' || this.state === 'react') {
      // the box turned onto another face, or the floor too steep for long: he falls
      if (this.faceChanged) { this.faceChanged = false; return this.startFall('the box turned over'); }
      // the first 1.2 s after a reset the phone's "level" is still being taken (the page averages the held angle over
      // 0.4 s): the raw angle read as a 36 deg slope and he fell at once (phone test, version 6)
      if (this.t - (this.resetT || 0) < 1.2) { this.tilted = 0; const w2 = Math.max(1, gn) / this.hc; this.c = [-gl[0] / w2, -gl[1] / w2]; this.cv = [0, 0]; }   // already leaning as the slope asks, no jolt when it ends
      else this.tilted = tilt > this.fallTilt ? this.tilted + dt : 0;
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
    const w2 = Math.max(1, gn) / this.hc, w = Math.sqrt(w2), lk = 1 - BR.lean * (this.braceW || 0), gt = [gl[0] * lk, gl[1] * lk];   // braced: the walls take part of the slope's pull, he leans less
    const { f, l } = this.facing();
    const m = this.W.mass || 70, Fp = this.pushF;
    const fp = [Fp[0] / m, Fp[1] / m];
    const n = Math.max(1, Math.ceil(dt / 0.004)), h = dt / n;
    let out = 0, dir = 0;
    for (let i = 0; i < n; i++) {
      const xi = [this.c[0] + this.cv[0] / w + gt[0] / w2, this.c[1] + this.cv[1] / w + gt[1] / w2];
      const xf = xi[0] * f[0] + xi[1] * f[1], xl = xi[0] * l[0] + xi[1] * l[1];
      const K = 2;
      const bw = (this.braceW || 0) * BR.base, sb = this.sup.back - 0.08 * bw, sfw = this.sup.fwd + 0.08 * bw, ss = this.sup.side + 0.2 * bw;   // hands on the walls: a wider base
      const pf = clamp((1 + K) * xf, sb, sfw), pl = clamp((1 + K) * xl, -ss, ss);
      const p = ctl ? [pf * f[0] + pl * l[0], pf * f[1] + pl * l[1]] : [0, 0];
      for (let k = 0; k < 2; k++) { const a = w2 * (this.c[k] - p[k]) + gt[k] + fp[k]; this.cv[k] += a * h; this.c[k] += this.cv[k] * h; }
      const ef = Math.max(xf - sfw, sb - xf, 0), el = Math.max(Math.abs(xl) - ss, 0);
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
    if (this.t - (this.resetT || 0) < 1.2) { this.advance(dt); return; }   // (see update: level not taken yet)
    const b = this.balance(dt, gl, gn);
    const xr = this.xiPrev ? Math.hypot(this.xi[0] - this.xiPrev[0], this.xi[1] - this.xiPrev[1]) / Math.max(dt, 1e-3) : 0;
    this.xiPrev = this.xi.slice(); this.xiRate = xr;
    // (feet planted, user 18:50) a step only when balance is really lost: past the margin for 0.15 s, or far past it
    this.outT = b.out > this.stepMargin ? (this.outT || 0) + dt : 0;
    if (this.outT > 0.15 || b.out > 3 * this.stepMargin) { this.outT = 0; return this.startReact(b.dir, b.out, xr); }
    // round 6: he turns back to face the glass, slowly (a person standing re-squares to what they look at): when calm and
    // more than 12 deg off, the recording's facing turns ~6 deg/s about his hips; planted feet pivot with the IK
    if (Math.hypot(this.cv[0], this.cv[1]) < 0.05 && (this.G.name === 'floor' || this.G.name === 'ceil')) {
      const want = qv(this.G.q, [0, -1, 0]), wy = Math.atan2(want[1], want[0]), dy = wrap(wy - this.facing().yaw);
      if (Math.abs(dy) > 0.21) this.turnAnchor(Math.sign(dy) * Math.min(Math.abs(dy), 0.1 * dt));
    }
    // calm: the next idle recording now and then
    if ((this.idleT -= dt) <= 0 && Math.hypot(this.cv[0], this.cv[1]) < 0.05) {
      // another idle, entered where it already looks like him (feet where his are); none close enough: keep this one
      const b = this.bestIdle(this.idleK);
      if (b && b.d < 0.035) this.toIdle(0.5, 'next idle', this.idleK);
      this.idleT = b && b.d < 0.035 ? 25 + Math.random() * 20 : 3;
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
    let c, x0, from = null;
    if (this.physOk && this.ext) {
      // neighbouring directions too (a 70 deg push can use a 45 or 90 recording); the score prefers the closest
      const near = R.filter(r => (r.kind === kind || want === 180) && Math.abs(wrap((r.ang - ang) * Math.PI / 180)) < 0.9);
      const b = this.pickReact(near.length ? near : pool, ang); c = b.c; x0 = b.x; from = this.physUnleaned();
    } else { c = pool[(this.reactN = (this.reactN || 0) + 1) % pool.length]; x0 = (c.push_frame || 3) + 4; }
    this.play(c.clip, x0, 'once', { hl: 0.25, keepUpper: true, fromPose: from || undefined, fromVel: from ? this.physVel : undefined });
    this.state = 'react'; this.steps.push(this.t); this.steps = this.steps.filter(t => this.t - t < 5);
    this.event(`${kind} ${Math.round(ang)} deg (${c.src || c.file})`);
  }
  // ---------- his real body, as the engine sees it (steps worker, round 8) ----------
  // The network's body lags and differs from the recording (20-30 cm during steps). Recoveries and the returns to
  // idle used to start from the ghost: the new recording was anchored where the ghost was, not where he was, so he
  // started each one 20-30 cm off and the engine had to 'catch up with his body' after. Now the driver hands the
  // engine his real pose every frame (setPhys), and those switches start from it.
  setPhys(q, d, dt) {
    if (!this.physPose) { this.physPose = new Float64Array(73); this.physPrev = new Float64Array(73); this.physVel = new Float64Array(72); this.physFeat = new Float64Array(16); this._pv = new Float64Array(72); }
    this.physPrev.set(this.physPose); this.localPose(q, this.physPose);
    if (this.physOk && dt > 0) { poseDiff(this.physPrev, this.physPose, this._pv); for (let k = 0; k < 72; k++) this.physVel[k] += (this._pv[k] / dt - this.physVel[k]) * 0.5; }
    this.physOk = true;
    const ids = this.W.bodyNames.map(n => this.W.bodyId[n]);
    featOf(b => this.toLocal([d.xpos[3 * ids[b]], d.xpos[3 * ids[b] + 1], d.xpos[3 * ids[b] + 2]]), [this.physPose[3], this.physPose[4], this.physPose[5], this.physPose[6]], this.physFeat);
    // his feet's middle (for taking the lean back out of his pose)
    const L = this.W.bodyId.LeftFoot, R = this.W.bodyId.RightFoot;
    this.physMid = this.toLocal([(d.xpos[3 * L] + d.xpos[3 * R]) / 2, (d.xpos[3 * L + 1] + d.xpos[3 * R + 1]) / 2, (d.xpos[3 * L + 2] + d.xpos[3 * R + 2]) / 2]);
  }
  // his real pose without the lean the engine adds on top (c): compose() will add it again, so it isn't counted twice
  physUnleaned() {
    const pl = this.physPose.slice(), pv = this.physMid, th = [this.c[0] / this.hc, this.c[1] / this.hc], a = Math.hypot(th[0], th[1]);
    if (a > 1e-6) {
      const R = qc(rv2q([-th[1], th[0], 0])), rel = [pl[0] - pv[0], pl[1] - pv[1], pl[2] - pv[2]], nr = qv(R, rel);
      pl[0] = pv[0] + nr[0]; pl[1] = pv[1] + nr[1]; pl[2] = pv[2] + nr[2];
      const q = qn(qm(R, [pl[3], pl[4], pl[5], pl[6]])); pl[3] = q[0]; pl[4] = q[1]; pl[5] = q[2]; pl[6] = q[3];
    }
    return pl;
  }
  // the recovery recording (and frame) to enter: right direction first, then the one whose frame looks most like him
  // now (feet, hands, head vs hips) and whose hips move at his speed, searched over its first 0.5 s after the push
  pickReact(pool, ang) {
    const f = this.physFeat, v = this.physVel, fac = this.facing(), yaw = fac.yaw;
    const vf = v[0] * Math.cos(yaw) + v[1] * Math.sin(yaw), vl = -v[0] * Math.sin(yaw) + v[1] * Math.cos(yaw);
    let best = null;
    for (const c of pool) {
      const FF = c.clip.features(), B = c.clip.buf, p0 = c.push_frame || 3;
      for (let i = p0; i < Math.min(c.clip.n - 20, p0 + 15); i++) {
        // the clip's hips speed at frame i, in its own facing
        const o = i * 376, q = [B[o + 6], B[o + 3], B[o + 4], B[o + 5]], cy = yawOf(q), cvx = B[o + 7], cvy = B[o + 8];
        const cf = cvx * Math.cos(cy) + cvy * Math.sin(cy), cl = -cvx * Math.sin(cy) + cvy * Math.cos(cy);
        const dv = Math.hypot(cf - vf, cl - vl), da = Math.abs(wrap((c.ang - ang) * Math.PI / 180));
        const score = featDist(f, FF[i]) + 0.12 * dv + 0.15 * da;
        if (!best || score < best.s) best = { c, x: i, s: score };
      }
    }
    return best;
  }
  // the ghost starts again from his real body's pose (world qpos q): it had drifted from him (a recovery on a slope, a
  // push the network handled its own way) and following it would pull him over. Into the idle frame that matches.
  resync(q, why = 'caught up with his body') {
    this.localPose(q, this.cur); this.vel.fill(0); this.c = [0, 0]; this.cv = [0, 0];
    this.writeWorld(this.cur); this.mj.mj_kinematics(this.W.m, this.W.d);
    this.state = 'stand'; this.havePre = false; this.toIdle(0.35, why); this.idleT = 8 + Math.random() * 8;
  }
  // his real body says his weight is past his feet (sim/springs.js): a step that way, as if thrown out fast
  requestStep(ang, out) { if (this.state === 'stand') this.startReact(ang, Math.max(out, 0.07), 1); }
  updateReact(dt, gl, gn) {
    // the recording carries him; his lean drifts to what the slope asks for (the step put his feet back under him)
    const w2 = Math.max(1, gn) / this.hc, ceq = [-gl[0] / w2, -gl[1] / w2], k = 1 - Math.exp(-dt / 0.35);
    for (let i = 0; i < 2; i++) { this.c[i] += (ceq[i] - this.c[i]) * k; this.cv[i] = 0; }
    if (this.advance(dt)) {
      // three recoveries in 5 s and still going: he can't keep it up
      if (this.steps.length >= 4) return this.startFall('too many recovery steps');
      this.state = 'stand';
      this.toIdle(0.4, 'back to idle', -1, !!this.ext);
      this.idleT = 8 + Math.random() * 10;
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
    // the pose before the lean and the planted feet (round 5: easing into the next recording started from the pose
    // AFTER the lean, and the lean was then added again: on a tilted floor every switch jumped by the slope's angle,
    // 32-60 deg at the spine in one frame)
    if (!this.pre) { this.pre = new Float64Array(73); this.preVel = new Float64Array(72); this.prePrev = new Float64Array(73); }
    this.prePrev.set(this.pre); this.pre.set(this.tmp);
    if (dt > 0 && this.havePre) { poseDiff(this.prePrev, this.pre, this.preVel); for (let k = 0; k < 72; k++) this.preVel[k] /= dt; }
    this.havePre = true;
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
      this.braceArms(dt);
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
  // ---------- hands on the walls (user 18:21: "use the arms to support himself and hold, so he doesn't move if the
  // phone is rotating"). When the floor tilts or the phone turns, he puts his hands flat on the walls beside him (the
  // box is narrow: the side walls are within reach), a little below his shoulders, pressing 2 cm into the wood, and
  // keeps them there until it has been calm for 1.5 s. His physical body then really leans on them (contacts).
  braceArms(dt) {
    if (this.noBrace) { this.braceW = 0; this.hands = null; return; }
    const W = this.W, d = W.d, g = W.m.opt.gravity, gl = qv(this.G.q, [g[0], g[1], g[2]]), G = Math.hypot(...gl) || 9.81;
    const up = [-gl[0] / G, -gl[1] / G, -gl[2] / G];
    const rate = this.upPrev ? Math.acos(clamp(up[0] * this.upPrev[0] + up[1] * this.upPrev[1] + up[2] * this.upPrev[2], -1, 1)) / Math.max(dt, 1e-3) : 0;
    this.upPrev = up;
    this.upRate = (this.upRate || 0) + (rate - (this.upRate || 0)) * (1 - Math.exp(-dt / 0.15));   // smoothed: sensor noise is not "the phone turning"
    const tilt = Math.acos(clamp(up[2], -1, 1));
    const room = !W.open && (this.G.name === 'floor' || this.G.name === 'ceil') && this.state === 'stand';
    const want = room && this.t - (this.resetT || 0) > 2 && (tilt > BR.tilt || this.upRate > BR.rate);   // not in the first 2 s (the phone settling at the start made him brace)
    this.calmT = want ? 0 : (this.calmT || 0) + dt;
    // keep holding only after a real brace (a tiny brace weight from a stray frame used to keep itself going: arms out at the start)
    if (want) this.wanted = true; else if (!(this.braceW > 0)) this.wanted = false;
    const on = want || (room && this.wanted && this.calmT < BR.hold);
    this.braceW = clamp((this.braceW || 0) + (on ? dt / BR.reach : -dt / BR.release), 0, 1);
    if (this.braceW <= 0) { this.hands = null; return; }
    const w = this.braceW * this.braceW * (3 - 2 * this.braceW);
    // where each hand goes, from where his shoulder is NOW (round 1 of the walls work: the targets were fixed in the
    // world when he reached, his body then moved with the idle and the network, the arms were stretched or crushed and
    // the hands slammed the wood: 0-1170 N): out to his side and a little forward, a hand-span under the shoulder,
    // palm flat on the wall, fingers up; the target follows his shoulder slowly (a hand doesn't skate on wood)
    const { f, l } = this.facing(), b = this.W.box, ids = this.W.bodyId;
    if (!this.hands) this.hands = {};
    for (const side of ['Right', 'Left']) {
      const sh = this.toLocal([d.xpos[3 * ids[side + 'Arm']], d.xpos[3 * ids[side + 'Arm'] + 1], d.xpos[3 * ids[side + 'Arm'] + 2]]);
      const s = side === 'Left' ? 1 : -1, dir = [l[0] * s, l[1] * s];
      // the wall on that side (x = +-w/2 or y = +-D/2 in the ground frame, origin under the box's middle)
      let best = null;
      for (const [ax, lim] of [[0, b.w / 2], [1, b.D / 2]]) for (const sg of [1, -1]) {
        if (dir[ax] * sg <= 0.5) continue;
        const t = (sg * lim - sh[ax]) / dir[ax]; if (t > 0 && (!best || t < best.t)) best = { t, ax, sg, lim };
      }
      if (!best || best.t > BR.maxReach) { delete this.hands[side]; continue; }
      // palm centre on the wall: forward of the shoulder, below it; the palm 1.5 cm off the wood (he touches and leans)
      const nW = [0, 0, 0]; nW[best.ax] = best.sg;                    // into the wall (ground frame)
      const pc = [sh[0] + f[0] * BR.fwd, sh[1] + f[1] * BR.fwd, sh[2] - BR.down];
      pc[best.ax] = best.sg * (best.lim - BR.gap);
      // fingers up, tilted forward a little; palm facing the wall
      let fd = [f[0] * 0.35, f[1] * 0.35, 1]; const fl = Math.hypot(...fd); fd = fd.map(v => v / fl);
      const dn = fd[0] * nW[0] + fd[1] * nW[1] + fd[2] * nW[2]; fd = [fd[0] - dn * nW[0], fd[1] - dn * nW[1], fd[2] - dn * nW[2]];
      const fl2 = Math.hypot(...fd); fd = fd.map(v => v / fl2);
      const wrist = [pc[0] - fd[0] * BR.palm, pc[1] - fd[1] * BR.palm, pc[2] - fd[2] * BR.palm];
      const tw = this.toWorld(wrist), fdw = qv(qc(this.G.q), fd), nw = qv(qc(this.G.q), nW);
      // hand frame: palm normal = local -z, fingers = local -x (right) / +x (left), in this body's rest (T) pose
      const xw = side === 'Right' ? fdw.map(v => -v) : fdw, zw = nw.map(v => -v);
      const yw = [zw[1] * xw[2] - zw[2] * xw[1], zw[2] * xw[0] - zw[0] * xw[2], zw[0] * xw[1] - zw[1] * xw[0]];
      const q = matQ(xw, yw, zw);
      const H = this.hands[side];
      if (!H) this.hands[side] = { p: tw, q };
      else {   // follow the shoulder slowly: at most 12 cm/s along the wall
        const e = [tw[0] - H.p[0], tw[1] - H.p[1], tw[2] - H.p[2]], el = Math.hypot(...e), k = Math.min(1, 0.12 * dt / Math.max(el, 1e-6));
        for (let i = 0; i < 3; i++) H.p[i] += e[i] * k; H.q = q;
      }
    }
    this.armIK(w);
  }
  armIK(w) {
    const W = this.W, mj = this.mj, m = W.m, d = W.d, nv = this.nv;
    const act = Object.keys(this.hands || {}); if (!act.length) return;
    mj.mj_kinematics(m, d);
    const start = {};
    for (const side of act) { const b = W.bodyId[ARMS[side].hand]; start[side] = { p: [d.xpos[3 * b], d.xpos[3 * b + 1], d.xpos[3 * b + 2]], q: [d.xquat[4 * b], d.xquat[4 * b + 1], d.xquat[4 * b + 2], d.xquat[4 * b + 3]] }; }
    for (let it = 0; it < 10; it++) {
      mj.mj_kinematics(m, d); mj.mj_comPos(m, d);
      for (const side of act) {
        const A = ARMS[side], b = W.bodyId[A.hand], H = this.hands[side], S0 = start[side];
        const tp = [0, 1, 2].map(i => S0.p[i] + (H.p[i] - S0.p[i]) * w);
        const cq = [d.xquat[4 * b], d.xquat[4 * b + 1], d.xquat[4 * b + 2], d.xquat[4 * b + 3]];
        const tq = qn(qm(rv2q(q2rv(qm(H.q, qc(S0.q))).map(v => v * w)), S0.q));
        const e = [tp[0] - d.xpos[3 * b], tp[1] - d.xpos[3 * b + 1], tp[2] - d.xpos[3 * b + 2], ...q2rv(qm(tq, qc(cq))).map(v => v * BR.rotW)];
        mj.mj_jacBody(m, d, this.jp, this.jr, b);
        const JP = this.jp.GetView(), JR = this.jr.GetView(), N = ARM_W.length;
        const J = []; for (let r = 0; r < 6; r++) { const row = []; for (let c = 0; c < N; c++) row.push(r < 3 ? JP[r * nv + A.dof + c] : JR[(r - 3) * nv + A.dof + c] * BR.rotW); J.push(row); }
        const M = []; for (let i = 0; i < 6; i++) { M.push([]); for (let j = 0; j < 6; j++) { let s = 0; for (let c = 0; c < N; c++) s += J[i][c] * ARM_W[c] * J[j][c]; M[i].push(s + (i === j ? 0.004 : 0)); } }
        const y = solve6(M, e);
        for (let c = 0; c < N; c++) { let s = 0; for (let i = 0; i < 6; i++) s += J[i][c] * y[i]; d.qpos[A.dof + c + 1] += clamp(ARM_W[c] * s, -0.15, 0.15); }
      }
    }
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
    this.havePre = false; this.fallWhy = why;
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
    // (falls worker) arms forward = RightArm_z + / LeftArm_z -, elbows bend = RightForeArm_z + / LeftForeArm_z - (signs were reversed)
    set('RightArm_y', -0.7); set('RightArm_z', 0.9); set('LeftArm_y', 0.7); set('LeftArm_z', -0.9);
    set('RightForeArm_z', 0.5); set('LeftForeArm_z', -0.5);
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
    // (user 19:06: "if the phone is rotated like this and he has less space, he would still get on his knees") a floor
    // with less than standing room (the back wall when the phone lies flat: ~1.3-1.5 m) still gets a get-up, held where
    // his head would reach the opposite wall (kneeling), see updateGetup
    if (this.fallT > 1.0 && this.restT > 0.6 && tilt < this.fallTilt * 0.75 && this.G.room >= 1.15) this.startGetup();
  }

  // ---------- getting up ----------
  startGetup() {
    const W = this.W, d = W.d;
    this.restoreGains();
    const hipsZ = this.cur[2];
    // his trunk upright (chest's up near the floor's up) and hips off the floor: he is crouched or sitting against a
    // wall, not lying: he stands straight up from there (round 7: he was laid flat first, to play a lying get-up)
    const chq = this.W.d.xquat, cb = this.W.bodyId.Chest, cu = qv(this.G.q, qv([chq[4 * cb], chq[4 * cb + 1], chq[4 * cb + 2], chq[4 * cb + 3]], [0, 0, 1]));
    const hd = this.W.bodyId.Head, headZ = this.toLocal([this.W.d.xpos[3 * hd], this.W.d.xpos[3 * hd + 1], this.W.d.xpos[3 * hd + 2]])[2];
    // on his feet = hips up AND trunk upright (falls worker: leaning diagonally on a wall is not); or trunk fairly upright
    // with the head still high (get-up worker: sitting back against a wall)
    if ((hipsZ > 0.6 && cu[2] > 0.75) || (cu[2] > 0.6 && headZ > 1.2)) {   // he never went down (caught against a wall): straight back to standing
      this.state = 'stand'; this.c = [0, 0]; this.cv = [0, 0];
      this.toIdle(hipsZ > 0.75 ? 0.4 : 0.9, 'caught himself'); return;
    }
    // (falls worker) sitting on the floor (trunk upright, hips low - where a protective fall into a wall or corner now
    // ends): no lying down first. The get-up recordings all pass through sitting up: start at the frame of any of them
    // that looks most like him now (feet, hands, head relative to the hips, hips height), anchored by his hips.
    if (cu[2] > 0.6) {
      const f = this.curFeatures(); let best = null;
      for (const kind of Object.keys(this.lib.getups)) for (const g of this.lib.getups[kind]) {
        const FF = g.clip.features();
        for (let i = 0; i < Math.floor(FF.length * 0.75); i += 2) { const dd = featDist(f, FF[i]); if (!best || dd < best.d) best = { g, i, d: dd, kind }; }
      }
      if (best) {
        this.state = 'getup'; this.havePre = false;
        this.play(best.g.clip, best.i, 'once', { hl: 0.45, still: true, fromCur: true });
        return this.event(`get up from sitting (${best.g.file} at ${(best.i / FPS).toFixed(1)} s, pose gap ${(best.d * 100).toFixed(0)} cm)`);
      }
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
    // (worker 'getup') the get-up recording AND the moment in it that look most like him now (any clip of any kind:
    // hips height, his up direction seen from his hips, feet/hands/head relative to his hips in the hips' own frame),
    // from the first 70% of each clip; then entered there, turned so the hips' sideways axis (or the spine, on his
    // side) lines up. Starting every get-up at frame 0 laid him flat first even when he sat against a wall.
    const me = this.bodyFeat(i => { const id = W.bodyId[BODY23[i]]; return this.toLocal([d.xpos[3 * id], d.xpos[3 * id + 1], d.xpos[3 * id + 2]]); }, [this.cur[3], this.cur[4], this.cur[5], this.cur[6]], this.cur[2]);
    let best = null;
    for (const k of Object.keys(G)) for (const e of G[k]) {
      const F = this.getupFeats(e.clip);
      for (let i = 0; i < Math.floor(F.length * 0.7); i += 2) { let dd = 0; for (let j = 0; j < me.length; j++) dd += (me[j] - F[i][j]) ** 2; if (!best || dd < best.d) best = { e, i, d: dd, kind: k }; }
    }
    const c = best.e; kind = best.kind;
    this.state = 'getup'; this.getupHold = 0;
    this.havePre = false;
    const q0 = c.clip.q(best.i), lat0 = qv([q0[3], q0[4], q0[5], q0[6]], [1, 0, 0]), latN = qv([this.cur[3], this.cur[4], this.cur[5], this.cur[6]], [1, 0, 0]);
    const useLat = Math.abs(lat0[2]) < 0.7 && Math.abs(latN[2]) < 0.7;
    const yaw = useLat ? wrap(Math.atan2(latN[1], latN[0]) - Math.atan2(lat0[1], lat0[0])) : undefined;
    this.play(c.clip, best.i, 'once', { spine: !useLat, yaw, hl: 0.35, still: true, fromCur: true });
    this.event(`get up from ${kind} (${c.file} at ${(best.i / FPS).toFixed(1)} s, pose gap ${Math.sqrt(best.d / me.length * 3).toFixed(2)})`);
  }
  // pose fingerprint for get-ups (independent of which way he faces): parts relative to the hips in the hips' own frame,
  // the ground's up seen from the hips (x2), hips height (x2)
  bodyFeat(pos, rootQ, hipsZ) {
    const inv = qc(rootQ), h = pos(0), o = [];
    for (const [b, w] of [[21, 1], [17, 1], [14, 0.7], [10, 0.7], [6, 1]]) { const p = pos(b), r = qv(inv, [p[0] - h[0], p[1] - h[1], p[2] - h[2]]); o.push(r[0] * w, r[1] * w, r[2] * w); }
    const up = qv(inv, [0, 0, 1]); o.push(up[0] * 0.6, up[1] * 0.6, up[2] * 0.6, hipsZ * 2);
    return o;
  }
  getupFeats(clip) {
    if (clip._gf) return clip._gf;
    clip._gf = [];
    for (let i = 0; i < clip.n; i++) { const q = clip.q(i); clip._gf.push(this.bodyFeat(b => clip.body(i, b), [q[3], q[4], q[5], q[6]], q[2])); }
    return clip._gf;
  }
  updateGetup(dt) {
    if (this.faceChanged) { this.faceChanged = false; return this.startFall('the box turned over'); }
    // the recording waits for him (worker 'getup'): when his real hips are behind the recording's (getupLag, metres
    // along "up", set by the physics layer), it slows down, to a quarter speed at 33 cm behind; and it isn't over
    // until his hips are within 15 cm of standing (or 4 s past its end: then he is down, and falls again)
    const lag = this.getupLag || 0;
    this.anim.speed = clamp(1 - (lag - 0.08) / 0.33, 0.25, 1);
    // not enough room above him to stand: the recording waits at the moment its head would come within 25 cm of the
    // opposite wall (on his knees, or sitting up) until the room is there (the phone turned back)
    const nx = Math.min(this.anim.clip.n - 1, Math.floor(this.anim.x) + 2), headZ = this.anim.clip.body(nx, 6)[2];
    this.kneeling = this.G.room < 1.9 && headZ > this.G.room - 0.25;
    if (this.kneeling) { this.anim.speed = 0; this.getupHold = 0; if (this.last !== 'kneeling: no room to stand') this.event('kneeling: no room to stand'); return; }
    const ended = this.anim.x >= this.anim.clip.n - 1.01;
    this.getupHold = ended ? (this.getupHold || 0) + dt : 0;
    if (ended && lag > 0.15 && this.getupHold < 4) return;
    if (ended && lag > 0.15) { this.getupHold = 0; return this.startFall('the get-up failed'); }
    if (this.advance(dt)) {
      this.state = 'stand'; this.c = [0, 0]; this.cv = [0, 0]; this.steps = [];
      this.toIdle(0.5, 'standing again');
      this.idleT = 10 + Math.random() * 10;
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
