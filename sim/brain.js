// The Box - the trained brain: NVIDIA's pretrained SOMA motion tracker in the controller slot.
//
// Each control tick (30 Hz) it reads the body from MuJoCo, compares it with where a reference motion wants each of
// the 23 body parts one tick later, and runs the trained network (ONNX Runtime Web, ~1-5 ms), which answers with a
// target for each of the 66 joint hinges. The servos (PD, the gains it was trained with) pull toward those targets
// for the next 16 physics steps. Nothing else pushes the body. Rebuilt from NVIDIA ProtoMotions (Apache-2.0):
//   inputs (961) = self (343) + target poses one tick ahead (552) + previous network output (66)
//   network (in body/soma_tracker.onnx) = input normalisation -> 6 x 1024 ReLU -> 66
//   output -> tanh -> x pi = a rotation vector per 3-hinge joint -> the three hinge angles (x, y, z; left mirrored)
// Weights: ProtoMotions motion_tracker/soma-bones (IsaacLab fine-tune), NVIDIA Open Model License (body/NOTICE.md).
// Checked against the Python reference (tools/brain/brain_ref.py): same inputs and outputs to 1e-7.
//
// The box tilts: the network was trained with gravity straight down onto a flat floor. So it sees the world turned
// until gravity is straight down again (the "gravity frame"); a tilted box becomes a sloped floor.
//
// Use (see index.html):
//   const brain = await Brain.create(W, ort, 'body/');      // after the BoxWorld exists
//   brain.attach();                                           // servos get the training gains; brain.detach() undoes
//   each physics step:  if (brain.due()) await brain.think();  W.step(brain.control);
// The reference is a standing pose that follows him (brain.ref), so pushes and slopes are his to deal with.

export const NB = 23, ND = 66;
// the standing pose the brain follows (hinge angles from the body file's T-pose): arms down at the sides (Arm_y), a
// little in front (Arm_z). (Arm_z alone swings the arms forward like a sleepwalker.)
export const BRAIN_STAND = { RightArm_y: -1.25, LeftArm_y: 1.25, RightArm_z: -0.2, LeftArm_z: 0.2 };
export const BODY_NAMES = ['Hips', 'Spine1', 'Spine2', 'Chest', 'Neck1', 'Neck2', 'Head',
  'RightShoulder', 'RightArm', 'RightForeArm', 'RightHand', 'LeftShoulder', 'LeftArm', 'LeftForeArm', 'LeftHand',
  'RightLeg', 'RightShin', 'RightFoot', 'RightToeBase', 'LeftLeg', 'LeftShin', 'LeftFoot', 'LeftToeBase'];

// ---------- quaternions (x y z w) ----------
export function qmul(a, ai, b, bi, o, oi) {
  const ax = a[ai], ay = a[ai + 1], az = a[ai + 2], aw = a[ai + 3], bx = b[bi], by = b[bi + 1], bz = b[bi + 2], bw = b[bi + 3];
  o[oi] = aw * bx + ax * bw + ay * bz - az * by; o[oi + 1] = aw * by - ax * bz + ay * bw + az * bx;
  o[oi + 2] = aw * bz + ax * by - ay * bx + az * bw; o[oi + 3] = aw * bw - ax * bx - ay * by - az * bz;
}
export function qrot(q, qi, vx, vy, vz, o, oi) {
  const x = q[qi], y = q[qi + 1], z = q[qi + 2], w = q[qi + 3];
  const tx = 2 * (y * vz - z * vy), ty = 2 * (z * vx - x * vz), tz = 2 * (x * vy - y * vx);
  o[oi] = vx + w * tx + (y * tz - z * ty); o[oi + 1] = vy + w * ty + (z * tx - x * tz); o[oi + 2] = vz + w * tz + (x * ty - y * tx);
}
function heading(q, qi) { const t = [0, 0, 0]; qrot(q, qi, 1, 0, 0, t, 0); return Math.atan2(t[1], t[0]); }
function tanNorm(q, qi, o, oi) { qrot(q, qi, 1, 0, 0, o, oi); qrot(q, qi, 0, 0, 1, o, oi + 3); }
// the shortest turn taking unit vector a onto unit vector b
function qBetween(a, b, o) {
  const d = a[0] * b[0] + a[1] * b[1] + a[2] * b[2];
  if (d < -0.999999) { o[0] = 1; o[1] = 0; o[2] = 0; o[3] = 0; return o; }
  o[0] = a[1] * b[2] - a[2] * b[1]; o[1] = a[2] * b[0] - a[0] * b[2]; o[2] = a[0] * b[1] - a[1] * b[0]; o[3] = 1 + d;
  const n = Math.hypot(o[0], o[1], o[2], o[3]); o[0] /= n; o[1] /= n; o[2] /= n; o[3] /= n; return o;
}
// rotation vector -> the three hinge angles about x, then y, then z (R = Rx Ry Rz)
export function rotvecToXYZ(rx, ry, rz, out, oi) {
  const th = Math.hypot(rx, ry, rz);
  let x = 1, y = 0, z = 0;
  if (th > 1e-9) { x = rx / th; y = ry / th; z = rz / th; }
  const c = Math.cos(th), s = Math.sin(th), C = 1 - c;
  const R00 = c + x * x * C, R01 = x * y * C - z * s, R02 = x * z * C + y * s, R12 = y * z * C - x * s, R22 = c + z * z * C;
  out[oi] = Math.atan2(-R12, R22); out[oi + 1] = Math.asin(Math.max(-1, Math.min(1, R02))); out[oi + 2] = Math.atan2(-R01, R00);
}

const _tq = new Float64Array(4), _t6 = new Float64Array(6);

export class Brain {
  static async create(W, ort, dir = 'body/', opt = {}) {
    const [meta, bytes] = await Promise.all([
      fetchJSON(dir + 'soma_tracker.json'), fetchBytes(dir + 'soma_tracker.onnx')]);
    const session = await ort.InferenceSession.create(bytes, { executionProviders: ['wasm'], graphOptimizationLevel: 'all' });
    return new Brain(W, ort, session, meta, opt);
  }

  constructor(W, ort, session, meta, opt = {}) {
    this.W = W; this.ort = ort; this.session = session; this.meta = meta;
    const m = W.m, mj = W.mj;
    this.bodyId = BODY_NAMES.map(n => W.bodyId[n]);
    const root = this.bodyId[0], j0 = m.body_jntadr[root] + 1;
    this.qadr = m.jnt_qposadr[j0 - 1]; this.vadr = m.jnt_dofadr[j0 - 1];
    this.hingeQ = new Int32Array(ND); this.hingeV = new Int32Array(ND); this.sign = new Float64Array(ND); this.actIdx = new Int32Array(ND);
    for (let i = 0; i < ND; i++) {
      const j = j0 + i, ax = [m.jnt_axis[3 * j], m.jnt_axis[3 * j + 1], m.jnt_axis[3 * j + 2]];
      const k = [0, 1, 2].reduce((p, c) => Math.abs(ax[c]) > Math.abs(ax[p]) ? c : p, 0);
      this.hingeQ[i] = m.jnt_qposadr[j]; this.hingeV[i] = m.jnt_dofadr[j]; this.sign[i] = Math.sign(ax[k]);
    }
    for (let a = 0; a < m.nu; a++) { const i = m.actuator_trnid[2 * a] - j0; if (i >= 0 && i < ND) this.actIdx[i] = a; }
    // state in the gravity frame
    this.pos = new Float64Array(NB * 3); this.rot = new Float64Array(NB * 4); this.vel = new Float64Array(NB * 3); this.ang = new Float64Array(NB * 3);
    this.G = new Float64Array([0, 0, 0, 1]);      // world (box) -> gravity frame
    this.obs = new Float32Array(meta.n_obs); this.prev = new Float32Array(ND); this.out = new Float32Array(ND);
    this.targets = new Float64Array(m.nu);
    this.vbuf = new mj.DoubleBuffer(6);
    this.ref = { pos: new Float64Array(NB * 3), rot: new Float64Array(NB * 4), vel: new Float64Array(NB * 3), ang: new Float64Array(NB * 3) };
    this.stand = null;            // the standing pose's parts relative to the hips (gravity frame, facing +x), from FK
    this.refXY = null; this.refYaw = 0;
    // how the spot under him follows his feet: slowly (0.15/s, no dead zone). Fast (1.2/s, v15) and he chased it onto his
    // toes; never (0) and he stands 30 cm from it with one heel up. Measured over 3-14 s: heels 0.4 / 0.0 cm up.
    this.follow = opt.follow ?? 0.15;
    this.deadband = opt.deadband ?? 0;
    this.settleT = opt.settleT ?? 2.0;
    this.snap = opt.snap ?? false;
    // Known: in MuJoCo this tracker (trained in PhysX) shuffles ~35 cm backwards in its first 1.5 s with the arms down,
    // then stands steadily. Not a balance offset (leaning the reference forward changes nothing). The page starts him
    // that much nearer the glass (settleBack) so he settles in the middle.
    this.settleBack = 0.35;
    this.turn = opt.turn ?? 1;
    this.mass = opt.mass ?? 70;          // kg, while the brain drives him (attach/detach)
    this.turnRate = opt.turnRate ?? 2;          // how much the reference's heading follows his (and turns toward the glass)
    this.lost = 0;                    // seconds he has been far from the reference (fallen)
    this.hz = meta.control_hz;
    this.tNext = 0;
    this.msNet = 0;
    // W.step calls this at its control rate; it hands back the targets the last think() made
    const self = this;
    this.control = function control() { return self.targets; };
    this.control.hz = 1e9;   // W.step asks every step; we only change targets in think()
  }

  // servos get the gains the network was trained with (and give the old ones back on detach)
  attach() {
    const m = this.W.m, W = this.W;
    // real adult weight: the body file weighs 42.5 kg (density-1000 shapes); 70 kg reads as a person, not a doll.
    // The tracker copes (tested: stands, tilts, shoves; heels flatter: 0.2 cm vs 0.6 cm up on average).
    if (!this.massSaved) {
      this.massSaved = W.bodyNames.map(n => { const b = W.bodyId[n]; return [m.body_mass[b], m.body_inertia[3 * b], m.body_inertia[3 * b + 1], m.body_inertia[3 * b + 2]]; });
      const f = this.mass / W.mass;
      W.bodyNames.forEach(n => { const b = W.bodyId[n]; m.body_mass[b] *= f; for (let k = 0; k < 3; k++) m.body_inertia[3 * b + k] *= f; });
      this.massWas = W.mass; W.mass = this.mass;
    }
    this.saved = [];
    for (let i = 0; i < ND; i++) {
      const a = this.actIdx[i], kp = this.meta.kp[i], kd = this.meta.kd[i];
      this.saved.push([m.actuator_gainprm[10 * a], m.actuator_biasprm[10 * a + 1], m.actuator_biasprm[10 * a + 2]]);
      m.actuator_gainprm[10 * a] = kp; m.actuator_biasprm[10 * a + 1] = -kp; m.actuator_biasprm[10 * a + 2] = -kd;
    }
    for (let a = 0; a < m.nu; a++) this.targets[a] = this.W.d.qpos[m.jnt_qposadr[m.actuator_trnid[2 * a]]];
    this.prev.fill(0); this.refXY = null; this.settled = false; this.tNext = 0; this.lost = 0; this.down = false; this.tone = 1;
  }
  // muscle tone: 1 = the trained gains, less = looser (when he has fallen and the tracker has given up)
  setTone(f) {
    const m = this.W.m; this.tone = f;
    for (let i = 0; i < ND; i++) {
      const a = this.actIdx[i], kp = this.meta.kp[i] * f, kd = this.meta.kd[i] * f;
      m.actuator_gainprm[10 * a] = kp; m.actuator_biasprm[10 * a + 1] = -kp; m.actuator_biasprm[10 * a + 2] = -kd;
    }
  }
  detach() {
    const m = this.W.m; if (!this.saved) return;
    for (let i = 0; i < ND; i++) { const a = this.actIdx[i], s = this.saved[i]; m.actuator_gainprm[10 * a] = s[0]; m.actuator_biasprm[10 * a + 1] = s[1]; m.actuator_biasprm[10 * a + 2] = s[2]; }
    this.saved = null;
    if (this.massSaved) {
      const W = this.W;
      W.bodyNames.forEach((n, i) => { const b = W.bodyId[n], s = this.massSaved[i]; m.body_mass[b] = s[0]; for (let k = 0; k < 3; k++) m.body_inertia[3 * b + k] = s[k + 1]; });
      W.mass = this.massWas; this.massSaved = null;
    }
  }

  // the standing pose (hinge angles by joint name, e.g. physics STAND) measured once by forward kinematics
  setStandPose(pose) {
    const W = this.W, mj = W.mj, m = W.m, d2 = new mj.MjData(m);
    d2.qpos[this.qadr + 3] = 1;
    for (const a of W.act) d2.qpos[a.qadr] = pose[a.name] || 0;
    mj.mj_kinematics(m, d2);
    const P = new Float64Array(NB * 3), R = new Float64Array(NB * 4);
    let lo = 1e9;
    for (let b = 0; b < NB; b++) {
      const id = this.bodyId[b];
      for (let k = 0; k < 3; k++) P[3 * b + k] = d2.xpos[3 * id + k];
      R[4 * b] = d2.xquat[4 * id + 1]; R[4 * b + 1] = d2.xquat[4 * id + 2]; R[4 * b + 2] = d2.xquat[4 * id + 3]; R[4 * b + 3] = d2.xquat[4 * id];
    }
    // the soles: lowest point of the feet and toes, roughly (their geoms' bottoms)
    for (let g = 0; g < m.ngeom; g++) {
      const b = BODY_NAMES.indexOf(mj.mj_id2name(m, mj.mjtObj.mjOBJ_BODY.value, m.geom_bodyid[g]));
      if (b < 17) continue;
      lo = Math.min(lo, d2.geom_xpos[3 * g + 2] - lowestOffset(m, d2, g));
    }
    // relative to the hips, turned to face +x (heading 0)
    const h = heading(R, 0), q = [0, 0, Math.sin(-h / 2), Math.cos(-h / 2)], t = [0, 0, 0];
    this.stand = { pos: new Float64Array(NB * 3), rot: new Float64Array(NB * 4), height: P[2] - lo };
    for (let b = 0; b < NB; b++) {
      qrot(q, 0, P[3 * b] - P[0], P[3 * b + 1] - P[1], P[3 * b + 2] - P[2], t, 0);
      this.stand.pos.set(t, 3 * b);
      qmul(q, 0, R, 4 * b, this.stand.rot, 4 * b);
    }
    d2.delete();
  }

  // read the 23 parts and turn them into the gravity frame (about the floor point under the hips)
  readState() {
    const W = this.W, d = W.d, mj = W.mj, xp = d.xpos, xq = d.xquat, O = mj.mjtObj.mjOBJ_XBODY.value;
    const g = W.m.opt.gravity, gn = Math.hypot(g[0], g[1], g[2]) || 1;
    qBetween([-g[0] / gn, -g[1] / gn, -g[2] / gn], [0, 0, 1], this.G);
    const G = this.G, t = [0, 0, 0];
    for (let b = 0; b < NB; b++) {
      const id = this.bodyId[b];
      qrot(G, 0, xp[3 * id], xp[3 * id + 1], xp[3 * id + 2], this.pos, 3 * b);
      _tq[0] = xq[4 * id + 1]; _tq[1] = xq[4 * id + 2]; _tq[2] = xq[4 * id + 3]; _tq[3] = xq[4 * id];
      qmul(G, 0, _tq, 0, this.rot, 4 * b);
      mj.mj_objectVelocity(W.m, d, O, id, this.vbuf, 0);
      const v = this.vbuf.GetView();
      qrot(G, 0, v[0], v[1], v[2], this.ang, 3 * b);
      qrot(G, 0, v[3], v[4], v[5], this.vel, 3 * b);
    }
    // the floor's height under the hips, measured along gravity (the box's floor plane, turned the same way)
    const fz = -W.box.h / 2, n = [0, 0, 1], p0 = [xp[3 * this.bodyId[0]], xp[3 * this.bodyId[0] + 1], fz];
    qrot(G, 0, n[0], n[1], n[2], t, 0); const fn = t.slice();
    qrot(G, 0, p0[0], p0[1], p0[2], t, 0);
    const hx = this.pos[0], hy = this.pos[1];
    // plane through t with normal fn: z = t.z - (fn.x (x - t.x) + fn.y (y - t.y)) / fn.z
    this.ground = t[2] - (fn[0] * (hx - t[0]) + fn[1] * (hy - t[1])) / (fn[2] || 1);
  }

  // where the standing pose should be: under him, upright with gravity, turned toward the glass
  updateRef(dt) {
    const P = this.pos, s = this.stand;
    // the reference stands on his feet: the point between his feet (not his hips, which a lean carries away)
    const fx = (P[3 * 17] + P[3 * 21]) / 2, fy = (P[3 * 17 + 1] + P[3 * 21 + 1]) / 2;
    if (!this.refXY) { this.refXY = [fx, fy]; this.refYaw = heading(this.rot, 0); this.tRef = 0; }
    this.tRef += dt;
    // the spot stays put while he settles (he shuffles back once, ~1.5 s), then moves once to where he ended up;
    // after that it only follows when a push or slope has carried him more than `deadband` from it. A spot that
    // keeps following his feet gets chased: he rises onto his toes (user, v15: "on the tip of his feet").
    const ex = fx - this.refXY[0], ey = fy - this.refXY[1], e = Math.hypot(ex, ey);
    if (this.tRef > this.settleT && !this.settled) { if (this.snap) this.refXY = [fx, fy]; this.settled = true; }
    else if (this.settled && e > this.deadband) {
      const k = (1 - Math.exp(-this.follow * dt)) * (e - this.deadband) / e;
      this.refXY[0] += ex * k; this.refXY[1] += ey * k;
    }
    const k = 1 - Math.exp(-this.turnRate * dt);   // heading: slow
    // face the glass: the glass is at box y = 0, so "toward the viewer" is box -y, in the gravity frame
    const t = [0, 0, 0]; qrot(this.G, 0, 0, -1, 0, t, 0);
    const want = Math.atan2(t[1], t[0]) + Math.PI / 2;   // the body's forward is its -y (semantic_forward_axis 0,-1)
    let dy = want - this.refYaw; dy = Math.atan2(Math.sin(dy), Math.cos(dy));
    let dh = heading(this.rot, 0) - this.refYaw; dh = Math.atan2(Math.sin(dh), Math.cos(dh));
    this.refYaw += (0.6 * dh + 0.15 * dy) * k * 2 * this.turn;
    const q = [0, 0, Math.sin(this.refYaw / 2), Math.cos(this.refYaw / 2)], R = this.ref;
    const hz = this.ground + s.height;
    // the stand pose's own feet middle, turned like the reference (so its feet land on refXY)
    const fo = [0, 0, 0]; qrot(q, 0, (s.pos[3 * 17] + s.pos[3 * 21]) / 2, (s.pos[3 * 17 + 1] + s.pos[3 * 21 + 1]) / 2, 0, fo, 0);
    for (let b = 0; b < NB; b++) {
      qrot(q, 0, s.pos[3 * b], s.pos[3 * b + 1], s.pos[3 * b + 2], t, 0);
      R.pos[3 * b] = this.refXY[0] + t[0] - fo[0]; R.pos[3 * b + 1] = this.refXY[1] + t[1] - fo[1]; R.pos[3 * b + 2] = hz + t[2];
      qmul(q, 0, s.rot, 4 * b, R.rot, 4 * b);
      R.vel[3 * b] = R.vel[3 * b + 1] = R.vel[3 * b + 2] = 0; R.ang[3 * b] = R.ang[3 * b + 1] = R.ang[3 * b + 2] = 0;
    }
  }

  // the 961 inputs, exactly as in training
  buildObs(ref) {
    const o = this.obs, P = this.pos, R = this.rot, V = this.vel, A = this.ang, t = _t6;
    const h = heading(R, 0), hi = [0, 0, Math.sin(-h / 2), Math.cos(-h / 2)];
    let k = 0;
    o[k++] = P[2] - this.ground;
    for (let b = 1; b < NB; b++) { qrot(hi, 0, P[3 * b] - P[0], P[3 * b + 1] - P[1], P[3 * b + 2] - P[2], t, 0); o[k++] = t[0]; o[k++] = t[1]; o[k++] = t[2]; }
    for (let b = 0; b < NB; b++) { qmul(hi, 0, R, 4 * b, _tq, 0); tanNorm(_tq, 0, t, 0); for (let i = 0; i < 6; i++) o[k++] = t[i]; }
    for (let b = 0; b < NB; b++) { qrot(hi, 0, V[3 * b], V[3 * b + 1], V[3 * b + 2], t, 0); o[k++] = t[0]; o[k++] = t[1]; o[k++] = t[2]; }
    for (let b = 0; b < NB; b++) { qrot(hi, 0, A[3 * b], A[3 * b + 1], A[3 * b + 2], t, 0); o[k++] = t[0]; o[k++] = t[1]; o[k++] = t[2]; }
    const rp = ref.pos, rr = ref.rot, rv = ref.vel, ra = ref.ang;
    for (let b = 0; b < NB; b++) { qrot(hi, 0, rp[3 * b] - P[0], rp[3 * b + 1] - P[1], rp[3 * b + 2] - P[2], t, 0); o[k++] = t[0]; o[k++] = t[1]; o[k++] = t[2]; }
    for (let b = 0; b < NB; b++) { qrot(hi, 0, rp[3 * b] - P[3 * b], rp[3 * b + 1] - P[3 * b + 1], rp[3 * b + 2] - P[3 * b + 2], t, 0); o[k++] = t[0]; o[k++] = t[1]; o[k++] = t[2]; }
    for (let b = 0; b < NB; b++) { qmul(hi, 0, rr, 4 * b, _tq, 0); tanNorm(_tq, 0, t, 0); for (let i = 0; i < 6; i++) o[k++] = t[i]; }
    const cj = [0, 0, 0, 1];
    for (let b = 0; b < NB; b++) {
      cj[0] = -R[4 * b]; cj[1] = -R[4 * b + 1]; cj[2] = -R[4 * b + 2]; cj[3] = R[4 * b + 3];
      qmul(cj, 0, rr, 4 * b, _tq, 0); tanNorm(_tq, 0, t, 0); for (let i = 0; i < 6; i++) o[k++] = t[i];
    }
    for (let b = 0; b < NB; b++) { qrot(hi, 0, rv[3 * b] - V[3 * b], rv[3 * b + 1] - V[3 * b + 1], rv[3 * b + 2] - V[3 * b + 2], t, 0); o[k++] = t[0]; o[k++] = t[1]; o[k++] = t[2]; }
    for (let b = 0; b < NB; b++) { qrot(hi, 0, ra[3 * b] - A[3 * b], ra[3 * b + 1] - A[3 * b + 1], ra[3 * b + 2] - A[3 * b + 2], t, 0); o[k++] = t[0]; o[k++] = t[1]; o[k++] = t[2]; }
    for (let i = 0; i < ND; i++) o[k++] = this.prev[i];
    return o;
  }

  due() { return this.W.d.time >= this.tNext; }

  // one control tick: read him, place the reference, run the network, set the servo targets
  async think(ref) {
    const dt = 1 / this.hz;
    this.tNext = this.W.d.time + dt - 1e-9;
    this.readState();
    if (this.down) {
      // fallen: this tracker can't get up (it was trained to give up past 0.5 m), so he goes limp (4% tone, no shape
      // held): he slumps, and slides and rolls when the box tilts. Getting up needs our own training (plan step 5/6).
      const d = this.W.d;
      for (let i = 0; i < ND; i++) this.targets[this.actIdx[i]] = d.qpos[this.hingeQ[i]];
      return this.targets;
    }
    if (!ref) { this.updateRef(dt); ref = this.ref; }
    const x = this.buildObs(ref);
    const t0 = performance.now();
    const r = await this.session.run({ obs: new this.ort.Tensor('float32', x, [1, x.length]) });
    this.msNet += (performance.now() - t0 - this.msNet) * 0.1;
    const a = r.action.data;
    this.out.set(a); this.prev.set(a);
    const s = this.meta.scale, T = _hinge;
    for (let j = 0; j < ND; j += 3) rotvecToXYZ(s * Math.tanh(a[j]), s * Math.tanh(a[j + 1]), s * Math.tanh(a[j + 2]), T, j);
    for (let i = 0; i < ND; i++) this.targets[this.actIdx[i]] = T[i] * this.sign[i];
    // how far he is from where the reference wants him (the tracker was trained to give up past 0.5 m)
    let e = 0; for (let b = 0; b < NB; b++) e = Math.max(e, Math.hypot(this.pos[3 * b] - ref.pos[3 * b], this.pos[3 * b + 1] - ref.pos[3 * b + 1], this.pos[3 * b + 2] - ref.pos[3 * b + 2]));
    this.err = e;
    this.lost = e > 0.5 ? this.lost + dt : Math.max(0, this.lost - dt);
    if (this.lost > 0.6 && this.pos[2] - this.ground < 0.6) { this.down = true; this.setTone(0.04); }
    return this.targets;
  }
}
const _hinge = new Float64Array(ND);

function lowestOffset(m, d, g) {
  const t = m.geom_type[g], s = m.geom_size, R = d.geom_xmat;
  if (t === 6) { let dz = 0; for (let k = 0; k < 3; k++) dz += Math.abs(R[9 * g + 6 + k]) * s[3 * g + k]; return dz; }
  if (t === 3) return Math.abs(R[9 * g + 8]) * s[3 * g + 1] + s[3 * g];
  return s[3 * g];
}
async function fetchJSON(u) { if (typeof window === 'undefined') { const fs = await import('fs'); return JSON.parse(fs.readFileSync(u, 'utf8')); } return (await fetch(u)).json(); }
async function fetchBytes(u) { if (typeof window === 'undefined') { const fs = await import('fs'); return new Uint8Array(fs.readFileSync(u)); } return new Uint8Array(await (await fetch(u)).arrayBuffer()); }
