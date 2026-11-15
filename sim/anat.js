// AnatGuard: his ball joints stay inside the poses real people reach (user 14:32: "the shoulders are clearly in the
// wrong position ... It was impossible for the puppet to have the joints in incorrect places").
// Each ball joint of his body is 3 hinges (x, y, z) and MuJoCo stops each hinge separately at the extremes people reach
// (limits.json, applyHumanLimits). Every hinge alone at its extreme is something a person did, but 2-3 of them at
// their extremes AT ONCE is not (the collarbone pushed down AND back AND up by a wall). limits.json "_env" holds, per
// ball joint, the set of hinge-angle triples found in 17,162 recorded moves of 356 people on his body (an occupancy
// grid, tools/kin/anat/fit.py) plus a margin. Each physics step this guard measures how far each joint is outside that
// set (degrees, in hinge space) and pushes it back in with a stiff, damped torque along the way out - like a ligament
// or a bone stop, not a muscle (no strength cap besides a stability cap). The torque is added to qfrc_applied just
// before mj_step and taken out again just after it, so it never builds up and whatever the controllers write there
// (set or add) is left exactly as they left it.
// Use: const anat = new AnatGuard(W, L); anat.install();   (L = parsed limits.json; install wraps W.step)
//      anat.violations(qpos) -> [{ joint, depth (deg) }] for joints outside their envelope (tests)
const D2R = Math.PI / 180, R2D = 180 / Math.PI;

// exact Euclidean distance transform (Felzenszwalb & Huttenlocher), squared distances in cell units, in place
function edt1(f, n, z, v, out) {
  let k = 0; v[0] = 0; z[0] = -Infinity; z[1] = Infinity;
  for (let q = 1; q < n; q++) {
    let p = v[k], s = ((f[q] + q * q) - (f[p] + p * p)) / (2 * q - 2 * p);
    while (s <= z[k]) { k--; p = v[k]; s = ((f[q] + q * q) - (f[p] + p * p)) / (2 * q - 2 * p); }
    k++; v[k] = q; z[k] = s; z[k + 1] = Infinity;
  }
  k = 0;
  for (let q = 0; q < n; q++) { while (z[k + 1] < q) k++; const p = v[k]; out[q] = (q - p) * (q - p) + f[p]; }
}
function edt3(occ, n0, n1, n2) {
  const N = n0 * n1 * n2, BIG = 1e10, g = new Float64Array(N), m = Math.max(n0, n1, n2);
  const f = new Float64Array(m), o = new Float64Array(m), z = new Float64Array(m + 1), v = new Int32Array(m);
  for (let i = 0; i < N; i++) g[i] = occ[i] ? 0 : BIG;
  const pass = (n, stride, starts) => { for (const s0 of starts) { for (let i = 0; i < n; i++) f[i] = g[s0 + i * stride]; edt1(f, n, z, v, o); for (let i = 0; i < n; i++) g[s0 + i * stride] = o[i]; } };
  const st = []; for (let a = 0; a < n0; a++) for (let b = 0; b < n1; b++) st.push(a * n1 * n2 + b * n2); pass(n2, 1, st);
  st.length = 0; for (let a = 0; a < n0; a++) for (let c = 0; c < n2; c++) st.push(a * n1 * n2 + c); pass(n1, n2, st);
  st.length = 0; for (let b = 0; b < n1; b++) for (let c = 0; c < n2; c++) st.push(b * n2 + c); pass(n0, n1 * n2, st);
  const D = new Float32Array(N); for (let i = 0; i < N; i++) D[i] = Math.sqrt(Math.min(g[i], 1e8)); return D;
}

export const ANAT = {
  w: 60,          // rad/s: the stop's natural frequency (9.5 Hz) for the hinge's own inertia
  zeta: 1,        // damping ratio (only resists moving further out; never pulls him out)
  soft: 4,        // deg: the stop gets stiffer with depth: k * (1 + depth / soft)
  amax: 4000,     // rad/s^2: cap on the push (stability at 480 Hz with a deep start pose)
  minI: 0.01,     // kg m^2: floor on a hinge's inertia used to scale its push
};

export class AnatGuard {
  constructor(W, L, opt = {}) {
    this.W = W; const m = W.m, mj = W.mj, E = (L && L._env) || {};
    this.opt = Object.assign({}, ANAT, opt); this.on = true;
    this.J = [];
    for (const b in E) {
      const e = E[b], jid = ['_x', '_y', '_z'].map(a => mj.mj_name2id(m, mj.mjtObj.mjOBJ_JOINT.value, b + a));
      if (jid.some(j => j < 0)) continue;
      const [n0, n1, n2] = e.n, occ = new Uint8Array(n0 * n1 * n2); let p = 0, val = 0;
      for (const r of e.rle.split(',')) { const k = +r; if (val) occ.fill(1, p, p + k); p += k; val ^= 1; }
      const qa = jid.map(j => m.jnt_qposadr[j]), da = jid.map(j => m.jnt_dofadr[j]);
      // the hinge's own inertia with the rest of his body free to move (1 / dof_invweight0; NOT dof_M0, the diagonal of
      // the mass matrix: 30-160x larger for the trunk and hips - with it the push and its damping blew the sim up)
      const I = da.map(k => Math.max(this.opt.minI, 1 / Math.max(1e-6, m.dof_invweight0[k])));
      this.J.push({ name: b, h: e.h, mg: e.mg, lo: e.lo, n: e.n, s1: n2, s0: n1 * n2, D: edt3(occ, n0, n1, n2), qa, da, I });
    }
    this.tau = new Float64Array(3 * this.J.length); this.g = new Float64Array(3);
    this.stats = { steps: 0, out: 0, worst: 0, worstJ: '', perJ: this.J.map(() => 0) };
  }
  // depth (deg) of one joint outside its envelope at hinge angles a0..a2 (rad); the way out (unit, hinge space) in this.g
  depth(J, a0, a1, a2) {
    const h = J.h, n = J.n, g = this.g; let ext2 = 0, ex0 = 0, ex1 = 0, ex2 = 0;
    let u0 = (a0 * R2D - J.lo[0]) / h, u1 = (a1 * R2D - J.lo[1]) / h, u2 = (a2 * R2D - J.lo[2]) / h;
    const c0 = Math.min(Math.max(u0, 0), n[0] - 1.000001), c1 = Math.min(Math.max(u1, 0), n[1] - 1.000001), c2 = Math.min(Math.max(u2, 0), n[2] - 1.000001);
    ex0 = (u0 - c0) * h; ex1 = (u1 - c1) * h; ex2 = (u2 - c2) * h; ext2 = ex0 * ex0 + ex1 * ex1 + ex2 * ex2;
    const i0 = Math.floor(c0), i1 = Math.floor(c1), i2 = Math.floor(c2), f0 = c0 - i0, f1 = c1 - i1, f2 = c2 - i2;
    const D = J.D, b = i0 * J.s0 + i1 * J.s1 + i2, s0 = J.s0, s1 = J.s1;
    const d000 = D[b], d001 = D[b + 1], d010 = D[b + s1], d011 = D[b + s1 + 1], d100 = D[b + s0], d101 = D[b + s0 + 1], d110 = D[b + s0 + s1], d111 = D[b + s0 + s1 + 1];
    const x00 = d000 + (d001 - d000) * f2, x01 = d010 + (d011 - d010) * f2, x10 = d100 + (d101 - d100) * f2, x11 = d110 + (d111 - d110) * f2;
    const y0 = x00 + (x01 - x00) * f1, y1 = x10 + (x11 - x10) * f1, v = (y0 + (y1 - y0) * f0) * h;
    // gradient of the trilinear field (cells -> deg: d/du / h * h = d/du)
    g[0] = y1 - y0;
    g[1] = (1 - f0) * (x01 - x00) + f0 * (x11 - x10);
    g[2] = (1 - f0) * ((1 - f1) * (d001 - d000) + f1 * (d011 - d010)) + f0 * ((1 - f1) * (d101 - d100) + f1 * (d111 - d110));
    let dep = v - J.mg;
    if (ext2 > 0) { const e = Math.sqrt(ext2); dep += e; g[0] += ex0 / e; g[1] += ex1 / e; g[2] += ex2 / e; }
    const gn = Math.hypot(g[0], g[1], g[2]); if (gn > 1e-9) { g[0] /= gn; g[1] /= gn; g[2] /= gn; } else { g[0] = g[1] = g[2] = 0; }
    return dep;
  }
  // the torques for the state in d (adds them into d.qfrc_applied, remembers them in this.tau)
  push() {
    const d = this.W.d, q = d.qpos, qv = d.qvel, F = d.qfrc_applied, o = this.opt, tau = this.tau, g = this.g, S = this.stats;
    const w2 = o.w * o.w, c = 2 * o.zeta * o.w; let any = false; S.steps++;
    for (let j = 0; j < this.J.length; j++) {
      const J = this.J[j], qa = J.qa, k3 = 3 * j;
      const dep = this.depth(J, q[qa[0]], q[qa[1]], q[qa[2]]);
      if (dep <= 0 || (g[0] === 0 && g[1] === 0 && g[2] === 0)) { tau[k3] = tau[k3 + 1] = tau[k3 + 2] = 0; continue; }
      if (dep > S.worst) { S.worst = dep; S.worstJ = J.name; } S.perJ[j]++; any = true;
      const da = J.da, vn = g[0] * qv[da[0]] + g[1] * qv[da[1]] + g[2] * qv[da[2]];   // rad/s along the way out
      let a = w2 * dep * D2R * (1 + dep / o.soft) + c * Math.max(0, vn);
      if (a > o.amax) a = o.amax;
      for (let i = 0; i < 3; i++) { const t = -J.I[i] * a * g[i]; tau[k3 + i] = t; F[da[i]] += t; }
    }
    if (any) S.out++;
  }
  pop() {
    const F = this.W.d.qfrc_applied, tau = this.tau;
    for (let j = 0; j < this.J.length; j++) { const da = this.J[j].da; for (let i = 0; i < 3; i++) { F[da[i]] -= tau[3 * j + i]; tau[3 * j + i] = 0; } }
  }
  // wrap W.step: the guard's torque is in qfrc_applied only during MuJoCo's step
  install() {
    if (this.W.__anat) return this; const W = this.W, step = W.step.bind(W), self = this; W.__anat = this;
    W.step = function (control) { if (self.on) self.push(); step(control); if (self.on) self.pop(); };
    return this;
  }
  // joints outside their envelope at qpos (tests): [{ joint, depth deg }]
  violations(qpos, minDeg = 0) {
    const out = [];
    for (const J of this.J) { const dep = this.depth(J, qpos[J.qa[0]], qpos[J.qa[1]], qpos[J.qa[2]]); if (dep > minDeg) out.push({ joint: J.name, depth: dep }); }
    return out;
  }
}
