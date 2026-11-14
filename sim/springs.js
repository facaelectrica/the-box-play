// The Box - "Recordings + springs" (kin.html, user Nov 14 17:30: "now add physics. Being pulled towards the recordings
// by springs to see what happens").
//
// Two bodies:
//  - the ghost: the recordings engine (sim/kin.js) runs on its own copy of the world and makes the pose he should have
//    (idles, recoveries, lean, get-ups). It is never drawn.
//  - his real body: MuJoCo physics, gravity from the phone, contacts with the box. Each joint's muscle (the body's own
//    servos, NVIDIA's gains, scaled by "stiffness") pulls toward the ghost's angle, plus the torque that holds the limb
//    against gravity at that moment (joint torques only: internal, like muscles).
//  - "help": a spring and damper from the ghost's hips to his hips (force and turn). This is the invisible support the
//    north star rules out; its strength is a slider (0 = none: he has to stand on his own feet). It is what keeps a
//    recording that is not balanced for this body from tipping him over.
// If his real body can't follow (hips far from the ghost's, or his head low while the ghost stands), the ghost gives
// up too: he falls (help off, muscles soft, arms forward), and the get-up recording takes him up again (help on, eased in).

const qm = (a, b) => [a[0] * b[0] - a[1] * b[1] - a[2] * b[2] - a[3] * b[3], a[0] * b[1] + a[1] * b[0] + a[2] * b[3] - a[3] * b[2],
  a[0] * b[2] - a[1] * b[3] + a[2] * b[0] + a[3] * b[1], a[0] * b[3] + a[1] * b[2] - a[2] * b[1] + a[3] * b[0]];
const qc = a => [a[0], -a[1], -a[2], -a[3]];
function q2rv(q) {
  let w = q[0], x = q[1], y = q[2], z = q[3]; if (w < 0) { w = -w; x = -x; y = -y; z = -z; }
  const s = Math.hypot(x, y, z); if (s < 1e-9) return [2 * x, 2 * y, 2 * z];
  const t = 2 * Math.atan2(s, w); return [x / s * t, y / s * t, z / s * t];
}
function qv(q, v) {
  const w = q[0], x = q[1], y = q[2], z = q[3];
  const tx = 2 * (y * v[2] - z * v[1]), ty = 2 * (z * v[0] - x * v[2]), tz = 2 * (x * v[1] - y * v[0]);
  return [v[0] + w * tx + (y * tz - z * ty), v[1] + w * ty + (z * tx - x * tz), v[2] + w * tz + (x * ty - y * tx)];
}
const clamp = (v, a, b) => Math.max(a, Math.min(b, v));
// convex hull (2D) and how far a point lies outside it (0 inside), with the direction out of it
function hull(P) {
  const p = P.slice().sort((a, b) => a[0] - b[0] || a[1] - b[1]); if (p.length < 3) return p;
  const cr = (o, a, b) => (a[0] - o[0]) * (b[1] - o[1]) - (a[1] - o[1]) * (b[0] - o[0]), lo = [], up = [];
  for (const q of p) { while (lo.length >= 2 && cr(lo[lo.length - 2], lo[lo.length - 1], q) <= 0) lo.pop(); lo.push(q); }
  for (const q of p.reverse()) { while (up.length >= 2 && cr(up[up.length - 2], up[up.length - 1], q) <= 0) up.pop(); up.push(q); }
  return lo.slice(0, -1).concat(up.slice(0, -1));
}
function outside(H, x) {
  let inside = H.length >= 3, best = 1e9, dir = [0, 0];
  for (let i = 0; i < H.length; i++) {
    const a = H[i], b = H[(i + 1) % H.length], e = [b[0] - a[0], b[1] - a[1]], L2 = e[0] * e[0] + e[1] * e[1] || 1e-12;
    if (H.length >= 3 && e[0] * (x[1] - a[1]) - e[1] * (x[0] - a[0]) < 0) inside = false;
    const t = clamp(((x[0] - a[0]) * e[0] + (x[1] - a[1]) * e[1]) / L2, 0, 1), c = [a[0] + e[0] * t, a[1] + e[1] * t];
    const dd = Math.hypot(x[0] - c[0], x[1] - c[1]); if (dd < best) { best = dd; dir = [x[0] - c[0], x[1] - c[1]]; }
  }
  return inside ? { d: 0, dir: [0, 0] } : { d: best, dir };
}

function mjFwdCom(W) { /* subtree_com is filled by the last physics step's forward pass */ }

export class Springs {
  constructor(W, kin, opt = {}) {
    this.W = W; this.kin = kin; this.G = kin.W;          // his body, the engine, the ghost's world
    this.help = opt.help ?? 0.5;                          // 0..1.5: hips spring strength (1 = 1.5 Hz, well damped)
    this.stiff = opt.stiff ?? 1;                          // muscles: fraction of the servos' own gains
    this.vsign = opt.vsign ?? -1;
    this.vturn = opt.vturn ?? 1;
    this.virtual = opt.virtual ?? 1;                      // the hips pull is made by the standing legs' muscles (legPush), not from outside
    this.feetHelp = opt.feetHelp ?? 0;
    this.stance = opt.stance ?? 1;                        // standing legs carry the body above them (see stanceTorques)
    this.load = opt.load ?? 0;                            // muscles also carry what the floor pushes back (his weight on his legs; 1 was unstable: blew up in tests)
    this.gains = W.act.map(a => ({ kp: a.kp, kd: a.kd }));
    this.hips = W.bodyId.Hips; this.head = W.bodyId.Head;
    this.gPrev = null; this.gv = [0, 0, 0]; this.gw = [0, 0, 0];
    this.lostT = 0; this.helpK = 1; this.mode = 'follow'; this.stats = { help: 0, n: 0 };
    kin.ext = W;   // the engine's falls follow this body instead of simulating its own
  }
  setGains(s, sd) {
    const m = this.W.m;
    this.W.act.forEach((a, i) => { const g = this.gains[i]; m.actuator_gainprm[10 * i] = g.kp * s; m.actuator_biasprm[10 * i + 1] = -g.kp * s; m.actuator_biasprm[10 * i + 2] = -g.kd * sd; });
  }
  // his body takes the ghost's pose, still
  reset() {
    const W = this.W, d = W.d, g = this.G.d, mj = W.mj;
    for (let k = 0; k < 73; k++) d.qpos[k] = g.qpos[k];
    d.qvel.fill(0); d.qfrc_applied.fill(0); d.xfrc_applied.fill(0);
    mj.mj_forward(W.m, d); W.tControl = d.time;
    this.gPrev = null; this.lostT = 0; this.helpK = 1; this.mode = 'follow';
    this.setGains(this.stiff, Math.sqrt(this.stiff));
  }
  // the ghost's hips: where, how fast, how it turns (from frame to frame)
  ghostRoot(dt) {
    const q = this.G.d.qpos, p = [q[0], q[1], q[2]], r = [q[3], q[4], q[5], q[6]];
    if (this.gPrev && dt > 0) {
      const a = 1 - Math.exp(-dt / 0.05);   // a little smoothing: the recordings are 30 fps
      for (let i = 0; i < 3; i++) this.gv[i] += ((p[i] - this.gPrev.p[i]) / dt - this.gv[i]) * a;
      const w = q2rv(qm(r, qc(this.gPrev.r))); for (let i = 0; i < 3; i++) this.gw[i] += (w[i] / dt - this.gw[i]) * a;
    }
    this.gPrev = { p, r };
    return { p, r };
  }
  // one frame: dt of physics in 1/480 s steps; fingerPush (the page's) writes your finger's force each step
  update(dt, fingerPush) {
    const W = this.W, d = W.d, m = W.m, mj = W.mj, K = this.kin, PH = 1 / 480;
    const gr = this.ghostRoot(dt);
    const falling = K.state === 'fall';
    // entering / leaving a fall
    if (falling && this.mode !== 'fall') {
      this.mode = 'fall'; this.fallT = 0;
      this.target = W.act.map(a => d.qpos[a.qadr]);
      this.brace = this.target.slice();
      const set = (n, v) => { const i = W.act.findIndex(a => a.name === n); if (i >= 0) this.brace[i] = v; };
      set('RightArm_y', -0.7); set('RightArm_z', -0.9); set('LeftArm_y', 0.7); set('LeftArm_z', 0.9);
      set('RightForeArm_z', -0.5); set('LeftForeArm_z', 0.5);
    }
    if (!falling && this.mode === 'fall') { this.mode = 'follow'; this.helpK = 0; this.lostT = -1.5; }   // get-up: help eases in
    let tg;
    if (falling) {
      this.fallT += dt;
      const u = clamp(this.fallT / 1.2, 0, 1), s = 0.35 + (0.08 - 0.35) * u, kb = clamp(this.fallT / 0.35, 0, 1);
      this.setGains(s * this.stiff, Math.sqrt(s * this.stiff));
      tg = this.target.map((v, i) => v + (this.brace[i] - v) * kb * 0.7);
    } else {
      this.setGains(this.stiff, Math.sqrt(this.stiff));
      tg = W.act.map(a => this.G.d.qpos[a.qadr]);
      this.helpK = Math.min(1, this.helpK + dt / 0.6);
    }
    const ctl = () => tg; ctl.hz = 480;
    // hips spring constants (mass 70 kg, ~1.5 Hz, damping ~0.8); turning: ~10 kg m2
    const M = W.mass || 70, w0 = 2 * Math.PI * 1.5, kp = M * w0 * w0, kd = 2 * 0.8 * M * w0, kr = 10 * w0 * w0, kdr = 2 * 0.8 * 10 * w0;
    // (user 17:41: "How come he doesn't fall in a position like this?") where does his weight land? The line of gravity
    // through his real centre of mass, onto the floor, against the area under his feet that touch it. Past it, the
    // hips spring may not hold him: it fades out over 8 cm, he has to step, and if that doesn't bring his feet back
    // under him he falls.
    const sup = falling || K.state === 'getup' ? null : this.support();
    this.sup = sup;
    let fade = 1;
    if (sup) {
      // standing: past his feet, the help fades and he must step. While a recovery step is under way his weight is
      // meant to be outside the foot he stands on (a step is a controlled fall): the help stays, and only a weight far
      // outside (25 cm) or still outside once the step is over counts as lost.
      const stepping = K.state === 'react' && K.anim && K.anim.x < K.anim.clip.n - 8;
      fade = stepping ? clamp(1 - (sup.out - 0.25) / 0.1, 0, 1) : clamp(1 - sup.out / 0.08, 0, 1);
      if (sup.out > 0.03 && K.state === 'stand') K.requestStep(sup.ang, sup.out);
      this.offT = sup.out > (stepping ? 0.3 : 0.08) ? (this.offT || 0) + dt : 0;
      if (this.offT > 0.4) { this.offT = 0; K.startFall('his weight went past his feet'); }
    }
    const H = falling ? 0 : this.help * this.helpK * fade * (K.state === 'getup' ? 1.6 : 1);
    const n = Math.min(40, Math.round(dt / PH));
    let helpSum = 0;
    for (let i = 0; i < n; i++) {
      d.xfrc_applied.fill(0);
      if (fingerPush) fingerPush();
      // muscles hold the limbs against gravity (joint torques = the bias forces on the hinges, not on the free root)
      d.qfrc_applied.fill(0);
      if (!falling) {
        for (let k = 6; k < m.nv; k++) d.qfrc_applied[k] = (d.qfrc_bias[k] - this.load * d.qfrc_constraint[k]) * this.stiff;
        if (this.stance) this.stanceTorques();
      }
      if (H > 0) {
        const b = this.hips, x = d.xfrc_applied;
        const p = [d.xpos[3 * b], d.xpos[3 * b + 1], d.xpos[3 * b + 2]];
        const v = [d.qvel[0], d.qvel[1], d.qvel[2]];
        const r = [d.qpos[3], d.qpos[4], d.qpos[5], d.qpos[6]];
        const wl = [d.qvel[3], d.qvel[4], d.qvel[5]], w = qv(r, wl);   // free joint: local angular velocity -> world
        const e = q2rv(qm(gr.r, qc(r)));
        const Fv = [0, 0, 0], Tv = [0, 0, 0];
        for (let k = 0; k < 3; k++) {
          Fv[k] = clamp(H * (kp * (gr.p[k] - d.qpos[k]) + kd * (this.gv[k] - v[k])), -2.5 * M * 9.81, 2.5 * M * 9.81);
          Tv[k] = clamp(H * (kr * e[k] + kdr * (this.gw[k] - w[k])), -400, 400);
        }
        if (this.virtual && K.state !== 'getup') {   // (get-ups: lying, no standing leg to push with -> outside help)
          // round 4: no outside force. The same pull is made by the muscles of the legs that stand on the floor
          // (joint torques = J^T of the hips' wanted force and turn, through each standing leg from its foot), so
          // the floor pushes back and his weight stays on his feet: past his feet, the feet tip and he can't be held.
          this.legPush(Fv, this.vturn ? Tv : [0, 0, 0]);
        } else {
          for (let k = 0; k < 3; k++) { x[6 * b + k] += Fv[k]; x[6 * b + 3 + k] += Tv[k]; }
          helpSum += Math.hypot(Fv[0], Fv[1], Fv[2]);
        }
        void p;
      }
      // feet (round 3: in recoveries the recording stepped and his real feet stayed planted behind: a loaded foot can't
      // be lifted by its joints alone while his body lags the recording). A spring from each recorded foot to his foot,
      // only while the recording moves that foot (lifted or sliding) or his foot is more than 3 cm from it. Counted with
      // the hidden support.
      if (this.feetHelp && (H > 0 || K.state === 'getup')) this.footSprings(H > 0 ? H : this.help);
      W.step(ctl);
    }
    d.qfrc_applied.fill(0);
    if (n) { this.stats.help = helpSum / n; }
    // can his body follow? hips far from the ghost's, or his head low while the ghost stands
    if (!falling && K.state !== 'getup') {
      const gap = Math.hypot(d.qpos[0] - gr.p[0], d.qpos[1] - gr.p[1], d.qpos[2] - gr.p[2]);
      const hh = this.headHeight(d), gh = this.headHeight(this.G.d);
      this.lostT = gap > 0.35 || (gh - hh > 0.45) ? this.lostT + dt : Math.min(0, this.lostT + dt);
      if (this.lostT > 0.3) { this.lostT = 0; K.startFall('his body could not follow'); }
    }
  }
  // his weight vs his feet (ground frame): out = how far past the feet (m), ang = which way, relative to his facing
  support() {
    const W = this.W, d = W.d, K = this.kin, m = W.m;
    mjFwdCom(W);
    const c = K.toLocal([d.subtree_com[3 * this.hips], d.subtree_com[3 * this.hips + 1], d.subtree_com[3 * this.hips + 2]]);
    const g = W.m.opt.gravity, gl = K.dirLocal([g[0], g[1], g[2]]);
    if (gl[2] > -1) return null;   // no floor under him to speak of
    const x = [c[0] - gl[0] * c[2] / gl[2], c[1] - gl[1] * c[2] / gl[2]];
    const P = [];
    for (const side of ['Left', 'Right']) {
      const fa = W.bodyId[side + 'Foot'], ft = W.bodyId[side + 'ToeBase'];
      const A = K.toLocal([d.xpos[3 * fa], d.xpos[3 * fa + 1], d.xpos[3 * fa + 2]]), T = K.toLocal([d.xpos[3 * ft], d.xpos[3 * ft + 1], d.xpos[3 * ft + 2]]);
      if (A[2] > 0.14 || T[2] > 0.08) continue;   // that foot is off the floor
      let u = [T[0] - A[0], T[1] - A[1]]; const L = Math.hypot(u[0], u[1]) || 1; u = [u[0] / L, u[1] / L]; const n = [-u[1], u[0]];
      // where a person can still put the pressure: heel to just past the ball of the foot (not the toe tips), foot width
      for (const [a, s] of [[-0.05, 1], [-0.05, -1], [L + 0.03, 1], [L + 0.03, -1]]) P.push([A[0] + u[0] * a + n[0] * 0.04 * s, A[1] + u[1] * a + n[1] * 0.04 * s]);
    }
    if (!P.length) return null;   // both feet in the air (mid-step): nothing to judge
    const o = outside(hull(P), x), f = K.facing();
    const ang = Math.atan2(o.dir[0] * f.l[0] + o.dir[1] * f.l[1], o.dir[0] * f.f[0] + o.dir[1] * f.f[1]) * 180 / Math.PI;
    return { out: o.d, ang, com: x };
  }
  // A leg with its foot on the floor holds up everything above it: at each hip, knee and ankle hinge, the torque that
  // holds the rest of the body (all but that joint's own lower part) against gravity, shared between the standing feet
  // by where his weight is. Replaces the hanging-leg gravity torque there. (Without it the legs only straighten under
  // load by lagging behind the recording: he sagged and over-leaned 8 deg, and the hidden support carried him.)
  stanceTorques() {
    const W = this.W, m = W.m, d = W.d, g = m.opt.gravity, K = this.kin;
    if (!this.legs) {
      this.legs = ['Left', 'Right'].map(side => {
        const js = [];
        for (const part of ['Leg', 'Shin', 'Foot']) for (const ax of ['x', 'y', 'z']) {
          const j = W.mj.mj_name2id(m, W.mj.mjtObj.mjOBJ_JOINT.value, side + part + '_' + ax); js.push({ j, dof: m.jnt_dofadr[j], b: m.jnt_bodyid[j] });
        }
        return { side, js, foot: W.bodyId[side + 'Foot'], toe: W.bodyId[side + 'ToeBase'] };
      });
      this.Mtot = m.body_subtreemass[this.hips];
    }
    // which feet stand, and how the weight is shared (where his centre of mass is along the line between the ankles)
    const M = this.Mtot, C = [d.subtree_com[3 * this.hips], d.subtree_com[3 * this.hips + 1], d.subtree_com[3 * this.hips + 2]];
    const down = (b) => K.toLocal([d.xpos[3 * b], d.xpos[3 * b + 1], d.xpos[3 * b + 2]])[2];
    const on = this.legs.map(L => down(L.foot) < 0.13 && down(L.toe) < 0.07);
    const A = this.legs.map(L => [d.xpos[3 * L.foot], d.xpos[3 * L.foot + 1], d.xpos[3 * L.foot + 2]]);
    let w = [0, 0];
    if (on[0] && on[1]) {
      const e = [A[1][0] - A[0][0], A[1][1] - A[0][1], A[1][2] - A[0][2]], L2 = e[0] * e[0] + e[1] * e[1] + e[2] * e[2] || 1e-9;
      const t = clamp(((C[0] - A[0][0]) * e[0] + (C[1] - A[0][1]) * e[1] + (C[2] - A[0][2]) * e[2]) / L2, 0, 1);
      w = [1 - t, t];
    } else if (on[0]) w = [1, 0]; else if (on[1]) w = [0, 1];
    this.share = w;
    this.legs.forEach((L, k) => {
      if (w[k] <= 0) return;
      for (const J of L.js) {
        const ms = m.body_subtreemass[J.b], cs = [d.subtree_com[3 * J.b], d.subtree_com[3 * J.b + 1], d.subtree_com[3 * J.b + 2]];
        const mp = M - ms, cp = [0, 1, 2].map(i => (M * C[i] - ms * cs[i]) / mp);
        const an = [d.xanchor[3 * J.j], d.xanchor[3 * J.j + 1], d.xanchor[3 * J.j + 2]], ax = [d.xaxis[3 * J.j], d.xaxis[3 * J.j + 1], d.xaxis[3 * J.j + 2]];
        const r = [cp[0] - an[0], cp[1] - an[1], cp[2] - an[2]], F = [mp * g[0], mp * g[1], mp * g[2]];
        const T = [r[1] * F[2] - r[2] * F[1], r[2] * F[0] - r[0] * F[2], r[0] * F[1] - r[1] * F[0]];
        const tau = -this.stance * (ax[0] * T[0] + ax[1] * T[1] + ax[2] * T[2]);   // sign checked: standing support 8% -> 4%
        d.qfrc_applied[J.dof] = (1 - w[k]) * d.qfrc_applied[J.dof] + w[k] * tau * this.stiff;
      }
    });
  }
  // outside hips help toward the ghost's root (used by drive_brain for get-ups)
  hipsHelp(gr, H, kp, kd, kr, kdr) {
    const d = this.W.d, b = this.hips, x = d.xfrc_applied, M = this.W.mass || 70;
    const v = [d.qvel[0], d.qvel[1], d.qvel[2]], r = [d.qpos[3], d.qpos[4], d.qpos[5], d.qpos[6]], w = qv(r, [d.qvel[3], d.qvel[4], d.qvel[5]]);
    const e = q2rv(qm(gr.r, qc(r)));
    let f2 = 0;
    for (let k = 0; k < 3; k++) {
      const F = clamp(H * (kp * (gr.p[k] - d.qpos[k]) + kd * (this.gv[k] - v[k])), -2.5 * M * 9.81, 2.5 * M * 9.81);
      x[6 * b + k] += F; x[6 * b + 3 + k] += clamp(H * (kr * e[k] + kdr * (this.gw[k] - w[k])), -400, 400); f2 += F * F;
    }
    this.lastHelp = Math.sqrt(f2);
  }
  // a wanted force F and turn T on his hips, made by the standing legs: each standing leg's 9 hinges get -J^T [F;T]
  // (J = how its foot moves relative to the hips), shared by his weight over the feet (this.share from stanceTorques).
  // Torques on a hinge are capped at the servo's own limit (300 N m).
  legPush(F, T) {
    const W = this.W, m = W.m, d = W.d, nv = m.nv, mj = W.mj;
    if (!this.legs || !this.share) return;
    if (!this.jp) { this.jp = new mj.DoubleBuffer(3 * nv); this.jr = new mj.DoubleBuffer(3 * nv); }
    this.legs.forEach((L, k) => {
      const s = this.share[k]; if (s <= 0) return;
      // the push point: the middle of the sole on the floor (the ankle's own origin would leave the ankle muscles out)
      const fa = L.foot, ft = L.toe, K = this.kin;
      const mid = [(d.xpos[3 * fa] + d.xpos[3 * ft]) / 2, (d.xpos[3 * fa + 1] + d.xpos[3 * ft + 1]) / 2, (d.xpos[3 * fa + 2] + d.xpos[3 * ft + 2]) / 2];
      const lm = K.toLocal(mid); lm[2] = 0; const pt = K.toWorld(lm);
      mj.mj_jac(m, d, this.jp, this.jr, pt, fa);
      const P = this.jp.GetView(), R = this.jr.GetView();
      for (const J of L.js) {
        const c = J.dof;
        const tau = this.vsign * s * (P[c] * F[0] + P[nv + c] * F[1] + P[2 * nv + c] * F[2] + R[c] * T[0] + R[nv + c] * T[1] + R[2 * nv + c] * T[2]);
        d.qfrc_applied[c] = clamp(d.qfrc_applied[c] + tau, -300, 300);
      }
    });
    this.stats.legs = Math.hypot(F[0], F[1], F[2]);
  }
  footSprings(h) {
    const W = this.W, d = W.d, Gd = this.G.d, x = d.xfrc_applied, M = W.mass || 70;
    const w0 = 2 * Math.PI * 3, k = 8 * w0 * w0, c = 2 * 0.9 * 8 * w0;   // like an 8 kg foot on a 3 Hz spring
    if (!this.fPrev) this.fPrev = {};
    let sum = 0;
    for (const side of ['Left', 'Right']) {
      const b = W.bodyId[side + 'Foot'], gp = [Gd.xpos[3 * b], Gd.xpos[3 * b + 1], Gd.xpos[3 * b + 2]];
      const pv = this.fPrev[side] || gp, gv = [0, 1, 2].map(i => (gp[i] - pv[i]) * 480);
      this.fPrev[side] = gp;
      const p = [d.xpos[3 * b], d.xpos[3 * b + 1], d.xpos[3 * b + 2]];
      const e = Math.hypot(gp[0] - p[0], gp[1] - p[1], gp[2] - p[2]), moving = Math.hypot(...gv) > 0.15;
      if (!moving && e < 0.03) continue;
      // the foot body's speed (its origin), from MuJoCo's body velocity
      W.mj.mj_objectVelocity(W.m, d, W.mj.mjtObj.mjOBJ_BODY.value, b, this.vb || (this.vb = new W.mj.DoubleBuffer(6)), 0);
      const v6 = this.vb.GetView(), v = [v6[3], v6[4], v6[5]];
      for (let i = 0; i < 3; i++) { const f = clamp(h * (k * (gp[i] - p[i]) + c * (gv[i] - v[i])), -0.6 * M * 9.81, 0.6 * M * 9.81); x[6 * b + i] += f; sum += Math.abs(f); }
    }
    this.stats.foot = sum;
  }
  headHeight(dd) { const b = this.head, K = this.kin; return K.toLocal([dd.xpos[3 * b], dd.xpos[3 * b + 1], dd.xpos[3 * b + 2]])[2]; }
}
