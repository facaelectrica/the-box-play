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

export class Springs {
  constructor(W, kin, opt = {}) {
    this.W = W; this.kin = kin; this.G = kin.W;          // his body, the engine, the ghost's world
    this.help = opt.help ?? 0.5;                          // 0..1.5: hips spring strength (1 = 1.5 Hz, well damped)
    this.stiff = opt.stiff ?? 1;                          // muscles: fraction of the servos' own gains
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
    const H = falling ? 0 : this.help * this.helpK * (K.state === 'getup' ? 1.6 : 1);
    const n = Math.min(40, Math.round(dt / PH));
    let helpSum = 0;
    for (let i = 0; i < n; i++) {
      d.xfrc_applied.fill(0);
      if (fingerPush) fingerPush();
      // muscles hold the limbs against gravity (joint torques = the bias forces on the hinges, not on the free root)
      d.qfrc_applied.fill(0);
      if (!falling) for (let k = 6; k < m.nv; k++) d.qfrc_applied[k] = (d.qfrc_bias[k] - this.load * d.qfrc_constraint[k]) * this.stiff;
      if (H > 0) {
        const b = this.hips, x = d.xfrc_applied;
        const p = [d.xpos[3 * b], d.xpos[3 * b + 1], d.xpos[3 * b + 2]];
        const v = [d.qvel[0], d.qvel[1], d.qvel[2]];
        const r = [d.qpos[3], d.qpos[4], d.qpos[5], d.qpos[6]];
        const wl = [d.qvel[3], d.qvel[4], d.qvel[5]], w = qv(r, wl);   // free joint: local angular velocity -> world
        const e = q2rv(qm(gr.r, qc(r)));
        for (let k = 0; k < 3; k++) {
          const F = H * (kp * (gr.p[k] - d.qpos[k]) + kd * (this.gv[k] - v[k]));
          const T = H * (kr * e[k] + kdr * (this.gw[k] - w[k]));
          x[6 * b + k] += clamp(F, -2.5 * M * 9.81, 2.5 * M * 9.81); x[6 * b + 3 + k] += clamp(T, -400, 400);
        }
        helpSum += Math.hypot(x[6 * b], x[6 * b + 1], x[6 * b + 2]);
        void p;
      }
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
  headHeight(dd) { const b = this.head, K = this.kin; return K.toLocal([dd.xpos[3 * b], dd.xpos[3 * b + 1], dd.xpos[3 * b + 2]])[2]; }
}
