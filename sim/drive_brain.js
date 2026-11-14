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
export class BrainDrive {
  // springs (sim/springs.js), optional: the network was never taught to get up from the floor (it gives up past
  // 0.5 m), so during a get-up the springs and their help carry him through the get-up recording instead
  constructor(W, kin, brain, springs = null) {
    this.W = W; this.kin = kin; this.brain = brain; this.G = kin.W; this.springs = springs;
    kin.ext = W;   // the engine's falls follow this body
    this.ref = { pos: new Float64Array(NB * 3), rot: new Float64Array(NB * 4), vel: new Float64Array(NB * 3), ang: new Float64Array(NB * 3) };
    this.prevRef = null; this.lostT = 0; this.sup = null; this.stats = {}; this.getupBy = 'springs';
  }
  reset() {
    const W = this.W, d = W.d, g = this.G.d;
    for (let k = 0; k < 73; k++) d.qpos[k] = g.qpos[k];
    d.qvel.fill(0); d.xfrc_applied.fill(0); d.qfrc_applied.fill(0);
    W.mj.mj_forward(W.m, d); W.tControl = d.time;
    this.brain.attach(); this.brain.told = true; this.prevRef = null; this.lostT = 0; this.mode = 'follow';
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
  height(dd, b) { return this.kin.toLocal([dd.xpos[3 * b], dd.xpos[3 * b + 1], dd.xpos[3 * b + 2]])[2]; }
  async update(dt, fingerPush) {
    const W = this.W, d = W.d, K = this.kin, B = this.brain, PH = 1 / 480;
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
    if (this.springs && K.state === 'getup') {
      if (this.mode !== 'getup') { this.mode = 'getup'; const S = this.springs; S.mode = 'follow'; S.helpK = 0; S.gPrev = null; S.lostT = 0; S.setGains(S.stiff, Math.sqrt(S.stiff)); }
      this.springs.update(dt, fingerPush); this.stats.err = 0; this.stats.help = this.springs.stats.help; return;
    }
    this.stats.help = 0;
    if (this.mode === 'getup') { this.mode = 'follow'; d.qfrc_applied.fill(0); B.attach(); B.told = true; this.prevRef = null; this.lostT = 0; }
    this.followBody(dt);
    const falling = K.state === 'fall';
    if (falling && this.mode !== 'fall') { this.mode = 'fall'; B.down = true; B.setTone(0.12); this.fallT = 0; }
    if (!falling && this.mode === 'fall') { this.mode = 'follow'; B.attach(); B.told = true; this.prevRef = null; }   // get-up: the network takes over again
    if (falling) { this.fallT += dt; B.setTone(Math.max(0.04, 0.12 - 0.08 * this.fallT)); }
    const n = Math.min(40, Math.round(dt / PH));
    for (let i = 0; i < n; i++) {
      d.xfrc_applied.fill(0);
      if (fingerPush) fingerPush();
      if (B.due()) {
        if (!falling) { B.readState(); B.down = false; await B.think(this.buildRef(1 / B.hz)); if (B.down) { B.down = false; B.setTone(1); } }
        else await B.think();
      }
      W.step(B.control);
    }
    // losing it: far from the recording's pose for a while (the network was trained to give up past 0.5 m)
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
        this.driftT = (B.err || 0) > 0.35 ? (this.driftT || 0) + dt : 0;
        if (this.driftT > 0.3) { this.driftT = 0; K.resync(d.qpos); this.prevRef = null; }
      }
    }
    if (!falling && K.state === 'getup') {   // (no springs given) the network tries the get-up itself
      this.lostT = (B.err || 0) > 0.6 ? this.lostT + dt : Math.max(0, this.lostT - dt);
      if (this.lostT > 0.8) { this.lostT = 0; K.startFall('the get-up failed'); }
    }
    this.stats.err = B.err || 0;
  }
}
