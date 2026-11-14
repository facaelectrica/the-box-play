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
  // (falls worker) muscle tone while falling, by body part, as a fraction of the network's own gains: a falling person
  // keeps the head up and the trunk together (neck 0.9, spine 0.6), braces with the arms (0.55) and lets the legs give
  // (0.4); once on the floor it eases over 2 s to a resting tone that still holds his shape (0.7 / 0.45 / 0.35 / 0.3).
  // (Before: everything at 35% -> 25%, earlier 12% -> 4%: the neck folded, the body poured into corners.)
  groupTone(t) {
    const B = this.brain, m = this.W.m, u = Math.min(1, Math.max(0, (t - 0.8) / 2));
    const T = (a, b) => a + (b - a) * u;
    for (let i = 0; i < B.meta.kp.length; i++) {
      const a = B.actIdx[i], n = this.W.act[a].name;
      const f = /Neck|Head/.test(n) ? T(0.9, 0.7) : /Spine|Chest/.test(n) ? T(0.6, 0.45) : /Shoulder|Arm|Hand/.test(n) ? T(0.55, 0.35) : T(0.4, 0.3);
      const kp = B.meta.kp[i] * f, kd = B.meta.kd[i] * Math.sqrt(f);
      m.actuator_gainprm[10 * a] = kp; m.actuator_biasprm[10 * a + 1] = -kp; m.actuator_biasprm[10 * a + 2] = -kd;
    }
  }
  // (combined build) hands on the walls: the network was never trained with walls; following a reaching pose with
  // the hands hitting the wood, its arms bounced around. While he braces, his arm joints (collarbones to wrists)
  // follow the recording's arms directly (their own muscles, NVIDIA's gains), holding the arms' weight; the network
  // keeps everything else. Faded in and out with the brace.
  armsToWalls() {
    const W = this.W, B = this.brain, K = this.kin, w = K.braceW || 0;
    if (!this.armJ) this.armJ = W.act.map((a, j) => /(Shoulder|Arm|ForeArm|Hand)_/.test(a.name) ? j : -1).filter(j => j >= 0);
    this.armW = w * w * (3 - 2 * w);
    if (this.armW <= 0) return;
    for (const j of this.armJ) { const a = W.act[j]; B.targets[j] += (this.G.d.qpos[a.qadr] - B.targets[j]) * this.armW; }
  }
  armHold() {
    const W = this.W, d = W.d;
    if (!this.armJ) return;
    for (const j of this.armJ) { const dof = W.act[j].qadr - 1; d.qfrc_applied[dof] = (this.armW || 0) > 0 && this.kin.state !== 'fall' ? d.qfrc_bias[dof] * this.armW : 0; }
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
    // falling (user 18:23 "why does he get all floppy?"): his muscles stay at ~1/3, easing to 1/4 (was 12% -> 4%),
    // holding the pose he had when he went down, so he keeps his shape (with the joint limits) instead of folding
    if (falling && this.mode !== 'fall') {
      this.mode = 'fall'; B.down = true; this.fallT = 0;
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
      if (Math.abs(dir) < 55) {                 // forward
        set('RightArm_y', -0.55); set('RightArm_z', 1.0); set('LeftArm_y', 0.55); set('LeftArm_z', -1.0);
        set('RightForeArm_z', 0.35); set('LeftForeArm_z', -0.35); set('Neck1_x', -0.15); set('Head_x', -0.1);
        atLeast('RightLeg_x', -0.3); atLeast('LeftLeg_x', -0.3);
      } else if (Math.abs(dir) > 125) {         // backward
        set('RightArm_y', -0.95); set('RightArm_z', -0.55); set('LeftArm_y', 0.95); set('LeftArm_z', 0.55);
        set('RightForeArm_z', 0.3); set('LeftForeArm_z', -0.3);
        set('Neck1_x', 0.4); set('Neck2_x', 0.1); set('Head_x', 0.1); set('Spine1_x', 0.2); set('Spine2_x', 0.1);
        atLeast('RightLeg_x', -0.7); atLeast('LeftLeg_x', -0.7); atLeast('RightShin_x', 0.8); atLeast('LeftShin_x', 0.8);
      } else {                                  // sideways: dir > 0 = toward his left
        const L = dir > 0, out = L ? 'Left' : 'Right', other = L ? 'Right' : 'Left', so = L ? 1 : -1;
        set(out + 'Arm_y', 0.6 * so); set(out + 'Arm_z', -0.35 * so); set(out + 'ForeArm_z', -0.3 * so);
        set(other + 'Arm_y', 1.0 * -so); set(other + 'Arm_z', 0.8 * so); set(other + 'ForeArm_z', 0.9 * so);
        set('Neck1_y', 0.15 * -so);
      }
      this.groupTone(0);
    }
    if (!falling && this.mode === 'fall') { this.mode = 'follow'; B.attach(); B.told = true; this.prevRef = null; }   // get-up: the network takes over again
    if (falling) { this.fallT += dt; this.groupTone(this.fallT); }
    const n = Math.min(40, Math.round(dt / PH));
    for (let i = 0; i < n; i++) {
      d.xfrc_applied.fill(0);
      if (fingerPush) fingerPush();
      if (B.due()) {
        if (!falling) { B.readState(); B.down = false; await B.think(this.buildRef(1 / B.hz)); if (B.down) { B.down = false; B.setTone(1); } this.armsToWalls(); }
        else {   // falling: hold the shape (see above) instead of the network
          B.tNext = d.time + 1 / B.hz - 1e-9;
          const k = Math.min(1, this.fallT / 0.35);
          W.act.forEach((a, j) => { B.targets[j] = this.hold[j] + (this.brace[j] - this.hold[j]) * k; });
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
