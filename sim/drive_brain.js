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
  headDownNow() {
    const W = this.W, d = W.d, hb = W.bodyId.Head, hp = W.bodyId.Hips, g0 = W.m.opt.gravity, gn0 = Math.hypot(g0[0], g0[1], g0[2]) || 9.81;
    const ux = d.xpos[3 * hb] - d.xpos[3 * hp], uy = d.xpos[3 * hb + 1] - d.xpos[3 * hp + 1], uz = d.xpos[3 * hb + 2] - d.xpos[3 * hp + 2], ul = Math.hypot(ux, uy, uz) || 1;
    return (ux * g0[0] + uy * g0[1] + uz * g0[2]) / ul / gn0;
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
      this.mode = 'fall'; B.down = true; this.fallT = 0; this.curled = false; this.headFirst = false;
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
      this.groupTone(0);
    }
    if (!falling && this.mode === 'fall') { this.mode = 'follow'; B.attach(); B.told = true; this.prevRef = null; }   // get-up: the network takes over again
    if (falling) { this.fallT += dt; this.groupTone(this.fallT); }
    const n = Math.min(40, Math.round(dt / PH));
    for (let i = 0; i < n; i++) {
      d.xfrc_applied.fill(0);
      if (fingerPush) fingerPush();
      if (B.due()) {
        if (!falling) { B.readState(); B.down = false; await B.think(this.buildRef(1 / B.hz)); if (B.down) { B.down = false; B.setTone(1); } this.plantLegs(); this.flatFeet(); this.armsToWalls(); }
        else {   // falling: hold the shape (see above) instead of the network
          B.tNext = d.time + 1 / B.hz - 1e-9;
          // turning head-down during the fall (the phone flipped while he was still upright): protect the head now
          if (!this.headFirst && !this.curled && this.fallT < 2.5 && this.headDownNow() > 0.25) this.headBrace();
          const k = Math.min(1, (this.fallT - (this.headFirst ? this.headT || 0 : 0)) / (this.headFirst ? 0.25 : 0.35));
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
