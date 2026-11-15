// The Box - the physics controller: his muscles without a trained brain (V43, user Nov 15 13:05-13:09: "a reality where
// we could do this without any brain ... purely physics, springs, IK/FK and guidance, plus all these animation poses";
// "this physics controller would replace the brain, but we keep using everything else ... right?" - yes).
//
// Same job as the network in sim/brain.js, same place in the page: the director (sim/kin.js: balance model, recordings,
// steps, IK) moves a ghost; this works his real, physical body's joints so it follows the ghost. Nothing else moves him.
//
// Every control tick (inverse dynamics with contact forces, the textbook way robots and pre-learning game characters
// balance):
//  1. what accelerations would bring his body onto the ghost: each joint toward the ghost's angle (a critically damped
//     spring), the hips toward where the ghost's hips are relative to the ghost's feet, placed over HIS feet
//  2. MuJoCo's inverse dynamics (mj_inverse, contacts switched off for that call): the generalized forces those
//     accelerations need. The 6 for the hips can't come from any muscle - only from the floor through his feet
//  3. the floor's share: the push of every sole contact point, inside its friction cone (no sliding) and never pulling
//     (so the centre of pressure stays inside his feet), chosen to give the hips' 6 as closely as possible (small
//     non-negative least squares). What the feet can't give, he doesn't get: he leans, steps (the director) or falls
//  4. joint torques = what the accelerations need minus what the floor gives through the legs, capped at a strong
//     adult's strength (springs.js HUMAN_NM) - rule 4 by construction
//  5. handed to the joints' servos as targets (torque = kp (target - q) - kd qd, so the servo gives exactly that torque
//     now and stays a soft spring between ticks)
import { Brain, ND } from './brain.js';
import { humanCap } from './springs.js?v=55';

const cross = (a, b) => [a[1] * b[2] - a[2] * b[1], a[2] * b[0] - a[0] * b[2], a[0] * b[1] - a[1] * b[0]];
const qv = (q, v) => { const [w, x, y, z] = q, tx = 2 * (y * v[2] - z * v[1]), ty = 2 * (z * v[0] - x * v[2]), tz = 2 * (x * v[1] - y * v[0]); return [v[0] + w * tx + (y * tz - z * ty), v[1] + w * ty + (z * tx - x * tz), v[2] + w * tz + (x * ty - y * tx)]; };
const qc = q => [q[0], -q[1], -q[2], -q[3]];
const qmul = (a, b) => [a[0] * b[0] - a[1] * b[1] - a[2] * b[2] - a[3] * b[3], a[0] * b[1] + a[1] * b[0] + a[2] * b[3] - a[3] * b[2], a[0] * b[2] - a[1] * b[3] + a[2] * b[0] + a[3] * b[1], a[0] * b[3] + a[1] * b[2] - a[2] * b[1] + a[3] * b[0]];
function clampVec(a, i, mx) { const l = Math.hypot(a[i], a[i + 1], a[i + 2]); if (l > mx) for (let k = 0; k < 3; k++) a[i + k] *= mx / l; }
// 2-D convex hull (monotone chain) and signed distance of a point to it (positive inside)
function convexHull(P) {
  if (P.length < 3) return P.slice();
  const p = P.slice().sort((a, b) => a[0] - b[0] || a[1] - b[1]), cr = (o, a, b) => (a[0] - o[0]) * (b[1] - o[1]) - (a[1] - o[1]) * (b[0] - o[0]);
  const lo = [], up = [];
  for (const q of p) { while (lo.length >= 2 && cr(lo[lo.length - 2], lo[lo.length - 1], q) <= 0) lo.pop(); lo.push(q); }
  for (const q of p.reverse()) { while (up.length >= 2 && cr(up[up.length - 2], up[up.length - 1], q) <= 0) up.pop(); up.push(q); }
  return lo.slice(0, -1).concat(up.slice(0, -1));
}
function insideDist(H, x) {
  let inside = true, best = 1e9;
  for (let i = 0; i < H.length; i++) {
    const a = H[i], b = H[(i + 1) % H.length], e = [b[0] - a[0], b[1] - a[1]], L2 = e[0] * e[0] + e[1] * e[1] || 1e-12;
    if (e[0] * (x[1] - a[1]) - e[1] * (x[0] - a[0]) < 0) inside = false;
    const t = Math.max(0, Math.min(1, ((x[0] - a[0]) * e[0] + (x[1] - a[1]) * e[1]) / L2));
    best = Math.min(best, Math.hypot(x[0] - a[0] - e[0] * t, x[1] - a[1] - e[1] * t));
  }
  return inside ? best : -best;
}
const norm = a => { const l = Math.hypot(...a) || 1; return a.map(v => v / l); };

export class PhysBrain extends Brain {
  // W: his physical world; kin: the director (sim/kin.js; its world kin.W is the ghost)
  constructor(W, kin, opt = {}) {
    const G = kin.W;
    const meta = { n_obs: 961, kp: W.act.map(a => a.kp), kd: W.act.map(a => a.kd), scale: Math.PI, control_hz: opt.hz ?? 120, physics: true };
    super(W, null, null, meta, opt);
    const m = W.m, mj = W.mj;
    this.Gw = G; this.kin = kin; this.nv = m.nv;
    // the soles: the box geoms of each foot and its toes (their bottom corners are where the floor can push)
    this.soles = ['Left', 'Right'].map(side => ({ side, bodies: [W.bodyId[side + 'Foot'], W.bodyId[side + 'ToeBase']],
      geoms: [...Array(m.ngeom).keys()].filter(g => m.geom_type[g] === 6 && [W.bodyId[side + 'Foot'], W.bodyId[side + 'ToeBase']].includes(m.geom_bodyid[g])) }));
    this.onFloor = opt.onFloor ?? 0.03;
    // each leg's 9 joints from the hip to the ankle: with that foot on the floor they move only as the hips over it allow
    for (const S of this.soles) S.legDofs = W.act.filter(a => new RegExp('^' + S.side + '(Leg|Shin|Foot)_').test(a.name)).map(a => a.dof);
    this.wf = opt.wf ?? 12;         // a stepping foot toward the ghost's
    this.spineDofs = [];   // the foot's Jacobian has no columns outside root + its own leg
    this.jp6 = new mj.DoubleBuffer(3 * m.nv); this.jr6 = new mj.DoubleBuffer(3 * m.nv);   // a sole this close to the floor (m) counts as standing on it (the welds hold feet ~1 cm up)
    this.d2 = new mj.MjData(m);
    this.jp = new mj.DoubleBuffer(3 * m.nv); this.jr = new mj.DoubleBuffer(3 * m.nv);
    this.wn = opt.wn ?? 11;          // joints: natural frequency (rad/s) of the pull toward the ghost's angle
    this.wr = opt.wr ?? 5;           // hips: position over the feet
    this.wo = opt.wo ?? 9;           // hips: orientation
    this.mu = opt.mu ?? 0.6;         // friction used when sharing the load (the floor's real friction is 1.0: a margin)
    this.gain = opt.gain ?? 0.5;
    this.yawK = opt.yawK ?? 0.15;    // how much of the ghost's turning the hips are asked to follow while the feet stay planted
    this.maxLin = opt.maxLin ?? 6;   // m/s^2 and rad/s^2: the most the hips are asked to accelerate
    this.maxAng = opt.maxAng ?? 40;
    this.rowW = opt.rowW ?? [1, 1, 1, 1, 1, 1];   // when the floor can't give all the hips ask: what to keep first (push, then turn)
    this.giveWay = opt.giveWay ?? true;   // over a person's strength: ask for less acceleration rather than clip joints one by one
    this.wo1 = opt.wo1 ?? 0.5;         // hips' orientation gain on one foot, as a fraction
    this.noHands = opt.noHands ?? false;
    this.freeLegs = opt.freeLegs ?? false;
    this.maxJ = opt.maxJ ?? 60;      // rad/s^2 per joint (a hand stopped by a wall while the ghost's hand goes on: not a reason to heave the trunk)
    this.hands = new Set(['RightHand', 'LeftHand', 'RightForeArm', 'LeftForeArm'].map(n => W.bodyId[n]));     // servo stiffness between ticks, as a fraction of the body file's gains
    this.hcap = W.act.map(a => humanCap(a.name));
    this.extLoad = opt.extLoad ?? (globalThis.PH_EXT !== undefined ? !!+globalThis.PH_EXT : true);
    // held by every joint between them and the floor (extAll; extAll false: the shoulder girdle and arm joints only),
    // each push counted up to extMax N (a knock into the wall is not a load to hold)
    this.extAll = opt.extAll ?? (globalThis.PH_EXTALL !== undefined ? !!+globalThis.PH_EXTALL : true); this.extMax = opt.extMax ?? globalThis.PH_EXTMAX ?? 400;
    this.girdleDofs = W.act.filter(a => /(Shoulder|Arm)_/.test(a.name) && !/ForeArm/.test(a.name)).map(a => a.dof); this.allDofs = [...Array(m.nv).keys()];
    this.footB = { L: W.bodyId.LeftFoot, R: W.bodyId.RightFoot };
    this.prevGq = null; this.gqd = new Float64Array(m.nq);
    // get-ups (user 14:17, option A): the get-up recording is the guide, his muscles and real contacts do the work
    // (thinkGetup). The director stops placing the recording on his body when this is set (drive_brain)
    this.getupReady = opt.getupReady ?? false;   // (off until it stands him up as often as the recordings placed on him: 6/40 vs 27/40) this.getupMode = false; this.getupInfo = null;
    this.guWLeg = opt.guWLeg ?? 9; this.guWArm = opt.guWArm ?? 7; this.guWTrunk = opt.guWTrunk ?? 5; this.guWHead = opt.guWHead ?? 4;
    this.guWr = opt.guWr ?? 5; this.guWo = opt.guWo ?? 5; this.guMaxLin = opt.guMaxLin ?? 3; this.guMaxAng = opt.guMaxAng ?? 12;
    this.guUpMax = opt.guUpMax ?? 2.5;   // m/s^2 up at most (a quarter of g)
    this.guFootUp = opt.guFootUp ?? 0.06; this.guComIn = opt.guComIn ?? 0.03; this.guWc = opt.guWc ?? 3; this.guFollow = opt.guFollow ?? 3; this.guJw = opt.guJw ?? globalThis.PG_JW ?? 20; this.guFeetCon = opt.guFeetCon ?? !!globalThis.PG_FEETCON; this.guRelease = opt.guRelease ?? globalThis.PG_REL ?? 0.03;
    this.fig = W.isFig; this.bname = [...Array(m.nbody).keys()].map(b => mj.mj_id2name(m, mj.mjtObj.mjOBJ_BODY.value, b) || '');
    this.stats = { resid: 0, n: 0, tau: 0, over: 0, contacts: 0 };
    this.dsbl = mj.mjtDisableBit.mjDSBL_CONSTRAINT.value;
  }
  attach() {
    super.attach();
    const m = this.W.m;
    for (let i = 0; i < ND; i++) {   // softer servos: the feedforward torque does the work, the servo only steadies it
      const a = this.actIdx[i], kp = this.meta.kp[i] * this.gain, kd = this.meta.kd[i] * Math.sqrt(this.gain);
      m.actuator_gainprm[10 * a] = kp; m.actuator_biasprm[10 * a + 1] = -kp; m.actuator_biasprm[10 * a + 2] = -kd;
    }
    this.prevGq = null;
  }
  setTone(f) { super.setTone(f * (f >= 1 ? this.gain : 1)); }

  // the ghost's joint speeds, from tick to tick (it is kinematic: no qvel of its own)
  ghostVel(dt) {
    const gq = this.Gw.d.qpos, n = this.W.m.nq;
    if (this.prevGq && dt > 0) for (let k = 0; k < n; k++) { const v = (gq[k] - this.prevGq[k]) / dt; this.gqd[k] += (v - this.gqd[k]) * 0.5; }
    else this.gqd.fill(0);
    if (!this.prevGq) this.prevGq = new Float64Array(n);
    this.prevGq.set(gq.subarray(0, n));
  }

  soleHeight(S, d = this.W.d) {
    const W = this.W, m = W.m, K = this.kin; let lo = 9;
    for (const g of S.geoms) {
      const c = [d.geom_xpos[3 * g], d.geom_xpos[3 * g + 1], d.geom_xpos[3 * g + 2]], R = d.geom_xmat.subarray(9 * g, 9 * g + 9), sz = m.geom_size.subarray(3 * g, 3 * g + 3);
      for (let c8 = 0; c8 < 8; c8++) {
        const l = [(c8 & 1 ? 1 : -1) * sz[0], (c8 & 2 ? 1 : -1) * sz[1], (c8 & 4 ? 1 : -1) * sz[2]];
        lo = Math.min(lo, K.toLocal([0, 1, 2].map(r => c[r] + R[3 * r] * l[0] + R[3 * r + 1] * l[1] + R[3 * r + 2] * l[2]))[2]);
      }
    }
    return lo;
  }
  // foot (6 rows: its point and its turning) fixed: J_root a_root + J_leg a_leg = 0. a_leg = wish + J+ (rhs - J wish)
  // the stepping foot's wish, in the world: toward the ghost's foot (position and turn), critically damped
  swingAcc(S, dt) {
    const W = this.W, d = W.d, gd = this.Gw.d, b = W.bodyId[S.side + 'Foot'], w = this.wf;
    const p = [d.xpos[3 * b], d.xpos[3 * b + 1], d.xpos[3 * b + 2]], g = [gd.xpos[3 * b], gd.xpos[3 * b + 1], gd.xpos[3 * b + 2]];
    const prev = S.gPrev; S.gPrev = g;
    const gv = prev && dt > 0 ? g.map((v, k) => (v - prev[k]) / dt) : [0, 0, 0];
    const v = S.vNow || [0, 0, 0], om = S.wNow || [0, 0, 0];
    const q = [d.xquat[4 * b], d.xquat[4 * b + 1], d.xquat[4 * b + 2], d.xquat[4 * b + 3]], gq = [gd.xquat[4 * b], gd.xquat[4 * b + 1], gd.xquat[4 * b + 2], gd.xquat[4 * b + 3]];
    // world rotation error: gq * q^-1 as a rotation vector
    const e = qmul(gq, qc(q)); if (e[0] < 0) for (let k = 0; k < 4; k++) e[k] = -e[k];
    const sn = Math.hypot(e[1], e[2], e[3]), th = 2 * Math.atan2(sn, e[0]), f = sn > 1e-9 ? th / sn : 2;
    const a = new Float64Array(6);
    for (let k = 0; k < 3; k++) { a[k] = w * w * (g[k] - p[k]) + 2 * w * (gv[k] - v[k]); a[3 + k] = w * w * e[1 + k] * f - 2 * w * om[k]; }
    clampVec(a, 0, 40); clampVec(a, 3, 80);
    return a;
  }
  closeChain(S, qacc, ades = null) {
    const W = this.W, m = W.m, d = W.d, mj = W.mj, nv = this.nv, b = W.bodyId[S.side + 'Foot'], L = S.legDofs;
    mj.mj_jacBody(m, d, this.jp6, this.jr6, b);
    const P = this.jp6.GetView(), R = this.jr6.GetView();
    const row = (r, v) => r < 3 ? P[r * nv + v] : R[(r - 3) * nv + v];
    const J = [], rhs = new Float64Array(6);
    for (let r = 0; r < 6; r++) {
      let s = 0; for (let v = 0; v < 6; v++) s += row(r, v) * qacc[v];
      for (const v of L) s += row(r, v) * qacc[v];   // minus what the leg's own wish already does
      rhs[r] = (ades ? ades[r] : 0) - s; J.push(L.map(v => row(r, v)));
    }
    // the foot's own speed now (for the next swing wish's damping)
    { const vq = d.qvel; const vv = [0, 0, 0, 0, 0, 0]; for (let r = 0; r < 6; r++) { let x = 0; for (let v = 0; v < 6; v++) x += row(r, v) * vq[v]; for (const v of L) x += row(r, v) * vq[v]; for (const v of this.spineDofs) x += row(r, v) * vq[v]; vv[r] = x; } S.vNow = vv.slice(0, 3); S.wNow = vv.slice(3); }
    // dx = J^T (J J^T + l I)^-1 rhs
    const A = [...Array(6)].map((_, i) => [...Array(6)].map((_, j) => J[i].reduce((a, x, k) => a + x * J[j][k], 0) + (i === j ? 1e-4 : 0)));
    const y = solve6(A, rhs);
    if (!y) return;
    L.forEach((v, k) => { let s = 0; for (let r = 0; r < 6; r++) s += J[r][k] * y[r]; qacc[v] += s; });
  }

  // the box's present pushes on bodies other than his soles and hands (which the plan counts as support), as generalized
  // forces: moved to the known side (like gravity), so the joints between them and his hips hold them
  externalLoads(...needs) {
    const W = this.W, m = W.m, d = W.d, mj = W.mj, nv = this.nv, cf = this.cfb || (this.cfb = new mj.DoubleBuffer(6));
    const skip = b => this.hands.has(b) || this.soles.some(S => S.bodies.includes(b));
    let n = 0;
    for (let i = 0; i < d.ncon; i++) {
      const c = d.contact.get(i), g1 = c.geom1, g2 = c.geom2, pos = [c.pos[0], c.pos[1], c.pos[2]], fr = [0, 1, 2, 3, 4, 5, 6, 7, 8].map(k => c.frame[k]); c.delete();
      const b1 = m.geom_bodyid[g1], b2 = m.geom_bodyid[g2], f1 = W.isFig[b1], f2 = W.isFig[b2]; if (f1 === f2) continue;
      const b = f2 ? b2 : b1, sg = f2 ? 1 : -1; if (skip(b)) continue;
      mj.mj_contactForce(m, d, i, cf); const f = cf.GetView(); if (!(f[0] > 0)) continue;
      const fw = [0, 1, 2].map(k => sg * (fr[k] * f[0] + fr[3 + k] * f[1] + fr[6 + k] * f[2]));
      mj.mj_jac(m, d, this.jp, this.jr, pos, b); const J = this.jp.GetView();
      const fl = Math.hypot(...fw), k = fl > this.extMax ? this.extMax / fl : 1;   // a knock into the wall is not a load to hold
      for (const v of (this.extAll ? this.allDofs : this.girdleDofs)) { const g = k * (J[v] * fw[0] + J[nv + v] * fw[1] + J[2 * nv + v] * fw[2]); for (const N of needs) N[v] -= g; }
      n++;
    }
    this.stats.ext = n;
  }
  // 4. the plan for an ambition sc (1 = all the accelerations asked; less = gentler, same gravity): the contacts' share
  // (non-negative least squares on the hips' 6), then the joints' torques; too much for a person: ask less (a person
  // under a hard knock doesn't follow his plan exactly, he gives); then capped at a person's strength and handed to the
  // servos as targets
  applyPlan(need0, needA, cols, jw = 0) {
    const W = this.W, m = W.m, d = W.d, nv = this.nv;
    // jw > 0: the contacts' pushes are also chosen to spare his joints (each joint's torque counted as a fraction of a
    // person's strength): a hand on the floor takes weight where it relieves the elbow, not where it overloads it
    let solve = null;
    if (cols.length && jw > 0) { const rw = new Float64Array(nv); for (let v = 0; v < 6; v++) rw[v] = this.rowW[v]; W.act.forEach((a, i) => { rw[a.dof] = jw / this.hcap[i]; }); solve = nnlsPrep(cols, nv, 1e-6, rw); }
    else if (cols.length) solve = nnlsPrep(cols, 6, 1e-6, this.rowW);
    const plan = sc => {
      const b = need0.map((v, i) => v + sc * needA[i]);
      const w = solve ? solve(b) : null;
      const t = b;
      if (w) for (let k = 0; k < cols.length; k++) if (w[k] > 0) { const c = cols[k]; for (let v = 0; v < nv; v++) t[v] -= w[k] * c[v]; }
      let r = 0; W.act.forEach((a, i) => { r = Math.max(r, Math.abs(t[a.dof]) / this.hcap[i]); });
      return { t, r, sc };
    };
    let P = plan(1);
    if (P.r > 1.05 && this.giveWay) {
      let best = P;
      for (const sc of [0.6, 0.35, 0.15]) { const Q = plan(sc); if (Q.r < best.r) best = Q; if (Q.r <= 1.05) { best = Q; break; } }
      P = best;
    }
    const tau = P.t; this.planSc = P.sc; this.planR = P.r;
    let resid = 0; for (let v = 0; v < 6; v++) resid += tau[v] * tau[v];
    this.stats.resid += (Math.sqrt(resid) - this.stats.resid) * 0.1;
    // capped at a person's strength, handed to the servos
    let over = 0, tsum = 0;
    W.act.forEach((a, i) => {
      let t = tau[a.dof]; const H = this.hcap[i];
      if (Math.abs(t) > H) { over++; if (globalThis.PHOVER) (this.overN || (this.overN = {}))[a.name] = (this.overN[a.name] || 0) + 1; t = Math.sign(t) * H; }
      tsum += Math.abs(t);
      const kpa = m.actuator_gainprm[10 * i], kda = -m.actuator_biasprm[10 * i + 2];
      this.targets[i] = d.qpos[a.qadr] + (t + kda * d.qvel[a.dof]) / Math.max(1e-6, kpa);
    });
    this.stats.over = over; this.stats.tau = tsum; this.stats.n++;
    return tau;
  }

  // ---- get-ups (user 14:17, option A) ----
  // The director plays the get-up recording on the ghost (and waits for him: kin.getupLag); his body is not written to
  // it. Each tick, as think() but for a body lying, sitting, kneeling:
  //  - joints toward the recording's: legs firmly, arms less, trunk and head softly (a person rising doesn't hold his
  //    head and trunk exactly as the recording did; asking it every tick put 11x a person's strength on his neck)
  //  - hips: up and down as the recording's (height over the floor); along the floor where the recording has them
  //    relative to its planted feet, put over his planted feet
  //  - before rising: his centre of mass over what holds him once his hips leave the floor (feet, knees, hands); while it
  //    isn't, the hips are asked to go there first and to rise only a little
  //  - planted feet stay put (closed chains, as standing)
  //  - every contact with the box is support (feet, knees, hands, forearms, hips, back): pushes along its normal within
  //    friction; never asked to accelerate up harder than ~0.25 g (pushing too hard throws him off all his supports)
  thinkGetup(ref) {
    const W = this.W, m = W.m, d = W.d, mj = W.mj, nv = this.nv, dt = 1 / this.hz, gd = this.Gw.d, K = this.kin;
    const t0 = performance.now(), clip = K.anim && K.anim.clip;
    if (!this.gu || this.gu.clip !== clip) this.gu = { t: 0, h0: null, prog: 0, clip, stallT: 0 };
    const gu = this.gu; gu.t += dt;
    this.getupMode = true; this.down = false;
    this.ghostVel(dt);
    const o = K.toWorld([0, 0, 0]), z = K.toWorld([0, 0, 1]), up = [z[0] - o[0], z[1] - o[1], z[2] - o[2]];
    const hb = W.bodyId.Hips, ht = dd => (dd.xpos[3 * hb] - o[0]) * up[0] + (dd.xpos[3 * hb + 1] - o[1]) * up[1] + (dd.xpos[3 * hb + 2] - o[2]) * up[2];
    const hMe = ht(d), hGh = ht(gd); if (gu.h0 === null) gu.h0 = hMe;
    const qacc = new Float64Array(nv), mx = this.maxJ;
    // 1. joints
    for (const a of W.act) {
      const w = /(Leg|Shin|Foot|ToeBase)_/.test(a.name) ? this.guWLeg : /^(Spine|Chest)/.test(a.name) ? this.guWTrunk : /^(Neck|Head)/.test(a.name) ? this.guWHead : this.guWArm;
      let e = gd.qpos[a.qadr] - d.qpos[a.qadr]; e = Math.atan2(Math.sin(e), Math.cos(e));
      qacc[a.dof] = Math.max(-mx, Math.min(mx, w * w * e + 2 * w * (this.gqd[a.qadr] - d.qvel[a.dof])));
    }
    // his contacts with the box
    const sup = { feet: 0, hands: 0, knees: 0, other: 0 }, con = [], spts = [];
    for (let i = 0; i < d.ncon; i++) {
      const c = d.contact.get(i), g1 = c.geom1, g2 = c.geom2, dist = c.dist, pos = [c.pos[0], c.pos[1], c.pos[2]], fr = [0, 1, 2, 3, 4, 5, 6, 7, 8].map(k => c.frame[k]); c.delete();
      if (dist > 0.005) continue;
      const b1 = m.geom_bodyid[g1], b2 = m.geom_bodyid[g2], f1 = this.fig[b1], f2 = this.fig[b2]; if (f1 === f2) continue;
      const b = f2 ? b2 : b1, sg = f2 ? 1 : -1, nm = this.bname[b];   // the frame's normal points from geom1 to geom2: the push on geom2 is +n
      const kind = /Foot|ToeBase/.test(nm) ? 'feet' : /Hand|ForeArm/.test(nm) ? 'hands' : /Shin/.test(nm) ? 'knees' : 'other';
      const nn = [fr[0] * sg, fr[1] * sg, fr[2] * sg];
      // a floor contact the recording is lifting off (the same point of the recording's body over 3 cm above the floor):
      // not support - the floor can't push a knee or a hip that is going up (planning with it, he stayed on his knee
      // while the plan believed the floor was lifting him)
      if (this.guRelease > 0 && (kind === 'knees' || kind === 'other') && nn[0] * up[0] + nn[1] * up[1] + nn[2] * up[2] > 0.7) {
        const R = d.xmat.subarray(9 * b, 9 * b + 9), GR = gd.xmat.subarray(9 * b, 9 * b + 9), r = [0, 1, 2].map(k => pos[k] - d.xpos[3 * b + k]);
        const l = [0, 1, 2].map(j => R[j] * r[0] + R[3 + j] * r[1] + R[6 + j] * r[2]), pg = [0, 1, 2].map(k => gd.xpos[3 * b + k] + GR[3 * k] * l[0] + GR[3 * k + 1] * l[1] + GR[3 * k + 2] * l[2]);
        if ((pg[0] - pos[0]) * nn[0] + (pg[1] - pos[1]) * nn[1] + (pg[2] - pos[2]) * nn[2] > this.guRelease) { sup.off = (sup.off || 0) + 1; continue; }
      }
      sup[kind]++;
      con.push({ pos, b, kind, n: nn, u1: [fr[3], fr[4], fr[5]], u2: [fr[6], fr[7], fr[8]] });
      if (kind !== 'other') { const l = K.toLocal(pos); spts.push([l[0], l[1]]); }
    }
    // planted feet: his sole on the floor where the recording's is
    for (const S of this.soles) { S.h = this.soleHeight(S); S.gh = this.soleHeight(S, gd); S.down = S.h <= this.onFloor && S.gh <= this.guFootUp; }
    const planted = this.soles.filter(S => S.down);
    // his centre of mass (dropped along gravity onto the floor) and how far inside what will hold him it is
    const c0 = K.toLocal([d.subtree_com[3 * hb], d.subtree_com[3 * hb + 1], d.subtree_com[3 * hb + 2]]);
    const gl = K.dirLocal([m.opt.gravity[0], m.opt.gravity[1], m.opt.gravity[2]]), tg = gl[2] < -1e-3 ? c0[2] / -gl[2] : 0;
    const com = [c0[0] + gl[0] * tg, c0[1] + gl[1] * tg];
    const hull = convexHull(spts);
    let comOver = -1;
    if (hull.length >= 3) comOver = insideDist(hull, com);
    else if (spts.length) comOver = -Math.min(...spts.map(p => Math.hypot(p[0] - com[0], p[1] - com[1])));
    // 2. hips
    const dz = hGh - hMe;
    let hor = [0, 0, 0];
    if (planted.length) {
      const fb = planted.map(S => W.bodyId[S.side + 'Foot']), mid = dd => { const p = [0, 0, 0]; for (const b of fb) for (let k = 0; k < 3; k++) p[k] += dd.xpos[3 * b + k] / fb.length; return p; };
      const me = mid(d), gh = mid(gd), t = [0, 1, 2].map(k => me[k] + gd.qpos[k] - gh[k] - d.qpos[k]), tu = t[0] * up[0] + t[1] * up[1] + t[2] * up[2];
      hor = t.map((v, k) => v - tu * up[k]);
    }
    // rising while his weight is outside what holds him: the hips go over it first, and up only a little
    let dzE = dz, acom = [0, 0, 0];
    const rising = dz > 0.02 && sup.other > 0;   // the recording lifts his hips while something besides feet/knees/hands still holds him
    if (rising && comOver < this.guComIn && spts.length) {
      const k = Math.max(0, Math.min(1, (comOver + 0.1) / (0.1 + this.guComIn)));
      dzE = Math.min(dz, 0.03 + dz * k * k);
      // toward the middle of his feet (or of all his supports)
      const fp = con.filter(q => q.kind === 'feet').map(q => K.toLocal(q.pos)), P = fp.length ? fp : spts.map(p => [p[0], p[1], 0]);
      const cx = P.reduce((a, p) => a + p[0], 0) / P.length, cy = P.reduce((a, p) => a + p[1], 0) / P.length;
      const wc = this.guWc, al = [wc * wc * (cx - com[0]), wc * wc * (cy - com[1]), 0], l = Math.hypot(al[0], al[1]);
      if (l > 3) { al[0] *= 3 / l; al[1] *= 3 / l; }
      const aw = K.toWorld(al); acom = [aw[0] - o[0], aw[1] - o[1], aw[2] - o[2]];
    }
    gu.wait = rising && dzE < dz - 0.01;
    const wr = this.guWr;
    for (let k = 0; k < 3; k++) qacc[k] = wr * wr * (dzE * up[k] + hor[k]) + 2 * wr * (this.gqd[k] - d.qvel[k]) + acom[k];
    {
      const q = [d.qpos[3], d.qpos[4], d.qpos[5], d.qpos[6]], g = [gd.qpos[3], gd.qpos[4], gd.qpos[5], gd.qpos[6]];
      const e = [q[0] * g[0] + q[1] * g[1] + q[2] * g[2] + q[3] * g[3], q[0] * g[1] - q[1] * g[0] - q[2] * g[3] + q[3] * g[2], q[0] * g[2] + q[1] * g[3] - q[2] * g[0] - q[3] * g[1], q[0] * g[3] - q[1] * g[2] + q[2] * g[1] - q[3] * g[0]];
      if (e[0] < 0) for (let k = 0; k < 4; k++) e[k] = -e[k];
      const sn = Math.hypot(e[1], e[2], e[3]), th = 2 * Math.atan2(sn, e[0]), f = sn > 1e-9 ? th / sn : 2, wo = this.guWo;
      for (let k = 0; k < 3; k++) qacc[3 + k] = wo * wo * e[1 + k] * f - 2 * wo * d.qvel[3 + k];
    }
    clampVec(qacc, 0, this.guMaxLin); clampVec(qacc, 3, this.guMaxAng);
    { const au = qacc[0] * up[0] + qacc[1] * up[1] + qacc[2] * up[2]; if (au > this.guUpMax) for (let k = 0; k < 3; k++) qacc[k] -= (au - this.guUpMax) * up[k]; }
    for (const S of planted) this.closeChain(S, qacc);
    // 3. inverse dynamics, contacts off
    const d2 = this.d2; d2.qpos.set(d.qpos.subarray(0, m.nq)); d2.qvel.set(d.qvel.subarray(0, nv)); d2.qacc.set(qacc);
    const fl = m.opt.disableflags; m.opt.disableflags = fl | this.dsbl; mj.mj_inverse(m, d2); m.opt.disableflags = fl;
    const needFull = Float64Array.from(d2.qfrc_inverse.subarray(0, nv));
    d2.qacc.fill(0); m.opt.disableflags = fl | this.dsbl; mj.mj_inverse(m, d2); m.opt.disableflags = fl;
    const need0 = Float64Array.from(d2.qfrc_inverse.subarray(0, nv)), needA = needFull.map((v, i) => v - need0[i]);
    // 4. what the box can give: planted soles at their bottom corners along the floor's normal, every other contact
    // along its own normal, within friction
    const cols = [], pushCol = (p, b, nn, u1, u2) => {
      mj.mj_jac(m, d, this.jp, this.jr, p, b); const J = this.jp.GetView();
      for (const [tt, s2] of [[u1, 1], [u1, -1], [u2, 1], [u2, -1]]) {
        const e = [nn[0] + this.mu * s2 * tt[0], nn[1] + this.mu * s2 * tt[1], nn[2] + this.mu * s2 * tt[2]], col = new Float64Array(nv);
        for (let v = 0; v < nv; v++) col[v] = J[v] * e[0] + J[nv + v] * e[1] + J[2 * nv + v] * e[2];
        cols.push(col);
      }
    };
    const ta = Math.abs(up[0]) < 0.9 ? [1, 0, 0] : [0, 1, 0], t1 = norm(cross(up, ta)), t2 = cross(up, t1), plantedB = new Set();
    for (const S of (this.guFeetCon ? [] : planted)) {
      for (const b of S.bodies) plantedB.add(b);
      const pts = [];
      for (const g of S.geoms) {
        const c = [d.geom_xpos[3 * g], d.geom_xpos[3 * g + 1], d.geom_xpos[3 * g + 2]], R = d.geom_xmat.subarray(9 * g, 9 * g + 9), sz = m.geom_size.subarray(3 * g, 3 * g + 3);
        for (let c8 = 0; c8 < 8; c8++) {
          const l = [(c8 & 1 ? 1 : -1) * sz[0], (c8 & 2 ? 1 : -1) * sz[1], (c8 & 4 ? 1 : -1) * sz[2]];
          const p = [0, 1, 2].map(r => c[r] + R[3 * r] * l[0] + R[3 * r + 1] * l[1] + R[3 * r + 2] * l[2]);
          pts.push({ p, h: K.toLocal(p)[2], b: m.geom_bodyid[g] });
        }
      }
      const lo = Math.min(...pts.map(q => q.h));
      for (const q of pts) if (q.h <= lo + 0.012) pushCol(q.p, q.b, up, t1, t2);
    }
    for (const q of con) if (!plantedB.has(q.b)) pushCol(q.pos, q.b, q.n, q.u1, q.u2);
    this.stats.contacts = cols.length / 4; this.stats.soles = planted.length; this.stats.hands = sup.hands;
    this.applyPlan(need0, needA, cols, this.guJw);
    // 5. the recording waits for him (kin.updateGetup reads getupLag), and its spot follows his hips along the floor
    const lag = Math.max(0, hGh - hMe);
    K.getupLag = gu.wait ? Math.max(lag, 0.2) : lag;
    if (K.anchor && this.guFollow > 0) {
      const A = K.anchor, me = K.toLocal([d.qpos[0], d.qpos[1], d.qpos[2]]), gh = K.toLocal([gd.qpos[0], gd.qpos[1], gd.qpos[2]]), k = 1 - Math.exp(-dt * this.guFollow);
      A.x += (me[0] - gh[0]) * k; A.y += (me[1] - gh[1]) * k;
    }
    const stalled = K.getupLag > 0.14;
    gu.stallT = stalled ? gu.stallT + dt : 0;
    const hEnd = clip ? clip.q(clip.n - 1)[2] : 0.9;
    gu.prog = Math.max(gu.prog, Math.max(0, Math.min(1, (hMe - gu.h0) / Math.max(0.1, hEnd - gu.h0))));
    this.getupInfo = { t: gu.t, support: { feet: sup.feet, hands: sup.hands, knees: sup.knees, other: sup.other }, comOver, height: hMe, stalled, stallT: gu.stallT, wait: gu.wait, progress: gu.prog };
    if (globalThis.PGDBG && this.stats.n % globalThis.PGDBG === 0) console.log('PG', d.time.toFixed(2), 'x', clip ? K.anim.x.toFixed(0) + '/' + clip.n : '-', 'h', hMe.toFixed(2), 'gh', hGh.toFixed(2), 'hEnd', hEnd.toFixed(2), 'com', comOver.toFixed(2), 'sup', JSON.stringify(sup), 'planted', planted.length, 'wait', gu.wait, 'sc', this.planSc, 'r', this.planR.toFixed(1), 'over', this.stats.over, 'resid', this.stats.resid.toFixed(0), JSON.stringify(this.overN || {})); this.overN = {};
    if (globalThis.PGJ) { const pv = this._pgv; if (pv && this.stats.n % globalThis.PGDBG === 0) console.log('PGA', d.time.toFixed(2), 'asked', pv.a.map(v => v.toFixed(1)).join(','), 'got', [0, 1, 2, 3, 4, 5].map(k => ((d.qvel[k] - pv.v[k]) / dt).toFixed(1)).join(','), 'jointErr', pv.j.toFixed(1)); let je = 0; for (const a of W.act) if (/Leg_x|Shin_x/.test(a.name)) je += Math.abs(qacc[a.dof]); this._pgv = { a: Array.from(qacc.slice(0, 6)), v: Array.from(d.qvel.slice(0, 6)), j: je }; }
    if (globalThis.PGJ && this.stats.n % globalThis.PGDBG === 0) console.log('PGJ', d.time.toFixed(2), 'qacc0-5', Array.from(qacc.slice(0, 6)).map(v => v.toFixed(1)).join(','), 'vel', Array.from(d.qvel.slice(0, 3)).map(v => v.toFixed(2)).join(','), W.act.filter(a => /Leg_x|Shin_x|Foot_x|Spine1_x|Chest_x/.test(a.name)).map(a => a.name.replace(/_/, '') + ':' + (d.qpos[a.qadr] * 57.3).toFixed(0) + '/' + (gd.qpos[a.qadr] * 57.3).toFixed(0) + '/' + d.actuator_force[W.act.indexOf(a)].toFixed(0) + '/' + this.hcap[W.act.indexOf(a)].toFixed(0)).join(' '));
    this.msNet += (performance.now() - t0 - this.msNet) * 0.1;
    let e = 0; for (let b = 0; b < this.bodyId.length; b++) e = Math.max(e, Math.hypot(this.pos[3 * b] - ref.pos[3 * b], this.pos[3 * b + 1] - ref.pos[3 * b + 1], this.pos[3 * b + 2] - ref.pos[3 * b + 2]));
    this.err = e; this.lost = 0;
    return this.targets;
  }

  async think(ref) {
    const W = this.W, m = W.m, d = W.d, mj = W.mj, nv = this.nv, dt = 1 / this.hz, gd = this.Gw.d;
    this.tNext = d.time + dt - 1e-9;
    this.readState();
    if (this.getupReady && this.kin.state === 'getup') return this.thinkGetup(ref);
    this.getupMode = false; this.gu = null;
    if (this.down) { for (let i = 0; i < ND; i++) this.targets[this.actIdx[i]] = d.qpos[this.hingeQ[i]]; return this.targets; }
    this.ghostVel(dt);
    const t0 = performance.now();
    const d2 = this.d2, qacc = new Float64Array(nv);
    // 1. accelerations toward the ghost
    const wn = this.wn, kp = wn * wn, kd = 2 * wn;
    const mj2 = this.maxJ;
    const goal = this.goal || (this.goal = new Float64Array(m.nq)); goal.set(gd.qpos.subarray(0, m.nq));
    // hands on a wall: the arms go for the drive's reach (sim/drive_brain.js physArms: wrist on the wood, palm flat)
    if (this.armW > 0 && this.armGoal && this.armJ) for (const j of this.armJ) { const qa = W.act[j].qadr; goal[qa] += (this.armGoal[qa] - goal[qa]) * this.armW; }
    for (const a of W.act) qacc[a.dof] = Math.max(-mj2, Math.min(mj2, kp * (goal[a.qadr] - d.qpos[a.qadr]) + kd * (this.gqd[a.qadr] - d.qvel[a.dof])));
    const K = this.kin, o = K.toWorld([0, 0, 0]), z = K.toWorld([0, 0, 1]), n = [z[0] - o[0], z[1] - o[1], z[2] - o[2]];   // the floor's normal
    // hips: the ghost's hips relative to the ghost's feet, put over his own feet
    const mid = (dd, ids) => { const p = [0, 0, 0]; for (const b of ids) for (let k = 0; k < 3; k++) p[k] += dd.xpos[3 * b + k] / ids.length; return p; };
    // which feet are planted: his sole is on the floor AND the director has that foot down (a stepping foot may lift)
    const dir = K.state === 'stand' || K.state === 'react';
    for (const S of this.soles) { const lk = K.locks && K.locks[S.side]; S.h = this.soleHeight(S); S.down = S.h <= this.onFloor && (!dir || !!(lk && lk.on)); }
    // neither foot planted by the director but a sole is on the floor (it let one foot go before the other landed): he
    // still stands on it
    if (!this.soles.some(S => S.down)) for (const S of this.soles) S.down = S.h <= this.onFloor;
    // the hips are placed relative to the planted feet (on one foot: that foot), his and the ghost's alike
    const st = this.soles.filter(S => S.down).map(S => W.bodyId[S.side + 'Foot']);
    const fb = st.length ? st : [this.footB.L, this.footB.R], me = mid(d, fb), gh = mid(gd, fb);
    const wr = this.wr;
    for (let k = 0; k < 3; k++) {
      const tgt = me[k] + gd.qpos[k] - gh[k];
      qacc[k] = wr * wr * (tgt - d.qpos[k]) + 2 * wr * (this.gqd[k] - d.qvel[k]);
    }
    // orientation: error rotation q_me^-1 * q_ghost, as a rotation vector in his hips' own frame (MuJoCo's free-joint
    // angular dofs are in the body frame)
    {
      const q = [d.qpos[3], d.qpos[4], d.qpos[5], d.qpos[6]], g = [gd.qpos[3], gd.qpos[4], gd.qpos[5], gd.qpos[6]];
      const e = [q[0] * g[0] + q[1] * g[1] + q[2] * g[2] + q[3] * g[3],
        q[0] * g[1] - q[1] * g[0] - q[2] * g[3] + q[3] * g[2],
        q[0] * g[2] + q[1] * g[3] - q[2] * g[0] - q[3] * g[1],
        q[0] * g[3] - q[1] * g[2] + q[2] * g[1] - q[3] * g[0]];
      if (e[0] < 0) for (let k = 0; k < 4; k++) e[k] = -e[k];
      const s = Math.hypot(e[1], e[2], e[3]), th = 2 * Math.atan2(s, e[0]), f = s > 1e-9 ? th / s : 2;
      const wo = this.wo * (this.soles.filter(S => S.down).length === 1 ? this.wo1 : 1);   // on one foot the hips may tip a little (a person's hip can't hold the pelvis level against everything)
      for (let k = 0; k < 3; k++) qacc[3 + k] = wo * wo * e[1 + k] * f - 2 * wo * d.qvel[3 + k];
      // turning about the floor's normal: only what planted feet can give (a slow follow; the director steps to turn)
      const zl = qv(qc(q), n), yaw = qacc[3] * zl[0] + qacc[4] * zl[1] + qacc[5] * zl[2];
      for (let k = 0; k < 3; k++) qacc[3 + k] -= (1 - this.yawK) * yaw * zl[k];
    }
    // no wild demands after a knock (an arm hitting a wall, a weld letting go): the floor can't give them anyway
    clampVec(qacc, 0, this.maxLin); clampVec(qacc, 3, this.maxAng);
    // the legs of feet on the floor: their joints accelerate so that the foot stays still while the hips do what was
    // asked (the leg's own wish toward the ghost only in what's left over). Otherwise the hips' wish and the legs' wish
    // fight, and the inverse dynamics plans for legs swinging in the air.
    for (const S of this.soles) {
      if (S.down && !this.freeLegs) this.closeChain(S, qacc);
      else if (dir && !this.freeLegs) this.closeChain(S, qacc, this.swingAcc(S, dt));   // a stepping foot goes where the ghost's goes
    }
    // 2. inverse dynamics, no contacts or welds: the total generalized force those accelerations need
    d2.qpos.set(d.qpos.subarray(0, m.nq)); d2.qvel.set(d.qvel.subarray(0, nv)); d2.qacc.set(qacc);
    const fl = m.opt.disableflags; m.opt.disableflags = fl | this.dsbl;
    mj.mj_inverse(m, d2);
    m.opt.disableflags = fl;
    const needFull = Float64Array.from(d2.qfrc_inverse.subarray(0, nv));
    // the same with no acceleration at all (gravity, speeds): what is left is the part we may scale down if it asks a
    // joint for more than a person's strength
    d2.qacc.fill(0); m.opt.disableflags = fl | this.dsbl; mj.mj_inverse(m, d2); m.opt.disableflags = fl;
    const need0 = Float64Array.from(d2.qfrc_inverse.subarray(0, nv)), needA = needFull.map((v, i) => v - need0[i]);
    // (user's snapshot V47: right shoulder leaning on the wall, his collarbone at its stop for 4 s while the recording
    // asked 14 deg less) the box pushing on parts of him the plan doesn't count as support (upper arm, shoulder, chest,
    // head on a wall): those pushes, as they are now, are loads his joints must hold - the collarbone holds the arm
    // against the wall instead of giving way to its stop (a person's shoulder girdle carries a lean into the ribcage)
    if (this.extLoad) this.externalLoads(need0, needFull);
    const need = needFull; this.need0 = need.slice(0, 6);
    // 3. the floor's share through his soles: each sole near the floor can be pushed at its corners, along the floor's
    // normal and within friction (a foot held by its weld is still a foot on the floor)
    const ta = Math.abs(n[0]) < 0.9 ? [1, 0, 0] : [0, 1, 0];
    const t1 = norm(cross(n, ta)), t2 = cross(n, t1);
    const cols = []; let nsole = 0; const supp = [];
    for (const S of this.soles) {
      const pts = [];
      for (const g of S.geoms) {
        const c = [d.geom_xpos[3 * g], d.geom_xpos[3 * g + 1], d.geom_xpos[3 * g + 2]], R = d.geom_xmat.subarray(9 * g, 9 * g + 9), sz = m.geom_size.subarray(3 * g, 3 * g + 3);
        for (let c8 = 0; c8 < 8; c8++) {
          const l = [(c8 & 1 ? 1 : -1) * sz[0], (c8 & 2 ? 1 : -1) * sz[1], (c8 & 4 ? 1 : -1) * sz[2]];
          const p = [0, 1, 2].map(r => c[r] + R[3 * r] * l[0] + R[3 * r + 1] * l[1] + R[3 * r + 2] * l[2]);
          pts.push({ p, h: K.toLocal(p)[2], b: m.geom_bodyid[g] });
        }
      }
      const lo = Math.min(...pts.map(q => q.h));
      if (!S.down) continue;
      nsole++;
      for (const q of pts) {
        if (q.h > lo + 0.012) continue;   // the bottom face only
        { const l = K.toLocal(q.p); supp.push([l[0], l[1]]); }
        mj.mj_jac(m, d, this.jp, this.jr, q.p, q.b);
        const J = this.jp.GetView();
        for (const [tt, sg] of [[t1, 1], [t1, -1], [t2, 1], [t2, -1]]) {
          const e = [n[0] + this.mu * sg * tt[0], n[1] + this.mu * sg * tt[1], n[2] + this.mu * sg * tt[2]];
          const col = new Float64Array(nv);
          for (let v = 0; v < nv; v++) col[v] = J[v] * e[0] + J[nv + v] * e[1] + J[2 * nv + v] * e[2];
          cols.push(col);
        }
      }
    }
    // hands (and forearms) on a wall or the floor: support too, where they really touch (rule 6: a hand on a wall is
    // support like a foot on the floor)
    let nh = 0;
    if (!this.noHands) for (let i = 0; i < d.ncon; i++) {
      const c = d.contact.get(i), g1 = c.geom1, g2 = c.geom2, dist = c.dist, pos = [c.pos[0], c.pos[1], c.pos[2]], fr = [0, 1, 2, 3, 4, 5, 6, 7, 8].map(k => c.frame[k]); c.delete();
      if (dist > 0.004) continue;
      const b1 = m.geom_bodyid[g1], b2 = m.geom_bodyid[g2];
      let b = -1, sg = 1;
      if (this.hands.has(b2) && !W.isFig[b1]) b = b2; else if (this.hands.has(b1) && !W.isFig[b2]) { b = b1; sg = -1; }
      if (b < 0) continue;
      nh++;
      const nn = [fr[0] * sg, fr[1] * sg, fr[2] * sg], u1 = [fr[3], fr[4], fr[5]], u2 = [fr[6], fr[7], fr[8]];
      mj.mj_jac(m, d, this.jp, this.jr, pos, b);
      const J = this.jp.GetView();
      for (const [tt, s2] of [[u1, 1], [u1, -1], [u2, 1], [u2, -1]]) {
        const e = [nn[0] + this.mu * s2 * tt[0], nn[1] + this.mu * s2 * tt[1], nn[2] + this.mu * s2 * tt[2]];
        const col = new Float64Array(nv);
        for (let v = 0; v < nv; v++) col[v] = J[v] * e[0] + J[nv + v] * e[1] + J[2 * nv + v] * e[2];
        cols.push(col);
      }
    }
    this.stats.soles = nsole; this.stats.hands = nh;
    // his REAL balance, for the director (coordination board, RB -> R3 14:20: the director decides steps from its own
    // pose; when his real body can't hold a lean the two drift apart and nobody steps): centre of mass, its speed, the
    // capture point (where his weight would have to go to stop), the support polygon of the planted soles, and how far
    // the capture point is inside it (negative = outside: he needs a step or a hand)
    {
      const hb = W.bodyId.Hips, c0 = K.toLocal([d.subtree_com[3 * hb], d.subtree_com[3 * hb + 1], d.subtree_com[3 * hb + 2]]);
      // dropped onto the floor along gravity (on a tilted floor that is not straight down the floor's normal)
      const gv = K.dirLocal([m.opt.gravity[0], m.opt.gravity[1], m.opt.gravity[2]]), tg = gv[2] < -1e-3 ? c0[2] / -gv[2] : 0;
      const c = [c0[0] + gv[0] * tg, c0[1] + gv[1] * tg, c0[2]];
      const pv = this.realBal && this.realBal.com, v = pv ? [0, 1].map(k => (c[k] - pv[k]) / dt) : [0, 0];
      const vs = this.realBal && this.realBal.v ? [0, 1].map(k => this.realBal.v[k] + (v[k] - this.realBal.v[k]) * 0.5) : v;
      const w0 = Math.sqrt(9.81 / Math.max(0.3, c[2])), cp = [c[0] + vs[0] / w0, c[1] + vs[1] / w0];
      const hull = convexHull(supp);
      this.realBal = { t: d.time, com: c, v: vs, cp, support: hull, margin: hull.length >= 3 ? insideDist(hull, cp) : -1, comMargin: hull.length >= 3 ? insideDist(hull, c) : -1 };
    }
    this.stats.contacts = cols.length / 4;
    const tau = this.applyPlan(need0, needA, cols);
    if (globalThis.PHJ && this.stats.n % globalThis.PHDBG === 0) console.log('PHJ', d.time.toFixed(2), W.act.filter(a => /Spine|Chest|Neck|Leg_x|Shin_x|Foot_x/.test(a.name)).map(a => a.name.replace(/_/, '') + ':' + (d.qpos[a.qadr] * 57.3).toFixed(0) + '/' + (goal[a.qadr] * 57.3).toFixed(0) + '/' + tau[a.dof].toFixed(0) + '/' + this.targets[W.act.indexOf(a)].toFixed(2)).join(' '));
    if (globalThis.PHDBG && this.stats.n % globalThis.PHDBG === 0) console.log('PH', d.time.toFixed(2), 'bal', this.realBal.margin.toFixed(2), this.realBal.comMargin.toFixed(2), 'soles', nsole, 'pts', cols.length / 4, 'need0', Array.from(this.need0).map(v => v.toFixed(0)).join(','), 'resid', Array.from(need.slice(0, 6)).map(v => v.toFixed(0)).join(','), 'root acc', Array.from(qacc.slice(0, 6)).map(v => v.toFixed(1)).join(','), 'hips', [0, 1, 2].map(k => (d.qpos[k] - me[k]).toFixed(2)).join(','), 'ghost', [0, 1, 2].map(k => (gd.qpos[k] - gh[k]).toFixed(2)).join(','), 'over', this.stats.over, 'g', Array.from(m.opt.gravity).map(v => v.toFixed(2)).join(','));
    this.msNet += (performance.now() - t0 - this.msNet) * 0.1;
    // how far he is from the ghost (the drive uses it to tell a fall)
    let e = 0; for (let b = 0; b < this.bodyId.length; b++) e = Math.max(e, Math.hypot(this.pos[3 * b] - ref.pos[3 * b], this.pos[3 * b + 1] - ref.pos[3 * b + 1], this.pos[3 * b + 2] - ref.pos[3 * b + 2]));
    this.err = e;
    this.lost = e > 0.5 ? this.lost + dt : Math.max(0, this.lost - dt);
    if (this.noRoom || (this.lost > 0.6 && this.pos[2] - this.ground < 0.6)) { this.down = true; this.setTone(0.04); }
    return this.targets;
  }
}

function solve6(A, b) {
  const n = 6, M = A.map((r, i) => [...r, b[i]]);
  for (let c = 0; c < n; c++) {
    let p = c; for (let r = c + 1; r < n; r++) if (Math.abs(M[r][c]) > Math.abs(M[p][c])) p = r;
    if (Math.abs(M[p][c]) < 1e-12) return null;
    [M[c], M[p]] = [M[p], M[c]];
    for (let r = 0; r < n; r++) if (r !== c) { const f = M[r][c] / M[c][c]; for (let k = c; k <= n; k++) M[r][k] -= f * M[c][k]; }
  }
  return M.map((r, i) => r[n] / r[i]);
}

// min |A w - b|^2 over the first `rows` rows (the hips' 6), w >= 0, small ridge, rows weighted; A given as columns.
// Prepared once per tick (A^T A), then solved for any b: accelerated projected gradient (FISTA), 150 iterations.
function nnlsPrep(cols, rows, ridge, rw = [1, 1, 1, 1, 1, 1]) {
  const K = cols.length;
  const AtA = new Float64Array(K * K);
  for (let i = 0; i < K; i++)
    for (let j = i; j < K; j++) { let s = 0; for (let r = 0; r < rows; r++) s += rw[r] * rw[r] * cols[i][r] * cols[j][r]; AtA[i * K + j] = s; AtA[j * K + i] = s; }
  let L = ridge; for (let i = 0; i < K * K; i++) L += AtA[i] * AtA[i]; L = Math.sqrt(L) + ridge;   // Frobenius bound
  return b => {
    const Atb = new Float64Array(K);
    for (let i = 0; i < K; i++) for (let r = 0; r < rows; r++) Atb[i] += rw[r] * rw[r] * cols[i][r] * b[r];
    const w = new Float64Array(K), y = new Float64Array(K), prev = new Float64Array(K);
    let t = 1;
    for (let it = 0; it < 150; it++) {
      prev.set(w);
      for (let i = 0; i < K; i++) {
        let g = -Atb[i] + ridge * y[i]; for (let j = 0; j < K; j++) g += AtA[i * K + j] * y[j];
        w[i] = Math.max(0, y[i] - g / L);
      }
      const tn = (1 + Math.sqrt(1 + 4 * t * t)) / 2, f = (t - 1) / tn; t = tn;
      for (let i = 0; i < K; i++) y[i] = w[i] + f * (w[i] - prev[i]);
    }
    return w;
  };
}
