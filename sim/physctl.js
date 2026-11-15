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
import { humanCap } from './springs.js?v=44';

const cross = (a, b) => [a[1] * b[2] - a[2] * b[1], a[2] * b[0] - a[0] * b[2], a[0] * b[1] - a[1] * b[0]];
const qv = (q, v) => { const [w, x, y, z] = q, tx = 2 * (y * v[2] - z * v[1]), ty = 2 * (z * v[0] - x * v[2]), tz = 2 * (x * v[1] - y * v[0]); return [v[0] + w * tx + (y * tz - z * ty), v[1] + w * ty + (z * tx - x * tz), v[2] + w * tz + (x * ty - y * tx)]; };
const qc = q => [q[0], -q[1], -q[2], -q[3]];
function clampVec(a, i, mx) { const l = Math.hypot(a[i], a[i + 1], a[i + 2]); if (l > mx) for (let k = 0; k < 3; k++) a[i + k] *= mx / l; }
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
    this.jp6 = new mj.DoubleBuffer(3 * m.nv); this.jr6 = new mj.DoubleBuffer(3 * m.nv);   // a sole this close to the floor (m) counts as standing on it (the welds hold feet ~1 cm up)
    this.d2 = new mj.MjData(m);
    this.jp = new mj.DoubleBuffer(3 * m.nv); this.jr = new mj.DoubleBuffer(3 * m.nv);
    this.wn = opt.wn ?? 11;          // joints: natural frequency (rad/s) of the pull toward the ghost's angle
    this.wr = opt.wr ?? 5;           // hips: position over the feet
    this.wo = opt.wo ?? 9;           // hips: orientation
    this.mu = opt.mu ?? 0.6;         // friction used when sharing the load (the floor's real friction is 1.0: a margin)
    this.gain = opt.gain ?? 0.5;
    this.yawK = opt.yawK ?? 0.15;    // how much of the ghost's turning the hips are asked to follow while the feet stay planted
    this.maxLin = opt.maxLin ?? 3;   // m/s^2 and rad/s^2: the most the hips are asked to accelerate
    this.maxAng = opt.maxAng ?? 15;
    this.noHands = opt.noHands ?? false;
    this.freeLegs = opt.freeLegs ?? false;
    this.maxJ = opt.maxJ ?? 60;      // rad/s^2 per joint (a hand stopped by a wall while the ghost's hand goes on: not a reason to heave the trunk)
    this.hands = new Set(['RightHand', 'LeftHand', 'RightForeArm', 'LeftForeArm'].map(n => W.bodyId[n]));     // servo stiffness between ticks, as a fraction of the body file's gains
    this.hcap = W.act.map(a => humanCap(a.name));
    this.footB = { L: W.bodyId.LeftFoot, R: W.bodyId.RightFoot };
    this.prevGq = null; this.gqd = new Float64Array(m.nq);
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

  soleHeight(S) {
    const W = this.W, m = W.m, d = W.d, K = this.kin; let lo = 9;
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
  closeChain(S, qacc) {
    const W = this.W, m = W.m, d = W.d, mj = W.mj, nv = this.nv, b = W.bodyId[S.side + 'Foot'], L = S.legDofs;
    mj.mj_jacBody(m, d, this.jp6, this.jr6, b);
    const P = this.jp6.GetView(), R = this.jr6.GetView();
    const row = (r, v) => r < 3 ? P[r * nv + v] : R[(r - 3) * nv + v];
    const J = [], rhs = new Float64Array(6);
    for (let r = 0; r < 6; r++) {
      let s = 0; for (let v = 0; v < 6; v++) s += row(r, v) * qacc[v];
      for (const v of L) s += row(r, v) * qacc[v];   // minus what the leg's own wish already does
      rhs[r] = -s; J.push(L.map(v => row(r, v)));
    }
    // dx = J^T (J J^T + l I)^-1 rhs
    const A = [...Array(6)].map((_, i) => [...Array(6)].map((_, j) => J[i].reduce((a, x, k) => a + x * J[j][k], 0) + (i === j ? 1e-4 : 0)));
    const y = solve6(A, rhs);
    if (!y) return;
    L.forEach((v, k) => { let s = 0; for (let r = 0; r < 6; r++) s += J[r][k] * y[r]; qacc[v] += s; });
  }

  async think(ref) {
    const W = this.W, m = W.m, d = W.d, mj = W.mj, nv = this.nv, dt = 1 / this.hz, gd = this.Gw.d;
    this.tNext = d.time + dt - 1e-9;
    this.readState();
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
    const fb = [this.footB.L, this.footB.R], me = mid(d, fb), gh = mid(gd, fb);
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
      const wo = this.wo;
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
      S.down = this.soleHeight(S) <= this.onFloor;
      if (S.down && !this.freeLegs) this.closeChain(S, qacc);
    }
    // 2. inverse dynamics, no contacts or welds: the total generalized force those accelerations need
    d2.qpos.set(d.qpos.subarray(0, m.nq)); d2.qvel.set(d.qvel.subarray(0, nv)); d2.qacc.set(qacc);
    const fl = m.opt.disableflags; m.opt.disableflags = fl | this.dsbl;
    mj.mj_inverse(m, d2);
    m.opt.disableflags = fl;
    const need = Float64Array.from(d2.qfrc_inverse.subarray(0, nv)); this.need0 = need.slice(0, 6);
    // 3. the floor's share through his soles: each sole near the floor can be pushed at its corners, along the floor's
    // normal and within friction (a foot held by its weld is still a foot on the floor)
    const ta = Math.abs(n[0]) < 0.9 ? [1, 0, 0] : [0, 1, 0];
    const t1 = norm(cross(n, ta)), t2 = cross(n, t1);
    const cols = []; let nsole = 0;
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
      if (lo > this.onFloor) continue;
      nsole++;
      for (const q of pts) {
        if (q.h > lo + 0.012) continue;   // the bottom face only
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
    this.stats.contacts = cols.length / 4;
    const w = cols.length ? nnls(cols, need, 6, 1e-6) : null;
    const tau = need;   // reuse: joint torques = need - floor's push
    if (w) for (let k = 0; k < cols.length; k++) if (w[k] > 0) { const c = cols[k]; for (let v = 0; v < nv; v++) tau[v] -= w[k] * c[v]; }
    let resid = 0; for (let v = 0; v < 6; v++) resid += tau[v] * tau[v];
    this.stats.resid += (Math.sqrt(resid) - this.stats.resid) * 0.1;
    // 4 + 5. capped at a person's strength, handed to the servos
    let over = 0, tsum = 0;
    W.act.forEach((a, i) => {
      let t = tau[a.dof]; const H = this.hcap[i];
      if (Math.abs(t) > H) { over++; t = Math.sign(t) * H; }
      tsum += Math.abs(t);
      const kpa = m.actuator_gainprm[10 * i], kda = -m.actuator_biasprm[10 * i + 2];
      this.targets[i] = d.qpos[a.qadr] + (t + kda * d.qvel[a.dof]) / Math.max(1e-6, kpa);
    });
    this.stats.over = over; this.stats.tau = tsum; this.stats.n++;
    if (globalThis.PHJ && this.stats.n % globalThis.PHDBG === 0) console.log('PHJ', d.time.toFixed(2), W.act.filter(a => /Spine|Chest|Neck|Leg_x|Shin_x|Foot_x/.test(a.name)).map(a => a.name.replace(/_/, '') + ':' + (d.qpos[a.qadr] * 57.3).toFixed(0) + '/' + (goal[a.qadr] * 57.3).toFixed(0) + '/' + tau[a.dof].toFixed(0) + '/' + this.targets[W.act.indexOf(a)].toFixed(2)).join(' '));
    if (globalThis.PHDBG && this.stats.n % globalThis.PHDBG === 0) console.log('PH', d.time.toFixed(2), 'soles', nsole, 'pts', cols.length / 4, 'need0', Array.from(this.need0).map(v => v.toFixed(0)).join(','), 'resid', Array.from(need.slice(0, 6)).map(v => v.toFixed(0)).join(','), 'root acc', Array.from(qacc.slice(0, 6)).map(v => v.toFixed(1)).join(','), 'hips', [0, 1, 2].map(k => (d.qpos[k] - me[k]).toFixed(2)).join(','), 'ghost', [0, 1, 2].map(k => (gd.qpos[k] - gh[k]).toFixed(2)).join(','), 'over', over, 'g', Array.from(m.opt.gravity).map(v => v.toFixed(2)).join(','));
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

// min |A w - b|^2 over the first `rows` rows (the hips' 6), w >= 0, small ridge; A given as columns. Accelerated
// projected gradient (FISTA), ~150 iterations: tiny (6 x a few dozen).
function nnls(cols, b, rows, ridge) {
  const K = cols.length;
  const AtA = new Float64Array(K * K), Atb = new Float64Array(K);
  for (let i = 0; i < K; i++) {
    for (let r = 0; r < rows; r++) Atb[i] += cols[i][r] * b[r];
    for (let j = i; j < K; j++) { let s = 0; for (let r = 0; r < rows; r++) s += cols[i][r] * cols[j][r]; AtA[i * K + j] = s; AtA[j * K + i] = s; }
  }
  let L = ridge; for (let i = 0; i < K * K; i++) L += AtA[i] * AtA[i]; L = Math.sqrt(L) + ridge;   // Frobenius bound
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
}
