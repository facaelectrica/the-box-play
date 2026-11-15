// The Box - is he balanced? (user 23:09: "Do you have ways of measuring that where the weight distribution goes
// along the legs and he's stable? Like, he often does positions that would make him fall in real life.")
//
// Measured on his real (physics) body, once per control tick, with nothing taken from the recording:
//  - his centre of mass, seen straight down along gravity (the plane across gravity, through nothing in particular:
//    only distances inside it matter);
//  - what holds him up: every part of him touching the box (feet, hands, knees, hips, a shoulder on a wall...) and
//    the force each carries along gravity (MuJoCo's contact forces), the parts that carry more than 5 N make the
//    support area (their contact points seen straight down, their outline);
//  - margin: how far his centre of mass is inside that outline (cm, negative = outside: he is falling, or something
//    not real holds him);
//  - capture point: where his weight would have to go to stop his current sway (centre of mass + its speed along the
//    plane * sqrt(height / g)); outside the outline = he can't stop without a step, a hand, or falling;
//  - load split: how much of his weight each foot, the hands, and the rest carry (sums to ~1 when he is still; more
//    = an impact, less = he is moving up/down or something else holds him).

const PARTS = [['LeftFoot', /^Left(Foot|ToeBase)$/], ['RightFoot', /^Right(Foot|ToeBase)$/], ['hands', /(Hand|ForeArm)$/], ['knees', /Shin$/]];

function hull(P) {
  const p = P.slice().sort((a, b) => a[0] - b[0] || a[1] - b[1]); if (p.length < 3) return p;
  const cr = (o, a, b) => (a[0] - o[0]) * (b[1] - o[1]) - (a[1] - o[1]) * (b[0] - o[0]), lo = [], up = [];
  for (const q of p) { while (lo.length >= 2 && cr(lo[lo.length - 2], lo[lo.length - 1], q) <= 0) lo.pop(); lo.push(q); }
  for (const q of p.reverse()) { while (up.length >= 2 && cr(up[up.length - 2], up[up.length - 1], q) <= 0) up.pop(); up.push(q); }
  return lo.slice(0, -1).concat(up.slice(0, -1));
}
// signed distance from x to the outline H (counter-clockwise): + inside, - outside (a point or a segment: - distance)
function margin(H, x) {
  if (!H.length) return -1;
  if (H.length === 1) return -Math.hypot(x[0] - H[0][0], x[1] - H[0][1]);
  let inside = H.length >= 3, best = 1e9;
  for (let i = 0; i < H.length; i++) {
    const a = H[i], b = H[(i + 1) % H.length], e = [b[0] - a[0], b[1] - a[1]], L2 = e[0] * e[0] + e[1] * e[1] || 1e-12;
    if (H.length >= 3 && e[0] * (x[1] - a[1]) - e[1] * (x[0] - a[0]) < 0) inside = false;
    const t = Math.max(0, Math.min(1, ((x[0] - a[0]) * e[0] + (x[1] - a[1]) * e[1]) / L2));
    best = Math.min(best, Math.hypot(x[0] - a[0] - e[0] * t, x[1] - a[1] - e[1] * t));
    if (H.length === 2) break;
  }
  return inside ? best : -best;
}

export class Balance {
  // opt.cap(jointName) -> a person's strength at that joint (N m; springs.js humanCap): for the effort measure
  constructor(W, opt = {}) {
    this.cap = opt.cap || null;
    this.W = W; const m = W.m, mj = W.mj;
    this.f6 = new mj.DoubleBuffer(6); this.prev = null; this.v = [0, 0, 0];
    this.hips = W.bodyId.Hips;
    this.part = new Int8Array(m.nbody).fill(-1);   // body -> which group (0..3), 4 = the rest of him, -1 = not him
    for (let b = 0; b < m.nbody; b++) {
      if (!W.isFig[b]) continue;
      const n = mj.mj_id2name(m, mj.mjtObj.mjOBJ_BODY.value, b) || '';
      const k = PARTS.findIndex(([, re]) => re.test(n)); this.part[b] = k >= 0 ? k : 4;
    }
    this.last = null;
  }
  // dt: time since the last call (for the centre of mass speed)
  // pins (sim/pins.js, optional): the welds that hold a planted foot or a hand; what they carry is counted apart
  update(dt, pins) {
    const W = this.W, m = W.m, d = W.d, mj = W.mj, g = m.opt.gravity, gn = Math.hypot(g[0], g[1], g[2]) || 9.81;
    const u = [-g[0] / gn, -g[1] / gn, -g[2] / gn];   // up
    // two directions across gravity
    const a0 = Math.abs(u[0]) < 0.9 ? [1, 0, 0] : [0, 1, 0];
    let e1 = [a0[1] * u[2] - a0[2] * u[1], a0[2] * u[0] - a0[0] * u[2], a0[0] * u[1] - a0[1] * u[0]]; const l1 = Math.hypot(...e1); e1 = e1.map(v => v / l1);
    const e2 = [u[1] * e1[2] - u[2] * e1[1], u[2] * e1[0] - u[0] * e1[2], u[0] * e1[1] - u[1] * e1[0]];
    const P = v => [v[0] * e1[0] + v[1] * e1[1] + v[2] * e1[2], v[0] * e2[0] + v[1] * e2[1] + v[2] * e2[2]];
    const U = v => v[0] * u[0] + v[1] * u[1] + v[2] * u[2];
    const h = this.hips, c = [d.subtree_com[3 * h], d.subtree_com[3 * h + 1], d.subtree_com[3 * h + 2]];
    if (this.prev && dt > 0) { const k = 1 - Math.exp(-dt / 0.05); for (let i = 0; i < 3; i++) this.v[i] += ((c[i] - this.prev[i]) / dt - this.v[i]) * k; }
    this.prev = c;
    const M = m.body_subtreemass[h], Wt = M * gn;
    const load = [0, 0, 0, 0, 0], pts = [], feetPts = [];   // feetPts: the support from his feet alone (for comparison)
    let low = 1e9;
    for (let i = 0; i < d.ncon; i++) {
      const cn = d.contact.get(i), g1 = cn.geom1, g2 = cn.geom2, dist = cn.dist, pos = [cn.pos[0], cn.pos[1], cn.pos[2]], fr = Array.from(cn.frame); cn.delete();
      if (dist > 0.002) continue;
      const b1 = m.geom_bodyid[g1], b2 = m.geom_bodyid[g2], f1 = W.isFig[b1], f2 = W.isFig[b2];
      if (f1 === f2) continue;   // self-contact or wall-wall
      mj.mj_contactForce(m, d, i, this.f6); const f = this.f6.GetView();
      const fw = [0, 1, 2].map(k => fr[k] * f[0] + fr[3 + k] * f[1] + fr[6 + k] * f[2]);
      const sg = f2 ? 1 : -1, b = f2 ? b2 : b1;   // the force on geom2 is +f (on geom1 -f)
      const up = sg * U(fw); const k = this.part[b]; if (up > 0) load[k] += up;
      // a support point: anything pressing on the box (a hand on a wall pushes sideways and still holds him up)
      if (Math.abs(f[0]) > 5) { pts.push(P(pos)); low = Math.min(low, U(pos)); if (k === 0 || k === 1) feetPts.push(P(pos)); }
      if (globalThis.BALDBG && globalThis.BALDBG(d.time)) console.log('BC', d.time.toFixed(3), mj.mj_id2name(m, mj.mjtObj.mjOBJ_BODY.value, b), mj.mj_id2name(m, mj.mjtObj.mjOBJ_BODY.value, f2 ? b1 : b2), 'fn', f[0].toFixed(0), 'up', up.toFixed(0), 'P', P(pos).map(v => (v * 100).toFixed(1)).join(','));
    }
    // the welds (sim/pins.js): a pinned hand or foot is held by a weld instead of (hands: as well as) its contacts.
    // What a weld does that a touch could do counts as support: pushing into the surface, sideways within friction
    // (0.8), turning within the sole/palm (10 cm x the press). The rest - pulling, gripping past friction, twisting -
    // is something a real hand or foot on a flat surface can't do ('held'): it is reported apart.
    let held = 0, heldT = 0, pinUp = 0, pinT = 0;
    if (pins && pins.on) { const gup = pins.ground().up;
      for (const n in pins.on) {
        const P0 = pins.on[n], f = pins.force(n), nrm = P0.out || gup, b = pins.id[n], k = /Foot/.test(n) ? (n[0] === 'L' ? 0 : 1) : 2;
        const fn = f[0] * nrm[0] + f[1] * nrm[1] + f[2] * nrm[2], ft = Math.hypot(f[0] - fn * nrm[0], f[1] - fn * nrm[1], f[2] - fn * nrm[2]);
        const N = Math.max(0, fn) + load[k] / Math.max(1, Object.keys(pins.on).filter(x => (/Foot/.test(x) ? (x[0] === 'L' ? 0 : 1) : 2) === k).length);
        const up = U(f); if (up > 0) load[k] += up; pinUp += up; pinT += Math.hypot(f[3], f[4], f[5]);
        if (fn > 5) { const p = pins.pos(b); pts.push(P(p)); low = Math.min(low, U(p) - 0.05); if (k !== 2) feetPts.push(P(p)); }
        held += Math.max(0, -fn) + Math.max(0, ft - 0.8 * N); heldT += Math.max(0, Math.hypot(f[3], f[4], f[5]) - 0.1 * N);
        if (globalThis.BALDBG && globalThis.BALDBG(d.time)) console.log('PIN', n, 'fn', fn.toFixed(0), 'ft', ft.toFixed(0), 'N', N.toFixed(0), 'T', Math.hypot(f[3], f[4], f[5]).toFixed(0));
      } }
    const H = hull(pts), x = P(c), vh = P(this.v), z = U(c) - (low < 1e8 ? low : U(c) - 1);
    const w = Math.sqrt(gn / Math.max(0.2, z)), cp = [x[0] + vh[0] / w, x[1] + vh[1] / w];
    if (globalThis.BALDBG && globalThis.BALDBG(d.time)) { let mm = 0; const cc = [0, 0, 0]; for (let b = 0; b < m.nbody; b++) if (W.isFig[b]) { mm += m.body_mass[b]; for (let k = 0; k < 3; k++) cc[k] += m.body_mass[b] * d.xipos[3 * b + k]; } console.log('MANUAL', P(cc.map(v => v / mm)).map(v => (v * 100).toFixed(1)).join(','), mm.toFixed(1), M.toFixed(1), 'hipsId', h, 'qfrc_app root', Array.from(d.qfrc_applied.slice(0, 6)).map(v => v.toFixed(0)).join(','), 'xfrc', Array.from(d.xfrc_applied).reduce((a, v) => a + Math.abs(v), 0).toFixed(0)); }
    if (globalThis.BALDBG && globalThis.BALDBG(d.time)) console.log('BCOM', P(c).map(v => (v * 100).toFixed(1)).join(','), 'hull', JSON.stringify(H.map(p => p.map(v => +(v * 100).toFixed(1)))));
    const r = {
      margin: pts.length ? margin(H, x) : -1, cpMargin: pts.length ? margin(H, cp) : -1, cpMarginFeet: feetPts.length ? margin(hull(feetPts), cp) : -1,
      L: load[0] / Wt, R: load[1] / Wt, hands: load[2] / Wt, knees: load[3] / Wt, rest: load[4] / Wt,
      speed: Math.hypot(...vh), n: pts.length, pins: pinUp / Wt, pinTorque: pinT, held: held / Wt, heldTorque: heldT,
      // for drawing: the outline and the two points, as world points at the lowest contact's height
      draw: () => { const o = low < 1e8 ? low : U(c) - 1, toW = p => [0, 1, 2].map(k => p[0] * e1[k] + p[1] * e2[k] + o * u[k]); return { hull: H.map(toW), com: toW(x), cp: toW(cp) }; },
    };
    if (this.cap) Object.assign(r, this.effort());
    this.last = r;
    return r;
  }
  // (user 23:17: "keeping your body stiff like this image is a lot of energy") how hard his muscles work: each joint's
  // torque in the last physics step (servo + every extra torque on that hinge = qfrc_smooth + qfrc_bias), against a
  // person's strength there. eff: the average share over all his joints (standing relaxed ~0.05-0.1), effMax: the
  // hardest-working joint's share (over 1 = more than a strong person can do), effJ: which joint, effNm: the sum (N m).
  effort() {
    const W = this.W, m = W.m, d = W.d;
    if (!this.caps) this.caps = W.act.map(a => this.cap(a.name));
    let s = 0, mx = 0, mj = -1, nm = 0;
    W.act.forEach((a, i) => {
      const tau = Math.abs(d.qfrc_smooth[a.dof] + d.qfrc_bias[a.dof]), sh = tau / this.caps[i];
      s += sh; nm += tau; if (sh > mx) { mx = sh; mj = i; }
    });
    return { eff: s / W.act.length, effMax: mx, effJ: mj >= 0 ? W.act[mj].name : '', effNm: nm };
  }
}
