// Pins: supporting hands and feet stay where they were put, and his body pivots over them.
// (user 20:24: "When we put our hands on the wall, we put the whole palm ... it should work like IK almost, where the
// hand is able to stay and the rest of the body pivots. The same with the feet ... the IK/FK that we see in animation
// where the foot stays in place.")
//
// In the physics: one MuJoCo weld per hand and per foot (added to the body file, off until needed). A pinned foot is
// held where it landed with the whole sole flat on the ground; a pinned hand is held where it touched the wall with the
// whole palm flat on the wood (its physical blob doesn't touch the wall meanwhile: the palm itself is what's placed).
// The weld is soft (it settles flat in about a tenth of a second) and only gives what real contact could give:
//  - it lets go when it would have to PULL (a sole or a palm can only push),
//  - it lets go when it would need more grip sideways than the wood/floor has (friction),
//  - a foot lets go when it would have to hold a turn bigger than his weight can hold over the sole (he would roll
//    onto the edge of the foot); a hand when the arm can't reach it any more.
// Feet are pinned while the recording has that foot planted (so steps still lift it); hands while he braces.
const SIDES = ['Left', 'Right'];
const PARTS = ['LeftFoot', 'RightFoot', 'LeftHand', 'RightHand'];
const SOLE = 0.045;      // foot origin above the ground when the sole is flat (the foot and toe boxes' bottom)
const PALM = 0.022;      // wrist origin off the wood with the palm flat on it

export function addPins(xml) {
  if (xml.includes('pin_LeftFoot')) return xml;
  // hands softer than feet: a braced arm gives a little (elbow and shoulder yield); stiff hands and stiff feet together
  // fought each other through his body (hands carrying 100%+ of his weight, feet slipping)
  const w = PARTS.map(b => `<weld name="pin_${b}" body1="${b}" active="false" solref="${/Hand/.test(b) ? (globalThis.HANDSOL || '0.06 1') : '0.035 1'}"/>`).join('\n  ');
  return xml.replace('</mujoco>', `<equality>\n  ${w}\n</equality>\n</mujoco>`);
}

const qmul = (a, b) => [a[0] * b[0] - a[1] * b[1] - a[2] * b[2] - a[3] * b[3], a[0] * b[1] + a[1] * b[0] + a[2] * b[3] - a[3] * b[2], a[0] * b[2] - a[1] * b[3] + a[2] * b[0] + a[3] * b[1], a[0] * b[3] + a[1] * b[2] - a[2] * b[1] + a[3] * b[0]];
const qconj = a => [a[0], -a[1], -a[2], -a[3]];
const qrot = (q, v) => { const r = qmul(qmul(q, [0, v[0], v[1], v[2]]), qconj(q)); return [r[1], r[2], r[3]]; };
const qnorm = q => { const n = Math.hypot(...q) || 1; return q.map(v => v / n); };
const dot = (a, b) => a[0] * b[0] + a[1] * b[1] + a[2] * b[2];
const cross = (a, b) => [a[1] * b[2] - a[2] * b[1], a[2] * b[0] - a[0] * b[2], a[0] * b[1] - a[1] * b[0]];
// the shortest turn taking unit vector a onto unit vector b
const qslerp = (a, b, u) => { let c = a[0] * b[0] + a[1] * b[1] + a[2] * b[2] + a[3] * b[3]; if (c < 0) { b = b.map(v => -v); c = -c; }
  if (c > 0.9995) return qnorm(a.map((v, i) => v + (b[i] - v) * u));
  const th = Math.acos(c), s = Math.sin(th), wa = Math.sin((1 - u) * th) / s, wb = Math.sin(u * th) / s; return a.map((v, i) => v * wa + b[i] * wb); };
const qbetween = (a, b) => { const c = cross(a, b), w = 1 + dot(a, b); return w < 1e-6 ? null : qnorm([w, c[0], c[1], c[2]]); };

export class Pins {
  constructor(W, kin) {
    this.W = W; this.kin = kin; const m = W.m, mj = W.mj;
    this.eq = {}; this.on = {}; this.cool = {}; this.bad = {}; this.stats = { pins: 0, letGo: {} };
    for (const n of PARTS) this.eq[n] = mj.mj_name2id(m, mj.mjtObj.mjOBJ_EQUALITY.value, 'pin_' + n);
    this.ok = PARTS.every(n => this.eq[n] >= 0);
    this.id = {}; for (const n of [...PARTS, 'LeftToeBase', 'RightToeBase', 'LeftArm', 'RightArm', 'LeftForeArm', 'RightForeArm']) this.id[n] = W.bodyId[n];
    // the hand's own blob(s): no contact with the wood while the palm is pinned
    this.handGeoms = {};
    for (const s of SIDES) { this.handGeoms[s] = []; for (let g = 0; g < m.ngeom; g++) if (m.geom_bodyid[g] === this.id[s + 'Hand']) this.handGeoms[s].push({ g, ct: m.geom_contype[g], ca: m.geom_conaffinity[g] }); }
    this.cfb = new mj.DoubleBuffer(6);
  }
  reset() { for (const n of PARTS) this.release(n, null); this.cool = {}; }
  // the ground's up (world) and a point on the ground
  ground() { const K = this.kin, o = K.toWorld([0, 0, 0]), u = K.toWorld([0, 0, 1]); return { o, up: [u[0] - o[0], u[1] - o[1], u[2] - o[2]] }; }
  pos(b) { const d = this.W.d; return [d.xpos[3 * b], d.xpos[3 * b + 1], d.xpos[3 * b + 2]]; }
  quat(b) { const d = this.W.d; return [d.xquat[4 * b], d.xquat[4 * b + 1], d.xquat[4 * b + 2], d.xquat[4 * b + 3]]; }
  setPose(n, p, q) { const m = this.W.m, e = this.eq[n]; m.eq_data.set([p[0], p[1], p[2], 0, 0, 0, ...qconj(q), 1], e * (m.eq_data.length / m.neq)); }
  pin(n, p, q, extra) {
    const d = this.W.d, e = this.eq[n];
    this.setPose(n, p, q);
    d.eq_active[e] = 1;
    this.on[n] = { p, q, t: 0, pull: 0, slip: 0, roll: 0, ...extra }; this.stats.pins++;
  }
  anchor(n, p) { const m = this.W.m, e = this.eq[n], k = e * (m.eq_data.length / m.neq); m.eq_data[k] = p[0]; m.eq_data[k + 1] = p[1]; m.eq_data[k + 2] = p[2]; }
  release(n, why) {
    const d = this.W.d, e = this.eq[n];
    if (e >= 0) d.eq_active[e] = 0;
    if (this.on[n] && why) { this.stats.letGo[why] = (this.stats.letGo[why] || 0) + 1; this.cool[n] = 0.25; }
    if (this.on[n] && /Hand/.test(n)) this.handOff = { ...(this.handOff || {}), [n.replace('Hand', '')]: 0.4 };
    delete this.on[n];
  }
  // the pin's force on the part (world, N) and its turning force (world, N m)
  force(n) {
    const d = this.W.d, e = this.eq[n], f = [0, 0, 0, 0, 0, 0]; let k = 0;
    for (let r = 0; r < d.nefc && k < 6; r++) if (d.efc_type[r] === 0 && d.efc_id[r] === e) f[k++] = d.efc_force[r];
    return f;
  }
  // the floor's/wall's own push on a body (contacts with the box), along n
  contactN(bodies, n) {
    const W = this.W, m = W.m, d = W.d, mj = W.mj; let s = 0;
    for (let i = 0; i < d.ncon; i++) {
      const c = d.contact.get(i), g1 = c.geom1, g2 = c.geom2, fr = Array.from(c.frame); c.delete();
      const b1 = m.geom_bodyid[g1], b2 = m.geom_bodyid[g2], in1 = bodies.includes(b1), in2 = bodies.includes(b2);
      if (in1 === in2 || (in1 ? W.isFig[b2] : W.isFig[b1])) continue;
      mj.mj_contactForce(m, d, i, this.cfb); const fn = this.cfb.GetView()[0];
      const sg = in2 ? 1 : -1;   // the frame's normal points from geom1 to geom2
      s += fn * sg * dot(fr, n);
    }
    return s;
  }
  // once per control tick (30 Hz), before the physics steps
  update(dt) {
    if (!this.ok || this.off) return;
    const K = this.kin, W = this.W, d = W.d, m = W.m, G = this.ground();
    for (const n in this.cool) this.cool[n] = Math.max(0, this.cool[n] - dt);
    // standing on that ground at all: the ground still under him (gravity within ~50 deg of straight into it). When
    // the box turns over, nothing holds his feet to what is becoming a wall or the ceiling.
    const g = m.opt.gravity, gl = Math.hypot(g[0], g[1], g[2]) || 1, into = -(g[0] * G.up[0] + g[1] * G.up[1] + g[2] * G.up[2]) / gl;
    const upright = (K.state === 'stand' || K.state === 'react') && into > 0.65;
    // ---------- feet
    for (const s of SIDES) {
      const n = s + 'Foot', b = this.id[n], toe = this.id[s + 'ToeBase'];
      const lk = K.locks && K.locks[s], want = upright && lk && lk.on;
      const P = this.on[n];
      if (P) {
        // the recording lifts that foot (a step): let go - but not for a flicker of its contact (0.1 s)
        P.unwant = want ? 0 : (P.unwant || 0) + dt;
        if (!upright || P.unwant > 0.1) { this.release(n, 'step'); if (lk) lk.pin = null; continue; }
        P.t += dt;
        const f = this.force(n), fn = dot(f, G.up), ft = Math.hypot(...[0, 1, 2].map(i => f[i] - fn * G.up[i]));
        const C = this.contactN([b, toe], G.up), N = Math.max(0, C + Math.max(0, fn));
        const tq = [f[3], f[4], f[5]], tn = dot(tq, G.up), th = Math.hypot(...[0, 1, 2].map(i => tq[i] - tn * G.up[i]));
        // up and down is the floor's job, not the pin's: the pin follows the foot's height (it holds it along the
        // ground and flat), so it never presses it into the floor or holds it down; the foot coming up = he lifts it
        const pn = this.pos(b), hNow = dot([pn[0] - P.p[0], pn[1] - P.p[1], pn[2] - P.p[2]], G.up);
        if (hNow > 0.015) { this.release(n, 'foot lift'); if (lk) lk.pin = null; continue; }
        this.anchor(n, [0, 1, 2].map(i => P.p[i] + hNow * G.up[i]));
        P.pull = fn + C < -(globalThis.PIN_PULL ?? 12) ? P.pull + dt : 0;             // floor + pin holding the foot DOWN (a sole can't)
        P.slip = ft > 0.8 * N + (globalThis.PIN_SLIP ?? 15) ? P.slip + dt : 0;         // more sideways grip than the floor has
        P.roll = th > 0.11 * N + (globalThis.PIN_ROLL ?? 8) ? P.roll + dt : 0;        // his weight past the sole's edge: it would roll
        // (Version 26; balance measure: the pins held him with up to 15 N m / 25 N / 20 N more than a sole on wood can) halved
        const why = P.pull > 0.08 ? 'pull' : P.slip > 0.08 ? 'slip' : P.roll > 0.12 ? 'roll' : null;
        if (why) { this.release(n, 'foot ' + why); if (lk) lk.pin = null; continue; }
        if (lk) lk.pin = { p: P.floor || P.p, q: P.q };   // the recording's planted foot = where his foot is pinned, on the floor
        continue;
      }
      if (!want || this.cool[n] > 0) continue;
      // put it down: near the ground (sole within 2.5 cm), not moving much
      const p = this.pos(b), h = dot([p[0] - G.o[0], p[1] - G.o[1], p[2] - G.o[2]], G.up) - SOLE;
      const vel = this.vel(b);
      // (user 21:40: "the feet are always a bit too high") only once the sole is really down (within 1 cm; was 2.5 cm)
      if (h > (globalThis.PIN_H ?? 0.025) || Math.hypot(...vel) > 0.35) continue;
      // flat: his foot turned the shortest way so its rest "up" (local z) is the ground's up; on the ground
      const q = this.quat(b), uz = qrot(q, [0, 0, 1]), r = qbetween(uz, G.up); if (!r) continue;
      const tq = qnorm(qmul(r, q)), tp = p.slice();   // where it is (the floor itself keeps it out of the wood)
      // (user 21:40: "the feet are always a bit too high") the recording's planted foot goes ON the floor (sole down),
      // not where his foot happened to be when pinned (up to 2.5 cm above it): the legs then put it down
      const floor = tp;   // (putting the target on the floor made the legs fight the pin: pushes ended in a fall)
      this.pin(n, tp, tq, { floor });
      if (lk) lk.pin = { p: floor, q: tq };
    }
    // ---------- hands on the floor while he gets up (user 21:16: "they use their arms as levers to get up ... this
    // guy is just waving them around"): where the recording has that hand on the floor (or puts it there within 0.3 s)
    // and his real hand is down there (wrist within 7 cm of touching, palm within 35 deg of facing the floor), it is
    // pinned flat where it is; his body then pushes up over it. Let go when the recording lifts the hand.
    if (this.handsOff) return;
    if (this.handOff) for (const s in this.handOff) { this.handOff[s] -= dt; }
    const gettingUp = K.state === 'getup' && into > 0.65 && K.anim && K.anim.clip;
    for (const s of SIDES) {
      const n = s + 'Hand', b = this.id[n], P = this.on[n];
      if (P && !P.ground) continue;
      const k = s === 'Left' ? 2 : 3, rec = gettingUp && (K.anim.clip.contact(K.anim.x, k) > 0.5 || K.anim.clip.contact(K.anim.x + 12, k) > 0.5 || K.anim.clip.contact(K.anim.x + 24, k) > 0.5);
      if (P) {
        P.unwant = rec ? 0 : (P.unwant || 0) + dt;
        if (!gettingUp || P.unwant > 0.1) { this.release(n, 'hand lifts'); continue; }
        P.t += dt;
        if (P.to && P.k < 1) { P.k = Math.min(1, P.k + dt / 0.12); const u = P.k * P.k * (3 - 2 * P.k);
          P.p = [0, 1, 2].map(i => P.from.p[i] + (P.to.p[i] - P.from.p[i]) * u); P.q = qslerp(P.from.q, P.to.q, u); this.setPose(n, P.p, P.q); continue; }
        // (snapshot 23:20: "Left arm is doing an impossible bend. The arm would be broken") the hand stays put while
        // his body turns around it during the get-up: once the arm is forced against its stops (wrist bent to 77 deg,
        // upper arm twisted 92 deg), the hand lets go and is placed again
        P.awk = this.armStops(s) ? (P.awk || 0) + dt : 0;
        if (P.awk > 0.12) { this.release(n, 'floor hand awkward'); this.cool[n] = 0.5; continue; }
        const f = this.force(n), fo = dot(f, G.up), ft = Math.hypot(...[0, 1, 2].map(i => f[i] - fo * G.up[i]));
        P.pull = fo < -30 ? P.pull + dt : 0;
        P.slip = ft > 0.8 * Math.max(0, fo) + 25 ? P.slip + dt : 0;
        const why = P.pull > 0.15 ? 'pull' : P.slip > 0.15 ? 'slip' : null;
        if (why) this.release(n, 'floor hand ' + why);
        continue;
      }
      if (!rec || this.cool[n] > 0) continue;
      const p = this.pos(b), gap = dot([p[0] - G.o[0], p[1] - G.o[1], p[2] - G.o[2]], G.up);
      if (gap > PALM + 0.07) continue;
      const q0 = this.quat(b), pz = qrot(q0, [0, 0, -1]), down = G.up.map(v => -v);
      if (dot(pz, down) < Math.cos(35 / 57.3) || this.armStops(s)) continue;
      const r = qbetween(pz, down), tq = r ? qnorm(qmul(r, q0)) : q0, tp = [0, 1, 2].map(i => p[i] - (gap - PALM) * G.up[i]);
      this.pin(n, p.slice(), q0, { out: G.up.slice(), ground: true, from: { p: p.slice(), q: q0 }, to: { p: tp, q: tq }, k: 0 });
      this.setHandContact(s, false);
      this.stats.floorHands = (this.stats.floorHands || 0) + 1;
    }
    // ---------- hands on a wall while he braces
    for (const s of SIDES) {
      const n = s + 'Hand', b = this.id[n], H = K.hands && K.hands[s];
      if (this.on[n] && this.on[n].ground) continue;
      const want = upright && H && (K.braceW || 0) > 0.4 && H.wall;
      const P = this.on[n];
      if (P) {
        const sh = this.pos(this.id[s + 'Arm']), reach = Math.hypot(...[0, 1, 2].map(i => P.p[i] - sh[i]));
        if (!want) { this.release(n, 'hand done'); if (H) H.pinned = false; continue; }
        if (reach > 0.64) { this.release(n, 'hand reach'); H.pinned = false; continue; }
        // (user 22:37 snapshot: "Left shoulder and arm is wrong": his body turned under a pinned hand until the hand was
        // up and behind his shoulder, all three shoulder hinges at their stops) a hand behind the shoulder, or a
        // shoulder pressed to its stops, lets go - a person would take the hand off and put it somewhere else
        const awk = this.awkward(s, P.p);
        P.awk = awk ? (P.awk || 0) + dt : 0;
        if (P.awk > 0.15) { this.release(n, 'hand awkward'); H.pinned = false; this.cool[n] = 0.6; continue; }
        P.t += dt;
        if (P.to && P.k < 1) {   // still placing the palm: no letting go for pulling or slipping yet
          P.k = Math.min(1, P.k + dt / 0.15); const u = P.k * P.k * (3 - 2 * P.k);
          P.p = [0, 1, 2].map(i => P.from.p[i] + (P.to.p[i] - P.from.p[i]) * u); P.q = qslerp(P.from.q, P.to.q, u);
          this.setPose(n, P.p, P.q); H.p = P.to.p.slice(); H.pinned = true; continue;
        }
        const f = this.force(n), out = P.out, fo = dot(f, out), ft = Math.hypot(...[0, 1, 2].map(i => f[i] - fo * out[i]));
        // (user 21:40: "the hands are always sliding ... like you do with IK, with a fixed point") a pressed palm grips:
        // skin on wood ~1.0, and only a slip that lasts (0.3 s) lets it go
        P.pull = fo < -30 ? P.pull + dt : 0;                 // holding the hand ON the wall (a palm can't)
        P.slip = ft > 1.0 * Math.max(0, fo) + 40 ? P.slip + dt : 0;
        const why = P.pull > 0.25 ? 'pull' : P.slip > 0.3 ? 'slip' : null;
        if (why) { this.release(n, 'hand ' + why); H.pinned = false; continue; }
        H.p = P.p.slice(); H.pinned = true;      // the recording's arm reaches exactly there
        { // how flat the palm really is on the wood, and how far the hand is from where it was put (for the tests)
          const S = this.stats.palm || (this.stats.palm = { n: 0, ang: 0, pos: 0, worst: 0 }), q = this.quat(b), pz = qrot(q, [0, 0, -1]);
          const a = Math.acos(Math.max(-1, Math.min(1, -dot(pz, out)))) * 57.3, pe = Math.hypot(...[0, 1, 2].map(i => this.pos(b)[i] - P.p[i])) * 100;
          S.n++; S.ang += a; S.pos += pe; S.worst = Math.max(S.worst, a);
        }
        continue;
      }
      if (!want || this.cool[n] > 0) continue;
      // touching that wall (hand or forearm contact, or the wrist within 6 cm of it): place the palm flat there
      const out = H.wall.out, p = this.pos(b), gap = dot([p[0] - H.wall.at[0], p[1] - H.wall.at[1], p[2] - H.wall.at[2]], out);
      // only once his own arm has brought the hand there: wrist within 5 cm of touching, palm within 25 deg of flat
      // (version 11 pulled it in from 16 cm: the weld dragged his whole body to the wall)
      if (gap > PALM + 0.05) continue;
      const q0 = this.quat(b), pz = qrot(q0, [0, 0, -1]), nin = out.map(v => -v);
      if (dot(pz, nin) < Math.cos(25 / 57.3)) continue;
      const tp = [0, 1, 2].map(i => p[i] - (gap - PALM) * out[i]);
      const sh = this.pos(this.id[s + 'Arm']); if (Math.hypot(tp[0] - sh[0], tp[1] - sh[1], tp[2] - sh[2]) > 0.62 || this.awkward(s, tp)) continue;
      // flat: his hand turned the shortest way so the palm faces the wood (fingers keep their way); placed over 0.15 s
      const r = qbetween(pz, nin), tq = r ? qnorm(qmul(r, q0)) : q0;
      this.pin(n, p.slice(), q0, { out, from: { p: p.slice(), q: q0 }, to: { p: tp, q: tq }, k: 0 });
      this.setHandContact(s, false);
      H.p = tp.slice(); H.pinned = true;
    }
    // hands back in contact with the wood once they're clear of it (or after 0.4 s)
    for (const s of SIDES) if (!this.on[s + 'Hand'] && this.handOff && this.handOff[s] !== undefined) {
      if (this.handOff[s] <= 0) { this.setHandContact(s, true); delete this.handOff[s]; }
    }
  }
  // the hand at p is behind his shoulder (more than 5 cm, along the chest's forward), or that shoulder's three hinges
  // are within 4 deg of a stop
  // that arm forced against its stops: the wrist within 8 deg of a stop, or two of the shoulder/elbow hinges within 6 deg
  armStops(side) {
    const W = this.W, m = W.m, d = W.d;
    if (!this.armAll) this.armAll = {};
    if (!this.armAll[side]) this.armAll[side] = ['Arm_x', 'Arm_y', 'Arm_z', 'ForeArm_z', 'Hand_x', 'Hand_y'].map(a => ({ a, j: W.mj.mj_name2id(m, W.mj.mjtObj.mjOBJ_JOINT.value, side + a) })).filter(e => e.j >= 0);
    let n = 0;
    for (const { a, j } of this.armAll[side]) {
      if (!m.jnt_limited[j]) continue;
      const v = d.qpos[m.jnt_qposadr[j]], lo = m.jnt_range[2 * j], hi = m.jnt_range[2 * j + 1], e = a.startsWith('Hand') ? 0.14 : 0.1;
      if (v < lo + e || v > hi - e) { if (a.startsWith('Hand')) return true; n++; }
    }
    return n >= 2;
  }
  awkward(side, p) {
    const W = this.W, m = W.m, d = W.d, ch = this.W.bodyId.Chest, sh = this.pos(this.id[side + 'Arm']);
    const q = this.quat(ch), fwd = qrot(q, [0, -1, 0]);
    if ((p[0] - sh[0]) * fwd[0] + (p[1] - sh[1]) * fwd[1] + (p[2] - sh[2]) * fwd[2] < -0.05) return true;
    if (!this.armJ) this.armJ = {};
    if (!this.armJ[side]) this.armJ[side] = ['x', 'y', 'z'].map(a => W.mj.mj_name2id(m, W.mj.mjtObj.mjOBJ_JOINT.value, side + 'Arm_' + a)).filter(j => j >= 0);
    let stops = 0;
    for (const j of this.armJ[side]) { if (!m.jnt_limited[j]) continue; const v = d.qpos[m.jnt_qposadr[j]], lo = m.jnt_range[2 * j], hi = m.jnt_range[2 * j + 1]; if (v < lo + 0.07 || v > hi - 0.07) stops++; }
    return stops >= 2;
  }
  vel(b) {
    const d = this.W.d, prev = this.prevPos || (this.prevPos = {}), p = this.pos(b), t = d.time, o = prev[b];
    prev[b] = { p, t };
    if (!o || t - o.t <= 0) return [0, 0, 0];
    return [0, 1, 2].map(i => (p[i] - o.p[i]) / (t - o.t));
  }
  setHandContact(s, on) {
    const m = this.W.m;
    for (const G of this.handGeoms[s]) { m.geom_contype[G.g] = on ? G.ct : 0; m.geom_conaffinity[G.g] = on ? G.ca : 0; }
  }
}
