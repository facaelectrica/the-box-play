// The Box - cracking mode (user, Nov 15 09:33): "every time the model hits walls and surfaces, if they hit strong
// enough, that body part starts cracking ... it has to respect the part where it was hit. After each hit, that part gets
// more cracked. Each part can be hit 3 times before it breaks. When it breaks ... the puppet keeps trying to move
// without the part that's now broken, and [the part keeps] moving around too in the physics. If the head gets broken,
// he stops moving completely." A button turns it on and off; a double tap makes him brand new.
//
// Three pieces, kept apart so the physics can run headless in the tests:
//  - Cracks: reads MuJoCo's contacts after each frame. A part arriving on a wall/floor/ceiling faster than CRACK.minV
//    (speed of the touching point into the surface, just before it touched) cracks there; the CRACK.hits-th crack
//    breaks the part off. A broken part and everything hanging from it (forearm -> hand) leave his physical body: no
//    collisions, almost no weight, no muscle; the brain still drives the rest of him. Losing the head (or the neck or
//    trunk above the hips, which carry it) = he stops: the rest of him goes limp, holding a little of his last pose.
//  - Debris: the broken-off piece (rigid, in the pose it broke in) and a few shards fly on in our own small rigid-body
//    physics: the box's six walls (or the open floor), the same gravity (phone tilt), his body as an obstacle.
//  - CrackView / CrackSound (page only): crack lines drawn on the part's surface where it was hit, the broken pieces,
//    and the sounds: a stronger thump + the shell cracking, a shatter when a part breaks, clinks of the pieces.
// Nothing here changes physics.js, the body file or training. The shell is drawn as glazed porcelain/ceramic.

export const CRACK = {
  minV: 2.4,          // m/s into the surface for a crack (footsteps ~0.3-1, catching a fall on the hands ~1-2)
  hits: 3,            // the 3rd cracking hit breaks the part off
  cool: 0.6,          // s: one impact (bounces, several walls in a corner) counts once per part
  away: 0.08,         // s: a part must have been off that surface this long for a new touch to count
  kind: { Foot: 1.25, Hand: 1.1 },   // feet and hands meet surfaces on purpose: a bit more needed
  mass: 0.03,         // what's left of a broken part's weight in his physics (its invisible stub)
};
const BREAK_ROOT = { Hips: 'Spine1' };   // the hips break = the upper body comes off them
const DEAD = new Set(['Hips', 'Spine1', 'Spine2', 'Chest', 'Neck1', 'Neck2', 'Head']);   // carry the head
export const pieceOf = n => n.replace('ToeBase', 'Foot').replace(/^(Left|Right)Shoulder$/, '$1Arm');

// ---------- small vector/quaternion helpers (quaternions w, x, y, z like MuJoCo) ----------
const dot = (a, b) => a[0] * b[0] + a[1] * b[1] + a[2] * b[2];
const cross = (a, b) => [a[1] * b[2] - a[2] * b[1], a[2] * b[0] - a[0] * b[2], a[0] * b[1] - a[1] * b[0]];
const add = (a, b, s = 1) => [a[0] + b[0] * s, a[1] + b[1] * s, a[2] + b[2] * s];
const sub = (a, b) => [a[0] - b[0], a[1] - b[1], a[2] - b[2]];
const len = a => Math.hypot(a[0], a[1], a[2]);
const mv = (R, v) => [R[0] * v[0] + R[1] * v[1] + R[2] * v[2], R[3] * v[0] + R[4] * v[1] + R[5] * v[2], R[6] * v[0] + R[7] * v[1] + R[8] * v[2]];
const mtv = (R, v) => [R[0] * v[0] + R[3] * v[1] + R[6] * v[2], R[1] * v[0] + R[4] * v[1] + R[7] * v[2], R[2] * v[0] + R[5] * v[1] + R[8] * v[2]];
const qmul = (a, b) => [a[0] * b[0] - a[1] * b[1] - a[2] * b[2] - a[3] * b[3], a[0] * b[1] + a[1] * b[0] + a[2] * b[3] - a[3] * b[2], a[0] * b[2] - a[1] * b[3] + a[2] * b[0] + a[3] * b[1], a[0] * b[3] + a[1] * b[2] - a[2] * b[1] + a[3] * b[0]];
const qnorm = q => { const n = Math.hypot(q[0], q[1], q[2], q[3]) || 1; return q.map(v => v / n); };
function q2m(q) {
  const [w, x, y, z] = q;
  return [1 - 2 * (y * y + z * z), 2 * (x * y - w * z), 2 * (x * z + w * y), 2 * (x * y + w * z), 1 - 2 * (x * x + z * z), 2 * (y * z - w * x), 2 * (x * z - w * y), 2 * (y * z + w * x), 1 - 2 * (x * x + y * y)];
}
function m2q(R) {
  const t = R[0] + R[4] + R[8]; let w, x, y, z;
  if (t > 0) { const s = Math.sqrt(t + 1) * 2; w = 0.25 * s; x = (R[7] - R[5]) / s; y = (R[2] - R[6]) / s; z = (R[3] - R[1]) / s; }
  else if (R[0] > R[4] && R[0] > R[8]) { const s = Math.sqrt(1 + R[0] - R[4] - R[8]) * 2; w = (R[7] - R[5]) / s; x = 0.25 * s; y = (R[1] + R[3]) / s; z = (R[2] + R[6]) / s; }
  else if (R[4] > R[8]) { const s = Math.sqrt(1 + R[4] - R[0] - R[8]) * 2; w = (R[2] - R[6]) / s; x = (R[1] + R[3]) / s; y = 0.25 * s; z = (R[5] + R[7]) / s; }
  else { const s = Math.sqrt(1 + R[8] - R[0] - R[4]) * 2; w = (R[3] - R[1]) / s; x = (R[2] + R[6]) / s; y = (R[5] + R[7]) / s; z = 0.25 * s; }
  return qnorm([w, x, y, z]);
}
const mmul = (A, B) => { const C = new Array(9); for (let i = 0; i < 3; i++) for (let j = 0; j < 3; j++) C[3 * i + j] = A[3 * i] * B[j] + A[3 * i + 1] * B[3 + j] + A[3 * i + 2] * B[6 + j]; return C; };
const mT = A => [A[0], A[3], A[6], A[1], A[4], A[7], A[2], A[5], A[8]];
function minv(A) {
  const [a, b, c, d, e, f, g, h, i] = A, C0 = e * i - f * h, C1 = -(d * i - f * g), C2 = d * h - e * g, det = a * C0 + b * C1 + c * C2 || 1e-12;
  return [C0 / det, -(b * i - c * h) / det, (b * f - c * e) / det, C1 / det, (a * i - c * g) / det, -(a * f - c * d) / det, C2 / det, -(a * h - b * g) / det, (a * e - b * d) / det];
}
// a small seeded random (the same hit draws the same crack again after the figure is swapped)
export function rng(seed) { let s = seed >>> 0 || 1; return () => { s = (s + 0x6D2B79F5) >>> 0; let t = s; t = Math.imul(t ^ (t >>> 15), t | 1); t ^= t + Math.imul(t ^ (t >>> 7), t | 61); return ((t ^ (t >>> 14)) >>> 0) / 4294967296; }; }

// ================= Debris: the broken pieces' own small physics =================
// Each piece is a rigid body made of spheres (from his body's collision shapes), against the box's flat walls (or the
// open floor), each other, and his body's shapes (they push the pieces, the pieces don't push him).
export class Debris {
  constructor(W) { this.W = W; this.list = []; this.events = []; }
  clear() { this.list.length = 0; this.events.length = 0; }
  add(p) { p.id = this.list.length; p.touch = new Set(); p.clinkT = 0; this.list.push(p); return p; }
  // his body's shapes as obstacles: [{a, b, r, body}] segments (sphere = a == b), skipping broken parts
  obstacles(skip) {
    const W = this.W, m = W.m, d = W.d, out = [];
    for (let g = 0; g < m.ngeom; g++) {
      const b = m.geom_bodyid[g]; if (!W.isFig[b] || (skip && skip.has(b))) continue;
      const t = m.geom_type[g], p = [d.geom_xpos[3 * g], d.geom_xpos[3 * g + 1], d.geom_xpos[3 * g + 2]], R = d.geom_xmat.subarray(9 * g, 9 * g + 9), s = m.geom_size.subarray(3 * g, 3 * g + 3);
      if (t === 2) out.push({ a: p, b: p, r: s[0], body: b });
      else if (t === 3) { const ax = [R[2], R[5], R[8]]; out.push({ a: add(p, ax, -s[1]), b: add(p, ax, s[1]), r: s[0], body: b }); }
      else if (t === 6) {   // a box: a capsule along its longest side, as thick as its thinnest
        let k = 0; if (s[1] > s[k]) k = 1; if (s[2] > s[k]) k = 2; const r = Math.min(s[0], s[1], s[2]), ax = [R[k], R[3 + k], R[6 + k]], h = Math.max(0, s[k] - r);
        out.push({ a: add(p, ax, -h), b: add(p, ax, h), r, body: b });
      }
    }
    return out;
  }
  step(dt, skip) {
    if (!this.list.length) return;
    const W = this.W, g = W.m.opt.gravity, planes = W.faces().map(f => ({ n: f.n, p: f.p, name: f.name }));
    const obs = this.obstacles(skip), n = Math.max(1, Math.ceil(dt / (1 / 300))), h = dt / n;
    for (let it = 0; it < n; it++) {
      for (const P of this.list) {
        P.v = add(P.v, g, h);
        P.R = q2m(P.q); P.Iinv = mmul(mmul(P.R, P.IinvB), mT(P.R));
        P.on = false;
      }
      for (const P of this.list) this.collide(P, planes, obs, h);
      // pieces against each other (bounding spheres first)
      for (let i = 0; i < this.list.length; i++) for (let j = i + 1; j < this.list.length; j++) this.pair(this.list[i], this.list[j]);
      for (const P of this.list) {
        // resting on something and nearly still: it stays put (no creeping from the contact pushes)
        if (P.on && len(P.v) < 0.04 && len(P.w) < 0.4) { P.v = [0, 0, 0]; P.w = [0, 0, 0]; continue; }
        const damp = P.on ? 1 - Math.min(0.5, 6 * h) : 1 - 0.05 * h;
        P.v = P.v.map(v => v * (P.on ? 1 - Math.min(0.3, 0.8 * h) : 1)); P.w = P.w.map(v => v * damp);
        P.x = add(P.x, P.v, h);
        const wq = [0, P.w[0], P.w[1], P.w[2]], dq = qmul(wq, P.q);
        P.q = qnorm(P.q.map((v, k) => v + 0.5 * h * dq[k]));
      }
    }
    for (const P of this.list) P.clinkT = Math.max(0, P.clinkT - dt);
  }
  impulse(P, rp, nrm, vrel, e, mu) {
    const invM = 1 / P.m, rn = cross(rp, nrm), k = invM + dot(nrm, cross(mv(P.Iinv, rn), rp));
    const vn = dot(vrel, nrm); if (vn >= 0) return 0;
    const jn = -(1 + e) * vn / k;
    P.v = add(P.v, nrm, jn * invM); P.w = add(P.w, mv(P.Iinv, rn), jn);
    const vp = add(P.v, cross(P.w, rp)), vt = add(vp, nrm, -dot(vp, nrm)), st = len(vt);
    if (st > 1e-6) {
      const t = vt.map(v => v / st), rt = cross(rp, t), kt = invM + dot(t, cross(mv(P.Iinv, rt), rp)), jt = Math.min(st / kt, mu * jn);
      P.v = add(P.v, t, -jt * invM); P.w = add(P.w, mv(P.Iinv, rt), -jt);
    }
    return -vn;
  }
  collide(P, planes, obs, h) {
    for (const s of P.s) {
      const c = add(P.x, mv(P.R, s.c));
      for (const pl of planes) {
        const dd = dot(sub(c, pl.p), pl.n) - s.r; if (dd >= 0) continue;
        const rp = add(sub(c, P.x), pl.n, -s.r), vp = add(P.v, cross(P.w, rp)), vn = -dot(vp, pl.n);
        const hit = this.impulse(P, rp, pl.n, vp, vn > 0.9 ? 0.25 : 0.02, 0.55);
        P.x = add(P.x, pl.n, -dd * 0.6); P.on = true;
        if (hit > 0.6 && P.clinkT <= 0) { P.clinkT = 0.07; this.events.push({ type: 'clink', v: hit, m: P.m }); }
      }
      if (P.age !== undefined && P.age < 0.06) continue;   // just broken off: let it clear his body first
      for (const o of obs) {
        const ab = sub(o.b, o.a), L2 = dot(ab, ab), u = L2 > 1e-12 ? Math.max(0, Math.min(1, dot(sub(c, o.a), ab) / L2)) : 0, q = add(o.a, ab, u);
        const dv = sub(c, q), dl = len(dv), pen = s.r + o.r - dl; if (pen <= 0 || dl < 1e-6) continue;
        const nrm = dv.map(v => v / dl), rp = add(sub(c, P.x), nrm, -s.r), vp = add(P.v, cross(P.w, rp));
        this.impulse(P, rp, nrm, vp, 0.1, 0.4); P.x = add(P.x, nrm, pen * 0.5); P.on = true;
      }
    }
  }
  pair(A, B) {
    if (len(sub(A.x, B.x)) > A.bound + B.bound) return;
    for (const sa of A.s) {
      const ca = add(A.x, mv(A.R, sa.c));
      for (const sb of B.s) {
        const cb = add(B.x, mv(B.R, sb.c)), dv = sub(ca, cb), dl = len(dv), pen = sa.r + sb.r - dl; if (pen <= 0 || dl < 1e-6) continue;
        const nrm = dv.map(v => v / dl), ra = add(sub(ca, A.x), nrm, -sa.r), rb = add(sub(cb, B.x), nrm, sb.r);
        const va = add(A.v, cross(A.w, ra)), vb = add(B.v, cross(B.w, rb)), vr = sub(va, vb), vn = dot(vr, nrm); if (vn >= 0) continue;
        const ka = 1 / A.m + dot(nrm, cross(mv(A.Iinv, cross(ra, nrm)), ra)), kb = 1 / B.m + dot(nrm, cross(mv(B.Iinv, cross(rb, nrm)), rb)), j = -1.15 * vn / (ka + kb);
        A.v = add(A.v, nrm, j / A.m); A.w = add(A.w, mv(A.Iinv, cross(ra, nrm)), j);
        B.v = add(B.v, nrm, -j / B.m); B.w = add(B.w, mv(B.Iinv, cross(rb, nrm)), -j);
        const tot = A.m + B.m; A.x = add(A.x, nrm, pen * 0.5 * B.m / tot); B.x = add(B.x, nrm, -pen * 0.5 * A.m / tot);
      }
    }
  }
}

// ================= Cracks: hits, cracks, breaks, his body without the broken parts =================
export class Cracks {
  constructor(W, opt = {}) {
    this.W = W; this.drive = opt.drive || null; this.on = false;
    const m = W.m, d = W.d;
    this.names = W.bodyNames; this.N = m.nbody;
    this.nameOf = []; for (const n of W.bodyNames) this.nameOf[W.bodyId[n]] = n;
    this.children = Array.from({ length: m.nbody }, () => []);
    for (const n of W.bodyNames) { const b = W.bodyId[n], p = m.body_parentid[b]; if (W.isFig[p]) this.children[p].push(b); }
    this.geomsOf = Array.from({ length: m.nbody }, () => []);
    for (let g = 0; g < m.ngeom; g++) if (W.isFig[m.geom_bodyid[g]]) this.geomsOf[m.geom_bodyid[g]].push(g);
    this.actOf = Array.from({ length: m.nbody }, () => []);
    W.act.forEach((a, i) => this.actOf[m.jnt_bodyid[a.joint]].push(i));
    // what to restore
    this.orig = { ct: Int32Array.from(m.geom_contype), ca: Int32Array.from(m.geom_conaffinity), mass: Float64Array.from(m.body_mass), inertia: Float64Array.from(m.body_inertia) };
    this.debris = new Debris(W);
    this.hist = [];
    this.reset();
    // his physics without the broken parts, enforced before every physics step (pins, tone changes and get-ups rewrite
    // collisions and muscle strengths)
    if (!W.__crackStep) {
      const step = W.step.bind(W); W.__crackStep = true;
      W.step = control => { if (this.broken.size || this.dead) this.preStep(); return step(control); };
    }
    const pins = this.drive && this.drive.pins;
    if (pins && !pins.__crack) {
      pins.__crack = true; const pin = pins.pin.bind(pins);
      pins.pin = (n, ...a) => (this.dead || this.broken.has(W.bodyId[n]) ? undefined : pin(n, ...a));
    }
  }
  subtree(b, out = []) { out.push(b); for (const c of this.children[b]) this.subtree(c, out); return out; }
  reset() {
    const W = this.W, m = W.m;
    if (this.broken && (this.broken.size || this.dead)) {
      m.geom_contype.set(this.orig.ct); m.geom_conaffinity.set(this.orig.ca);
      m.body_mass.set(this.orig.mass); m.body_inertia.set(this.orig.inertia);
      // the muscles: the drive's own reset (attach) puts the brain's strengths back
    }
    if (this.getupWas !== undefined && this.drive) { this.drive.getupBy = this.getupWas; this.getupWas = undefined; }
    this.parts = {};   // piece name -> { level, hits: [...] }
    this.broken = new Set(); this.dead = false; this.deadHold = null; this.deadT = 0;
    this.debris.clear(); this.events = []; this.touch = new Map(); this.lastHit = {};
    this.hist.length = 0; this.quiet = 0.4; this.t = 0; this.chunks = [];
    this.stats = { hits: [], cracks: 0, breaks: 0 }; this.pending = [];
  }
  part(name) { return this.parts[name] || (this.parts[name] = { level: 0, hits: [] }); }
  // turn the mode on/off (off = everything as new)
  setOn(on) { this.on = !!on; if (!on) this.reset(); }
  snapshot() {
    const d = this.W.d;
    return { t: this.t, x: Float64Array.from(d.xpos), R: Float64Array.from(d.xmat) };
  }
  // the speed of a point fixed in body b (local l) between the last two frames
  pointVel(b, l) {
    const H = this.hist; if (H.length < 2) return null;
    const A = H[H.length - 1], B = H[H.length - 2], dt = A.t - B.t; if (dt <= 1e-4) return null;
    const pa = add([A.x[3 * b], A.x[3 * b + 1], A.x[3 * b + 2]], mv(A.R.subarray(9 * b, 9 * b + 9), l));
    const pb = add([B.x[3 * b], B.x[3 * b + 1], B.x[3 * b + 2]], mv(B.R.subarray(9 * b, 9 * b + 9), l));
    const v = sub(pa, pb).map(x => x / dt);
    return len(v) > 30 ? null : v;   // a teleport (reset, get-up blend), not a motion
  }
  // after each frame's physics. skip = he is being placed by a recording (kinematic get-up): no hits then.
  update(dt, skip = false) {
    const W = this.W, m = W.m, d = W.d;
    this.t += dt; this.quiet = Math.max(0, this.quiet - dt);
    if (!this.on) { this.hist.length = 0; return []; }
    const ev = this.pending.splice(0);
    // which parts touch which wall now
    const now = new Map();
    for (let i = 0; i < d.ncon; i++) {
      const c = d.contact.get(i), g1 = c.geom1, g2 = c.geom2, dist = c.dist, fr = c.frame, pos = c.pos, n = [fr[0], fr[1], fr[2]], p = [pos[0], pos[1], pos[2]];
      c.delete();
      if (dist > 0.004) continue;
      const b1 = m.geom_bodyid[g1], b2 = m.geom_bodyid[g2], f1 = W.isFig[b1], f2 = W.isFig[b2];
      if (f1 === f2) continue;
      const b = f1 ? b1 : b2, wall = f1 ? g2 : g1, nOut = f1 ? n.map(v => -v) : n;   // out of the wall, toward the part
      if (this.broken.has(b)) continue;
      const key = b * 1024 + wall;
      if (!now.has(key)) now.set(key, { b, wall, p, n: nOut });
    }
    if (!skip && !this.quiet && !this.dead) {
      for (const [key, c] of now) {
        const last = this.touch.get(key);
        if (last !== undefined && this.t - last < CRACK.away + dt * 1.5) continue;   // still (or just) touching
        const b = c.b, R = d.xmat.subarray(9 * b, 9 * b + 9), x = [d.xpos[3 * b], d.xpos[3 * b + 1], d.xpos[3 * b + 2]];
        const l = mtv(R, sub(c.p, x)), v = this.pointVel(b, l); if (!v) continue;
        const vIn = -dot(v, c.n);
        const name = this.nameOf[b], piece = pieceOf(name), k = CRACK.kind[piece.replace(/^(Left|Right)/, '')] || 1;
        if (vIn > 1.0) this.stats.hits.push({ t: +this.t.toFixed(2), part: name, v: +vIn.toFixed(2) });
        if (vIn < CRACK.minV * k) continue;
        if (this.t - (this.lastHit[piece] || -9) < CRACK.cool) continue;
        this.lastHit[piece] = this.t;
        const pb = W.bodyId[piece];
        if (this.broken.has(pb)) continue;
        // where on the part: the touching point and the surface's outward normal, in the part's own frame
        const Rp = d.xmat.subarray(9 * pb, 9 * pb + 9), xp = [d.xpos[3 * pb], d.xpos[3 * pb + 1], d.xpos[3 * pb + 2]];
        const P = this.part(piece), s = Math.min(1, (vIn - CRACK.minV * k) / (2.5 * k));
        const hit = { body: pb, l: mtv(Rp, sub(c.p, xp)), n: mtv(Rp, c.n.map(v => -v)), v: vIn, s, seed: (Math.random() * 1e9) | 0, level: P.level + 1 };
        P.level++; P.hits.push(hit);
        if (P.level >= CRACK.hits) { this.stats.breaks++; ev.push({ type: 'break', piece, hit, ...this.breakOff(piece, c.n, vIn) }); }
        else { this.stats.cracks++; ev.push({ type: 'crack', piece, hit, level: P.level }); }
      }
    }
    for (const [key] of now) this.touch.set(key, this.t);
    this.hist.push(this.snapshot()); if (this.hist.length > 2) this.hist.shift();
    // the pieces fly on
    for (const P of this.debris.list) P.age = (P.age || 0) + dt;
    this.debris.step(dt, this.broken);
    for (const e of this.debris.events) ev.push(e); this.debris.events.length = 0;
    this.events = ev;
    return ev;
  }
  // a part breaks off: it and what hangs from it leave his body and become a piece of debris (+ shards)
  breakOff(piece, nWall, vIn) {
    const W = this.W, m = W.m, d = W.d, mj = W.mj;
    const root = W.bodyId[BREAK_ROOT[piece] || piece];
    const bodies = this.subtree(root).filter(b => !this.broken.has(b));
    const killed = bodies.some(b => DEAD.has(this.nameOf[b])) || DEAD.has(piece);
    // the piece: mass, centre, turning inertia, collision spheres, speed - from his body right now
    let M = 0, com = [0, 0, 0];
    for (const b of bodies) { const mb = this.orig.mass[b]; M += mb; com = add(com, [d.xipos[3 * b], d.xipos[3 * b + 1], d.xipos[3 * b + 2]], mb); }
    com = com.map(v => v / M);
    const Rr = Array.from(d.xmat.subarray(9 * root, 9 * root + 9)), Rt = mT(Rr);
    let Iw = [0, 0, 0, 0, 0, 0, 0, 0, 0];
    for (const b of bodies) {
      const Ri = Array.from(d.ximat.subarray(9 * b, 9 * b + 9)), I = this.orig.inertia.subarray(3 * b, 3 * b + 3), mb = this.orig.mass[b];
      const Ib = mmul(mmul(Ri, [I[0], 0, 0, 0, I[1], 0, 0, 0, I[2]]), mT(Ri)), r = sub([d.xipos[3 * b], d.xipos[3 * b + 1], d.xipos[3 * b + 2]], com), r2 = dot(r, r);
      for (let i = 0; i < 3; i++) for (let j = 0; j < 3; j++) Iw[3 * i + j] += Ib[3 * i + j] + mb * ((i === j ? r2 : 0) - r[i] * r[j]);
    }
    const IB = mmul(mmul(Rt, Iw), Rr);
    for (let i = 0; i < 3; i++) IB[4 * i] += 1e-5;
    const spheres = [];
    for (const b of bodies) for (const g of this.geomsOf[b]) {
      const t = m.geom_type[g], p = [d.geom_xpos[3 * g], d.geom_xpos[3 * g + 1], d.geom_xpos[3 * g + 2]], R = d.geom_xmat.subarray(9 * g, 9 * g + 9), s = m.geom_size.subarray(3 * g, 3 * g + 3);
      const put = (c, r) => spheres.push({ c: mtv(Rr, sub(c, com)), r });
      if (t === 2) put(p, s[0]);
      else if (t === 3) { const ax = [R[2], R[5], R[8]], k = Math.max(1, Math.ceil(2 * s[1] / s[0])); for (let i = 0; i <= k; i++) put(add(p, ax, -s[1] + 2 * s[1] * i / k), s[0]); }
      else if (t === 6) {
        const r = Math.min(s[0], s[1], s[2]), cnt = [0, 1, 2].map(i => Math.max(1, Math.ceil((s[i] - r) / r) + 1));
        for (let i = 0; i < cnt[0]; i++) for (let j = 0; j < cnt[1]; j++) for (let k = 0; k < cnt[2]; k++) {
          const o = [i, j, k].map((u, a) => cnt[a] > 1 ? -(s[a] - r) + 2 * (s[a] - r) * u / (cnt[a] - 1) : 0);
          put(add(p, mv(R, o)), r);
        }
      }
    }
    // its motion: the root's centre and turn between the last two frames
    let v = [0, 0, 0], w = [0, 0, 0];
    const H = this.hist;
    if (H.length >= 1) {
      const A = { t: this.t, x: d.xpos, R: d.xmat }, B = H[H.length - 1], dt = A.t - B.t;
      if (dt > 1e-4) {
        const lc = mtv(Rr, sub(com, [d.xpos[3 * root], d.xpos[3 * root + 1], d.xpos[3 * root + 2]]));
        const pb = add([B.x[3 * root], B.x[3 * root + 1], B.x[3 * root + 2]], mv(B.R.subarray(9 * root, 9 * root + 9), lc));
        v = sub(com, pb).map(x => x / dt);
        const dR = mmul(Rr, mT(Array.from(B.R.subarray(9 * root, 9 * root + 9)))), q = m2q(dR), ang = 2 * Math.acos(Math.min(1, Math.abs(q[0]))), sg = q[0] < 0 ? -1 : 1, sn = Math.hypot(q[1], q[2], q[3]);
        if (sn > 1e-8) w = [q[1], q[2], q[3]].map(x => sg * x / sn * ang / dt);
        if (len(v) > 12) v = v.map(x => x * 12 / len(v)); if (len(w) > 25) w = w.map(x => x * 25 / len(w));
      }
    }
    // the crack pops it off: a little kick away from what it hit, and away from his body
    const joint = [d.xpos[3 * root], d.xpos[3 * root + 1], d.xpos[3 * root + 2]];
    const away = sub(com, joint), al = len(away) || 1;
    v = add(add(v, nWall, 0.6 + 0.08 * vIn), away, 0.5 / al);
    const bound = Math.max(...spheres.map(s => len(s.c) + s.r));
    const chunk = this.debris.add({ kind: 'chunk', bodies, root, m: M, x: com, q: m2q(Rr), v, w, IinvB: minv(IB), s: spheres, bound, comL: mtv(Rr, sub(com, joint)), age: 0,
      rel: bodies.map(b => { const Rb = Array.from(d.xmat.subarray(9 * b, 9 * b + 9)); return { b, p: mtv(Rr, sub([d.xpos[3 * b], d.xpos[3 * b + 1], d.xpos[3 * b + 2]], com)), q: m2q(mmul(Rt, Rb)) }; }) });
    // shards from the break
    const shards = [], nS = 5 + Math.round(Math.random() * 4), R0 = Math.random;
    const brk = len(sub(joint, com)) < 0.02 ? com : joint;
    for (let i = 0; i < nS; i++) {
      const r = 0.006 + 0.008 * R0(), ms = 2600 * (4 / 3) * Math.PI * r * r * r * 0.5, I = 0.4 * ms * r * r;
      const dir = [R0() - 0.5, R0() - 0.5, R0() - 0.5], dl = len(dir) || 1;
      const sv = add(add(v, nWall, 0.8 + R0() * 1.4), dir, (0.6 + 1.6 * R0()) / dl);
      shards.push(this.debris.add({ kind: 'shard', m: ms, x: add(brk, dir, 0.03 / dl), q: qnorm([R0(), R0(), R0(), R0()]), v: sv, w: [R0() * 30 - 15, R0() * 30 - 15, R0() * 30 - 15], IinvB: [1 / I, 0, 0, 0, 1 / I, 0, 0, 0, 1 / I], s: [{ c: [0, 0, 0], r }], bound: r, size: r, age: 0, seed: (R0() * 1e9) | 0 }));
    }
    for (const b of bodies) this.broken.add(b);
    this.chunks.push(chunk);
    // a leg gone: get-ups can't be the recording played on his body any more (it would stand him on the missing leg):
    // his muscles try it in the physics instead, and fail like a person on one leg would
    if (this.drive && bodies.some(b => /(Leg|Shin|Foot|ToeBase)$/.test(this.nameOf[b])) && globalThis.CRACK_GETUP !== 'kin') { if (this.getupWas === undefined) this.getupWas = this.drive.getupBy; this.drive.getupBy = globalThis.CRACK_GETUP || 'springs'; }
    if (killed) this.die();
    this.preStep();
    mj.mj_forward(m, d);
    return { chunk, shards, bodies, dead: killed };
  }
  // (tests) break a part now
  forceBreak(piece) {
    const W = this.W, d = W.d, b = W.bodyId[piece], P = this.part(piece); P.level = CRACK.hits;
    const hit = { body: b, l: [0, 0, 0.05], n: [0, 0, 1], v: 3, s: 0.3, seed: 7, level: CRACK.hits };
    const e = { type: 'break', piece, hit, ...this.breakOff(piece, [0, 0, 1], 3) }; this.pending.push(e); return e;
  }
  die() {
    if (this.dead) return;
    const W = this.W, m = W.m, d = W.d;
    this.dead = true; this.deadT = 0;
    // he stops: every muscle lets go to a little tone holding his last pose (a puppet whose strings were cut, not jelly)
    this.deadHold = W.act.map(a => d.qpos[a.qadr]);
    this.deadGain = W.act.map((a, i) => [m.actuator_gainprm[10 * i] * 0.05, m.actuator_biasprm[10 * i + 2] * 0.25]);
    if (this.drive && this.drive.pins) this.drive.pins.reset();
  }
  // before every physics step: broken parts don't collide, weigh almost nothing and have no muscle
  preStep() {
    const W = this.W, m = W.m;
    if (this.dead) W.act.forEach((a, i) => { const [kp, kd] = this.deadGain[i]; m.actuator_gainprm[10 * i] = kp; m.actuator_biasprm[10 * i + 1] = -kp; m.actuator_biasprm[10 * i + 2] = kd; });
    for (const b of this.broken) {
      for (const g of this.geomsOf[b]) { m.geom_contype[g] = 0; m.geom_conaffinity[g] = 0; }
      m.body_mass[b] = this.orig.mass[b] * CRACK.mass;
      for (let k = 0; k < 3; k++) m.body_inertia[3 * b + k] = this.orig.inertia[3 * b + k] * CRACK.mass;
      for (const i of this.actOf[b]) { m.actuator_gainprm[10 * i] = 0; m.actuator_biasprm[10 * i + 1] = 0; m.actuator_biasprm[10 * i + 2] = Math.max(-0.3, m.actuator_biasprm[10 * i + 2] * 0.02); }
    }
    const pins = this.drive && this.drive.pins;
    if (pins) for (const n of ['LeftFoot', 'RightFoot', 'LeftHand', 'RightHand']) if (pins.on[n] && (this.dead || this.broken.has(W.bodyId[n]))) pins.release(n, 'broken');
  }
  // he has stopped: only physics (and your finger) moves him
  stepDead(dt, fingerPush) {
    const W = this.W, d = W.d, n = Math.min(40, Math.round(dt / (1 / 480)));
    this.deadT += dt;
    const hold = this.deadHold, ctl = () => hold; ctl.hz = 30;
    for (let i = 0; i < n; i++) { d.xfrc_applied.fill(0); if (fingerPush) fingerPush(); W.step(ctl); }
  }
  // the hidden part's world pose for drawing: a chunk's body b
  chunkBodyPose(P, rel) {
    const R = q2m(P.q);
    return { p: add(P.x, mv(R, rel.p)), q: qmul(P.q, rel.q) };
  }
}

// ================= CrackView: what you see (page only; THREE passed in) =================
export class CrackView {
  // groups(): MuJoCo body id -> THREE.Group of that part; isShell(mesh): the part's outer shell; root: figRoot
  constructor(THREE, root, groups, isShell, movers, shell) {
    this.T = THREE; this.shell = shell; this.root = root; this.groups = groups; this.isShell = isShell; this.movers = movers;
    this.cache = new Map(); this.crackMeshes = []; this.debris = new Map();
    this.ray = new THREE.Raycaster();
    this.dark = new THREE.MeshBasicMaterial({ color: 0x2a211a, side: THREE.DoubleSide, polygonOffset: true, polygonOffsetFactor: -6, polygonOffsetUnits: -6 });
    this.halo = new THREE.MeshBasicMaterial({ color: 0xfffaf2, transparent: true, opacity: 0.45, side: THREE.DoubleSide, depthWrite: false, polygonOffset: true, polygonOffsetFactor: -4, polygonOffsetUnits: -4 });
    this.inner = new THREE.MeshStandardMaterial({ color: 0xd9d2c4, roughness: 0.9, side: THREE.DoubleSide });   // the unglazed inside of a break
  }
  reset() {
    for (const m of this.crackMeshes) { if (m.parent) m.parent.remove(m); m.geometry.dispose(); }
    this.crackMeshes.length = 0;
    for (const [, o] of this.debris) { this.root.remove(o); const i = this.movers.indexOf(o); if (i >= 0) this.movers.splice(i, 1); }
    this.debris.clear();
    for (const g of this.groups()) if (g) g.visible = true;
    this.cache.clear();
  }
  // the part's shell meshes and a line (or point) inside it that rays start from
  info(g) {
    const T = this.T; let c = this.cache.get(g);
    if (c) return c;
    const meshes = []; g.traverse(o => { if (o.isMesh && this.isShell(o)) meshes.push(o); });
    const box = new T.Box3(); g.updateMatrixWorld(true);
    const inv = new T.Matrix4().copy(g.matrixWorld).invert(), M = new T.Matrix4();
    for (const me of meshes) { if (!me.geometry.boundingBox) me.geometry.computeBoundingBox(); M.multiplyMatrices(inv, me.matrixWorld); box.union(me.geometry.boundingBox.clone().applyMatrix4(M)); }
    const size = box.getSize(new T.Vector3()), ctr = box.getCenter(new T.Vector3()), dims = [size.x, size.y, size.z];
    let k = 0; if (dims[1] > dims[k]) k = 1; if (dims[2] > dims[k]) k = 2;
    const sorted = dims.slice().sort((a, b) => b - a), axis = new T.Vector3(); axis.setComponent(k, 1);
    const half = sorted[0] > 1.5 * sorted[1] ? 0.5 * (dims[k] - sorted[1]) : 0;   // long parts: a segment; round ones: the centre
    c = { meshes, a: ctr.clone().addScaledVector(axis, -half), b: ctr.clone().addScaledVector(axis, half), size: sorted[1] };
    this.cache.set(g, c); return c;
  }
  // a point near the part's surface -> the surface point straight out from the inner line, and its normal (part frame)
  project(g, I, p, prevN) {
    return this.radial(g, I, p, prevN) || this.inward(g, I, p, prevN);
  }
  // from outside, straight in along the normal (where the part's shell is open, e.g. the ends of a body segment)
  inward(g, I, p, n) {
    const T = this.T, ow = p.clone().addScaledVector(n, 0.12).applyMatrix4(g.matrixWorld), dw = n.clone().negate().transformDirection(g.matrixWorld);
    this.ray.set(ow, dw); this.ray.far = 10;
    const hits = this.ray.intersectObjects(I.meshes, false); if (!hits.length) return null;
    const h = hits[0], inv = this._inv || (this._inv = new T.Matrix4()); inv.copy(g.matrixWorld).invert();
    const pt = h.point.clone().applyMatrix4(inv), nn = h.face ? h.face.normal.clone().transformDirection(h.object.matrixWorld).transformDirection(inv) : n.clone();
    if (nn.dot(n) < 0) nn.negate();
    return { p: pt, n: nn.normalize() };
  }
  radial(g, I, p, prevN) {
    const T = this.T, ab = I.b.clone().sub(I.a), L2 = ab.lengthSq();
    const u = L2 > 1e-10 ? Math.max(0, Math.min(1, p.clone().sub(I.a).dot(ab) / L2)) : 0, o = I.a.clone().addScaledVector(ab, u);
    const dir = p.clone().sub(o); if (dir.lengthSq() < 1e-10) dir.copy(prevN); dir.normalize();
    const ow = o.clone().applyMatrix4(g.matrixWorld), dw = dir.clone().transformDirection(g.matrixWorld);
    this.ray.set(ow, dw); this.ray.far = 10;
    const hits = this.ray.intersectObjects(I.meshes, false); if (!hits.length) return null;
    const h = hits[hits.length - 1], inv = this._inv || (this._inv = new T.Matrix4());
    inv.copy(g.matrixWorld).invert();
    const pt = h.point.clone().applyMatrix4(inv);
    const n = h.face ? h.face.normal.clone().transformDirection(h.object.matrixWorld).transformDirection(inv) : dir.clone();
    if (n.dot(dir) < 0) n.negate();
    return { p: pt, n: n.normalize() };
  }
  // draw one crack where hit landed (seeded, so a swapped figure gets the same crack)
  addCrack(hit) {
    const T = this.T, gs = this.groups(); let g = gs[hit.body]; if (!g) return;
    this.root.updateMatrixWorld(true);
    let I = this.info(g), l = new T.Vector3(...hit.l), n0 = new T.Vector3(...hit.n).normalize();
    if (!I.meshes.length) {
      // a part drawn as part of its neighbour (e.g. the middle of the spine): the crack goes on the nearest drawn part
      const pw = l.clone().applyMatrix4(g.matrixWorld), nw = n0.clone().transformDirection(g.matrixWorld); let best = 1e9, bg = null;
      for (const o of gs) {
        if (!o || o === g) continue; const J = this.info(o); if (!J.meshes.length) continue;
        const a = J.a.clone().applyMatrix4(o.matrixWorld), b = J.b.clone().applyMatrix4(o.matrixWorld), ab = b.clone().sub(a), L2 = ab.lengthSq();
        const u = L2 > 1e-12 ? Math.max(0, Math.min(1, pw.clone().sub(a).dot(ab) / L2)) : 0, dd = pw.distanceTo(a.addScaledVector(ab, u));
        if (dd < best) { best = dd; bg = o; }
      }
      if (!bg) return;
      const inv = new T.Matrix4().copy(bg.matrixWorld).invert();
      g = bg; I = this.info(g); l = pw.applyMatrix4(inv); n0 = nw.transformDirection(inv);
    }
    const R = rng(hit.seed), lv = hit.level, s = hit.s;
    const start = this.inward(g, I, l, n0) || this.radial(g, I, l, n0); if (!start) return;
    // sizes: visible on a phone (1 cm is ~3 px), grows with the part's damage and the hit
    const S = Math.min(1, Math.max(0.55, I.size / 0.12));
    const Lmain = (0.045 + 0.035 * (lv - 1) + 0.03 * s) * S, W0 = (0.0042 + 0.0012 * (lv - 1)) * S, step = 0.006 * S;
    const lines = [];
    const walk = (p0, n0, dir, L, w0, depth) => {
      const pts = [{ p: p0.clone(), n: n0.clone(), w: w0 }]; let p = p0.clone(), n = n0.clone(), d = dir.clone(), went = 0;
      while (went < L) {
        d.addScaledVector(n, -d.dot(n)).normalize();
        d.applyAxisAngle(n, (R() - 0.5) * 0.7);
        const h = this.project(g, I, p.clone().addScaledVector(d, step).addScaledVector(n, 0.002), n); if (!h) break;
        const seg = h.p.distanceTo(p); if (seg < 1e-5) break;
        d = h.p.clone().sub(p).normalize(); p = h.p; n = h.n; went += seg;
        pts.push({ p: p.clone(), n: n.clone(), w: w0 * (1 - 0.75 * went / L) });
        if (depth < 2 && R() < 0.1 + 0.04 * lv && L - went > 0.015) walk(p, n, d.clone().applyAxisAngle(n, (R() < 0.5 ? -1 : 1) * (0.5 + 0.5 * R())), (L - went) * (0.45 + 0.3 * R()), w0 * 0.6, depth + 1);
      }
      if (pts.length > 1) lines.push(pts);
    };
    const t0 = new T.Vector3(1, 0, 0); if (Math.abs(t0.dot(start.n)) > 0.9) t0.set(0, 1, 0);
    t0.addScaledVector(start.n, -t0.dot(start.n)).normalize();
    const B = 3 + lv + (R() < 0.5 ? 1 : 0), a0 = R() * Math.PI * 2;
    for (let i = 0; i < B; i++) {
      const dir = t0.clone().applyAxisAngle(start.n, a0 + (i + (R() - 0.5) * 0.6) * 2 * Math.PI / B);
      walk(start.p, start.n, dir, Lmain * (0.55 + 0.6 * R()), W0, 0);
    }
    // the chipped spot where it hit
    const chip = [], rc = (0.006 + 0.003 * lv) * S;
    for (let i = 0; i < 9; i++) {
      const dir = t0.clone().applyAxisAngle(start.n, i / 9 * Math.PI * 2), q = this.project(g, I, start.p.clone().addScaledVector(dir, rc * (0.6 + 0.6 * R())).addScaledVector(start.n, 0.002), start.n);
      chip.push(q ? q.p : start.p.clone().addScaledVector(dir, rc));
    }
    for (const [mat, wk, lift, ord] of [[this.halo, 2.4, 0.0007, 0], [this.dark, 1, 0.0012, 1]]) {
      const pos = [];
      for (const L of lines) for (let i = 0; i < L.length - 1; i++) {
        const A = L[i], Bp = L[i + 1], t = Bp.p.clone().sub(A.p).normalize();
        const sa = A.n.clone().cross(t).normalize().multiplyScalar(A.w * wk / 2), sb = Bp.n.clone().cross(t).normalize().multiplyScalar(Bp.w * wk / 2);
        const a = A.p.clone().addScaledVector(A.n, lift), b = Bp.p.clone().addScaledVector(Bp.n, lift);
        const v = [a.clone().add(sa), a.clone().sub(sa), b.clone().add(sb), b.clone().sub(sb)];
        pos.push(...v[0].toArray(), ...v[1].toArray(), ...v[2].toArray(), ...v[1].toArray(), ...v[3].toArray(), ...v[2].toArray());
      }
      if (wk === 1) {   // the chip: a dark fan
        const c = start.p.clone().addScaledVector(start.n, lift);
        for (let i = 0; i < chip.length; i++) { const a = chip[i].clone().addScaledVector(start.n, lift), b = chip[(i + 1) % chip.length].clone().addScaledVector(start.n, lift); pos.push(...c.toArray(), ...a.toArray(), ...b.toArray()); }
      }
      if (!pos.length) continue;
      const geo = new T.BufferGeometry(); geo.setAttribute('position', new T.Float32BufferAttribute(pos, 3));
      const mesh = new T.Mesh(geo, mat); mesh.renderOrder = 2 + ord; mesh.userData.crack = true; mesh.castShadow = false; mesh.receiveShadow = false;
      g.add(mesh); this.crackMeshes.push(mesh);
    }
  }
  // the broken-off pieces: copies of the parts (with their cracks), and the shards
  addDebris(P) {
    const T = this.T, o = new T.Group();
    if (P.kind === 'chunk') {
      const gs = this.groups();
      for (const r of P.rel) {
        const src = gs[r.b]; if (!src) continue;
        const c = new T.Group();
        for (const ch of src.children) { const k = ch.clone(true); if (ch.userData.hiddenRigid) k.visible = true; c.add(k); }   // (the trunk is drawn skinned on him; the piece shows its rigid copy)
        c.position.set(...r.p); c.quaternion.set(r.q[1], r.q[2], r.q[3], r.q[0]);
        c.userData.body = r.b; o.add(c);
      }
      // the broken face: a rough disc of unglazed ceramic where it came off
      const jl = P.comL.map(v => -v), cap = new T.Mesh(new T.IcosahedronGeometry(0.028, 1), this.inner);
      { const pa = cap.geometry.attributes.position, R = rng(P.id * 7919 + 13); for (let i = 0; i < pa.count; i++) pa.setXYZ(i, pa.getX(i) * (0.8 + 0.4 * R()), pa.getY(i) * (0.8 + 0.4 * R()), pa.getZ(i) * (0.35 + 0.3 * R())); cap.geometry.computeVertexNormals(); }
      cap.position.set(...jl); cap.castShadow = true; o.add(cap);
    } else {
      const geo = new T.IcosahedronGeometry(P.size * 1.4, 0), pa = geo.attributes.position, R = rng(P.seed);
      for (let i = 0; i < pa.count; i++) pa.setXYZ(i, pa.getX(i) * (0.6 + 0.8 * R()), pa.getY(i) * (0.5 + 0.7 * R()), pa.getZ(i) * (0.3 + 0.4 * R()));
      geo.computeVertexNormals();
      const me = new T.Mesh(geo, this.shell); me.castShadow = true; o.add(me);
    }
    o.userData.P = P; this.root.add(o); this.movers.push(o); this.debris.set(P, o);
    return o;
  }
  // after the figure is rebuilt (another figure picked): cracks and pieces again from the stored hits
  rebuild(C) {
    const keep = [...this.debris.keys()];
    this.reset();
    for (const k in C.parts) for (const h of C.parts[k].hits) this.addCrack(h);
    for (const P of keep) this.addDebris(P);
  }
  sync(C) {
    const gs = this.groups();
    for (const b of C.broken) if (gs[b]) gs[b].visible = false;
    for (const [P, o] of this.debris) {
      if (P.kind === 'chunk') {
        o.position.set(P.x[0], P.x[1], P.x[2]); o.quaternion.set(P.q[1], P.q[2], P.q[3], P.q[0]);
      } else { o.position.set(P.x[0], P.x[1], P.x[2]); o.quaternion.set(P.q[1], P.q[2], P.q[3], P.q[0]); }
    }
  }
}

// ================= CrackSound: a stronger thump, the shell cracking, a break, the pieces' clinks =================
// Made from noise and tones (nothing to download). Its own path to the speakers: the figure's other sounds go through
// a soft low-pass (muffled wood), but a crack lives in the highs.
export class CrackSound {
  constructor(actx) { this.a = actx; this.out = null; this.last = -1; }
  init() {
    const a = this.a; if (!a || this.out) return !!this.out;
    const comp = a.createDynamicsCompressor(); comp.threshold.value = -14; comp.ratio.value = 5; comp.attack.value = 0.001; comp.release.value = 0.15;
    this.out = a.createGain(); this.out.gain.value = 1.4; this.out.connect(comp).connect(a.destination);
    const n = a.sampleRate, buf = a.createBuffer(1, n, a.sampleRate), d = buf.getChannelData(0);
    for (let i = 0; i < n; i++) d[i] = Math.random() * 2 - 1;
    this.noise = buf; return true;
  }
  burst(t, dur, type, f, q, gain) {
    const a = this.a, src = a.createBufferSource(); src.buffer = this.noise;
    const fl = a.createBiquadFilter(); fl.type = type; fl.frequency.value = f; fl.Q.value = q;
    const g = a.createGain(); g.gain.setValueAtTime(0.0001, t); g.gain.exponentialRampToValueAtTime(gain, t + 0.0015); g.gain.exponentialRampToValueAtTime(0.0001, t + dur);
    src.connect(fl).connect(g).connect(this.out); src.start(t, Math.random() * 0.9); src.stop(t + dur + 0.02);
  }
  tone(t, f, dur, gain, type = 'sine') {
    const a = this.a, o = a.createOscillator(), g = a.createGain(); o.type = type; o.frequency.setValueAtTime(f, t);
    g.gain.setValueAtTime(0.0001, t); g.gain.exponentialRampToValueAtTime(gain, t + 0.002); g.gain.exponentialRampToValueAtTime(0.0001, t + dur);
    o.connect(g).connect(this.out); o.start(t); o.stop(t + dur + 0.02);
  }
  thump(t, s) {   // the hit itself, stronger than an ordinary bump
    const a = this.a, o = a.createOscillator(), g = a.createGain();
    o.frequency.setValueAtTime(75, t); o.frequency.exponentialRampToValueAtTime(42, t + 0.2);
    g.gain.setValueAtTime(0.0001, t); g.gain.exponentialRampToValueAtTime(0.5 * s, t + 0.006); g.gain.exponentialRampToValueAtTime(0.0001, t + 0.22);
    o.connect(g).connect(this.out); o.start(t); o.stop(t + 0.25);
    this.burst(t, 0.12, 'lowpass', 500, 0.5, 0.55 * s);
  }
  crackle(t, s, n, span) {   // the shell splitting: a sharp snap, then little ticks running along the crack
    this.burst(t, 0.03, 'highpass', 2200, 0.7, 0.55 * s);
    this.burst(t, 0.05, 'bandpass', 1400, 1.2, 0.35 * s);
    for (let i = 0; i < n; i++) { const u = Math.pow(Math.random(), 1.6) * span; this.burst(t + 0.01 + u, 0.006 + 0.008 * Math.random(), 'bandpass', 2500 + Math.random() * 4500, 2.5, (0.12 + 0.25 * Math.random()) * s * (1 - u / span * 0.6)); }
    // the glaze rings a little (porcelain)
    const f = 2300 + Math.random() * 900; this.tone(t + 0.004, f, 0.14, 0.05 * s); this.tone(t + 0.004, f * 2.71, 0.08, 0.03 * s);
  }
  crack(strength, level) {
    if (!this.init()) return; const t = this.a.currentTime + 0.01, s = Math.min(1, 0.55 + 0.45 * strength);
    this.thump(t, s); this.crackle(t + 0.006, s * (0.8 + 0.15 * level), 6 + 4 * level, 0.09 + 0.05 * level);
  }
  shatter(strength) {
    if (!this.init()) return; const t = this.a.currentTime + 0.01, s = Math.min(1, 0.7 + 0.3 * strength);
    this.thump(t, s * 1.1); this.crackle(t + 0.004, s * 1.1, 22, 0.25);
    this.burst(t + 0.01, 0.35, 'highpass', 3000, 0.5, 0.3 * s);
    for (let i = 0; i < 18; i++) { const u = 0.03 + Math.pow(Math.random(), 1.4) * 0.6, f = 2600 + Math.random() * 5200; this.tone(t + u, f, 0.04 + 0.08 * Math.random(), (0.04 + 0.06 * Math.random()) * s * (1 - u)); this.burst(t + u, 0.008, 'bandpass', f, 3, 0.12 * s * (1 - u)); }
  }
  clink(v, m) {   // a piece landing
    if (!this.init()) return; const now = this.a.currentTime; if (now - this.last < 0.03) return; this.last = now;
    const t = now + 0.005, s = Math.min(1, v / 4) * (m > 0.5 ? 1 : 0.6), f = (m > 0.5 ? 1700 : 3200) + Math.random() * 1800;
    this.burst(t, 0.012, 'bandpass', f, 2, 0.9 * s); this.tone(t, f * 1.03, m > 0.5 ? 0.09 : 0.05, 0.14 * s);
    if (m > 0.5) this.burst(t, 0.08, 'lowpass', 400, 0.5, 0.3 * s);
  }
}
