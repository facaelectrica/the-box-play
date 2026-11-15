// The Box - cracking mode (user, Nov 15 09:33): "every time the model hits walls and surfaces, if they hit strong
// enough, that body part starts cracking ... it has to respect the part where it was hit. After each hit, that part gets
// more cracked. Each part can be hit 3 times before it breaks. When it breaks ... the puppet keeps trying to move
// without the part that's now broken, and [the part keeps] moving around too in the physics. If the head gets broken,
// he stops moving completely." A button turns it on and off; a double tap makes him brand new.
//
// Three pieces, kept apart so the physics can run headless in the tests:
//  - Cracks: reads MuJoCo's contacts after each frame. A part arriving on a wall/floor/ceiling faster than CRACK.minV
//    (speed of the touching point into the surface, just before it touched) cracks there, as deep as the speed says;
//    when a part's damage adds up to CRACK.breakAt
//    breaks the part off. A broken part and everything hanging from it (forearm -> hand) leave his physical body: no
//    collisions, almost no weight, no muscle; the brain still drives the rest of him. Losing the head (or the neck or
//    trunk above the hips, which carry it) = he stops: the rest of him goes limp, holding a little of his last pose.
//  - Debris: the broken-off piece (rigid, in the pose it broke in) and a few shards fly on in our own small rigid-body
//    physics: the box's six walls (or the open floor), the same gravity (phone tilt), his body as an obstacle.
//  - CrackView / CrackSound (page only): crack lines drawn on the part's surface where it was hit, the broken pieces,
//    and the sounds: a stronger thump + the shell cracking, a shatter when a part breaks, clinks of the pieces.
// Nothing here changes physics.js, the body file or training. The shell is drawn as glazed porcelain/ceramic.

export const CRACK = {
  // (user 12:22: "the speed at which he hits a surface should define how much damage it does ... a small distance and
  // not much velocity, it should not break") damage per hit = (speed into the surface - minV) / per, at most maxHit;
  // nothing below minV; a part breaks when its damage adds up to breakAt. Speed = the physics' own velocity of the
  // touching point just before it touched (not position jumps: the recordings move him around kinematically).
  minV: 2.8,          // m/s: below this a hit does no damage (footsteps ~0.3-1, a fall from standing ~2-3, the phone flipped ~5)
  per: 2.5,           // m/s above minV for one unit of damage
  maxHit: 1.5,        // the most one hit can do
  breakAt: 2.5,       // units of damage that break a part
  hits: 3,            // (kept for the tests' forced breaks)
  cool: 0.6,          // s: one impact (bounces, several walls in a corner) counts once per part
  away: 0.08,         // s: a part must have been off that surface this long for a new touch to count
  kind: { Foot: 1.5, Hand: 1.5 },   // feet and hands meet surfaces on purpose (support, catching himself): much more needed
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
  constructor(W) { this.W = W; this.list = []; this.events = []; this.removed = []; this.nid = 0; }
  clear() { this.list.length = 0; this.events.length = 0; this.removed.length = 0; }
  add(p) {
    p.id = this.nid++; p.touch = new Set(); p.clinkT = 0; this.list.push(p);
    // (user 16:30: nothing is ever erased until reset) every shard stays; resting ones sleep (no work), and the view
    // draws them all in a few shared batches
    return p;
  }
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
    dt = Math.min(dt, 1 / 30);   // a slow frame: the pieces go a bit slower instead of tiny shards taking huge steps and flying apart
    const W = this.W, g = W.m.opt.gravity, planes = W.faces().map(f => ({ n: f.n, p: f.p, name: f.name }));
    const obs = this.obstacles(skip), n = Math.min(6, Math.max(1, Math.ceil(dt / (1 / 200))));
    // his body's box (pieces far from it skip the body checks)
    const lo = [1e9, 1e9, 1e9], hi = [-1e9, -1e9, -1e9];
    for (const o of obs) for (const q of [o.a, o.b]) for (let k = 0; k < 3; k++) { lo[k] = Math.min(lo[k], q[k] - o.r); hi[k] = Math.max(hi[k], q[k] + o.r); }
    this.box = { lo, hi };
    // (user 11:04: no freezes) pieces lying still sleep: no work until the phone turns or his body comes near
    // a piece asleep wakes when gravity has turned more than ~4 deg since it fell asleep (a hand holding the phone moves it
    // a little all the time: waking everything at every tiny change kept hundreds of resting pieces busy)
    const gl = Math.hypot(g[0], g[1], g[2]) || 1, gd = [g[0] / gl, g[1] / gl, g[2] / gl];
    for (const P of this.list) if (P.sleep) {
      const sg = P.sleepG; if (!sg || gd[0] * sg[0] + gd[1] * sg[1] + gd[2] * sg[2] < 0.9976) { P.sleep = false; P.stillT = 0; continue; }
      const x = P.x, mg = P.bound + 0.01; if (x[0] < lo[0] - mg || x[0] > hi[0] + mg || x[1] < lo[1] - mg || x[1] > hi[1] + mg || x[2] < lo[2] - mg || x[2] > hi[2] + mg) continue;   // nowhere near him
      for (const o of obs) { const ab = sub(o.b, o.a), L2 = dot(ab, ab), u = L2 > 1e-12 ? Math.max(0, Math.min(1, dot(sub(P.x, o.a), ab) / L2)) : 0; if (len(sub(P.x, add(o.a, ab, u))) < P.bound + o.r + 0.004) { P.sleep = false; P.stillT = 0; break; } }
    }
    const awake = this.list.filter(P => !P.sleep), big = this.list.filter(P => P.kind !== 'shard');
    if (!awake.length) return;
    const run = (awake, ns) => { if (!awake.length) return; const h = dt / ns; for (let it = 0; it < ns; it++) {
      for (const P of awake) {
        P.v = add(P.v, g, h);
        P.R = q2m(P.q); P.Iinv = mmul(mmul(P.R, P.IinvB), mT(P.R));
        P.on = false; P.held = false;
      }
      for (const P of awake) this.collide(P, planes, obs, h);
      // pieces against each other (bounding spheres first; little shards don't bother each other)
      // (only the big pieces are walked: hundreds of shards would make this shards x shards per step)
      for (const A of awake) for (const B of big) { if (A === B || (!B.sleep && B.id < A.id && A.kind !== 'shard')) continue; this.pair(A, B); }
      for (const P of awake) {
        // resting ON something that holds it up against gravity (floor, a slope its friction can hold) and nearly still:
        // it stays put (no creeping from the contact pushes). Against a wall or a steep slope it keeps sliding/falling
        // (user 10:34: "they seem to end up glued on the surface")
        if (P.held && (P.kind === 'shard' ? len(P.v) < 0.15 : len(P.v) < 0.04 && len(P.w) < 0.4)) {   // (a shard is a flat splinter, not the ball it is simulated as: it doesn't roll down a gentle slope)
        P.v = [0, 0, 0]; P.w = [0, 0, 0]; P.stillT = (P.stillT || 0) + h; if (P.stillT > 0.4) { P.sleep = true; P.sleepG = gd; } continue; }
        if (len(P.v) > 0.3) P.stillT = 0;   // (only a real move restarts the count: a piece wedged in a corner jiggles a little)
        const damp = P.on ? 1 - Math.min(0.5, 6 * h) : 1 - 0.05 * h;
        P.v = P.v.map(v => v * (P.on ? 1 - Math.min(0.3, 0.8 * h) : 1)); P.w = P.w.map(v => v * damp);
        P.x = add(P.x, P.v, h);
        const wq = [0, P.w[0], P.w[1], P.w[2]], dq = qmul(wq, P.q);
        P.q = qnorm(P.q.map((v, k) => v + 0.5 * h * dq[k]));
      }
    } };
    // the big pieces with full substeps; the little shards with at most 2 (hundreds of them stay cheap)
    run(awake.filter(P => P.kind !== 'shard'), n); run(awake.filter(P => P.kind === 'shard'), Math.min(n, 2));
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
    const g = this.W.m.opt.gravity, gl = Math.hypot(g[0], g[1], g[2]) || 1, up = [-g[0] / gl, -g[1] / gl, -g[2] / gl];
    for (const s of P.s) {
      const c = add(P.x, mv(P.R, s.c));
      for (const pl of planes) {
        const dd = dot(sub(c, pl.p), pl.n) - s.r; if (dd >= 0) continue;
        const rp = add(sub(c, P.x), pl.n, -s.r), vp = add(P.v, cross(P.w, rp)), vn = -dot(vp, pl.n);
        const hit = this.impulse(P, rp, pl.n, vp, vn > 0.9 ? 0.25 : 0.02, 0.55);
        P.x = add(P.x, pl.n, -dd * 0.6); P.on = true;
        if (dot(pl.n, up) > 0.88) P.held = true;   // within the friction cone (0.55 -> 29 deg)
        if (hit > 0.6 && P.clinkT <= 0) { P.clinkT = 0.07; this.events.push({ type: 'clink', v: hit, m: P.m }); }
        if (P.kind === 'chunk' && s.b !== undefined && hit > 1.5) this.events.push({ type: 'phit', P, b: s.b, v: hit, p: add(c, pl.n, -s.r), n: pl.n.slice(), wall: pl.name });
      }
      if (P.age !== undefined && P.age < 0.06) continue;   // just broken off: let it clear his body first
      const B = this.box; if (c[0] < B.lo[0] - s.r || c[0] > B.hi[0] + s.r || c[1] < B.lo[1] - s.r || c[1] > B.hi[1] + s.r || c[2] < B.lo[2] - s.r || c[2] > B.hi[2] + s.r) continue;
      for (const o of obs) {
        const ab = sub(o.b, o.a), L2 = dot(ab, ab), u = L2 > 1e-12 ? Math.max(0, Math.min(1, dot(sub(c, o.a), ab) / L2)) : 0, q = add(o.a, ab, u);
        const dv = sub(c, q), dl = len(dv), pen = s.r + o.r - dl; if (pen <= 0 || dl < 1e-6) continue;
        const nrm = dv.map(v => v / dl), rp = add(sub(c, P.x), nrm, -s.r), vp = add(P.v, cross(P.w, rp));
        this.impulse(P, rp, nrm, vp, 0.1, 0.4); P.x = add(P.x, nrm, pen * 0.5); P.on = true;
        if (P.kind === 'shard' && dot(nrm, up) > 0.88) P.held = true;   // a splinter lying on top of him can rest too (it wakes when he moves into it)
      }
    }
  }
  pair(A, B) {
    if (len(sub(A.x, B.x)) > A.bound + B.bound) return;
    if (B.sleep) { B.sleep = false; B.stillT = 0; B.R = q2m(B.q); B.Iinv = mmul(mmul(B.R, B.IinvB), mT(B.R)); }
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
    this.wallName = {}; for (const n in W.wall) this.wallName[W.wall[n]] = n;
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
    const d = this.W.d, r = this.W.m.body_rootid[this.W.bodyId.Hips];
    return { t: this.t, x: Float64Array.from(d.xpos), R: Float64Array.from(d.xmat), cv: Float64Array.from(d.cvel), c0: [d.subtree_com[3 * r], d.subtree_com[3 * r + 1], d.subtree_com[3 * r + 2]] };
  }
  // the speed of a point fixed in body b (local l) between the last two frames
  pointVel(b, l) {
    const H = this.hist; if (H.length < 2) return null;
    const A = H[H.length - 1], B = H[H.length - 2], dt = A.t - B.t; if (dt <= 1e-4) return null;
    const pa = add([A.x[3 * b], A.x[3 * b + 1], A.x[3 * b + 2]], mv(A.R.subarray(9 * b, 9 * b + 9), l));
    const pb = add([B.x[3 * b], B.x[3 * b + 1], B.x[3 * b + 2]], mv(B.R.subarray(9 * b, 9 * b + 9), l));
    // the physics' own velocity of that point at the end of the last frame (com-based: [turn; move] about the root's centre)
    const cv = A.cv.subarray(6 * b, 6 * b + 6), w = [cv[0], cv[1], cv[2]], v = add([cv[3], cv[4], cv[5]], cross(w, sub(pa, A.c0)));
    void pb;
    return len(v) > 30 ? null : v;
  }
  // after each frame's physics. skip = he is being placed by a recording (kinematic get-up): no hits then.
  // support: he is on his feet (standing / recovering): feet on the ground and hands bracing do no damage then
  update(dt, skip = false, support = false) {
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
    if (!skip && !this.quiet) {   // (user 11:07) also once he has stopped: bounced around, he keeps cracking and breaking
      for (const [key, c] of now) {
        const last = this.touch.get(key);
        if (last !== undefined && this.t - last < CRACK.away + dt * 1.5) continue;   // still (or just) touching
        const b = c.b, R = d.xmat.subarray(9 * b, 9 * b + 9), x = [d.xpos[3 * b], d.xpos[3 * b + 1], d.xpos[3 * b + 2]];
        const l = mtv(R, sub(c.p, x)), v = this.pointVel(b, l); if (!v) continue;
        const vIn = -dot(v, c.n);
        const name = this.nameOf[b], piece = pieceOf(name), kind = piece.replace(/^(Left|Right)/, ''), k = CRACK.kind[kind] || 1;
        if (vIn > 1.0) this.stats.hits.push({ t: +this.t.toFixed(2), part: name, v: +vIn.toFixed(2) });
        // (user 12:22-12:24) feet and hands doing their job - standing, stepping, bracing on the walls - never get hurt
        if (support && (kind === 'Foot' || kind === 'Hand')) continue;
        const dmg = Math.min(CRACK.maxHit, (vIn - CRACK.minV * k) / CRACK.per); if (dmg < 0.05) continue;
        if (this.t - (this.lastHit[piece] || -9) < CRACK.cool) continue;
        this.lastHit[piece] = this.t;
        const pb = W.bodyId[piece];
        if (this.broken.has(pb)) continue;
        // where on the part: the touching point and the surface's outward normal, in the part's own frame
        const Rp = d.xmat.subarray(9 * pb, 9 * pb + 9), xp = [d.xpos[3 * pb], d.xpos[3 * pb + 1], d.xpos[3 * pb + 2]];
        const P = this.part(piece), s = Math.min(1, dmg / CRACK.maxHit);
        const hit = { body: pb, l: mtv(Rp, sub(c.p, xp)), n: mtv(Rp, c.n.map(v => -v)), v: vIn, s, dmg, seed: (Math.random() * 1e9) | 0, level: P.level + 1 };
        // where on the box: the wall's name and the spot on its face (relative to the face's centre, so it follows the box)
        { const wn = this.wallName[c.wall], f = wn && W.faces().find(q => q.name === wn);
          if (f) { const d0 = dot(sub(c.p, f.p), f.n), q = add(c.p, f.n, -d0); hit.wall = { name: wn, n: f.n.slice(), off: sub(q, f.p) }; } }
        P.level++; P.dmg = (P.dmg || 0) + dmg; P.hits.push(hit);
        if (P.dmg >= CRACK.breakAt) { this.stats.breaks++; ev.push({ type: 'break', piece, hit, ...this.breakOff(piece, c.n, vIn) }); }
        else { this.stats.cracks++; ev.push({ type: 'crack', piece, hit, level: P.level, chips: this.chips(c.p, c.n, v, Math.max(1, Math.round(dmg * 2))) }); }
      }
    }
    for (const [key] of now) this.touch.set(key, this.t);
    this.hist.push(this.snapshot()); if (this.hist.length > 2) this.hist.shift();
    // the pieces fly on
    for (const P of this.debris.list) P.age = (P.age || 0) + dt;
    this.debris.step(dt, this.broken);
    for (const e of this.debris.events.splice(0)) { if (e.type === 'phit') { const r = this.pieceHit(e); if (r) ev.push(r); } else ev.push(e); }
    this.events = ev;
    return ev;
  }
  // (user 10:43: "when it cracks, you could have little pieces that come off each time") chips off the shell and the
  // wood under it, thrown off the spot that hit
  chips(p, nOut, vPart, n) { return this.frags(add(p, nOut, 0.012), nOut, vPart.map(x => x * 0.15), n, 0.003, 0.016); }
  // fragments thrown from p: sizes from rMin to rMax, mostly small (a few big ones)
  frags(p, nOut, vBase, n, rMin, rMax) {
    const out = [], R0 = Math.random;
    for (let i = 0; i < n; i++) {
      const r = rMin + (rMax - rMin) * Math.pow(R0(), 2.2), ms = 700 * (4 / 3) * Math.PI * r * r * r, I = 0.4 * ms * r * r;
      const dir = [R0() - 0.5, R0() - 0.5, R0() - 0.5], dl = len(dir) || 1;
      const v = add(add(vBase, nOut, 0.8 + 1.8 * R0()), dir, (0.4 + 1.2 * R0()) / dl);
      out.push(this.debris.add({ kind: 'shard', m: ms, x: add(p, dir, (0.01 + r) / dl), q: qnorm([R0(), R0(), R0(), R0()]), v, w: [R0() * 40 - 20, R0() * 40 - 20, R0() * 40 - 20], IinvB: [1 / I, 0, 0, 0, 1 / I, 0, 0, 0, 1 / I], s: [{ c: [0, 0, 0], r }], bound: r, size: r, age: 0, seed: (R0() * 1e9) | 0, wood: R0() < 0.5 }));
    }
    return out;
  }
  // a part breaks off: it and what hangs from it leave his body and become a piece of debris (+ shards)
  breakOff(piece, nWall, vIn) {
    const W = this.W, m = W.m, d = W.d, mj = W.mj;
    const root = W.bodyId[BREAK_ROOT[piece] || piece];
    const bodies = this.subtree(root).filter(b => !this.broken.has(b));
    const killed = bodies.some(b => DEAD.has(this.nameOf[b])) || DEAD.has(piece);
    // the piece: mass, centre, turning inertia, collision spheres - from his body right now
    const pose = b => ({ x: [d.xpos[3 * b], d.xpos[3 * b + 1], d.xpos[3 * b + 2]], R: Array.from(d.xmat.subarray(9 * b, 9 * b + 9)) });
    const chunk = this.buildChunk(bodies, root, pose), com = chunk.x, Rr = q2m(chunk.q);
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
    chunk.v = v; chunk.w = w; this.debris.add(chunk);
    // a few fragments from the break, small to big (user 11:07: "less, with more size variation")
    const brk = len(sub(joint, com)) < 0.02 ? com : joint;
    const shards = this.frags(brk, nWall, v, 2 + Math.round(Math.random() * 2), 0.005, 0.03);
    // the part that broke starts over: hit again, it cracks again, and its 3rd crack breaks it further (user 11:07)
    this.part(piece).level = 0; this.part(piece).dmg = 0;
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
  // a rigid piece made of these bodies at these poses (b -> { x, R } in the world): mass, centre, turning inertia, collision
  // spheres (from the body file's own shapes), and where each body sits in it (for drawing)
  buildChunk(bodies, root, pose) {
    const m = this.W.m;
    let M = 0, com = [0, 0, 0]; const P = {};
    for (const b of bodies) { const ps = pose(b), ip = add(ps.x, mv(ps.R, [m.body_ipos[3 * b], m.body_ipos[3 * b + 1], m.body_ipos[3 * b + 2]])); P[b] = { ...ps, ip }; const mb = this.orig.mass[b]; M += mb; com = add(com, ip, mb); }
    com = com.map(v => v / M);
    const Rr = P[root].R, Rt = mT(Rr);
    const Iw = [0, 0, 0, 0, 0, 0, 0, 0, 0];
    for (const b of bodies) {
      const Ri = mmul(P[b].R, q2m([m.body_iquat[4 * b], m.body_iquat[4 * b + 1], m.body_iquat[4 * b + 2], m.body_iquat[4 * b + 3]])), I = this.orig.inertia.subarray(3 * b, 3 * b + 3), mb = this.orig.mass[b];
      const Ib = mmul(mmul(Ri, [I[0], 0, 0, 0, I[1], 0, 0, 0, I[2]]), mT(Ri)), r = sub(P[b].ip, com), r2 = dot(r, r);
      for (let i = 0; i < 3; i++) for (let j = 0; j < 3; j++) Iw[3 * i + j] += Ib[3 * i + j] + mb * ((i === j ? r2 : 0) - r[i] * r[j]);
    }
    const IB = mmul(mmul(Rt, Iw), Rr); for (let i = 0; i < 3; i++) IB[4 * i] += 1e-5;
    const spheres = [];
    for (const b of bodies) for (const g of this.geomsOf[b]) {
      const t = m.geom_type[g], s = m.geom_size.subarray(3 * g, 3 * g + 3);
      const p = add(P[b].x, mv(P[b].R, [m.geom_pos[3 * g], m.geom_pos[3 * g + 1], m.geom_pos[3 * g + 2]])), R = mmul(P[b].R, q2m([m.geom_quat[4 * g], m.geom_quat[4 * g + 1], m.geom_quat[4 * g + 2], m.geom_quat[4 * g + 3]]));
      const put = (c, r) => spheres.push({ c: mtv(Rr, sub(c, com)), r, b });
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
    const bound = Math.max(...spheres.map(q => len(q.c) + q.r));
    return { kind: 'chunk', bodies, root, m: M, x: com, q: m2q(Rr), v: [0, 0, 0], w: [0, 0, 0], IinvB: minv(IB), s: spheres, bound, comL: mtv(Rr, sub(com, P[root].x)), age: 0,
      rel: bodies.map(b => ({ b, p: mtv(Rr, sub(P[b].x, com)), q: m2q(mmul(Rt, P[b].R)) })) };
  }
  // a broken-off piece hitting the box hard: it cracks where it hit like his body does, and the 3rd crack on a part
  // splits the piece there (an arm at the elbow, then the hand...); a single part shatters into a few fragments
  pieceHit(e) {
    const P = e.P, W = this.W; if (!this.debris.list.includes(P)) return null;
    const name = this.nameOf[e.b], piece = pieceOf(name), pb = W.bodyId[piece], k = CRACK.kind[piece.replace(/^(Left|Right)/, '')] || 1;
    const dmg = Math.min(CRACK.maxHit, (e.v - CRACK.minV * k) / CRACK.per);
    if (!P.bodies.includes(pb) || dmg < 0.05 || this.t - (this.lastHit[piece] || -9) < CRACK.cool) return null;
    this.lastHit[piece] = this.t;
    const rel = P.rel.find(r => r.b === pb), bp = this.chunkBodyPose(P, rel), Rb = q2m(bp.q);
    const Pt = this.part(piece), s = Math.min(1, dmg / CRACK.maxHit);
    const hit = { body: pb, l: mtv(Rb, sub(e.p, bp.p)), n: mtv(Rb, e.n.map(v => -v)), v: e.v, s, dmg, seed: (Math.random() * 1e9) | 0, level: Pt.level + 1, chunk: P };
    const f = W.faces().find(q => q.name === e.wall); if (f) { const d0 = dot(sub(e.p, f.p), f.n), q = add(e.p, f.n, -d0); hit.wall = { name: e.wall, n: f.n.slice(), off: sub(q, f.p) }; }
    Pt.level++; Pt.dmg = (Pt.dmg || 0) + dmg; Pt.hits.push(hit);
    const vp = add(P.v, cross(P.w, sub(e.p, P.x)));
    if (Pt.dmg < CRACK.breakAt) { this.stats.cracks++; return { type: 'crack', piece, hit, level: Pt.level, chips: this.chips(e.p, e.n, vp, 1 + (Math.random() < 0.5 ? 1 : 0)) }; }
    this.stats.breaks++; Pt.level = 0; Pt.dmg = 0;
    const R = q2m(P.q), pose = b => { const r = P.rel.find(q => q.b === b), bpz = this.chunkBodyPose(P, r); return { x: bpz.p, R: q2m(bpz.q) }; };
    const inP = b => P.bodies.includes(b), groupsOf = [];
    if (pb !== P.root) { const sub1 = this.subtree(pb).filter(inP); groupsOf.push(sub1, P.bodies.filter(b => !sub1.includes(b))); }
    else { for (const c of this.children[pb]) { const st = this.subtree(c).filter(inP); if (st.length) groupsOf.push(st); } if (groupsOf.length) groupsOf.unshift([pb]); }
    this.debris.list.splice(this.debris.list.indexOf(P), 1);
    if (!groupsOf.length) {   // a single part: it shatters
      const frs = this.frags(P.x, e.n, P.v, 3 + Math.round(Math.random() * 2), 0.008, 0.035);
      return { type: 'shatter', piece, hit, P, shards: frs };
    }
    const parts = groupsOf.map(bs => {
      const top = bs.find(b => !bs.includes(this.W.m.body_parentid[b])), C = this.buildChunk(bs, top, pose);
      const kick = sub(C.x, P.x), kl = len(kick) || 1;
      C.v = add(add(P.v, cross(P.w, sub(C.x, P.x))), kick, 0.4 / kl); C.w = P.w.slice(); C.age = 0;
      return this.debris.add(C);
    });
    const frs = this.frags(e.p, e.n, P.v, 1 + Math.round(Math.random() * 2), 0.005, 0.025);
    return { type: 'split', piece, hit, P, parts, shards: frs };
  }
  // (tests) break a part now
  forceBreak(piece) {
    const W = this.W, d = W.d, b = W.bodyId[piece], P = this.part(piece); P.level = CRACK.hits; P.dmg = CRACK.breakAt;
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
    this.inner = new THREE.MeshStandardMaterial({ color: 0xc98a4a, roughness: 0.95, side: THREE.DoubleSide });   // the raw wood inside a break
    // (user 14:15: "simulate depth in the geometry ... versus something that gets blacker and just looks painted on")
    // a crack is a real groove: walls and a bottom built below the surface and lit by the scene's lights. The part's
    // shell is intact, so the groove is seen through its opening: the opening is first marked in the stencil buffer
    // (crackMask, drawn at the surface, hidden wherever something is in front of it), then the groove is drawn only
    // inside that mark, pulled toward the camera along each view ray so the shell doesn't hide it (moving a point along
    // its own view ray keeps it on the same pixel: the walls stay exactly where they are in 3D, depth order among them
    // is kept, and the light/shadows are taken at their real position)
    this.crackMask = new THREE.MeshBasicMaterial({ colorWrite: false, depthWrite: false, polygonOffset: true, polygonOffsetFactor: -4, polygonOffsetUnits: -4,
      stencilWrite: true, stencilRef: 1, stencilFunc: THREE.AlwaysStencilFunc, stencilZPass: THREE.ReplaceStencilOp });
    this.crackMat = new THREE.MeshStandardMaterial({ vertexColors: true, roughness: 0.82, metalness: 0,
      stencilWrite: true, stencilRef: 1, stencilFunc: THREE.EqualStencilFunc, stencilFail: THREE.KeepStencilOp, stencilZFail: THREE.KeepStencilOp, stencilZPass: THREE.KeepStencilOp });
    this.crackMat.onBeforeCompile = sh => { sh.vertexShader = sh.vertexShader.replace('#include <project_vertex>', '#include <project_vertex>\n\tmvPosition.xyz += normalize( - mvPosition.xyz ) * 0.03;\n\tgl_Position = projectionMatrix * mvPosition;'); };
    this.crackMat.customProgramCacheKey = () => 'crackGroove';
    this.splMat = new THREE.MeshStandardMaterial({ vertexColors: true, roughness: 0.9, side: THREE.DoubleSide });   // splinters at a hole's rim
    this.shardMat = new THREE.MeshStandardMaterial({ vertexColors: true, roughness: 0.75 });   // the fragments (instanced)
    this.marks = [];
  }
  // heavy drawing work waits for the next frames (a few ms per frame), so a hit never stalls a frame
  later(fn) { (this.jobs || (this.jobs = [])).push(fn); }
  runJobs(ms = 5) { const J = this.jobs; if (!J || !J.length) return; const t0 = performance.now(); do { try { J.shift()(); } catch (e) { console.warn('crack job', e); } } while (J.length && performance.now() - t0 < ms); }
  // the first crack, wall mark and piece would each make the phone prepare a new shader mid-game (a visible hitch): draw
  // one tiny copy of each while the page starts, so they are ready
  prewarm(at) {
    const T = this.T, g = new T.Group(); g.position.set(...at); g.scale.setScalar(0.001);
    const tri = new T.BufferGeometry(); tri.setAttribute('position', new T.Float32BufferAttribute([0, 0, 0, 1, 0, 0, 0, 1, 0], 3)); tri.setAttribute('normal', new T.Float32BufferAttribute([0, 0, 1, 0, 0, 1, 0, 0, 1], 3)); tri.setAttribute('color', new T.Float32BufferAttribute([1, 1, 1, 1, 1, 1, 1, 1, 1], 3));
    g.add(new T.Mesh(tri, this.crackMask)); g.add(new T.Mesh(tri, this.crackMat)); g.add(new T.Mesh(tri, this.splMat)); { const im = new T.InstancedMesh(tri, this.shardMat, 1); im.castShadow = true; g.add(im); }
    for (const m of [this.inner, this.shell]) { const me = new T.Mesh(new T.IcosahedronGeometry(1, 0), m); me.castShadow = true; me.receiveShadow = true; g.add(me); }
    this.markMats = [0, 1, 2, 3].map(k => this.markMat(0.4 + 0.2 * k, 1000 + k * 77));
    for (const mt of this.markMats) { const mm = new T.Mesh(new T.PlaneGeometry(1, 1), mt); mm.receiveShadow = true; g.add(mm); }   // (also sends their textures to the graphics chip now)
    g.traverse(o => { o.frustumCulled = false; });
    this.root.add(g); this.warm = { g, n: 0 };
  }
  reset(all = true) {
    if (all) { if (this.walls) for (const W of this.walls.values()) { this.root.remove(W.grp); for (const B of [W.mask, W.groove, W.spl]) if (B.mesh) B.mesh.geometry.dispose(); } this.walls = new Map(); this.marks.length = 0; }
    for (const m of this.crackMeshes) { if (m.parent) m.parent.remove(m); m.traverse(o => { if (o.geometry) o.geometry.dispose(); }); }
    this.crackMeshes.length = 0;
    this.nets = new Map();
    for (const [, o] of this.debris) if (!o.shard) { this.root.remove(o); const i = this.movers.indexOf(o); if (i >= 0) this.movers.splice(i, 1); }
    this.debris.clear();
    if (this.shards) { for (const B of this.shards.batch) if (B.mesh) { this.root.remove(B.mesh); const i = this.movers.indexOf(B.mesh); if (i >= 0) this.movers.splice(i, 1); B.mesh.dispose(); } this.shards = null; }
    for (const g of this.groups()) if (g) g.visible = true;
    if (all && this.jobs) this.jobs = this.jobs.filter(f => f.keep);   // (the parts' surface lists stay queued)
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
    // (user 11:04: no freezes) the shell's surface as points with normals in the part's own frame, in a 1 cm grid: a
    // crack walks over the surface by snapping to the nearest point's tangent plane (constant time per step; casting
    // rays at every triangle cost up to half a second per crack)
    const vp = [], vn = [], v = new T.Vector3(), nv = new T.Vector3(), NM = new T.Matrix3();
    for (const me of meshes) {
      M.multiplyMatrices(inv, me.matrixWorld); NM.getNormalMatrix(M); const P = me.geometry.attributes.position, N = me.geometry.attributes.normal; if (!N) continue;
      for (let i = 0; i < P.count; i++) { v.fromBufferAttribute(P, i).applyMatrix4(M); nv.fromBufferAttribute(N, i).applyMatrix3(NM).normalize(); vp.push(v.x, v.y, v.z); vn.push(nv.x, nv.y, nv.z); }
    }
    const grid = new Map(), CS = 0.01, key = (x, y, z) => (Math.floor(x / CS) + 512) * 1048576 + (Math.floor(y / CS) + 512) * 1024 + (Math.floor(z / CS) + 512);
    for (let i = 0; i < vp.length / 3; i++) { const k = key(vp[3 * i], vp[3 * i + 1], vp[3 * i + 2]); let a = grid.get(k); if (!a) grid.set(k, a = []); a.push(i); }
    c = { meshes, a: ctr.clone().addScaledVector(axis, -half), b: ctr.clone().addScaledVector(axis, half), size: sorted[1], vp: new Float32Array(vp), vn: new Float32Array(vn), grid, CS };
    this.cache.set(g, c); return c;
  }
  // a point near the part's surface -> the nearest surface point (on the tangent plane of the nearest shell point)
  snap(I, p, maxD) {
    const CS = I.CS, cx = Math.floor(p.x / CS), cy = Math.floor(p.y / CS), cz = Math.floor(p.z / CS), rr = Math.ceil(maxD / CS);
    let best = maxD * maxD, bi = -1;
    for (let r = 0; r <= rr && bi < 0; r++) {   // grow the search ring until something is found
      for (let dx = -r; dx <= r; dx++) for (let dy = -r; dy <= r; dy++) for (let dz = -r; dz <= r; dz++) {
        if (Math.max(Math.abs(dx), Math.abs(dy), Math.abs(dz)) !== r) continue;
        const a = I.grid.get((cx + dx + 512) * 1048576 + (cy + dy + 512) * 1024 + (cz + dz + 512)); if (!a) continue;
        for (const i of a) { const ex = I.vp[3 * i] - p.x, ey = I.vp[3 * i + 1] - p.y, ez = I.vp[3 * i + 2] - p.z, d2 = ex * ex + ey * ey + ez * ez; if (d2 < best) { best = d2; bi = i; } }
      }
      if (bi >= 0 && r === 0) { r++; for (let dx = -1; dx <= 1; dx++) for (let dy = -1; dy <= 1; dy++) for (let dz = -1; dz <= 1; dz++) {   // the neighbours too: the nearest may be across a cell edge
        const a = I.grid.get((cx + dx + 512) * 1048576 + (cy + dy + 512) * 1024 + (cz + dz + 512)); if (!a) continue;
        for (const i of a) { const ex = I.vp[3 * i] - p.x, ey = I.vp[3 * i + 1] - p.y, ez = I.vp[3 * i + 2] - p.z, d2 = ex * ex + ey * ey + ez * ez; if (d2 < best) { best = d2; bi = i; } } } }
    }
    if (bi < 0) return null;
    const T = this.T, n = new T.Vector3(I.vn[3 * bi], I.vn[3 * bi + 1], I.vn[3 * bi + 2]), v = new T.Vector3(I.vp[3 * bi], I.vp[3 * bi + 1], I.vp[3 * bi + 2]);
    { const ab = I.b.clone().sub(I.a), L2 = ab.lengthSq(), u = L2 > 1e-10 ? Math.max(0, Math.min(1, v.clone().sub(I.a).dot(ab) / L2)) : 0; if (v.clone().sub(I.a.clone().addScaledVector(ab, u)).dot(n) < 0) n.negate(); }   // outward
    return { p: p.clone().addScaledVector(n, -p.clone().sub(v).dot(n)), n };
  }
  project(g, I, p, prevN) { return this.snap(I, p, 0.02); }
  // (user 13:58: "what's already a crack gets more cracks. Not overlapping") each part keeps ONE crack network that
  // grows like a real fracture: a hit opens the cracks near it wider, pushes their tips further, and starts new cracks
  // only where the surface is still whole; a growing crack stops where it runs into another one (it can't cross it);
  // a part can only hold so much cracking - after that hits only open the cracks and chip flakes off. Seeded per hit, so
  // swapping the figure re-grows the same network.
  net(key) { const N = this.nets || (this.nets = new Map()); let n = N.get(key); if (!n) N.set(key, n = { lines: [], chips: [], grid: new Map(), len: 0, mesh: null }); return n; }
  addCrack(hit, target = null) {   // target: the drawn copy of that part on a broken-off piece
    const T = this.T, gs = this.groups(); let g = gs[hit.body]; if (!g) return;
    this.root.updateMatrixWorld(true);
    let I = this.info(g), l = new T.Vector3(...hit.l), n0 = new T.Vector3(...hit.n).normalize(), key = hit.body;
    if (!I.meshes.length) {
      if (target) return;
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
      g = bg; I = this.info(g); l = pw.applyMatrix4(inv); n0 = nw.transformDirection(inv); key = g.userData.body;
    }
    const start = this.snap(I, l, 0.06); if (!start) return;
    const N = this.net(key);
    this.grow(N, I, start, hit);
    // the part's one crack mesh, redrawn; on a broken-off piece it goes on that piece's copy of the part. Built a couple
    // of ms per frame (a part full of cracks took up to ~10 ms in one go); the old mesh stays until the new one is done
    const gen = N.gen = (N.gen || 0) + 1, it = this.drawNetIter(N);
    const step = () => {
      if (N.gen !== gen || this.nets.get(key) !== N) return;   // a newer hit (or a reset) took over
      const t0 = performance.now(); let r;
      do r = it.next(); while (!r.done && performance.now() - t0 < 2);
      if (!r.done) { this.later(step); return; }
      for (const o of [g, target].filter(Boolean)) for (const c of o.children.slice()) if (c.userData.crackOf === key) o.remove(c);
      if (N.mesh) { N.mesh.traverse(o => { if (o.geometry) o.geometry.dispose(); }); const k = this.crackMeshes.indexOf(N.mesh); if (k >= 0) this.crackMeshes.splice(k, 1); }
      N.mesh = r.value;
      if (N.mesh) { N.mesh.userData.crackOf = key; (target || g).add(N.mesh); this.crackMeshes.push(N.mesh); }
    };
    step();
  }
  // the network's segments in a 1 cm grid, for "does this new crack run into an old one"
  segAdd(N, li, si) { const L = N.lines[li], a = L.pts[si].p, b = L.pts[si + 1].p, CS = 0.01; for (const q of [a, b, a.clone().add(b).multiplyScalar(0.5)]) { const k = `${Math.floor(q.x / CS)},${Math.floor(q.y / CS)},${Math.floor(q.z / CS)}`; let A = N.grid.get(k); if (!A) N.grid.set(k, A = []); if (!A.some(e => e[0] === li && e[1] === si)) A.push([li, si]); } N.len += a.distanceTo(b); }
  segNear(N, p, R, skip) {
    const CS = 0.01, cx = Math.floor(p.x / CS), cy = Math.floor(p.y / CS), cz = Math.floor(p.z / CS); let best = { d: R, q: null };
    for (let dx = -1; dx <= 1; dx++) for (let dy = -1; dy <= 1; dy++) for (let dz = -1; dz <= 1; dz++) {
      const A = N.grid.get(`${cx + dx},${cy + dy},${cz + dz}`); if (!A) continue;
      for (const [li, si] of A) {
        if (skip && skip(li, si)) continue;
        const a = N.lines[li].pts[si].p, b = N.lines[li].pts[si + 1].p, ab = b.clone().sub(a), L2 = ab.lengthSq(), u = L2 > 1e-12 ? Math.max(0, Math.min(1, p.clone().sub(a).dot(ab) / L2)) : 0, q = a.clone().addScaledVector(ab, u), d = q.distanceTo(p);
        if (d < best.d) best = { d, q, li, si, u };
      }
    }
    return best;
  }
  // one crack running over the surface from p0 along dir for up to L, tapering from w0; stops where it meets another
  // crack (joined to it) or runs off the shell; may fork
  run(N, I, R, p0, n0, dir, L, w0, S, depth, li0 = -1) {
    // (user 14:16: "too veiny ... sharper, like a more solid material breaking ... not so many little curves") a crack
    // runs straight in legs of 0.8-2.5 cm and turns sharply between them (7-30 deg); forks only at a turn
    const line = { pts: [{ p: p0.clone(), n: n0.clone(), w: w0 }], side: R() < 0.5 ? 1 : -1, open: true }, li = N.lines.length; N.lines.push(line);
    let p = p0.clone(), n = n0.clone(), d = dir.clone(), went = 0, leg = (0.008 + 0.017 * R()) * S; const step = 0.003 * S;
    const wAt = x => w0 * Math.max(0.25, 1 - 0.75 * x / L);
    while (went < L) {
      d.addScaledVector(n, -d.dot(n)).normalize();
      if (went >= leg) {
        d.applyAxisAngle(n, (R() < 0.5 ? -1 : 1) * (0.12 + 0.4 * R())); leg = went + (0.008 + 0.017 * R()) * S;
        if (depth < 2 && R() < 0.3 && L - went > 0.012 * S) this.run(N, I, R, p, n, d.clone().applyAxisAngle(n, (R() < 0.5 ? -1 : 1) * (0.45 + 0.5 * R())), (L - went) * (0.3 + 0.35 * R()), wAt(went) * 0.7, S, depth + 1, li);
      }
      const want = p.clone().addScaledVector(d, step), h = this.snap(I, want.clone().addScaledVector(n, 0.0015), 0.02); if (!h) break;
      // the surface turned away or the nearest surface is somewhere else (the next finger, the other side of a thin part):
      // the crack ends at the edge instead of jumping across
      if (h.n.dot(n) < 0.5 || h.p.distanceTo(want) > 0.6 * step + 0.0012) break;
      const seg = h.p.distanceTo(p); if (seg < 1e-5) break;
      // ran into another crack: join it and stop (cracks don't cross)
      const own = line.pts.length;
      const hit = this.segNear(N, h.p, 0.0022 * S, (a, b) => (a === li && b >= own - 3) || (a === li0 && went < 0.006 * S));
      p = hit.q ? hit.q : h.p; n = h.n; went += seg;
      line.pts.push({ p: p.clone(), n: n.clone(), w: wAt(went) });
      this.segAdd(N, li, line.pts.length - 2);
      if (hit.q) { line.open = false; break; }
    }
    if (line.pts.length < 2) { N.lines.pop(); return null; }
    return line;
  }
  grow(N, I, start, hit) {
    const T = this.T, R = rng(hit.seed), E = Math.max(0.15, Math.min(1.5, hit.dmg !== undefined ? hit.dmg : 0.3 + hit.s));
    const S = Math.min(1, Math.max(0.55, I.size / 0.12)), c = start.p, nC = start.n; N.I = I;
    const CAP = 0.55 * S;   // the most crack a part can hold (m); after that hits only open cracks and chip
    // 1. the cracks near the hit open wider (more for a harder hit, less further away)
    const Rw = (0.05 + 0.03 * E) * S;
    for (const L of N.lines) for (const q of L.pts) { const dd = q.p.distanceTo(c); if (dd < Rw) { const f = 1 - dd / Rw; q.w = Math.min(0.0028 * S, q.w * (1 + 0.35 * E * f) + 0.00025 * E * f * S); } }
    // 2. their open tips nearby run further
    const near = this.segNear(N, c, 0.03 * S);
    if (N.len < CAP) for (const L of N.lines.slice()) {
      if (!L.open) continue; const tip = L.pts[L.pts.length - 1], dd = tip.p.distanceTo(c); if (dd > 0.12 * S) continue;
      const prev = L.pts[L.pts.length - 2], dir = tip.p.clone().sub(prev.p).normalize();
      L.open = false;   // it continues as a new piece of line from its tip
      this.run(N, I, R, tip.p, tip.n, dir, (0.02 + 0.03 * E) * S * (1 - dd / (0.12 * S)), tip.w, S, 1, N.lines.indexOf(L));
    }
    // 3. new cracks from the hit: a star on whole surface; near an old crack only a couple of short ones that run into it
    if (N.len < CAP) {
      const fresh = !near.q, B = fresh ? 3 + Math.round(R() * 2) : 1 + Math.round(R()), Lm = (fresh ? 0.03 + 0.05 * E : 0.015 + 0.025 * E) * S, w0 = (0.0010 + 0.0012 * E) * S;
      const t0 = new T.Vector3(1, 0, 0); if (Math.abs(t0.dot(nC)) > 0.9) t0.set(0, 1, 0); t0.addScaledVector(nC, -t0.dot(nC)).normalize();
      const a0 = R() * Math.PI * 2;
      for (let i = 0; i < B; i++) this.run(N, I, R, c, nC, t0.clone().applyAxisAngle(nC, a0 + (i + (R() - 0.5) * 0.7) * 2 * Math.PI / B), Lm * (0.5 + 0.7 * R()), w0, S, 0);
    }
    // 4. (user 14:41: "more of the material underneath ... cracking the outer layer") the outer coat flakes off: most hits
    // knock a flake off at the hit (or make the flake already there bigger), harder ones another along the edge of a crack
    // nearby, where the coat has nothing holding it; the raw wood shows underneath
    const flake = (p, n, r, seed) => {
      const rMax = Math.min(0.02 * S, 0.3 * I.size); r = Math.min(r, rMax);
      for (const f of N.chips) if (f.p.distanceTo(p) < 0.8 * (f.r + r)) { f.r = Math.min(rMax, Math.sqrt(f.r * f.r + 0.6 * r * r)); f.grow = (f.grow || 0) + 1; return; }
      if (N.chips.length < Math.round(5 + 11 * S)) N.chips.push({ p: p.clone(), n: n.clone(), r, seed });
    };
    if (E > 0.3) flake(c, nC, (0.003 + 0.005 * E) * S, hit.seed);
    if (E > 0.6) {
      const q = this.segNear(N, c.clone().addScaledVector(nC, 0), 0.05 * S);
      if (q.q) {
        const L = N.lines[q.li], A = L.pts[q.si], B = L.pts[q.si + 1], t = B.p.clone().sub(A.p).normalize(), side = A.n.clone().cross(t).normalize().multiplyScalar(R() < 0.5 ? 1 : -1);
        const r = (0.003 + 0.004 * E) * S, h = this.snap(I, q.q.clone().addScaledVector(side, 0.7 * r + 0.5 * A.w), 0.02);
        if (h) flake(h.p, h.n, r, hit.seed * 7 + 3);
      }
    }
  }
  // the network drawn as real grooves (see crackMat): across each crack a chipped bevel on each side (the paint broken
  // off its edge), two walls going down to a bottom; flat-shaded faces with their true normals, so the light picks out
  // the lit wall and leaves the other in its own shade; only a little darkening toward the bottom where light can't get
  // in. A wider crack is also deeper and shows more wall, not more black. Flakes = shallow pits with sloped sides and a
  // floor of raw wood.
  drawNet(N) { const it = this.drawNetIter(N); let r; do r = it.next(); while (!r.done); return r.value; }
  *drawNetIter(N) {
    const T = this.T, pos = [], nor = [], col = [], mpos = []; let cnt = 0;
    const put = (p, n, c) => { pos.push(p.x, p.y, p.z); nor.push(n.x, n.y, n.z); col.push(c.r, c.g, c.b); };
    const face = (a, b, c, na, nb, nc, ca, cb, cc) => {   // a triangle facing the way its normals point
      const up = na.clone().add(nb).add(nc); if (b.clone().sub(a).cross(c.clone().sub(a)).dot(up) >= 0) { put(a, na, ca); put(b, nb, cb); put(c, nc, cc); } else { put(a, na, ca); put(c, nc, cc); put(b, nb, cb); }
    };
    const mask = (a, b, c) => mpos.push(a.x, a.y, a.z, b.x, b.y, b.z, c.x, c.y, c.z);
    const BEV = new T.Color(0xe2c49c), WALL = new T.Color(0xc4905c), BOT = new T.Color(0x6e4a2c), FLOOR = new T.Color(0xd9aa72);
    const LIFT = 0.0005, hash = (i, k) => { const x = Math.sin(i * 127.1 + k * 311.7) * 43758.5453; return x - Math.floor(x); };
    let pid = 0;
    // the cross-section of a crack at one point: [surface L, bevel L, bottom, bevel R, surface R] + the band normals
    const COAT = 0.0011;   // the outer coat's thickness: a flake's floor is the wood under it
    const inFlake = p => { for (const f of N.chips) if (f.p.distanceTo(p) < 0.6 * f.r) return true; return false; };
    const section = (q, s, t, j) => {
      const w = q.w, h = 0.5 * w, D = 0.9 * w + 0.0004 + COAT, bL = (0.2 + 0.45 * hash(j, 1)) * w + 0.00015, bR = (0.2 + 0.45 * hash(j, 2)) * w + 0.00015;
      if (inFlake(q.p)) q = { p: q.p.clone().addScaledVector(q.n, -COAT), n: q.n, w };   // inside a flake the crack runs in the wood
      const P = [q.p.clone().addScaledVector(s, -(h + bL)), q.p.clone().addScaledVector(s, -h).addScaledVector(q.n, -0.6 * bL),
        q.p.clone().addScaledVector(s, (hash(j, 3) - 0.5) * 0.5 * h).addScaledVector(q.n, -D),
        q.p.clone().addScaledVector(s, h).addScaledVector(q.n, -0.6 * bR), q.p.clone().addScaledVector(s, h + bR)];
      const Nb = []; for (let k = 0; k < 4; k++) { const v = P[k + 1].clone().sub(P[k]), m = t.clone().cross(v).normalize(); if (m.dot(q.n) < 0) m.negate(); Nb.push(m); }
      return { P, Nb };
    };
    const C = [BEV, WALL, BOT, WALL, BEV];
    for (const L of N.lines) {
      const secs = L.pts.map((q, i) => {
        const a = L.pts[Math.max(0, i - 1)].p, b = L.pts[Math.min(L.pts.length - 1, i + 1)].p, t = b.clone().sub(a).normalize();
        const s = q.n.clone().cross(t).normalize(); return section(q, s, t, pid++);
      });
      for (let i = 0; i < secs.length - 1; i++) {
        const A = secs[i], B = secs[i + 1];
        for (let k = 0; k < 4; k++) {   // bands: bevel, wall, wall, bevel - each flat-shaded with its own normal
          face(A.P[k], A.P[k + 1], B.P[k], A.Nb[k], A.Nb[k], B.Nb[k], C[k], C[k + 1], C[k]);
          face(A.P[k + 1], B.P[k + 1], B.P[k], A.Nb[k], B.Nb[k], B.Nb[k], C[k + 1], C[k + 1], C[k]);
        }
        const nA = L.pts[i].n, nB = L.pts[i + 1].n, up = (p, n) => p.clone().addScaledVector(n, LIFT);
        if (++cnt % 25 === 0) yield;
        mask(up(A.P[0], nA), up(B.P[0], nB), up(A.P[4], nA)); mask(up(A.P[4], nA), up(B.P[0], nB), up(B.P[4], nB));   // (wound to face out of the surface)
      }
    }
    // flakes: the coat's broken edge (a thin step, coat colour fading into wood) around a floor of raw wood one coat
    // below the surface; rim, floor and the hole's mark all follow the part's curved surface (snapped rings)
    const COATC = new T.Color(0xd6cdbf), STEP = new T.Color(0x9a6a40), FLOORM = FLOOR.clone().multiplyScalar(0.88), FLOORE = FLOOR.clone().multiplyScalar(0.55), I = N.I;
    const sn = p => (I && this.snap(I, p, 0.02)) || null;
    for (const ch of N.chips) {
      const R = rng(ch.seed * 3 + 1), nn = ch.n, t0 = new T.Vector3(1, 0, 0); if (Math.abs(t0.dot(nn)) > 0.9) t0.set(0, 1, 0); t0.addScaledVector(nn, -t0.dot(nn)).normalize();
      const K = 7 + Math.round(R() * 3), a0 = R() * 6.28, c0 = sn(ch.p) || { p: ch.p, n: nn };
      const rim = [], mid = [];
      // each rim point is walked out over the surface in 4 steps; at an edge (the surface turns away or jumps to another
      // finger) the flake stops there
      for (let i = 0; i < K; i++) {
        const dir = t0.clone().applyAxisAngle(nn, a0 + (i + (R() - 0.5) * 0.5) / K * Math.PI * 2), r = ch.r * (0.55 + 0.6 * R());
        let q = c0, m = c0;
        for (let k = 1; k <= 4; k++) {
          const dd = dir.clone().addScaledVector(q.n, -dir.dot(q.n)).normalize(), want = q.p.clone().addScaledVector(dd, r / 4), h = sn(want.clone().addScaledVector(q.n, 0.001));
          if (!h || h.n.dot(q.n) < 0.6 || h.p.distanceTo(want) > 0.5 * r / 4 + 0.001) break;
          q = h; if (k === 2) m = h;
        }
        if (m === c0 && q !== c0) m = { p: c0.p.clone().lerp(q.p, 0.5), n: c0.n.clone().add(q.n).normalize() };
        rim.push(q); mid.push(m);
      }
      const dn = q => q.p.clone().addScaledVector(q.n, -COAT), upv = q => q.p.clone().addScaledVector(q.n, LIFT);
      const fc = dn(c0), fm = mid.map(q => dn(q).addScaledVector(q.n, -0.35 * COAT * R())), fr = rim.map(q => dn(q).addScaledVector(c0.p.clone().sub(q.p).normalize(), 0.6 * COAT));
      for (let i = 0; i < K; i++) {
        const j = (i + 1) % K, a = rim[i].p, b = rim[j].p, fa = fr[i], fb = fr[j];
        let m = b.clone().sub(a).cross(fa.clone().sub(a)).normalize(); if (m.dot(nn) < 0) m.negate();   // the step, facing up and inward
        // the floor is darker right under the coat's edge (its lip shades it) and torn a little unevenly
        face(a, b, fa, m, m, m, COATC, COATC, STEP); face(b, fb, fa, m, m, m, COATC, STEP, STEP);
        face(fc, fm[i], fm[j], c0.n, mid[i].n, mid[j].n, FLOOR, FLOORM, FLOORM);
        face(fm[i], fa, fb, mid[i].n, rim[i].n, rim[j].n, FLOORM, FLOORE, FLOORE); face(fm[i], fb, fm[j], mid[i].n, rim[j].n, mid[j].n, FLOORM, FLOORE, FLOORM);
        const um = mid.map(upv);   // the mark, wound to face out of the surface (rings go counter-clockwise about nn)
        mask(upv(c0), um[i], um[j]); mask(um[i], upv(rim[i]), upv(rim[j])); mask(um[i], upv(rim[j]), um[j]);
        if (++cnt % 25 === 0) yield;
      }
    }
    if (!pos.length) return null;
    const geo = new T.BufferGeometry();
    geo.setAttribute('position', new T.Float32BufferAttribute(pos, 3)); geo.setAttribute('normal', new T.Float32BufferAttribute(nor, 3)); geo.setAttribute('color', new T.Float32BufferAttribute(col, 3));
    const mg = new T.BufferGeometry(); mg.setAttribute('position', new T.Float32BufferAttribute(mpos, 3));
    const grp = new T.Group(), mk = new T.Mesh(mg, this.crackMask), gv = new T.Mesh(geo, this.crackMat);
    mk.renderOrder = 5; gv.renderOrder = 6;
    for (const o of [grp, mk, gv]) { o.userData.crack = true; o.castShadow = false; }
    gv.receiveShadow = true; grp.add(mk, gv);
    return grp;
  }
  // (user 10:43: "the wood has some marks from being hit ... we want to see the wood reacting to the hits too") a dent
  // in the box's wood where he hit: a dark pit, light splintered fibres around it, cracks running mostly along the grain.
  // Drawn as a decal with a colour map and a normal map made from a height map, so it takes the light like a real dent.
  // (user 15:41: "the marks on the wood should match the shape that hits it") at the moment of the hit, the part's own
  // surface (its drawn mesh, not the collision capsule) is pressed into the wood: every point of it within a few mm of
  // the wall (more for a harder hit) makes the dent there, deeper where it pressed deeper - a hand leaves a hand, a heel
  // a heel. Taken at once (cheap), the picture is made a frame or two later.
  footprint(hit) {
    const w = hit && hit.wall; if (!w || w.name === 'glass') return null;
    const T = this.T, gs = this.groups(), src = gs[hit.body]; if (!src) return null;
    const pose = (hit.chunk && this.chunkPart(hit.chunk, hit.body)) || src, I = this.info(src); if (!I.vp.length) return null;
    this.root.updateMatrixWorld(true); pose.updateWorldMatrix(true, false);
    const M = new T.Matrix4().copy(this.root.matrixWorld).invert().multiply(pose.matrixWorld), e = M.elements;
    const n = new T.Vector3(...w.n), up = Math.abs(n.z) > 0.7 ? new T.Vector3(1, 0, 0) : new T.Vector3(0, 0, 1), rt = up.clone().cross(n).normalize(); up.copy(n).cross(rt).normalize();
    const vp = I.vp, nv = vp.length / 3, D = new Float32Array(nv), X = new Float32Array(nv), Y = new Float32Array(nv); let dmin = 1e9;
    for (let i = 0; i < nv; i++) {
      const x = vp[3 * i], y = vp[3 * i + 1], z = vp[3 * i + 2];
      const wx = e[0] * x + e[4] * y + e[8] * z + e[12], wy = e[1] * x + e[5] * y + e[9] * z + e[13], wz = e[2] * x + e[6] * y + e[10] * z + e[14];
      D[i] = wx * n.x + wy * n.y + wz * n.z; X[i] = wx * rt.x + wy * rt.y + wz * rt.z; Y[i] = wx * up.x + wy * up.y + wz * up.z; if (D[i] < dmin) dmin = D[i];
    }
    const depth = 0.003 + 0.006 * Math.min(1, hit.s || 0.5), pts = []; let cx = 0, cy = 0, sw = 0;
    for (let i = 0; i < nv; i++) { const h = 1 - (D[i] - dmin) / depth; if (h <= 0) continue; pts.push(X[i], Y[i], h); cx += X[i] * h; cy += Y[i] * h; sw += h; }
    if (!sw) return null; cx /= sw; cy /= sw;
    let ext = 0.008; for (let i = 0; i < pts.length; i += 3) ext = Math.max(ext, Math.abs(pts[i] - cx), Math.abs(pts[i + 1] - cy));
    const size = Math.min(0.45, 2 * ext * 1.6 + 0.03);
    // where the contact point was vs where the print's middle is, along the wall
    const hl = new T.Vector3(...hit.l).applyMatrix4(M), hx = hl.dot(rt), hy = hl.dot(up);
    const off = new T.Vector3(...w.off).addScaledVector(rt, cx - hx).addScaledVector(up, cy - hy);
    return { size, off: [off.x, off.y, off.z], pts, cx, cy };
  }
  // (user 16:30: "All marks have to stay on the walls until reset. You cannot erase any of them.") every hole stays.
  // So they cost almost nothing to keep: all the holes on one wall share three growing meshes (the holes' mark, their
  // insides + cracks, the splinters) - 3 draw calls per wall however many holes; a new hole is copied into the end of
  // the buffers (only that part is sent to the graphics chip).
  wallMark(w, s, seed, fp = null) {
    if (!w || w.name === 'glass') return;
    const T = this.T;
    if (!fp) {   // a part with no drawn surface of its own (inside the trunk): a small round print
      const pts = []; for (let i = 0; i < 24; i++) { const a = i / 24 * 2 * Math.PI; for (const r of [0.004, 0.008, 0.012]) pts.push(Math.cos(a) * r, Math.sin(a) * r, 1 - r / 0.02); }
      fp = { pts, cx: 0, cy: 0, size: 0.06, off: w.off };
    }
    const n = new T.Vector3(...w.n), up = Math.abs(n.z) > 0.7 ? new T.Vector3(1, 0, 0) : new T.Vector3(0, 0, 1), rt = up.clone().cross(n).normalize(); up.copy(n).cross(rt).normalize();
    const M = new T.Matrix4().makeBasis(rt, up, n).setPosition(new T.Vector3(...fp.off).addScaledVector(n, 0.0006));
    const W = this.wallSet(w.name), g = this.holeMesh(fp, s, seed);
    g.traverse(o => {
      if (!o.isMesh) return;
      const geo = o.geometry; geo.applyMatrix4(M);
      const B = o.material === this.crackMask ? W.mask : o.material === this.splMat ? W.spl : W.groove;
      B.add(geo.attributes.position.array, geo.attributes.normal ? geo.attributes.normal.array : null, geo.attributes.color ? geo.attributes.color.array : null);
      geo.dispose();
    });
    this.marks.push({ name: w.name });
  }
  // one wall's holes: a group that follows the wall, with the three shared meshes
  wallSet(name) {
    const T = this.T, Ws = this.walls || (this.walls = new Map()); let W = Ws.get(name); if (W) return W;
    const grp = new T.Group(); grp.userData.wall = name; this.root.add(grp);
    const mk = (mat, ro, normals, colors) => {
      const B = { n: 0, cap: 0, mesh: null, normals, colors };
      B.grow = need => {
        let cap = Math.max(3072, B.cap); while (cap < need) cap *= 2; if (cap === B.cap) return;
        const geo = new T.BufferGeometry(), old = B.mesh ? B.mesh.geometry : null;
        const at = (k, w) => { const A = new Float32Array(cap * w); if (old) A.set(old.attributes[k].array.subarray(0, B.n * w)); const ba = new T.BufferAttribute(A, w); ba.setUsage(T.DynamicDrawUsage); geo.setAttribute(k, ba); };
        at('position', 3); if (normals) at('normal', 3); if (colors) at('color', 3);
        geo.setDrawRange(0, B.n);
        if (B.mesh) { B.mesh.geometry = geo; old.dispose(); }
        else { B.mesh = new T.Mesh(geo, mat); B.mesh.renderOrder = ro; B.mesh.frustumCulled = false; B.mesh.receiveShadow = mat !== this.crackMask; B.mesh.castShadow = mat === this.splMat; grp.add(B.mesh); }
        B.cap = cap;
      };
      B.add = (pos, nor, col) => {
        const k = pos.length / 3; B.grow(B.n + k); const geo = B.mesh.geometry;
        const put = (name, src, w) => { const at = geo.attributes[name]; if (!at) return; at.array.set(src || new Float32Array(k * w).fill(1), B.n * w); at.addUpdateRange(B.n * w, k * w); at.needsUpdate = true; };
        put('position', pos, 3); put('normal', nor, 3); put('color', col, 3);
        B.n += k; geo.setDrawRange(0, B.n);
      };
      return B;
    };
    W = { grp, mask: mk(this.crackMask, 5, false, false), groove: mk(this.crackMat, 6, true, true), spl: mk(this.splMat, 4, true, true) };
    Ws.set(name, W); return W;
  }
  // (user 15:45, photo of a hole knocked into plywood) the hole in the wood, in the mark's frame (x along the wall, y =
  // the grain, z out of the wall): its outline = the print's outline; inside, the plies torn away layer by layer, each
  // ply's floor stepping in toward the middle, the torn ply edges as walls; a few splinters lifted at the rim along the
  // grain; cracks running out along the grain. Seen through the wall like the cracks on him (crackMask + crackMat), the
  // splinters drawn normally.
  holeMesh(fp, s, seed) {
    const T = this.T, R = rng(seed * 5 + 11), g = new T.Group(), K = 28, PLY = 0.0022;
    // the outline: per direction, the furthest pressed point (a star-shaped outline of the print), a little torn
    const rad = new Float32Array(K).fill(0);
    for (let i = 0; i < fp.pts.length; i += 3) {
      const x = fp.pts[i] - fp.cx, y = fp.pts[i + 1] - fp.cy; if (fp.pts[i + 2] < 0.15) continue;
      const a = Math.atan2(y, x), k = ((Math.round(a / (2 * Math.PI) * K) % K) + K) % K, r = Math.hypot(x, y);
      for (const kk of [k, (k + 1) % K, (k + K - 1) % K]) if (r > rad[kk]) rad[kk] = r * (kk === k ? 1 : 0.92);
    }
    let rm = 0; for (let k = 0; k < K; k++) rm = Math.max(rm, rad[k]); if (rm < 0.004) for (let k = 0; k < K; k++) rad[k] = 0.006;
    // torn wood: the outline runs out in splintered points along the grain
    for (let k = 0; k < K; k++) { const ay = Math.abs(Math.sin(k / K * 2 * Math.PI)); rad[k] = Math.max(rad[k], 0.35 * rm, 0.004) * (0.9 + 0.2 * R() + (R() < 0.35 ? 0.5 * R() * ay * ay : 0)) + 0.0015; }
    const depth = 0.003 + 0.009 * Math.min(1, s), L = Math.max(1, Math.round(depth / PLY));
    const dirs = []; for (let k = 0; k < K; k++) { const a = k / K * 2 * Math.PI; dirs.push([Math.cos(a), Math.sin(a)]); }
    // contour k (k = 0 at the surface): shrinks toward the middle with each ply, torn unevenly (more along the grain)
    const C = []; for (let l = 0; l <= L; l++) { const f = l === 0 ? 1 : Math.pow(1 - l / (L + 1), 0.8); C.push(dirs.map((d, k) => { const j = l === 0 ? 1 : (0.8 + 0.35 * R()) * (1 + 0.25 * Math.abs(d[1])); return [d[0] * rad[k] * f * Math.min(1.1, j), d[1] * rad[k] * f * j]; })); }
    const pos = [], nor = [], col = [], mpos = [];
    const V = (x, y, z) => new T.Vector3(x, y, z);
    const tri = (a, b, c, n, ca, cb, cc) => { if (b.clone().sub(a).cross(c.clone().sub(a)).dot(n) < 0) { const t = b; b = c; c = t; const tc = cb; cb = cc; cc = tc; } for (const [p, q] of [[a, ca], [b, cb], [c, cc]]) { pos.push(p.x, p.y, p.z); nor.push(n.x, n.y, n.z); col.push(q.r, q.g, q.b); } };
    const PLYA = new T.Color(0xffae5c), PLYB = new T.Color(0xe88a3c), EDGE = new T.Color(0x9a5222), DEEP = new T.Color(0x4a220a), up = V(0, 0, 1);
    for (let l = 1; l <= L; l++) {
      const zt = -(l - 1) * PLY, zb = -l * PLY, A = C[l - 1], B2 = C[l], shade = 1 - 0.45 * (l / L);
      const ply = (l % 2 ? PLYA : PLYB).clone().multiplyScalar(shade), edge = EDGE.clone().multiplyScalar(shade), deep = DEEP.clone().multiplyScalar(0.8 + 0.2 * shade);
      for (let k = 0; k < K; k++) {
        const j = (k + 1) % K, a0 = V(A[k][0], A[k][1], zt), a1 = V(A[j][0], A[j][1], zt), b0 = V(A[k][0], A[k][1], zb), b1 = V(A[j][0], A[j][1], zb);
        // the torn edge of this ply: a wall facing into the hole
        const wn = V(-(a0.y + a1.y) / 2, (a0.x + a1.x) / 2, 0); const tg = a1.clone().sub(a0); const inw = V(-tg.y, tg.x, 0).normalize(); if (inw.dot(V(-(a0.x + a1.x), -(a0.y + a1.y), 0)) < 0) inw.negate();
        tri(a0, a1, b0, inw, edge, edge, deep); tri(a1, b1, b0, inw, edge, deep, deep);
        // the floor of this ply where it wasn't torn further (between this outline and the next)
        const c0 = V(B2[k][0], B2[k][1], zb), c1 = V(B2[j][0], B2[j][1], zb);
        tri(b0, b1, c0, up, ply, ply, ply); tri(b1, c1, c0, up, ply, ply, ply);
        if (l === L) tri(V(0, 0, zb), c0, c1, up, deep, ply, ply);
        if (l === 1) { const LI = 0.0004; mpos.push(0, 0, LI, A[k][0], A[k][1], LI, A[j][0], A[j][1], LI); }
      }
    }
    // the hole's mark wound to face out of the wall
    for (let i = 0; i < mpos.length; i += 9) { const ax = mpos[i + 3], ay = mpos[i + 4], bx = mpos[i + 6], by = mpos[i + 7]; if (ax * by - ay * bx < 0) { mpos[i + 3] = bx; mpos[i + 4] = by; mpos[i + 6] = ax; mpos[i + 7] = ay; } }
    // cracks out along the grain (y), from points on the rim, as grooves (the same drawing as his cracks)
    const N2 = { lines: [], chips: [], grid: new Map(), len: 0 }, nC = 1 + Math.round(R() * (1 + 2 * s));
    for (let c = 0; c < nC; c++) {
      const sgn = R() < 0.5 ? 1 : -1, k = (Math.round((sgn > 0 ? 0.25 : 0.75) * K + (R() - 0.5) * 6) + K) % K, Lm = (0.03 + 0.09 * s) * (0.5 + R());
      let p = V(C[0][k][0], C[0][k][1], 0), d = V((R() - 0.5) * 0.25, sgn, 0).normalize(), went = 0, leg = 0.01 + 0.02 * R(); const w0 = 0.0008 + 0.0012 * s * R(), pts = [{ p: p.clone(), n: up, w: w0 }];
      while (went < Lm) { if (went > leg) { d.x = (R() - 0.5) * 0.3; d.normalize(); leg = went + 0.01 + 0.02 * R(); } p = p.clone().addScaledVector(d, 0.004); went += 0.004; pts.push({ p, n: up, w: w0 * Math.max(0.25, 1 - went / Lm) }); }
      N2.lines.push({ pts, side: 1, open: true });
    }
    const pit = new T.BufferGeometry(); pit.setAttribute('position', new T.Float32BufferAttribute(pos, 3)); pit.setAttribute('normal', new T.Float32BufferAttribute(nor, 3)); pit.setAttribute('color', new T.Float32BufferAttribute(col, 3));
    const mg = new T.BufferGeometry(); mg.setAttribute('position', new T.Float32BufferAttribute(mpos, 3));
    const pm = new T.Mesh(pit, this.crackMat), mk = new T.Mesh(mg, this.crackMask); pm.renderOrder = 6; mk.renderOrder = 5; pm.receiveShadow = true; g.add(mk, pm);
    const cr = this.drawNet(N2); if (cr) g.add(cr);
    // splinters: thin torn fibres lifted at the rim, pointing along the grain
    const sp = [], sn = [], sc = [], nS = 2 + Math.round(R() * (2 + 4 * s)), FIBC = new T.Color(0xf4c088);
    for (let i = 0; i < nS; i++) {
      const k = (Math.round((R() < 0.5 ? 0.25 : 0.75) * K + (R() - 0.5) * 8) + K) % K, b = C[0][k], sg = b[1] >= 0 ? 1 : -1;
      const base = V(b[0], b[1], 0), w2 = 0.0008 + 0.0015 * R(), len = 0.004 + 0.012 * R() * (0.5 + s), lift = 0.001 + 0.004 * R() * s;
      const a = base.clone().add(V(-w2, 0, 0)), c2 = base.clone().add(V(w2, 0, 0)), tip = base.clone().add(V((R() - 0.5) * 0.004, -sg * len * 0.6, lift));
      const nn = c2.clone().sub(a).cross(tip.clone().sub(a)).normalize(); if (nn.z < 0) nn.negate();
      for (const q of [a, c2, tip]) { sp.push(q.x, q.y, q.z); sn.push(nn.x, nn.y, nn.z); sc.push(FIBC.r, FIBC.g, FIBC.b); }
    }
    const sg2 = new T.BufferGeometry(); sg2.setAttribute('position', new T.Float32BufferAttribute(sp, 3)); sg2.setAttribute('normal', new T.Float32BufferAttribute(sn, 3)); sg2.setAttribute('color', new T.Float32BufferAttribute(sc, 3));
    const spl = new T.Mesh(sg2, this.splMat); spl.castShadow = true; spl.receiveShadow = true; g.add(spl);
    return g;
  }
  // one dent's look (colour + normal map from a height map); 4 are made while the page starts (no work during play)
  markMat(s, seed, fp = null) {
    const T = this.T, R = rng(seed), N = 96, H = new Float32Array(N * N), A = new Float32Array(N * N), C = new Float32Array(N * N * 3);
    const mix = (k, c, a, h) => { const i3 = 3 * k; const t = Math.max(A[k], a); C[i3] = C[i3] * (1 - a) + c[0] * a; C[i3 + 1] = C[i3 + 1] * (1 - a) + c[1] * a; C[i3 + 2] = C[i3 + 2] * (1 - a) + c[2] * a; A[k] = t; H[k] += h; };
    const DENT = [0.30, 0.13, 0.04], PIT = [0.10, 0.04, 0.015], FIB = [0.98, 0.84, 0.60], CRK = [0.08, 0.035, 0.012];
    const r0 = 0.15 + 0.06 * s, wob = [0, 1, 2, 3].map(() => [R() * 6.28, 0.15 + 0.25 * R()]);
    // the print: the pressed points splatted into the picture's grid (u, v = -1..1 across the mark), smoothed; F = how
    // deep it pressed, B = a wider soft copy for the crushed wood around it
    let F = null, B = null;
    if (fp) {
      F = new Float32Array(N * N); const half = fp.size / 2, rp = Math.max(1, Math.ceil(0.0035 / (fp.size / N)));
      for (let i = 0; i < fp.pts.length; i += 3) {
        const px = Math.round(((fp.pts[i] - fp.cx) / half + 1) / 2 * N - 0.5), py = Math.round(((fp.pts[i + 1] - fp.cy) / half + 1) / 2 * N - 0.5), h = fp.pts[i + 2];
        for (let dy = -rp; dy <= rp; dy++) for (let dx = -rp; dx <= rp; dx++) { const x = px + dx, y = py + dy; if (x < 0 || y < 0 || x >= N || y >= N || dx * dx + dy * dy > rp * rp) continue; const k = y * N + x; if (h > F[k]) F[k] = h; }
      }
      const blur = (A, r) => { const O = new Float32Array(N * N); for (let y = 0; y < N; y++) for (let x = 0; x < N; x++) { let t = 0, c = 0; for (let dy = -r; dy <= r; dy++) for (let dx = -r; dx <= r; dx++) { const xx = x + dx, yy = y + dy; if (xx < 0 || yy < 0 || xx >= N || yy >= N) continue; t += A[yy * N + xx]; c++; } O[y * N + x] = t / c; } return O; };
      F = blur(F, 1); B = blur(blur(F, 3), 3);
    }
    // splinters (along the grain = v) and cracks, as tapered strokes from the centre
    const strokes = [];
    for (let i = 0; i < 12; i++) { const ang = (R() < 0.75 ? (R() < 0.5 ? 0 : Math.PI) : R() * 6.28) + (R() - 0.5) * 0.7; strokes.push({ ang, len: 0.25 + 0.25 * R() + 0.1 * s, w: 0.025 + 0.035 * R(), c: FIB, h: 0.35, a: 1 }); }
    for (let i = 0; i < 5 + Math.round(3 * s); i++) { const ang = (R() < 0.8 ? (R() < 0.5 ? 0 : Math.PI) : R() * 6.28) + (R() - 0.5) * 0.5; strokes.push({ ang, len: 0.45 + 0.3 * R() + 0.15 * s, w: 0.012 + 0.01 * R(), c: CRK, h: -0.9, a: 1, jag: R() * 9 }); }
    for (const st of strokes) { st.ux = Math.sin(st.ang); st.uy = Math.cos(st.ang); }
    if (F) strokes.length = 0;   // (with a print, the hole, its splinters and its cracks are real geometry: holeMesh)
    for (let y = 0; y < N; y++) for (let x = 0; x < N; x++) {
      const k = y * N + x, u = (x + 0.5) / N * 2 - 1, v = (y + 0.5) / N * 2 - 1, r = Math.hypot(u, v), th = Math.atan2(u, v);
      // the dent: the wood pressed in and darkened around the hit
      if (F) { const b = Math.min(1, B[k] * 1.6); if (b > 0) mix(k, DENT, 0.35 * b, -0.3 * b); }
      else { const dr = 0.5 + 0.1 * s; if (r < dr) { const f = 1 - r / dr; mix(k, DENT, 0.55 * f * f, -0.5 * f * f); } }
      for (const st of strokes) {
        const uu = u - (st.ox || 0), vv = v - (st.oy || 0);
        if (Math.hypot(uu, vv) > st.len + 0.05) continue;
        const along = uu * st.ux + vv * st.uy; if (along < 0 || along > st.len) continue;
        const across = uu * st.uy - vv * st.ux + (st.jag ? 0.012 * Math.sin(along * 40 + st.jag) : 0);
        const wd = st.w * (1 - along / st.len) + 0.002; if (Math.abs(across) > wd) continue;
        const e = 1 - Math.abs(across) / wd; mix(k, st.c, Math.min(1, e * 2) * st.a, st.h * e);
      }
      // the pit itself: the print (deeper and darker where it pressed harder, crushed fibres at its edge), or a round pit
      if (F) continue;
      let rr = r0; for (const [ph, am] of wob) rr += 0.03 * am * Math.sin(th * 3 + ph);
      if (r < rr) { const f = 1 - r / rr; mix(k, PIT, Math.min(1, 0.6 + f), -1.2 * Math.sqrt(f)); }
      else if (r < rr * 1.35) { const f = 1 - (r - rr) / (rr * 0.35); mix(k, FIB, 0.8 * f, 0.25 * f); }   // the torn rim
    }
    const cv = document.createElement('canvas'); cv.width = cv.height = N; const cx = cv.getContext('2d'), im = cx.createImageData(N, N);
    const nd = new Uint8Array(N * N * 4), K = 2.2;
    for (let y = 0; y < N; y++) for (let x = 0; x < N; x++) {
      const k = y * N + x, o = 4 * ((N - 1 - y) * N + x);   // canvas rows go down
      im.data[o] = 255 * C[3 * k]; im.data[o + 1] = 255 * C[3 * k + 1]; im.data[o + 2] = 255 * C[3 * k + 2]; im.data[o + 3] = 255 * Math.min(1, A[k]);
      const hx = H[y * N + Math.min(N - 1, x + 1)] - H[y * N + Math.max(0, x - 1)], hy = H[Math.min(N - 1, y + 1) * N + x] - H[Math.max(0, y - 1) * N + x];
      let nx = -hx * K, ny = -hy * K, nz = 1; const l = Math.hypot(nx, ny, nz); nx /= l; ny /= l; nz /= l;
      const q = 4 * k; nd[q] = 127.5 + 127.5 * nx; nd[q + 1] = 127.5 + 127.5 * ny; nd[q + 2] = 127.5 + 127.5 * nz; nd[q + 3] = 255;
    }
    cx.putImageData(im, 0, 0);
    const map = new T.CanvasTexture(cv); map.colorSpace = T.SRGBColorSpace;
    const nmap = new T.DataTexture(nd, N, N, T.RGBAFormat); nmap.needsUpdate = true;
    return new T.MeshStandardMaterial({ map, normalMap: nmap, normalScale: new T.Vector2(1.6, 1.6), transparent: true, depthWrite: false, roughness: 0.9, polygonOffset: true, polygonOffsetFactor: -4, polygonOffsetUnits: -4 });
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
      o.add(this.cap(P));   // the broken face: raw wood where it came off
    } else {
      // (user 15:41: "a bunch of triangles all bunched together") a fragment = one solid splinter (see shardSet); every
      // fragment stays until reset (user 16:30), so they're drawn as instances: 16 batches for all of them
      const R = rng(P.seed), sc = [P.size * (1.3 + 1.1 * R()), P.size * (0.55 + 0.45 * R()), P.size * (0.25 + 0.3 * R())];
      const sh = this.shardSet(), B = sh.batch[(P.seed >>> 0) % 8 + (P.wood ? 0 : 8)], idx = B.take();
      const prox = { shard: true, B, idx, sc, last: null };
      this.debris.set(P, prox); return prox;
    }
    o.userData.P = P; this.root.add(o); this.movers.push(o); this.debris.set(P, o);
    return o;
  }
  // the drawn copy of body b on a broken-off piece
  chunkPart(P, b) { const o = this.debris.get(P); return o ? o.children.find(c => c.userData.body === b) || null : null; }
  removeDebris(P) { const o = this.debris.get(P); if (!o) return; this.dropObj(o); this.debris.delete(P); }
  dropObj(o) {
    if (o.shard) { o.B.free(o.idx); return; }
    this.root.remove(o); const i = this.movers.indexOf(o); if (i >= 0) this.movers.splice(i, 1);
  }
  // the fragments' batches: 8 splinter shapes x (raw wood / with the white coat on one face), each an InstancedMesh
  // that grows as needed; a fragment = one slot (its own size, place and turn); a freed slot is scaled to nothing
  shardSet() {
    if (this.shards) return this.shards;
    const T = this.T, S = { batch: [] }, SHELLC = new T.Color(0xebe5da), WOODC = new T.Color(0xc98a4a), ZERO = new T.Matrix4().makeScale(0, 0, 0);
    for (let v = 0; v < 2; v++) for (let k = 0; k < 8; k++) {
      const R = rng(4242 + k * 31);
      let geo = new T.BoxGeometry(1, 1, 1); const pa = geo.attributes.position, off = new Map();
      for (let i = 0; i < pa.count; i++) {
        const x = pa.getX(i), y = pa.getY(i), z = pa.getZ(i), key = `${x > 0}${y > 0}${z > 0}`;
        if (!off.has(key)) off.set(key, [(R() - 0.5) * 0.5, (R() - 0.5) * 0.35, (R() - 0.5) * 0.3]);
        const d = off.get(key); pa.setXYZ(i, x + d[0], y + d[1], z + d[2]);
      }
      geo = geo.toNonIndexed(); geo.computeVertexNormals(); geo.clearGroups();
      const col = []; for (let i = 0; i < geo.attributes.position.count; i++) { const c = (v === 1 && i >= 24 && i < 30) ? SHELLC : WOODC; col.push(c.r, c.g, c.b); }
      geo.setAttribute('color', new T.Float32BufferAttribute(col, 3));
      const B = { geo, mesh: null, cap: 0, used: 0, freeList: [] };
      B.make = cap => {
        const m = new T.InstancedMesh(geo, this.shardMat, cap); m.count = 0; m.castShadow = true; m.receiveShadow = true; m.frustumCulled = false;
        if (B.mesh) { m.instanceMatrix.array.set(B.mesh.instanceMatrix.array); m.count = B.mesh.count; this.root.remove(B.mesh); const i = this.movers.indexOf(B.mesh); if (i >= 0) this.movers.splice(i, 1); B.mesh.dispose(); }
        B.mesh = m; B.cap = cap; this.root.add(m); this.movers.push(m);
      };
      B.free = i => { B.mesh.setMatrixAt(i, ZERO); B.mesh.instanceMatrix.needsUpdate = true; B.freeList.push(i); };
      B.take = () => { if (B.freeList.length) return B.freeList.pop(); if (!B.mesh || B.used >= B.cap) B.make(Math.max(32, B.cap * 2)); const i = B.used++; B.mesh.count = B.used; return i; };
      S.batch.push(B);
    }
    return (this.shards = S);
  }
  // a piece split in two (or more): its drawn parts move to the new pieces, each with a raw-wood face where it broke
  split(e) {
    const T = this.T, o = this.debris.get(e.P); if (!o) { for (const Q of e.parts) this.addDebris(Q); return; }
    for (const Q of e.parts) {
      const n = new T.Group();
      for (const r of Q.rel) { const c = o.children.find(k => k.userData.body === r.b); if (!c) continue; c.position.set(...r.p); c.quaternion.set(r.q[1], r.q[2], r.q[3], r.q[0]); n.add(c); }
      n.add(this.cap(Q)); n.userData.P = Q; this.root.add(n); this.movers.push(n); this.debris.set(Q, n);
      n.position.set(...Q.x); n.quaternion.set(Q.q[1], Q.q[2], Q.q[3], Q.q[0]);
    }
    this.removeDebris(e.P);
  }
  // the broken end of a piece: the peg that held it at the joint, snapped - a short stub of raw wood with a jagged,
  // splintered top (one closed shape; the old version was a ball whose loose triangles flew apart, user 15:41)
  cap(P) {
    const T = this.T, R = rng(P.id * 7919 + 13), cl = new T.Vector3(...P.comL), out = cl.lengthSq() > 1e-10 ? cl.clone().negate().normalize() : new T.Vector3(0, 1, 0);
    const root = P.rel && P.rel[0] ? this.groups()[P.rel[0].b] : null, I = root ? this.info(root) : null;
    const r = Math.max(0.006, Math.min(0.016, (I && I.size ? I.size : 0.06) * 0.16)), L = r * 2.2, K = 9;
    const pos = [], O = new T.Vector3(0, -0.15 * L, 0), add = (a, b, c) => {   // each triangle wound to face away from the middle
      const nrm = b.clone().sub(a).cross(c.clone().sub(a)), mid = a.clone().add(b).add(c).multiplyScalar(1 / 3).sub(O);
      if (nrm.dot(mid) < 0) { const t = b; b = c; c = t; }
      pos.push(a.x, a.y, a.z, b.x, b.y, b.z, c.x, c.y, c.z);
    };
    const bot = [], top = [];
    for (let i = 0; i < K; i++) {
      const th = (i + (R() - 0.5) * 0.4) / K * Math.PI * 2, rr = r * (0.85 + 0.3 * R());
      bot.push(new T.Vector3(Math.cos(th) * r, -0.6 * L, Math.sin(th) * r));
      top.push(new T.Vector3(Math.cos(th) * rr * 0.9, L * (0.25 + 0.75 * R() * R()), Math.sin(th) * rr * 0.9));
    }
    const ctop = new T.Vector3(0, L * 0.15, 0), cbot = new T.Vector3(0, -0.6 * L, 0);
    for (let i = 0; i < K; i++) { const j = (i + 1) % K; add(bot[i], top[i], top[j]); add(bot[i], top[j], bot[j]); add(ctop, top[j], top[i]); add(cbot, bot[i], bot[j]); }
    const geo = new T.BufferGeometry(); geo.setAttribute('position', new T.Float32BufferAttribute(pos, 3)); geo.computeVertexNormals();
    const cap = new T.Mesh(geo, this.inner);
    cap.quaternion.setFromUnitVectors(new T.Vector3(0, 1, 0), out); cap.position.copy(cl).negate();
    cap.castShadow = true; cap.receiveShadow = true; return cap;
  }
  // after the figure is rebuilt (another figure picked): cracks and pieces again from the stored hits
  rebuild(C) {
    const keep = [...this.debris.keys()];
    this.reset(false); this.cache.clear();
    // each part's triangles, ready before the first hit (a part per frame)
    for (const g of this.groups()) if (g) { const f = () => this.info(g); f.keep = true; this.later(f); }
    for (const k in C.parts) for (const h of C.parts[k].hits) this.addCrack(h);
    for (const P of keep) this.addDebris(P);
  }
  sync(C) {
    const gs = this.groups();
    if (this.warm && ++this.warm.n > 3) { this.root.remove(this.warm.g); this.warm = null; }
    this.runJobs();
    if (this.walls && this.walls.size) { const F = {}; for (const f of C.W.faces()) F[f.name] = f; for (const W of this.walls.values()) { const f = F[W.grp.userData.wall]; W.grp.visible = !!f; if (f) W.grp.position.set(f.p[0], f.p[1], f.p[2]); } }
    for (const P of C.debris.removed.splice(0)) this.removeDebris(P);
    for (const b of C.broken) if (gs[b]) gs[b].visible = false;
    for (const [P, o] of this.debris) {
      if (P.kind === 'chunk') {
        o.position.set(P.x[0], P.x[1], P.x[2]); o.quaternion.set(P.q[1], P.q[2], P.q[3], P.q[0]);
      } else if (o.shard) {   // only fragments that moved are written (resting ones cost nothing)
        const L = o.last; if (L && L[0] === P.x[0] && L[1] === P.x[1] && L[2] === P.x[2] && L[3] === P.q[0] && L[4] === P.q[1] && L[5] === P.q[2]) continue;
        o.last = [P.x[0], P.x[1], P.x[2], P.q[0], P.q[1], P.q[2]];
        const m = this._m || (this._m = new this.T.Matrix4()), q = this._q || (this._q = new this.T.Quaternion()), v = this._v || (this._v = new this.T.Vector3()), sv = this._s || (this._s = new this.T.Vector3());
        m.compose(v.set(P.x[0], P.x[1], P.x[2]), q.set(P.q[1], P.q[2], P.q[3], P.q[0]), sv.set(o.sc[0], o.sc[1], o.sc[2]));
        o.B.mesh.setMatrixAt(o.idx, m); o.B.mesh.instanceMatrix.needsUpdate = true;
      }
    }
  }
}

// ================= CrackSound: wood cracking (user 10:32: "it should feel like wood cracking, not porcelain ... the
// sound of cracking should be stronger") =================
// Made from noise and tones (nothing to download), on its own path to the speakers (the figure's other sounds go through
// a soft low-pass). A crack = a heavy thump, the hollow knock of the wooden part, a dry snap and the fibres tearing (a
// burst of short ticks, each ringing a little like wood); a deep hit creaks first. A break = a bigger snap and a long
// splintering crunch. Pieces landing = small wooden knocks.
export class CrackSound {
  constructor(actx) { this.a = actx; this.out = null; this.last = -1; }
  init() {
    const a = this.a; if (!a || this.out) return !!this.out;
    const comp = a.createDynamicsCompressor(); comp.threshold.value = -8; comp.knee.value = 6; comp.ratio.value = 6; comp.attack.value = 0.001; comp.release.value = 0.12;
    this.out = a.createGain(); this.out.gain.value = 1.8;
    const sh = a.createWaveShaper(), cv = new Float32Array(1025); for (let i = 0; i < 1025; i++) { const x = (i - 512) / 512 * 1.5; cv[i] = 0.95 * Math.tanh(x / 0.95); }
    sh.curve = cv;   // never past full scale: the strongest cracks saturate a little instead of clipping
    this.out.connect(comp).connect(sh).connect(a.destination);
    const n = a.sampleRate, buf = a.createBuffer(1, n, a.sampleRate), d = buf.getChannelData(0);
    for (let i = 0; i < n; i++) d[i] = Math.random() * 2 - 1;
    this.noise = buf; return true;
  }
  burst(t, dur, type, f, q, gain) {   // filtered noise: a short burst (high q = it rings at f, like a struck board)
    const a = this.a, src = a.createBufferSource(); src.buffer = this.noise;
    const fl = a.createBiquadFilter(); fl.type = type; fl.frequency.value = f; fl.Q.value = q;
    const g = a.createGain(); g.gain.setValueAtTime(0.0001, t); g.gain.exponentialRampToValueAtTime(gain, t + 0.001); g.gain.exponentialRampToValueAtTime(0.0001, t + dur);
    src.connect(fl).connect(g).connect(this.out); src.start(t, Math.random() * 0.9); src.stop(t + dur + 0.05);
  }
  ring(t, f, q, dur, gain) {   // a click into a resonance: the wood's own note, dying away
    const a = this.a, src = a.createBufferSource(); src.buffer = this.noise;
    const g0 = a.createGain(); g0.gain.setValueAtTime(1, t); g0.gain.setValueAtTime(0, t + 0.003);
    const fl = a.createBiquadFilter(); fl.type = 'bandpass'; fl.frequency.value = f; fl.Q.value = q;
    const g = a.createGain(); g.gain.setValueAtTime(gain, t); g.gain.exponentialRampToValueAtTime(0.0001, t + dur);
    src.connect(g0).connect(fl).connect(g).connect(this.out); src.start(t, Math.random() * 0.9); src.stop(t + dur + 0.05);
  }
  thump(t, s) {   // the hit itself: heavy and low
    const a = this.a, o = a.createOscillator(), g = a.createGain();
    o.frequency.setValueAtTime(95, t); o.frequency.exponentialRampToValueAtTime(48, t + 0.22);
    g.gain.setValueAtTime(0.0001, t); g.gain.exponentialRampToValueAtTime(0.9 * s, t + 0.005); g.gain.exponentialRampToValueAtTime(0.0001, t + 0.26);
    o.connect(g).connect(this.out); o.start(t); o.stop(t + 0.3);
    this.burst(t, 0.14, 'lowpass', 700, 0.6, 0.9 * s);
  }
  knock(t, s, low = 1) {   // the hollow knock of a wooden part
    const f = (190 + Math.random() * 60) * low;
    this.ring(t, f, 9, 0.16, 2.2 * s); this.ring(t, f * 2.3, 10, 0.09, 1.3 * s); this.ring(t, f * 4.1, 12, 0.05, 0.7 * s);
  }
  creak(t, s, dur) {   // fibres straining just before they give
    const a = this.a, o = a.createOscillator(), fl = a.createBiquadFilter(), g = a.createGain();
    o.type = 'sawtooth'; o.frequency.setValueAtTime(70 + Math.random() * 30, t); o.frequency.linearRampToValueAtTime(110 + Math.random() * 50, t + dur);
    fl.type = 'bandpass'; fl.frequency.value = 900; fl.Q.value = 2.5;
    g.gain.setValueAtTime(0.0001, t);
    for (let k = 0, n = Math.round(dur / 0.012); k < n; k++) g.gain.setValueAtTime((0.15 + 0.35 * Math.random()) * s * (0.4 + 0.6 * k / n), t + k * 0.012);
    g.gain.setValueAtTime(0.0001, t + dur);
    o.connect(fl).connect(g).connect(this.out); o.start(t); o.stop(t + dur + 0.02);
  }
  snap(t, s) {   // the split: a dry, sharp crack with the wood ringing under it
    this.burst(t, 0.022, 'bandpass', 1700, 0.7, 1.6 * s);
    this.burst(t, 0.05, 'bandpass', 650, 1.0, 1.1 * s);
    this.burst(t + 0.002, 0.012, 'highpass', 3500, 0.7, 0.5 * s);
    this.ring(t, 420 + Math.random() * 120, 7, 0.1, 1.6 * s);
  }
  tear(t, s, n, span) {   // fibres tearing along the crack: irregular ticks, thinning out
    for (let i = 0; i < n; i++) {
      const u = Math.pow(Math.random(), 1.7) * span, k = 1 - 0.7 * u / span, f = 600 + Math.random() * 2200;
      this.burst(t + u, 0.004 + 0.01 * Math.random(), 'bandpass', f, 2.2, (0.35 + 0.6 * Math.random()) * s * k);
      if (Math.random() < 0.35) this.ring(t + u, 300 + Math.random() * 500, 8, 0.04, 0.6 * s * k);
    }
  }
  crack(strength, level) {
    if (!this.init()) return; const now = this.a.currentTime + 0.01, s = Math.min(1, 0.65 + 0.35 * strength);
    let t = now;
    if (level >= 2) { this.creak(t, 0.8 * s, 0.07); t += 0.06; }   // a part already cracked creaks before it splits more
    this.thump(t, s); this.knock(t, s); this.snap(t + 0.004, s * (0.9 + 0.1 * level)); this.tear(t + 0.01, s, 8 + 5 * level, 0.1 + 0.05 * level);
  }
  shatter(strength) {   // the part breaks off: a big snap and a splintering crunch
    if (!this.init()) return; const t = this.a.currentTime + 0.01, s = Math.min(1, 0.8 + 0.2 * strength);
    this.creak(t, s, 0.09);
    const t1 = t + 0.08;
    this.thump(t1, s * 1.1); this.knock(t1, s, 0.85); this.snap(t1, s * 1.25); this.snap(t1 + 0.035, s * 0.8);
    this.tear(t1 + 0.01, s * 1.1, 26, 0.45);
    this.burst(t1 + 0.01, 0.32, 'bandpass', 1100, 0.6, 0.45 * s);   // the crunch
  }
  clink(v, m) {   // a piece landing: a small wooden knock (bigger piece = lower, louder)
    if (!this.init()) return; const now = this.a.currentTime; if (now - this.last < 0.03) return; this.last = now;
    const t = now + 0.005, big = m > 0.5, s = Math.min(1, v / 3.5) * (big ? 1 : 0.55);
    this.ring(t, big ? 210 + Math.random() * 60 : 520 + Math.random() * 400, 8, big ? 0.12 : 0.05, 1.6 * s);
    this.burst(t, 0.01, 'bandpass', big ? 900 : 1800, 1.5, 0.6 * s);
    if (big) this.burst(t, 0.08, 'lowpass', 400, 0.5, 0.5 * s);
  }
}
