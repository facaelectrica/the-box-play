// Camera tracking -> his pose. Input: the 33 body points of MediaPipe's Pose Landmarker ("world" points: metres, centred
// on the hips, x to the image's right, y down, z away from the camera). Output: the game's 73 numbers for each frame.
// Every bone is aimed along the tracked limb; the twist of an upper arm or thigh comes from the plane its elbow or
// knee bends in. Points the camera sees poorly are trusted less: the joint drifts to the tracker's guess, and
// keeps its last angles when the point is lost. Legs are fitted to the performer's size. The hips travel by keeping each foot
// still while it is on the floor, because the tracked points carry no position in the room.
import { M, V, fk, standPose } from './skel.js?v=1';

export const LM = { nose: 0, lEar: 7, rEar: 8, lSh: 11, rSh: 12, lEl: 13, rEl: 14, lWr: 15, rWr: 16, lPk: 17, rPk: 18, lIx: 19, rIx: 20,
  lHip: 23, rHip: 24, lKn: 25, rKn: 26, lAn: 27, rAn: 28, lHe: 29, rHe: 30, lTo: 31, rTo: 32 };
const clamp01 = v => Math.max(0, Math.min(1, v));
// rotation taking the rest pair (direction d0, side axis f0) onto the tracked pair (d, f)
const basis = (d, f) => { const u = V.norm(d), w = V.norm(V.perp(f, u)); return M.cols(u, V.cross(w, u), w); };
const aim = (d, f, d0, f0) => M.mul(basis(d, f), M.T(basis(d0, f0)));
const ok3 = v => Number.isFinite(v[0] + v[1] + v[2]) && V.len(v) > 1e-6;

// smoothing that follows fast motion and steadies slow motion (the "1 euro" filter), one per number
class Euro { constructor(minCut, beta) { this.mc = minCut; this.beta = beta; this.x = null; this.dx = 0; }
  step(x, dt) { if (this.x === null || !(dt > 0)) { this.x = x; return x; }
    const al = c => { const r = 2 * Math.PI * c * dt; return r / (r + 1); }, d = (x - this.x) / dt; this.dx += al(1) * (d - this.dx);
    this.x += al(this.mc + this.beta * Math.abs(this.dx)) * (x - this.x); return this.x; } }

export function makeSolver(S, opt = {}) {
  const o = Object.assign({ smooth: 1, hands: true, minVis: 0.3, travel: true, fit: true }, opt), id = S.id, nb = S.names.length;
  const rest = n => V.norm(S.off[id[n]]), Z0 = [0, 0, 1], X0 = [1, 0, 0];
  const SIDE = [   // right, then left: the points and bodies of one arm and one leg
    { s: -1, sh: LM.rSh, el: LM.rEl, wr: LM.rWr, pk: LM.rPk, ix: LM.rIx, hip: LM.rHip, kn: LM.rKn, an: LM.rAn, he: LM.rHe, to: LM.rTo, A: 'RightArm', FA: 'RightForeArm', H: 'RightHand', SH: 'RightShoulder', LG: 'RightLeg', SN: 'RightShin', FT: 'RightFoot', TB: 'RightToeBase' },
    { s: 1, sh: LM.lSh, el: LM.lEl, wr: LM.lWr, pk: LM.lPk, ix: LM.lIx, hip: LM.lHip, kn: LM.lKn, an: LM.lAn, he: LM.lHe, to: LM.lTo, A: 'LeftArm', FA: 'LeftForeArm', H: 'LeftHand', SH: 'LeftShoulder', LG: 'LeftLeg', SN: 'LeftShin', FT: 'LeftFoot', TB: 'LeftToeBase' } ];
  // distance from each joint down to the body's surface. Feet and toes: the game body's own soles (4.5 cm under the
  // ankle joint; the toe joint lies 0.6 cm under the sole plane). The rest are rough.
  const RAD = S.names.map(n => /Toe/.test(n) ? -0.006 : /Foot/.test(n) ? 0.045 : n === 'Head' ? 0.11 : /Hand|ForeArm/.test(n) ? 0.04 : 0.08);
  let q, cal, headCal, footCal, filt, tPrev, mem, root, perf;
  // The tracker's heel point sits higher than its toe point: on a flat foot the line from heel to toe tilts about 26
  // degrees down (measured on the user's camera, Nov 16). This is the default correction; Calibrate measures it per foot.
  const FOOT0 = M.euler(0.46, 0, 0);
  const LEG = V.len(S.off[id.RightShin]) + V.len(S.off[id.RightFoot]);   // his own leg, hip joint to ankle
  function reset() { q = standPose(S); cal = M.I(); headCal = M.I(); footCal = [FOOT0, FOOT0]; filt = null; tPrev = null; mem = { fw: {}, tw: [0, 0], foot: [null, null], hf: [null, null] }; root = { xy: [0, 0], lock: [null, null], toe: [false, false], g: null, pt: null }; perf = { leg: null }; }
  reset();

  // tracked points -> the game's axes (x to his left, y behind him, z up), steadied
  function points(lm, t) {
    const dt = tPrev === null ? 0 : t - tPrev; tPrev = t;
    if (!filt) filt = Array.from({ length: 33 }, (_, i) => { const hand = i >= 17 && i <= 22, c = (hand ? 1.0 : 1.4) / o.smooth, bt = hand ? 1.5 : 0.6; return [new Euro(c, bt), new Euro(c, bt), new Euro(c, bt)]; });   // hand points: steadied when still, but quick wrist moves pass
    const P = new Array(33), vis = new Array(33);
    for (let i = 0; i < 33; i++) { const l = lm[i] || { x: 0, y: 0, z: 0, visibility: 0 }, g = [l.x, l.z, -l.y];
      P[i] = o.smooth > 0 ? [filt[i][0].step(g[0], dt), filt[i][1].step(g[1], dt), filt[i][2].step(g[2], dt)] : g; vis[i] = l.visibility === undefined ? 1 : l.visibility; }
    return { P, vis };
  }
  const torso = P => { const HM = V.mid(P[LM.lHip], P[LM.rHip]), SM = V.mid(P[LM.lSh], P[LM.rSh]), up = V.sub(SM, HM);
    const fr = (l, r) => { const X = V.norm(V.sub(P[l], P[r])), Z = V.norm(V.perp(up, X)), Y = V.cross(Z, X); return M.cols(X, Y, Z); };
    return { pelvis: fr(LM.lHip, LM.rHip), chest: fr(LM.lSh, LM.rSh) }; };
  const headOf = P => { const X = V.norm(V.sub(P[LM.lEar], P[LM.rEar])), f = V.sub(P[LM.nose], V.mid(P[LM.lEar], P[LM.rEar])), Y = V.mul(V.norm(V.perp(f, X)), -1); return M.cols(X, Y, V.cross(X, Y)); };

  // Stand straight, facing the camera: from here on the hips' frame at this moment counts as upright and facing
  // forward (this removes the camera's own tilt), and the head's angle at this moment counts as level.
  function calibrate(lm) { const P = lm.map(l => [l.x, l.z, -l.y]), T = torso(P); if (!T.pelvis.every(Number.isFinite)) return false;
    cal = M.T(T.pelvis); const Pc = P.map(p => M.vec(cal, p)), T2 = torso(Pc), H = headOf(Pc);
    headCal = H.every(Number.isFinite) ? M.mul(M.T(T2.chest), H) : M.I(); filt = null; mem.fw = {}; perf.leg = null;
    SIDE.forEach((d, i) => { const Rt = footOf(Pc, d), f = Rt && V.perp([-Rt[1], -Rt[4], -Rt[7]], Z0); if (!Rt || !ok3(f)) return;   // he stands flat now: what the tracker shows is the correction
      const C = M.mul(M.T(aim(f, Z0, [0, -1, 0], Z0)), Rt), ang = Math.acos(Math.max(-1, Math.min(1, (C[0] + C[4] + C[8] - 1) / 2))); if (ang < 1) footCal[i] = C; });
    return true; }

  const footOf = (P, d) => { const fwd = V.sub(P[d.to], P[d.he]); if (!ok3(fwd)) return null; const up = V.perp(V.sub(P[d.an], P[d.he]), V.norm(fwd)); return ok3(up) ? aim(fwd, up, [0, -1, 0], Z0) : null; };
  // two-bone IK for a leg: hip joint H, ankle wanted at Tg, bending around the side axis f0
  function legIK(d, H, Tg, f0) { const sn = id[d.SN], ft = id[d.FT], L1 = V.len(S.off[sn]), L2 = V.len(S.off[ft]), D = V.sub(Tg, H), dl = V.len(D); if (dl < 1e-4) return null;
    const dd = Math.max(Math.abs(L1 - L2) + 1e-3, Math.min(L1 + L2 - 1e-3, dl)), u = V.mul(D, 1 / dl), fp = V.perp(f0, u); if (!ok3(fp)) return null;
    const f = V.norm(fp), p = V.cross(u, f), ca = (L1 * L1 + dd * dd - L2 * L2) / (2 * L1 * dd), sa = Math.sqrt(Math.max(0, 1 - ca * ca));
    const a = V.add(V.mul(u, ca), V.mul(p, sa)), kn = V.add(H, V.mul(a, L1)), b = V.norm(V.sub(V.add(H, V.mul(u, dd)), kn));
    return [aim(a, f, rest(d.SN), X0), aim(b, f, rest(d.FT), X0)]; }
  // how far to trust a group of points: 1 when clearly seen, falling to 0 as the tracker loses them. A limb the
  // camera cannot see well (the far arm and leg when he stands sideways) still follows the tracker's guess, slowly.
  const trust = (vis, k) => clamp01((Math.min(...k.map(i => vis[i])) - 0.05) / 0.25);

  // How still each foot is in the camera picture (1 = not moving, 0 = clearly moving). The picture shows whether a
  // foot moves far more reliably than the tracker's height estimate shows whether it is lifted. img: the 33 points in
  // the picture (x, y from 0 to 1), aspect: picture width / height. Speeds are turned into metres a second with the
  // least foreshortened body segment as the ruler.
  function stillness(lm, img, aspect, t) {
    const dt = mem.tF === undefined ? 0 : t - mem.tF; mem.tF = t; if (!img) return [1, 1];
    const px = i => [img[i].x * aspect, img[i].y]; let ppm = 0;
    for (const [a, b] of [[11, 23], [12, 24], [23, 25], [24, 26], [25, 27], [26, 28]]) { const lw = Math.hypot(lm[a].x - lm[b].x, lm[a].y - lm[b].y, lm[a].z - lm[b].z), A = px(a), B = px(b); if (lw > 0.15) ppm = Math.max(ppm, Math.hypot(A[0] - B[0], A[1] - B[1]) / lw); }
    if (!(ppm > 0)) return [1, 1]; mem.ppm = mem.ppm ? mem.ppm + 0.1 * (ppm - mem.ppm) : ppm;
    return footStill(SIDE.map(d => { const p = [d.an, d.he, d.to].map(px); return [(p[0][0] + p[1][0] + p[2][0]) / 3, (p[0][1] + p[1][1] + p[2][1]) / 3]; }), mem.ppm, dt);
  }
  // c: where each foot is in the picture (right, left), ppm: picture units per metre at his distance
  function footStill(c, ppm, dt) {
    return c.map((ci, i) => { const m = mem.foot[i]; let st = 1;
      if (m && dt > 0) { const a = 1 - Math.exp(-dt / 0.08); for (let k = 0; k < 2; k++) m.v[k] += a * ((ci[k] - m.c[k]) / dt / ppm - m.v[k]); m.speed = Math.hypot(m.v[0], m.v[1]); st = clamp01((0.35 - m.speed) / 0.23); }
      mem.foot[i] = { c: ci, v: m ? m.v : [0, 0], speed: m ? m.speed : 0 }; return st; });
  }

  // The hand's direction and its knuckle line (little finger to index), in the game's axes. From the hand tracker's
  // 21 points when it found the hand (wrist 0, index knuckle 5, middle knuckle 9, little-finger knuckle 17), otherwise
  // from the body tracker's three rough hand points.
  function handOf(d, si, hands, P, vis) {
    const hp = hands && (d.s < 0 ? hands.right : hands.left), g = p => M.vec(cal, [p.x, p.z, -p.y]);
    if (hp && hp.length >= 18) { const w = g(hp[0]), raw = [V.sub(g(hp[9]), w), V.sub(g(hp[5]), g(hp[17]))], f = mem.hf[si] || (mem.hf[si] = [0, 1, 2, 3, 4, 5].map(() => new Euro(2.5 / Math.max(0.25, o.smooth), 1.5)));
      const dt = mem.dtH || 0, sm = raw.map((v, a) => v.map((x, k) => o.smooth > 0 ? f[3 * a + k].step(x * 10, dt) : x)); return { dir: sm[0], across: sm[1], sure: true }; }
    mem.hf[si] = null; return vis[d.pk] >= o.minVis && vis[d.ix] >= o.minVis ? { dir: V.sub(V.mid(P[d.ix], P[d.pk]), P[d.wr]), across: V.sub(P[d.ix], P[d.pk]), sure: false } : null;
  }

  function solve(lm, t, img = null, aspect = 1, hands = null) {
    mem.dtH = mem.tH === undefined ? 0 : t - mem.tH; mem.tH = t;
    const still = stillness(lm, img, aspect, t), { P: raw, vis } = points(lm, t), P = raw.map(p => M.vec(cal, p)), see = (...k) => k.every(i => vis[i] >= o.minVis);
    const tgt = new Array(nb).fill(null), wt = new Array(nb).fill(1), T = torso(P), Rp = T.pelvis, Rc = T.chest;
    if (!Rp.every(Number.isFinite) || !Rc.every(Number.isFinite)) return q;
    // trunk: the hips carry the trunk's lean; the turn between hips and shoulders is shared by the three spine joints
    const rel = M.mul(M.T(Rp), Rc); tgt[0] = Rp; tgt[id.Spine1] = M.mul(Rp, M.part(rel, 1 / 3)); tgt[id.Spine2] = M.mul(Rp, M.part(rel, 2 / 3)); tgt[id.Chest] = Rc;
    const wh = trust(vis, [LM.nose, LM.lEar, LM.rEar]); wt[id.Neck1] = wt[id.Neck2] = wt[id.Head] = wh;
    if (wh > 0) { const Rh = M.mul(headOf(P), M.T(headCal));
      if (Rh.every(Number.isFinite)) { const rh = M.mul(M.T(Rc), Rh); tgt[id.Neck1] = M.mul(Rc, M.part(rh, 1 / 3)); tgt[id.Neck2] = M.mul(Rc, M.part(rh, 2 / 3)); tgt[id.Head] = Rh; } }
    const Fc = [-Rc[1], -Rc[4], -Rc[7]], Zc = [Rc[2], Rc[5], Rc[8]], Fp = [-Rp[1], -Rp[4], -Rp[7]], Xp = [Rp[0], Rp[3], Rp[6]];
    // the side axis an elbow or knee bends around: from the bend itself when it is bent enough, otherwise the last
    // one, drifting to the natural one (elbows and knees bend forward)
    const bendAxis = (key, a, b, sgn, nat, strict) => { const n = V.cross(a, b), sn = V.len(n); let w = clamp01((sn - 0.15) / 0.35), dir = V.mul(n, sgn / (sn || 1));
      const fb = ok3(V.perp(nat, a)) ? V.norm(V.perp(nat, a)) : null, prev = mem.fw[key];
      if (strict && fb && V.dot(dir, fb) < -0.2) w = 0;   // a knee never bends backwards: that is tracking noise
      let base = prev && fb ? V.perp(V.add(V.mul(prev, 0.85), V.mul(fb, 0.15)), a) : prev ? V.perp(prev, a) : fb; if (!base || !ok3(base)) base = fb || dir; base = V.norm(base);
      let f = V.perp(V.add(V.mul(dir, w), V.mul(base, 1 - w)), a); if (!ok3(f)) f = base; f = V.norm(f); mem.fw[key] = f; return f; };
    SIDE.forEach((d, si) => {
      tgt[id[d.SH]] = Rc;
      const wa = trust(vis, [d.sh, d.el, d.wr]), wl = trust(vis, [d.hip, d.kn, d.an]); wt[id[d.A]] = wt[id[d.FA]] = wa; const hv = o.hands ? handOf(d, si, hands, P, vis) : null; wt[id[d.H]] = hv && hv.sure ? wa : 0.8 * Math.min(wa, trust(vis, [d.pk, d.ix]));   // rough hand points: the wrist eases to its target wt[id[d.LG]] = wt[id[d.SN]] = wt[id[d.FT]] = wl;
      if (wa > 0) { const a = V.norm(V.sub(P[d.el], P[d.sh])), b = V.norm(V.sub(P[d.wr], P[d.el])), sgn = -d.s;
        const f = bendAxis(d.A, a, b, sgn, V.add(V.mul(V.cross(a, Fc), sgn), V.mul(Zc, 0.3)), false);
        tgt[id[d.A]] = aim(a, f, rest(d.FA), Z0); let Rf = aim(b, f, rest(d.H), Z0);
        if (hv) { const dir = hv.dir, nrm = V.mul(V.cross(dir, hv.across), -d.s), rate = hv.sure ? 0.8 : 0.35;
          if (ok3(dir) && ok3(V.perp(nrm, V.norm(dir)))) { const Rh = aim(dir, nrm, [d.s, 0, 0], Z0), qh = M.toQuat(M.mul(M.T(Rf), Rh));
            // The hand's turn around the forearm goes into the forearm. Zero is palm down in the T pose, which is the
            // forearm turned fully inward; palm up is half a turn the other way. So the forearm's range is one-sided:
            // about -20 to +170 degrees for the left arm and the mirror of that for the right. The angle is measured
            // from the middle of that range (75 degrees), which puts the +-180 seam where no forearm can go; measured
            // from zero, a palm-up hand sat on the seam and the figure's hand flipped over.
            const mid = d.s * 1.3; let tw = 2 * Math.atan2(qh[1], qh[0]); tw = mid + (tw - mid) - 2 * Math.PI * Math.round((tw - mid) / (2 * Math.PI));
            tw = Math.max(mid - 2.2, Math.min(mid + 2.2, tw)); tw = mem.tw[si] + Math.max(-rate, Math.min(rate, tw - mem.tw[si])); mem.tw[si] = tw; Rf = M.mul(Rf, M.euler(tw, 0, 0)); tgt[id[d.H]] = Rh; } }
        else Rf = M.mul(Rf, M.euler(mem.tw[si], 0, 0));
        tgt[id[d.FA]] = Rf; }
      if (wl > 0) { const a = V.norm(V.sub(P[d.kn], P[d.hip])), b = V.norm(V.sub(P[d.an], P[d.kn])), foot = see(d.he, d.to), fwd = foot ? V.sub(P[d.to], P[d.he]) : Fp;
        const upF = foot ? V.perp(V.sub(P[d.an], P[d.he]), V.norm(fwd)) : null, latF = upF && ok3(upF) ? V.norm(V.cross(upF, fwd)) : [0, 0, 0];
        const f = bendAxis(d.LG, a, b, 1, V.add(Xp, latF), true);   // a knee's axis lies along the hips' and the foot's sideways axes
        // The performer's bones are not his. The tracked leg length gives the scale between the two bodies, and the
        // ankle is placed at the performer's ankle position (from the middle of the hips) times that scale, so feet
        // that stand still for the performer stand still for him. The knee keeps its tracked bend axis.
        if (see(d.hip, d.kn, d.an) && vis[d.kn] > 0.6) { const len = V.len(V.sub(P[d.kn], P[d.hip])) + V.len(V.sub(P[d.an], P[d.kn])); if (len > 0.3 && len < 1.5) perf.leg = perf.leg === null ? len : perf.leg + 0.02 * (len - perf.leg); }
        const sc = o.fit && perf.leg ? Math.max(0.7, Math.min(1.45, LEG / perf.leg)) : 1, Hr = M.vec(Rp, S.off[id.RightLeg]), Hl = M.vec(Rp, S.off[id.LeftLeg]);
        const ik = o.fit ? legIK(d, d.s < 0 ? Hr : Hl, V.add(V.mid(Hr, Hl), V.mul(V.sub(P[d.an], V.mid(P[LM.lHip], P[LM.rHip])), sc)), f) : null;
        if (ik) { tgt[id[d.LG]] = ik[0]; tgt[id[d.SN]] = ik[1]; } else { tgt[id[d.LG]] = aim(a, f, rest(d.SN), X0); tgt[id[d.SN]] = aim(b, f, rest(d.FT), X0); }
        if (foot) { const Rt = footOf(P, d); if (Rt) tgt[id[d.FT]] = M.mul(Rt, M.T(footCal[si])); } }
    });
    return finish(tgt, wt, still, !!img);
  }

  // From each body's wanted world rotation (tgt, with trust wt) to the pose: hinge angles, then the hips in the room and
  // the feet on the floor. Shared by both kinds of tracker. still: how still each foot is in the picture; pic: whether that is known.
  function finish(tgt, wt, still, pic) {
    // world rotations -> hinge angles, joint by joint; a joint without a tracked rotation keeps its angles
    const out = new Float64Array(q), Rw = new Array(nb); Rw[0] = tgt[0];
    let qr = M.toQuat(Rw[0]); if (qr[0] * q[3] + qr[1] * q[4] + qr[2] * q[5] + qr[3] * q[6] < 0) qr = qr.map(v => -v);
    out[3] = qr[0]; out[4] = qr[1]; out[5] = qr[2]; out[6] = qr[3];
    for (let b = 1; b < nb; b++) { const h = 7 + 3 * (b - 1), pa = Rw[S.parent[b]], near = [q[h], q[h + 1], q[h + 2]];
      if (tgt[b]) { let R = tgt[b]; if (wt[b] < 1) { const held = M.mul(pa, M.euler(near[0], near[1], near[2])); R = M.mul(held, M.part(M.mul(M.T(held), R), wt[b] * wt[b])); }   // weakly seen: drift toward the tracker's guess
        const e = M.toEuler(M.mul(M.T(pa), R), near); out[h] = e[0]; out[h + 1] = e[1]; out[h + 2] = e[2]; Rw[b] = R; }
      else Rw[b] = M.mul(pa, M.euler(near[0], near[1], near[2])); }
    // The hips in the room. The lowest part of the body rests on the floor. Each foot is held where it is while it
    // is on the floor; the hips go where the held feet put them, and each held leg is then bent to reach its spot
    // exactly (two-bone IK around the same bend axis), so a standing foot cannot slide.
    const euler = b => { const h = 7 + 3 * (b - 1), e = M.toEuler(M.mul(M.T(Rw[S.parent[b]]), Rw[b]), [q[h], q[h + 1], q[h + 2]]); out[h] = e[0]; out[h + 1] = e[1]; out[h + 2] = e[2]; };
    out[0] = 0; out[1] = 0; out[2] = 0; const K = fk(S, out); let low = 1e9;
    for (let b = 0; b < nb; b++) low = Math.min(low, K.p[b][2] - RAD[b]);
    // How firmly each foot stands: 1 when its lowest part is within 2.5 cm of the floor, 0 from 5 cm up, with no
    // stickiness, so a foot is free as soon as it lifts. The held point is the part of the foot on the floor: the
    // ball of the foot when the heel is up, otherwise the ankle. The ankle can then roll forward over a planted toe.
    const pv = [null, null], g = [0, 0];
    SIDE.forEach((d, i) => { const ft = id[d.FT], tb = id[d.TB], zf = K.p[ft][2] - RAD[ft], zt = K.p[tb][2] - RAD[tb], h = Math.min(zf, zt) - low;
      const shinUp = K.p[id[d.SN]][2] - K.p[ft][2] > 0.2;   // the knee well above the ankle: a foot he stands on, not a leg lying on the floor
      g[i] = o.travel && !!tgt[id[d.LG]] && wt[id[d.LG]] > 0.5 && shinUp ? (pic ? clamp01((0.10 - h) / 0.05) * still[i] : clamp01((0.05 - h) / 0.025)) : 0;   // with the picture: low and not moving; without it: low
      const toe = root.toe[i] ? zt < zf + 0.005 : zt < zf - 0.01; if (toe !== root.toe[i]) { root.toe[i] = toe; root.lock[i] = null; }   // the held point changed: hold the new one where it is now
      pv[i] = toe ? K.p[tb] : K.p[ft]; if (!g[i]) root.lock[i] = null; });
    let sx = 0, sy = 0, sw = 0; SIDE.forEach((d, i) => { const L = root.lock[i]; if (L) { sx += g[i] * (L[0] - pv[i][0]); sy += g[i] * (L[1] - pv[i][1]); sw += g[i]; } });
    if (sw > 0) root.xy = [sx / sw, sy / sw];
    SIDE.forEach((d, i) => { if (!g[i]) return; const at = [root.xy[0] + pv[i][0], root.xy[1] + pv[i][1]], L = root.lock[i];
      if (!L) root.lock[i] = at;   // a foot that has just landed is held where it landed
      else { const e = Math.hypot(at[0] - L[0], at[1] - L[1]), k = e > 0.2 ? 1 : Math.max(1 - g[i], clamp01((e - 0.03) / 0.1) * 0.4); L[0] += (at[0] - L[0]) * k; L[1] += (at[1] - L[1]) * k; } });   // a lifting or dragged foot takes its spot with it
    out[0] = root.xy[0]; out[1] = root.xy[1]; out[2] = -low;
    SIDE.forEach((d, i) => { const L = root.lock[i]; if (!L || !g[i]) return; const RL = Rw[id[d.LG]], f0 = [RL[0], RL[3], RL[6]];   // the knee's hinge axis: the thigh's own x axis
      const lg = id[d.LG], sn = id[d.SN], ft = id[d.FT], tb = id[d.TB], H = [root.xy[0] + K.p[lg][0], root.xy[1] + K.p[lg][1], K.p[lg][2] - low];
      // a held foot that is nearly flat is flat (the tracked foot angle wobbles by several degrees), and it rests on the floor
      const fwd = [-Rw[ft][1], -Rw[ft][4], -Rw[ft][7]], fh = V.perp(fwd, Z0);
      if (ok3(fh)) { const flat = aim(fh, Z0, [0, -1, 0], Z0), rel = M.mul(M.T(Rw[ft]), flat), ang = Math.acos(Math.max(-1, Math.min(1, (rel[0] + rel[4] + rel[8] - 1) / 2))); Rw[ft] = M.mul(Rw[ft], M.part(rel, g[i] * clamp01((0.44 - ang) / 0.17))); }   // fully flat within 15 degrees, tracked as it is from 25
      const gap = Math.min(K.p[ft][2] - RAD[ft], K.p[ft][2] + M.vec(Rw[ft], S.off[tb])[2] - RAD[tb]) - low;   // how far the sole is above the floor
      const dx = (L[0] - root.xy[0] - pv[i][0]) * g[i], dy = (L[1] - root.xy[1] - pv[i][1]) * g[i], ik = legIK(d, H, [root.xy[0] + K.p[ft][0] + dx, root.xy[1] + K.p[ft][1] + dy, K.p[ft][2] - low - g[i] * gap], f0); if (!ik) return;
      Rw[lg] = ik[0]; Rw[sn] = ik[1]; euler(lg); euler(sn); euler(ft); });
    root.g = g; root.pt = SIDE.map((d, i) => { if (!root.lock[i]) return null; const Kf = fk(S, out), p = Kf.p[root.toe[i] ? id[d.TB] : id[d.FT]]; return [p[0], p[1], root.toe[i] ? 1 : 0]; });   // where each held point ended up (for the checks)
    q = out; return out;
  }
  const home = () => { root.xy = [0, 0]; root.lock = [null, null]; };   // the hips back over the origin (a new recording starts here)
  // A tracker that gives joint rotations (InstantHMR): W = each body's world rotation in the game's axes, as the camera
  // sees him. feet: where the right and the left foot are in the picture (pixels), ppm: pixels per metre at his distance.
  function solveRot(W, t, feet = null, ppm = 0) {
    const dt = mem.tF === undefined ? 0 : t - mem.tF; mem.tF = t;
    const still = feet && ppm > 0 ? footStill(feet, ppm, dt) : [1, 1];
    return finish(W.map(R => R ? M.mul(cal, R) : null), new Array(nb).fill(1), still, !!(feet && ppm > 0));
  }
  // he stands straight, facing the camera: the hips' rotation at this moment counts as upright and facing forward
  function calibrateRot(W) { if (!W[0] || !W[0].every(Number.isFinite)) return false; cal = M.T(W[0]); return true; }
  return { solve, calibrate, solveRot, calibrateRot, reset, home, opt: o, get pose() { return q; }, get twist() { return mem.tw.slice(); }, get footSpeed() { return mem.foot.map(m => m ? m.speed : 0); }, get footCal() { return footCal; }, get performerLeg() { return perf.leg; }, get held() { return SIDE.filter((d, i) => root.lock[i] && root.g && root.g[i] > 0.5).map(d => id[d.FT]); }, get contacts() { return SIDE.map((d, i) => ({ g: root.g ? root.g[i] : 0, pt: root.pt ? root.pt[i] : null })); } };
}

// A finished recording: frames taken at the camera's own times -> 30 frames a second, lightly smoothed.
// frames = [{ t (seconds), q (73) }]. Returns { T, C, q } like libq.js decodeQ. Joints are mixed as rotations, not as
// hinge angles, so a frame between two camera frames is always a real pose.
export function toMove(frames, fps = 30, soften = 1) {
  const C = 73, NJ = 23, n = frames.length, Q = frames.map(f => { const a = new Float64Array(4 * NJ); a.set([f.q[3], f.q[4], f.q[5], f.q[6]]);
    for (let j = 1; j < NJ; j++) a.set(M.toQuat(M.euler(f.q[4 + 3 * j], f.q[5 + 3 * j], f.q[6 + 3 * j])), 4 * j); return a; });
  for (let i = 1; i < n; i++) for (let j = 0; j < NJ; j++) { const o = 4 * j, A = Q[i - 1], B = Q[i]; if (A[o] * B[o] + A[o + 1] * B[o + 1] + A[o + 2] * B[o + 2] + A[o + 3] * B[o + 3] < 0) for (let c = 0; c < 4; c++) B[o + c] = -B[o + c]; }
  const t0 = frames[0].t, T = Math.max(1, Math.floor((frames[n - 1].t - t0) * fps + 1e-6) + 1), rp = new Float64Array(T * 3), rq = new Float64Array(T * 4 * NJ); let k = 0;
  for (let i = 0; i < T; i++) { const t = t0 + i / fps; while (k < n - 2 && frames[k + 1].t < t) k++;
    const a = k, b = Math.min(n - 1, k + 1), u = frames[b].t > frames[a].t ? clamp01((t - frames[a].t) / (frames[b].t - frames[a].t)) : 0;
    for (let c = 0; c < 3; c++) rp[i * 3 + c] = frames[a].q[c] + (frames[b].q[c] - frames[a].q[c]) * u;
    for (let c = 0; c < 4 * NJ; c++) rq[i * 4 * NJ + c] = Q[a][c] + (Q[b][c] - Q[a][c]) * u; }
  const q = new Float64Array(T * C), w = 0.25 * soften, mix = (src, i, c, W) => { const a = Math.max(0, i - 1), b = Math.min(T - 1, i + 1); return w * src[a * W + c] + (1 - 2 * w) * src[i * W + c] + w * src[b * W + c]; };
  for (let i = 0; i < T; i++) { const o = i * C; for (let c = 0; c < 3; c++) q[o + c] = mix(rp, i, c, 3);
    for (let j = 0; j < NJ; j++) { const v = [0, 1, 2, 3].map(c => mix(rq, i, 4 * j + c, 4 * NJ)), l = Math.hypot(...v) || 1;
      if (j === 0) for (let c = 0; c < 4; c++) q[o + 3 + c] = v[c] / l;
      else { const h = o + 4 + 3 * j, e = M.toEuler(M.fromQuat(v[0], v[1], v[2], v[3]), i ? [q[h - C], q[h + 1 - C], q[h + 2 - C]] : [frames[0].q[4 + 3 * j], frames[0].q[5 + 3 * j], frames[0].q[6 + 3 * j]]); q[h] = e[0]; q[h + 1] = e[1]; q[h + 2] = e[2]; } } }
  return { T, C, q };
}
