// The Box - the mannequin's look (drawing only), fitted onto the SOMA body.
//
// The sculpted black-and-white mannequin of v12.1 (shapes from v13/index.html: `loft`, `sculpt`) re-fitted to the
// SOMA 23-body humanoid: each old shape is stretched to its new bone and placed by SOMA's joints in its rest pose
// (a T-pose; MuJoCo frame: z up, the body faces -y). Collisions still come from the MJCF; nothing here touches physics.
//
// buildLook(THREE, mergeGeometries, W) -> { BodyName: [{ geo, mat: 'shell' | 'dark' | 'dark2' }] }, each geometry in
// its MuJoCo body's own frame (metres), ready to add to that body's group.
//
// Old shapes were made in "figure units" for a 1.75 m figure: y up, z toward the glass (front), x to the side.
// Old "R" parts sit on +x, which is SOMA's Left side, so SOMA Left* parts use the old R shapes and Right* the L ones.

const PI = Math.PI;
const gau = (x, s) => Math.exp(-(x * x) / (s * s));
const angTo = (th, a) => { let d = th - a; while (d > PI) d -= 2 * PI; while (d < -PI) d += 2 * PI; return d; };

export function buildLook(THREE, mergeGeometries, W) {
  // ---------- v12.1's sculpting tools (unchanged) ----------
  // A smooth body made of stacked superellipse rings [y, half width, half depth, z shift, x shift], domed at the ends.
  // bump(angle, y) pushes the surface out (angle 0 = front, +PI/2 = +x).
  function loft(secs, o = {}) {
    const e = 2 / (o.n || 2), seg = o.seg || 36, sub = o.sub || 4;
    const sec = secs.map(s => [s[0], s[1], s[2], s[3] || 0, s[4] || 0]);
    const L = sec.length, at = i => sec[Math.max(0, Math.min(L - 1, i))];
    const cr = (a, b, c, d, t) => 0.5 * (2 * b + (c - a) * t + (2 * a - 5 * b + 4 * c - d) * t * t + (3 * b - a - 3 * c + d) * t * t * t);
    let rings = [];
    for (let i = 0; i < L - 1; i++) for (let k = 0; k < sub; k++) {
      const r = []; for (let c = 0; c < 5; c++) r.push(cr(at(i - 1)[c], at(i)[c], at(i + 1)[c], at(i + 2)[c], k / sub)); rings.push(r);
    }
    rings.push(sec[L - 1]);
    const cap = (r, h, sg) => { const out = []; for (let k = 1; k <= 5; k++) { const a = k / 6 * PI / 2, c = Math.cos(a); out.push([r[0] + sg * h * Math.sin(a), r[1] * c, r[2] * c, r[3], r[4]]); } return out; };
    const cb = o.capB || 0, ct = o.capT || 0;
    const r0 = rings[0], r1 = rings[rings.length - 1];
    if (cb) rings = cap(r0, cb, -1).reverse().concat(rings);
    if (ct) rings = rings.concat(cap(r1, ct, 1));
    const pos = [r0[4], r0[0] - cb, r0[3]];
    for (const [y, w, d, oz, ox] of rings) for (let j = 0; j < seg; j++) {
      let th = j / seg * 2 * PI; const s = Math.sin(th), c = Math.cos(th); if (th > PI) th -= 2 * PI;
      const b = o.bump ? o.bump(th, y) : 0;
      pos.push(ox + w * Math.sign(s) * Math.pow(Math.abs(s), e) + b * s, y, oz + d * Math.sign(c) * Math.pow(Math.abs(c), e) + b * c);
    }
    pos.push(r1[4], r1[0] + ct, r1[3]);
    const idx = [], R = rings.length, top = 1 + R * seg, v = (i, j) => 1 + i * seg + (j % seg);
    for (let j = 0; j < seg; j++) {
      idx.push(v(0, j), 0, v(0, j + 1));
      for (let i = 0; i < R - 1; i++) idx.push(v(i, j), v(i, j + 1), v(i + 1, j), v(i, j + 1), v(i + 1, j + 1), v(i + 1, j));
      idx.push(v(R - 1, j), v(R - 1, j + 1), top);
    }
    const g = new THREE.BufferGeometry();
    g.setAttribute('position', new THREE.Float32BufferAttribute(pos, 3)); g.setIndex(idx);
    return g;
  }
  function seg3(p, dir, len, r) {
    const g = new THREE.CapsuleGeometry(r, len, 4, 10); g.deleteAttribute('uv'); g.deleteAttribute('normal');
    const d = new THREE.Vector3(...dir).normalize();
    g.applyQuaternion(new THREE.Quaternion().setFromUnitVectors(new THREE.Vector3(0, 1, 0), d));
    g.translate(p[0] + d.x * len / 2, p[1] + d.y * len / 2, p[2] + d.z * len / 2);
    return { g, end: [p[0] + d.x * len, p[1] + d.y * len, p[2] + d.z * len] };
  }
  function finger(base, dir, lens, r, curl) {
    const out = []; let p = base; const d = new THREE.Vector3(...dir).normalize();
    for (const l of lens) { const s = seg3(p, [d.x, d.y, d.z], l, r); out.push(s.g); p = s.end; d.applyAxisAngle(new THREE.Vector3(1, 0, 0), -curl); }
    return out;
  }
  const bare = g => { for (const k of ['uv', 'normal']) if (g.getAttribute(k)) g.deleteAttribute(k); return g; };

  // the v12.1 shapes, in old figure units, each centred on its old part (see v13/index.html sculpt())
  function sculpt(base, side) {
    switch (base) {
      case 'chest': return loft([
          [-0.110, 0.104, 0.074], [-0.080, 0.114, 0.082, 0.004], [-0.040, 0.130, 0.090, 0.006], [0.000, 0.146, 0.096, 0.006],
          [0.040, 0.160, 0.098, 0.004], [0.075, 0.172, 0.094], [0.098, 0.168, 0.084, -0.006], [0.115, 0.130, 0.070, -0.010],
          [0.128, 0.082, 0.056, -0.012]],
          { n: 2.6, capB: 0.004, capT: 0.010, seg: 56, sub: 5, bump: (th, y) => {
            const pec = gau(y - 0.035, y < 0.035 ? 0.022 : 0.05);
            return 0.016 * (gau(angTo(th, 0.42), 0.3) + gau(angTo(th, -0.42), 0.3)) * pec
              - 0.005 * gau(th, 0.07) * gau(y - 0.02, 0.07)
              + 0.010 * (gau(angTo(th, PI - 0.55), 0.33) + gau(angTo(th, -PI + 0.55), 0.33)) * gau(y - 0.045, 0.04)
              + 0.006 * (gau(angTo(th, PI / 2 + 0.35), 0.35) + gau(angTo(th, -PI / 2 - 0.35), 0.35)) * gau(y - 0.02, 0.05)
              - 0.004 * gau(angTo(th, PI), 0.07);
          } });
      case 'belly': return loft([[-0.062, 0.110, 0.078], [-0.030, 0.100, 0.074, 0.002], [0.000, 0.097, 0.073, 0.003],
          [0.030, 0.099, 0.074, 0.003], [0.062, 0.104, 0.075]],
          { n: 2.3, capB: 0.006, capT: 0.006, seg: 48, bump: (th, y) =>
            0.004 * (gau(angTo(th, 0.2), 0.13) + gau(angTo(th, -0.2), 0.13)) * (1 - 0.35 * gau(y + 0.005, 0.012))
            - 0.004 * gau(angTo(th, PI), 0.07) });
      case 'pelvis': return loft([
          [-0.074, 0.050, 0.052, 0.008], [-0.062, 0.088, 0.070, 0.004], [-0.040, 0.122, 0.082, 0.001], [-0.008, 0.134, 0.086, -0.002],
          [0.030, 0.126, 0.084, -0.002], [0.055, 0.114, 0.079], [0.070, 0.110, 0.078]],
          { n: 2.4, capB: 0.010, capT: 0.003, seg: 48, sub: 5, bump: (th, y) =>
            0.013 * (gau(angTo(th, PI - 0.5), 0.38) + gau(angTo(th, -PI + 0.5), 0.38)) * gau(y + 0.03, 0.035)
            - 0.006 * gau(angTo(th, PI), 0.06) * gau(y + 0.045, 0.035)
            - (Math.cos(th) > 0 ? 0.004 * gau(Math.abs(th) - (0.3 + 8 * (y + 0.07)), 0.09) * gau(y + 0.03, 0.04) : 0) });
      case 'neck': return loft([[-0.045, 0.050, 0.046], [-0.015, 0.045, 0.043], [0.015, 0.043, 0.042], [0.045, 0.045, 0.044]],
          { capB: 0.008, capT: 0.008, seg: 28 });
      case 'head': {
        const r = 0.10;
        const g = loft([
          [-1.20, 0.18, 0.12, 0.40], [-1.12, 0.38, 0.27, 0.30], [-0.98, 0.56, 0.46, 0.16], [-0.72, 0.66, 0.63, 0.07],
          [-0.40, 0.76, 0.76, 0.02], [-0.05, 0.83, 0.86, -0.02], [0.25, 0.86, 0.90, -0.05], [0.50, 0.835, 0.88, -0.07]],
          { capB: 0.03, capT: 0.52, seg: 64, sub: 6, bump: (th, y) => {
            const ny = gau(y + 0.32, y > -0.32 ? 0.36 : 0.15);
            return 0.14 * gau(th, 0.075 + 0.05 * ny) * ny
              + 0.05 * gau(th, 0.6) * gau(y - 0.15, 0.07)
              - 0.035 * (gau(angTo(th, 0.36), 0.15) + gau(angTo(th, -0.36), 0.15)) * gau(y - 0.0, 0.1)
              + 0.025 * (gau(angTo(th, 0.75), 0.2) + gau(angTo(th, -0.75), 0.2)) * gau(y + 0.22, 0.12)
              + 0.012 * gau(th, 0.3) * gau(y + 0.66, 0.09)
              + 0.035 * gau(th, 0.28) * gau(y + 1.02, 0.09);
          } });
        g.scale(r, r, r);
        const ears = [-1, 1].map(sx => { const e = bare(new THREE.SphereGeometry(1, 16, 12)); e.scale(0.11 * r, 0.27 * r, 0.19 * r); e.rotateY(sx * 0.25); e.translate(sx * 0.82 * r, -0.12 * r, -0.10 * r); return e; });
        return mergeGeometries([g, ...ears]);
      }
      case 'cap': {   // dark shoulder cap over the ball, centred on the shoulder joint
        const g = bare(new THREE.SphereGeometry(0.058, 24, 10, 0, 2 * PI, 0, 0.40 * PI));
        g.scale(1, 0.8, 0.95); g.rotateZ(-side * 0.55);
        return g;
      }
      case 'uArm': return loft([[-0.092, 0.030, 0.032], [-0.050, 0.034, 0.037], [0.010, 0.040, 0.044], [0.060, 0.045, 0.046], [0.086, 0.043, 0.043]],
          { capB: 0.016, capT: 0.02, bump: (th, y) => 0.004 * gau(th, 0.6) * gau(y - 0.0, 0.04) });
      case 'fArm': return loft([[-0.088, 0.023, 0.020], [-0.050, 0.028, 0.026], [0.0, 0.034, 0.033], [0.050, 0.038, 0.036], [0.078, 0.035, 0.034]],
          { capB: 0.012, capT: 0.018 });
      case 'thigh': return loft([[-0.150, 0.040, 0.043], [-0.100, 0.046, 0.050, 0.002], [-0.030, 0.054, 0.059, 0.004], [0.050, 0.061, 0.065, 0.002],
          [0.120, 0.065, 0.066], [0.150, 0.064, 0.063], [0.178, 0.056, 0.056, 0, side * 0.012]],
          { capB: 0.022, capT: 0.018, bump: (th, y) => 0.005 * gau(th, 0.5) * gau(y + 0.02, 0.07) });
      case 'shin': return loft([[-0.150, 0.026, 0.028], [-0.100, 0.031, 0.034, -0.003], [-0.020, 0.039, 0.045, -0.008], [0.060, 0.044, 0.050, -0.010],
          [0.110, 0.044, 0.045, -0.004], [0.140, 0.041, 0.041]],
          { capB: 0.018, capT: 0.022 });
      case 'hand': {
        const sx = side;
        const parts = [loft([[0.046, 0.019, 0.012], [0.030, 0.026, 0.0140], [0.004, 0.029, 0.0145], [-0.008, 0.028, 0.0135]],
          { n: 3, capB: 0.004, capT: 0.007, seg: 28 })];
        const fx = [-0.0195, -0.0066, 0.0066, 0.0195], fl = [0.034, 0.042, 0.046, 0.043];
        fx.forEach((x, i) => parts.push(...finger([sx * x, -0.008, 0.001], [0.02 * sx * Math.sign(x), -1, 0], [fl[i] * 0.48, fl[i] * 0.32, fl[i] * 0.22], 0.0071, 0.22)));
        parts.push(...finger([sx * 0.021, 0.012, 0.011], [sx * 0.04, -0.95, 0.3], [0.018, 0.015], 0.0078, 0.12));
        const g = mergeGeometries(parts.map(bare));
        g.rotateY(-sx * PI / 2);
        return g;
      }
      case 'foot': {
        const g = loft([[-0.070, 0.028, 0.025, 0.004], [-0.040, 0.034, 0.029, 0.0], [0.0, 0.038, 0.030, 0.0], [0.040, 0.041, 0.025, 0.005], [0.075, 0.042, 0.020, 0.010]],
          { n: 3, capB: 0.012, seg: 32 });
        g.rotateX(PI / 2);
        return g;
      }
      case 'toe': {
        const g = loft([[-0.035, 0.041, 0.020, 0.002], [0.0, 0.040, 0.018, 0.0045], [0.024, 0.035, 0.0135, 0.009]], { n: 3, capT: 0.012, seg: 32 });
        g.rotateX(PI / 2);
        return g;
      }
    }
  }

  // ---------- SOMA's rest pose (T-pose, standing at the origin) ----------
  const mj = W.mj, m = W.m, d2 = new mj.MjData(m);
  const root = m.body_jntadr[W.bodyId.Hips], qa = m.jnt_qposadr[root];
  d2.qpos[qa + 3] = 1;
  mj.mj_kinematics(m, d2);
  const P = {};
  for (const n of W.bodyNames) { const b = W.bodyId[n]; P[n] = new THREE.Vector3(d2.xpos[3 * b], d2.xpos[3 * b + 1], d2.xpos[3 * b + 2]); }
  d2.delete();
  const V = (x, y, z) => new THREE.Vector3(x, y, z);
  const FRONT = V(0, -1, 0), UP = V(0, 0, 1);

  // an old shape onto SOMA: scale (old x side, y along the bone, z front), turn old y -> u and old z -> front,
  // then put old point `anchor` (old units, after scaling) at world point `at`
  function place(g, { u = UP, f = FRONT, s = [1, 1, 1], at, anchor = [0, 0, 0] }) {
    g.scale(s[0], s[1], s[2]);
    const uu = u.clone().normalize(), ff = f.clone().sub(uu.clone().multiplyScalar(f.dot(uu))).normalize();
    const xx = new THREE.Vector3().crossVectors(uu, ff);
    const M = new THREE.Matrix4().makeBasis(xx, uu, ff);
    const a = new THREE.Vector3(anchor[0] * s[0], anchor[1] * s[1], anchor[2] * s[2]).applyMatrix4(M);
    g.applyMatrix4(M);
    g.translate(at.x - a.x, at.y - a.y, at.z - a.z);
    return g;
  }
  // a limb along its bone: old part centred between its joints; scaled so its old joint-to-joint length fits
  function limb(base, side, from, to, oldLen, widen = 1) {
    const bone = P[from].clone().sub(P[to]), L = bone.length();
    const g = sculpt(base, side);
    return place(g, { u: bone, s: [widen, L / oldLen, widen], at: P[from].clone().add(P[to]).multiplyScalar(0.5) });
  }

  const out = {}; for (const n of W.bodyNames) out[n] = [];
  const add = (body, geo, mat = 'shell') => { geo.computeVertexNormals(); out[body].push({ geo, mat, at: P[body] }); };
  const ball = (body, r) => add(body, place(bare(new THREE.SphereGeometry(r, 20, 14)), { at: P[body] }), 'dark');
  const band = (body, rx, rz, dy) => { const g = bare(new THREE.CylinderGeometry(1, 1, 0.014, 40)); g.scale(rx, 1, rz); add(body, place(g, { at: P[body].clone().add(V(0, dy, 0)) }), 'dark'); };

  // torso: pelvis on the hips, belly on Spine2 (between Spine1 and Chest), chest up to the neck; dark waist bands
  const hipJ = P.RightLeg.clone().add(P.LeftLeg).multiplyScalar(0.5), hipW = P.LeftLeg.x - P.RightLeg.x;
  add('Hips', place(sculpt('pelvis', 1), { s: [hipW / 0.18, 0.95, 1], at: V(0, hipJ.y - 0.004, hipJ.z + 0.08 * 0.95) }));
  const bellyLen = P.Chest.z - P.Spine1.z;
  add('Spine2', place(sculpt('belly', 1), { s: [0.97, bellyLen / 0.145, 0.98], at: V(0, P.Spine2.y - 0.035, (P.Spine1.z + P.Chest.z) / 2) }));
  band('Spine1', 0.104, 0.074, -0.035);
  band('Chest', 0.098, 0.070, -0.035);
  const chestLen = P.Neck1.z - P.Chest.z, chestS = chestLen / 0.225;
  add('Chest', place(sculpt('chest', 1), { s: [0.88, chestS, 1.0], at: V(0, P.Chest.y - 0.028, P.Chest.z + 0.115 * chestS) }));
  // neck in two pieces (Neck1 -> Neck2 -> Head), dark like v12.1
  // the neck sits a little behind the joint line and stops under the skull (in front it showed as a dark beard)
  add('Neck1', limb('neck', 1, 'Neck1', 'Neck2', 0.10, 0.92).translate(0, 0.008, 0), 'dark');
  add('Neck2', limb('neck', 1, 'Neck2', 'Head', 0.13, 0.88).translate(0, 0.016, -0.006), 'dark');
  add('Head', place(sculpt('head', 1), { s: [0.97, 0.97, 0.97], at: P.Head.clone().add(V(0, 0.022, 0.075)) }));
  ball('Head', 0.035);
  // arms (old R shapes on SOMA's Left, old L on Right)
  for (const [S, side] of [['Left', 1], ['Right', -1]]) {
    add(S + 'Shoulder', place(sculpt('cap', side), { at: P[S + 'Arm'], s: [0.9, 0.9, 0.9] }), 'dark2');
    ball(S + 'Arm', 0.046);
    add(S + 'Arm', limb('uArm', side, S + 'Arm', S + 'ForeArm', 0.26));
    ball(S + 'ForeArm', 0.036);
    add(S + 'ForeArm', limb('fArm', side, S + 'ForeArm', S + 'Hand', 0.24));
    ball(S + 'Hand', 0.024);
    const u = P[S + 'ForeArm'].clone().sub(P[S + 'Hand']);
    add(S + 'Hand', place(sculpt('hand', side), { u, s: [1.08, 1.08, 1.08], at: P[S + 'Hand'], anchor: [0, 0.058, 0] }), 'dark');
    // legs
    ball(S + 'Leg', 0.046);
    add(S + 'Leg', limb('thigh', side, S + 'Leg', S + 'Shin', 0.40));
    ball(S + 'Shin', 0.047);
    add(S + 'Shin', limb('shin', side, S + 'Shin', S + 'Foot', 0.385));
    ball(S + 'Foot', 0.033);
    // shoes: on SOMA's foot and toe blocks (foot: 15 cm long, sole 4.5 cm under the ankle; toe from 13 cm ahead)
    add(S + 'Foot', place(sculpt('foot', side), { s: [1.05, 1, 1], at: P[S + 'Foot'].clone().add(V(0, -0.048, -0.017)) }), 'dark');
    add(S + 'ToeBase', place(sculpt('toe', side), { s: [1.1, 1, 1], at: P[S + 'ToeBase'].clone().add(V(0, -0.014, 0.025)) }), 'dark');
  }
  // into each body's own frame (all SOMA body frames are world-aligned in the rest pose)
  for (const n of W.bodyNames) for (const it of out[n]) { it.geo.translate(-it.at.x, -it.at.y, -it.at.z); delete it.at; }
  return out;
}
