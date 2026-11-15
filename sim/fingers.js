// Fingers for the figure's hands (user 21:33: "when the hand goes against the wall, it spreads the fingers a bit, versus
// when it's more relaxed and the fingers are more close together"). The physics hand is one piece; this is the look
// only. Each GLB hand is cut, once, into the palm and 3 pieces per finger (2-3 for the thumb), split through the
// middle of the mannequin's joint balls (a ball turned about its own centre stays a ball, so the cuts don't show), and
// hung on pivots at those ball centres. Self-calibrating: the fingers are found as 4 bands across the hand, the thumb
// as the part off its side, so it works on both figures and both hands.
export function rigHand(THREE, grp, side) {
  const meshes = grp.children.filter(o => o.isMesh);
  if (!meshes.length) return null;
  const all = [];
  for (const m of meshes) { const p = m.geometry.attributes.position; for (let i = 0; i < p.count; i++) all.push([p.getX(i), p.getY(i), p.getZ(i)]); }
  const far = all.filter(p => Math.abs(p[0]) > 0.06), s = far.reduce((a, p) => a + p[0], 0) > 0 ? 1 : -1;   // fingers along +x (left) / -x (right)
  const U = p => s * p[0];
  const isThumb = (u, y) => y < -0.04 && u > 0.03;
  // the 4 finger bands: 1-D k-means on y of the points well out on the fingers
  const fy = all.filter(p => U(p) > 0.105 && !isThumb(U(p), p[1])).map(p => p[1]).sort((a, b) => a - b);
  if (fy.length < 40) return null;
  let c = [0.125, 0.375, 0.625, 0.875].map(q => fy[Math.floor(q * (fy.length - 1))]);
  for (let it = 0; it < 20; it++) {
    const sum = [0, 0, 0, 0], n = [0, 0, 0, 0];
    for (const y of fy) { let k = 0; for (let j = 1; j < 4; j++) if (Math.abs(y - c[j]) < Math.abs(y - c[k])) k = j; sum[k] += y; n[k]++; }
    c = c.map((v, k) => n[k] ? sum[k] / n[k] : v);
  }
  const near = y => { let k = 0; for (let j = 1; j < 4; j++) if (Math.abs(y - c[j]) < Math.abs(y - c[k])) k = j; return k; };
  // each finger's joints along it: knuckle ball just inside where the band starts, then ~42% and ~72% to the tip
  const zAt = (pts, u) => { const q = pts.filter(p => Math.abs(U(p) - u) < 0.01); return q.length ? q.reduce((a, p) => a + p[2], 0) / q.length : 0; };
  const fingers = c.map((y, k) => {
    const pts = all.filter(p => U(p) > 0.06 && !isThumb(U(p), p[1]) && near(p[1]) === k && Math.abs(p[1] - y) < 0.016);
    const u0 = Math.min(...pts.map(U)), u1 = Math.max(...pts.map(U)), uk = u0 + 0.012;
    const J = [uk, uk + 0.42 * (u1 - uk), uk + 0.72 * (u1 - uk)];
    return { y, J, P: J.map(u => [s * u, y, zAt(pts, u)]) };
  });
  const tp = all.filter(p => isThumb(U(p), p[1]));
  let thumb = null;
  if (tp.length > 30) {
    const t0 = Math.min(...tp.map(U)), ty = tp.reduce((a, p) => a + p[1], 0) / tp.length, J = [t0 + 0.015, t0 + 0.045, t0 + 0.065];
    thumb = { y: ty, J, P: J.map(u => [s * u, ty, zAt(tp, u)]) };
  }
  // which piece each triangle goes to (by its centre): 0 palm; 1 + 3f + seg for finger f (0..3); 13 + seg thumb
  const pieceOf = (u, y) => {
    if (thumb && isThumb(u, y)) { const J = thumb.J; return u < J[0] ? 0 : 13 + (u < J[1] ? 0 : u < J[2] ? 1 : 2); }
    if (u < 0.07) return 0;
    const f = near(y), J = fingers[f].J;
    return u < J[0] ? 0 : 1 + 3 * f + (u < J[1] ? 0 : u < J[2] ? 1 : 2);
  };
  // pivots: chains of groups; piece meshes inside them, shifted so the geometry keeps its place
  const ctl = { s, fingers, thumb, piv: [], tpiv: [], side };
  const chain = (P) => {
    const g0 = new THREE.Group(); g0.position.set(...P[0]); grp.add(g0);
    const g1 = new THREE.Group(); g1.position.set(P[1][0] - P[0][0], P[1][1] - P[0][1], P[1][2] - P[0][2]); g0.add(g1);
    const g2 = new THREE.Group(); g2.position.set(P[2][0] - P[1][0], P[2][1] - P[1][1], P[2][2] - P[1][2]); g1.add(g2);
    return [g0, g1, g2];
  };
  fingers.forEach(f => ctl.piv.push(chain(f.P)));
  if (thumb) ctl.tpiv = chain(thumb.P);
  const holder = k => k === 0 ? null : k < 13 ? ctl.piv[Math.floor((k - 1) / 3)][(k - 1) % 3] : ctl.tpiv[k - 13];
  const jointOf = k => k < 13 ? fingers[Math.floor((k - 1) / 3)].P[(k - 1) % 3] : thumb.P[k - 13];
  for (const m of meshes) {
    const geo = m.geometry, pos = geo.attributes.position;
    const idx = geo.index ? geo.index.array : Array.from({ length: pos.count }, (_, i) => i);
    const parts = new Map();
    for (let t = 0; t < idx.length; t += 3) {
      let cx = 0, cy = 0;
      for (let j = 0; j < 3; j++) { cx += pos.getX(idx[t + j]); cy += pos.getY(idx[t + j]); }
      const k = pieceOf(s * cx / 3, cy / 3);
      if (!parts.has(k)) parts.set(k, []);
      parts.get(k).push(idx[t], idx[t + 1], idx[t + 2]);
    }
    if (parts.size < 2) continue;
    for (const [k, list] of parts) {
      if (k === 0) { const g = geo.clone(); g.setIndex(list); m.geometry = g; continue; }
      const g = geo.clone(); g.setIndex(list);
      const piece = new THREE.Mesh(g, m.material); piece.castShadow = m.castShadow; piece.receiveShadow = m.receiveShadow;
      piece.userData = m.userData;
      const J = jointOf(k); piece.position.set(-J[0], -J[1], -J[2]);
      // the piece's own geometry is in hand coordinates: undo the chain's offsets
      const h = holder(k); let o = h, acc = [0, 0, 0];
      while (o && o !== grp) { acc[0] += o.position.x; acc[1] += o.position.y; acc[2] += o.position.z; o = o.parent; }
      piece.position.set(-acc[0], -acc[1], -acc[2]);
      h.add(piece);
    }
  }
  ctl.spread = 0; ctl.curl = 0.3;
  return ctl;
}
// spread 0..1 (fingers fanned ~8 deg apart each), curl in radians at each finger joint (0 = flat)
export function poseHand(ctl, spread, curl) {
  if (!ctl) return;
  const s = ctl.s, mid = (ctl.fingers[1].y + ctl.fingers[2].y) / 2;
  ctl.fingers.forEach((f, k) => {
    const [g0, g1, g2] = ctl.piv[k], a = s * (f.y - mid) / 0.025 * spread * 0.14;
    g0.rotation.set(0, s * curl * 0.8, a, 'ZYX'); g1.rotation.set(0, s * curl * 1.1, 0); g2.rotation.set(0, s * curl * 0.9, 0);
  });
  if (ctl.tpiv.length) { const [t0, t1, t2] = ctl.tpiv; t0.rotation.set(0, 0, -s * spread * 0.25); t1.rotation.set(0, s * curl * 0.5, 0); t2.rotation.set(0, s * curl * 0.6, 0); }
}
