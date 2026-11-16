// Fingers for the game's figure (user Nov 16 16:16): the hand in the figure file is already made of separate solid
// pieces (the palm, a ball at every knuckle, and the finger segments between them), so each piece is hung on its own
// joint as it is: nothing is cut. A finger = knuckle ball, segment, ball, segment, ball, segment (the thumb has two of
// each); each segment turns about the centre of the ball before it. The rig has the same shape as sim/fingers.js's,
// so its poseHand works unchanged. A hand that isn't made of such pieces falls back to sim/fingers.js (cutting).
import { rigHand as rigHandCut, poseHand } from '../sim/fingers.js?v=57';
export { poseHand };

// the solid pieces of one mesh: triangles grouped by shared corners (corners at the same place count as one)
function islands(geo) {
  const pos = geo.attributes.position, n = pos.count, key = new Map(), weld = new Int32Array(n);
  for (let i = 0; i < n; i++) { const k = Math.round(pos.getX(i) * 1e5) + ',' + Math.round(pos.getY(i) * 1e5) + ',' + Math.round(pos.getZ(i) * 1e5); let w = key.get(k); if (w === undefined) key.set(k, w = key.size); weld[i] = w; }
  const par = Int32Array.from({ length: key.size }, (_, i) => i), find = x => { while (par[x] !== x) { par[x] = par[par[x]]; x = par[x]; } return x; };
  const idx = geo.index ? geo.index.array : Array.from({ length: n }, (_, i) => i);
  for (let t = 0; t < idx.length; t += 3) { const a = find(weld[idx[t]]); par[find(weld[idx[t + 1]])] = a; par[find(weld[idx[t + 2]])] = a; }
  const byRoot = new Map();
  for (let t = 0; t < idx.length; t += 3) { const r = find(weld[idx[t]]); if (!byRoot.has(r)) byRoot.set(r, { tris: [], verts: new Set() }); const I = byRoot.get(r); I.tris.push(idx[t], idx[t + 1], idx[t + 2]); for (let j = 0; j < 3; j++) I.verts.add(idx[t + j]); }
  return [...byRoot.values()].map(I => {
    let c = [0, 0, 0], lo = [1e9, 1e9, 1e9], hi = [-1e9, -1e9, -1e9];
    for (const v of I.verts) { const p = [pos.getX(v), pos.getY(v), pos.getZ(v)]; for (let k = 0; k < 3; k++) { c[k] += p[k]; lo[k] = Math.min(lo[k], p[k]); hi[k] = Math.max(hi[k], p[k]); } }
    c = c.map(x => x / I.verts.size); const size = hi.map((h, k) => h - lo[k]);
    // a ball: about as wide every way, every corner about the same distance from the centre
    let m = 0, m2 = 0; for (const v of I.verts) { const d = Math.hypot(pos.getX(v) - c[0], pos.getY(v) - c[1], pos.getZ(v) - c[2]); m += d; m2 += d * d; }
    m /= I.verts.size; const sd = Math.sqrt(Math.max(0, m2 / I.verts.size - m * m));
    const ball = Math.max(...size) / Math.max(1e-6, Math.min(...size)) < 1.15 && sd / Math.max(1e-6, m) < 0.06;
    return { tris: I.tris, n: I.verts.size, c, size, ball };
  });
}

export function rigHand(THREE, grp, side) {
  try { const r = rigByPieces(THREE, grp, side); if (r) { console.info('fingers: ' + side + ' hand rigged from its own pieces'); return r; } } catch (e) { console.warn('fingers by pieces', e); }
  console.info('fingers: ' + side + ' hand cut (no separate pieces found)');
  return rigHandCut(THREE, grp, side);
}

function rigByPieces(THREE, grp, side) {
  const meshes = grp.children.filter(o => o.isMesh);
  if (!meshes.length) return null;
  const m = meshes.reduce((a, b) => b.geometry.attributes.position.count > a.geometry.attributes.position.count ? b : a);   // the hand's shell
  const isl = islands(m.geometry); if (isl.length < 20) return null;
  const s = isl.reduce((a, I) => a + I.c[0] * I.n, 0) > 0 ? 1 : -1, U = I => s * I.c[0];   // fingers along +x (left) / -x (right)
  const wrist = isl.filter(I => I.ball && Math.hypot(...I.c) < 0.012);
  const rest = isl.filter(I => !wrist.includes(I));
  const palm = rest.reduce((a, b) => b.n > a.n && !b.ball ? b : a, rest.find(I => !I.ball));
  const parts = rest.filter(I => I !== palm);
  const thumbP = parts.filter(I => I.c[1] < -0.04), finP = parts.filter(I => I.c[1] >= -0.04);
  // the 4 fingers: 1-D k-means on the pieces' sideways place
  const ys = finP.map(I => I.c[1]).sort((a, b) => a - b); if (ys.length < 12) return null;
  let cy = [0.125, 0.375, 0.625, 0.875].map(q => ys[Math.floor(q * (ys.length - 1))]);
  const near = y => { let k = 0; for (let j = 1; j < 4; j++) if (Math.abs(y - cy[j]) < Math.abs(y - cy[k])) k = j; return k; };
  for (let it = 0; it < 20; it++) { const sum = [0, 0, 0, 0], n = [0, 0, 0, 0]; for (const y of ys) { const k = near(y); sum[k] += y; n[k]++; } cy = cy.map((v, k) => n[k] ? sum[k] / n[k] : v); }
  // a finger, in order from the palm: [ball, segment, ball, segment, ...]; its pivots are the balls' centres
  const order = list => list.slice().sort((a, b) => U(a) - U(b));
  const fingersP = [0, 1, 2, 3].map(k => order(finP.filter(I => near(I.c[1]) === k)));
  const thumbOrd = order(thumbP);
  const okChain = (L, nb) => L.length === 2 * nb && L.every((I, i) => I.ball === (i % 2 === 0));
  if (!fingersP.every(L => okChain(L, 3)) || !okChain(thumbOrd, 2)) return null;
  // pieces keep their place: each piece's geometry is in hand coordinates, so it sits at minus its group's offset
  const piece = (I, g) => { const geo = m.geometry.clone(); geo.setIndex(I.tris); const me = new THREE.Mesh(geo, m.material);
    me.castShadow = m.castShadow; me.receiveShadow = m.receiveShadow; me.userData = m.userData;
    let o = g, acc = [0, 0, 0]; while (o && o !== grp) { acc[0] += o.position.x; acc[1] += o.position.y; acc[2] += o.position.z; o = o.parent; }
    me.position.set(-acc[0], -acc[1], -acc[2]); g.add(me); };
  const chain = (L, parent, depth) => {   // L = [ball, seg, ball, seg, ...]; 3 groups as sim/fingers.js (an empty last one for the thumb)
    const gs = []; let par = parent, prev = [0, 0, 0];
    for (let j = 0; j < depth; j++) {
      const b = L[2 * j], c = b ? b.c : prev, g = new THREE.Group();
      g.position.set(c[0] - prev[0], c[1] - prev[1], c[2] - prev[2]); par.add(g); gs.push(g); prev = c; par = g;
      if (b) { piece(b, g); piece(L[2 * j + 1], g); }
    }
    return gs;
  };
  // the palm (and the wrist ball) stay in the hand itself
  const pg = m.geometry.clone(); pg.setIndex([...palm.tris, ...wrist.flatMap(I => I.tris)]); m.geometry = pg;
  const ctl = { s, side, piv: [], tpiv: [], fingers: cy.map(y => ({ y })) };
  fingersP.forEach(L => ctl.piv.push(chain(L, grp, 3)));
  ctl.tpiv = chain(thumbOrd, grp, 3);
  ctl.spread = 0; ctl.curl = 0.3; ctl.byPieces = true;
  return ctl;
}
