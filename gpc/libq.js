// Library moves (.q files, see the-box-library README): decode, and expand into the game's recording layout
// (376 floats a frame, the same as tools/kin/npy2bin.py makes: 23 bodies x [pos 3, quat xyzw 4, vel 3, angvel 3],
// qpos 73, contacts 4), using the game's own physics model for the body positions.

// .q bytes (gzip) -> { T, q: Float64Array(T * 73) }: root position, root quaternion (w x y z), 66 hinges, 30 fps
export async function decodeQ(ab) {
  const raw = await new Response(new Blob([ab]).stream().pipeThrough(new DecompressionStream('gzip'))).arrayBuffer();
  const v = new DataView(raw);
  if (v.getUint32(0, true) !== 0x00315142) throw new Error('not a library move');
  const T = v.getUint32(4, true), C = v.getUint16(8, true);
  const sc = new Float32Array(raw.slice(12, 12 + 4 * C)), md = new Uint8Array(raw, 12 + 4 * C, C);
  let o = 12 + 5 * C; o += (4 - (o % 4)) % 4;
  const A = new Int16Array(raw, o, T * C), q = new Float64Array(T * C);
  for (let c = 0; c < C; c++) {
    let s = 0; const k = 1 / sc[c];
    for (let t = 0; t < T; t++) { const x = A[c * T + t]; s = md[c] ? x : s + x; q[t * C + c] = s * k; }
  }
  return { T, C, q };
}

// lowest point of one geom (sphere, capsule, box), as the converter and the director measure it
function geomBottom(m, d, g) {
  const t = m.geom_type[g], s0 = m.geom_size[3 * g], s1 = m.geom_size[3 * g + 1], s2 = m.geom_size[3 * g + 2];
  const r6 = d.geom_xmat[9 * g + 6], r7 = d.geom_xmat[9 * g + 7], r8 = d.geom_xmat[9 * g + 8];
  const dz = t === 6 ? Math.abs(r6) * s0 + Math.abs(r7) * s1 + Math.abs(r8) * s2 : t === 3 ? Math.abs(r8) * s1 + s0 : s0;
  return d.geom_xpos[3 * g + 2] - dz;
}

// q (decoded) -> Float32Array(T * 376). ids = the 23 body ids in the game's body order, names = their names.
// The floor goes to z = 0 at the 2nd percentile of the body's lowest point, and the lowest point is glued to the floor
// (smoothed over 0.3 s), exactly as npy2bin.py does.
export function expandQ(mj, m, dec, ids, names) {
  const { T, C, q } = dec, F = 376, NB = ids.length, d = new mj.MjData(m), out = new Float32Array(T * F);
  const of = new Map(ids.map((id, b) => [id, b])), geoms = [], part = [[], [], [], []];
  const P4 = [['LeftFoot', 'LeftToeBase'], ['RightFoot', 'RightToeBase'], ['LeftHand'], ['RightHand']];
  for (let g = 0; g < m.ngeom; g++) { const b = of.get(m.geom_bodyid[g]); if (b === undefined) continue; geoms.push(g);
    for (let k = 0; k < 4; k++) if (P4[k].includes(names[b])) part[k].push(g); }
  const pose = (t, dz) => { for (let c = 0; c < C; c++) d.qpos[c] = q[t * C + c]; d.qpos[2] += dz; mj.mj_kinematics(m, d); };
  const low = new Float64Array(T);
  for (let t = 0; t < T; t++) { pose(t, 0); let lo = 1e9; for (const g of geoms) lo = Math.min(lo, geomBottom(m, d, g)); low[t] = lo; }
  const srt = Array.from(low).sort((a, b) => a - b), pi = 0.02 * (T - 1), i0 = Math.floor(pi);
  const floor = srt[i0] + (srt[Math.min(T - 1, i0 + 1)] - srt[i0]) * (pi - i0);
  const k = 9, h = 4, sh = new Float64Array(T);
  for (let t = 0; t < T; t++) { let s = 0; for (let j = -h; j <= h; j++) s += -(low[Math.min(T - 1, Math.max(0, t + j))] - floor); sh[t] = s / k; }
  const P = new Float64Array(T * NB * 3);
  for (let t = 0; t < T; t++) {
    const dz = -floor + sh[t]; pose(t, dz); const o = t * F;
    for (let b = 0; b < NB; b++) { const id = ids[b], a = o + 13 * b;
      for (let j = 0; j < 3; j++) { out[a + j] = d.xpos[3 * id + j]; P[(t * NB + b) * 3 + j] = d.xpos[3 * id + j]; }
      out[a + 3] = d.xquat[4 * id + 1]; out[a + 4] = d.xquat[4 * id + 2]; out[a + 5] = d.xquat[4 * id + 3]; out[a + 6] = d.xquat[4 * id]; }
    for (let c = 0; c < C; c++) out[o + 299 + c] = q[t * C + c] + (c === 2 ? dz : 0);
    for (let j = 0; j < 4; j++) { let lo = 1e9; for (const g of part[j]) lo = Math.min(lo, geomBottom(m, d, g)); out[o + 372 + j] = lo < 0.03 ? 1 : 0; }
  }
  // velocities: central differences (one-sided at the ends), as numpy.gradient; angular velocities stay 0 (as npy2bin)
  for (let t = 0; t < T; t++) { const a = Math.max(0, t - 1), b = Math.min(T - 1, t + 1), s = b > a ? 30 / (b - a) : 0;
    for (let n = 0; n < NB; n++) for (let j = 0; j < 3; j++) out[t * F + 13 * n + 7 + j] = (P[(b * NB + n) * 3 + j] - P[(a * NB + n) * 3 + j]) * s; }
  if (d.delete) d.delete();
  return out;
}

// where a recording would put his body if it started now: every 3rd frame of every body, anchored like the brain's
// clipRef (the first frame's hips on his hips, its facing turned to his), in the box's own frame. Returns how far the
// body would go through the walls, floor or ceiling (m, the worst point; 2 cm is allowed), and the share of points outside.
export function overflow(buf, n, me, box, ground) {
  const F = 376, NB = 23, X = outside(me, box, ground);
  for (let f = 0; f < n; f += 3) for (let b = 0; b < NB; b++) { const a = f * F + 13 * b; X.add(buf[a], buf[a + 1], buf[a + 2]); }
  return X.done();
}
// the same, from a move's index entry (only the points on the outside of all those points: the worst point is always
// one of them, so the result is the same number)
export function overflowIx(E, me, box, ground) {
  const X = outside(me, box, ground);
  for (let i = IX_PTS; i < E.length; i += 3) X.add(E[i], E[i + 1], E[i + 2]);
  return X.done();
}
function outside(me, box, ground) {
  const { G, xy, yaw, h0, mid } = me, c = Math.cos(yaw - h0), s = Math.sin(yaw - h0);
  const Gi = [-G[0], -G[1], -G[2], G[3]], t = [0, 0, 0];
  let worst = 0, out = 0, all = 0;
  return {
    add(px, py, pz) {
      const dx = px - mid[0], dy = py - mid[1];
      const x = xy[0] + c * dx - s * dy, y = xy[1] + s * dx + c * dy, z = ground + pz;
      qrot(Gi, x, y, z, t);
      const e = Math.max(Math.abs(t[0]) - box.w / 2, -t[1], t[1] - box.D, t[2] - box.h / 2, 0);
      all++; if (e > 0.02) out++; if (e > worst) worst = e;
    },
    done: () => ({ worst, share: all ? out / all : 0 }) };
}

// A move's index entry (gpc/clips/index.json, version 2): what the director reads from a move to pick and place one,
// without downloading and expanding the move. Float32Array: [first-frame hips x, y; frame count; first-frame hips
// quaternion x y z w; first-frame qpos (73); first-frame x y of the 23 bodies (46); then the points (x y z) of every 3rd
// frame of every body that lie on the outside of all those points (their convex hull, plus any point within 0.1 mm of
// it)]. Values are copied, not recomputed, so they are exactly the ones the full move gives.
export const IX_VERSION = 2, IX_Q0 = 7, IX_B0 = 80, IX_PTS = 126;
export async function indexEntry(buf, n) {
  const { ConvexHull } = await import('three/addons/math/ConvexHull.js'), { Vector3 } = await import('three');
  const F = 376, NB = 23, P = [];
  for (let f = 0; f < n; f += 3) for (let b = 0; b < NB; b++) { const a = f * F + 13 * b; P.push([buf[a], buf[a + 1], buf[a + 2]]); }
  let keep = P;
  try {
    const H = new ConvexHull().setFromPoints(P.map(p => new Vector3(p[0], p[1], p[2])));
    const planes = H.faces.map(f => [f.normal.x, f.normal.y, f.normal.z, f.constant]);
    if (planes.length >= 4) keep = P.filter(p => planes.some(([x, y, z, k]) => x * p[0] + y * p[1] + z * p[2] - k > -1e-4));
  } catch (_) {}
  const E = new Float32Array(IX_PTS + 3 * keep.length);
  E[0] = buf[0]; E[1] = buf[1]; E[2] = n; for (let j = 0; j < 4; j++) E[3 + j] = buf[3 + j];
  for (let j = 0; j < 73; j++) E[IX_Q0 + j] = buf[299 + j];
  for (let b = 0; b < 23; b++) { E[IX_B0 + 2 * b] = buf[13 * b]; E[IX_B0 + 2 * b + 1] = buf[13 * b + 1]; }
  keep.forEach((p, i) => { E[IX_PTS + 3 * i] = p[0]; E[IX_PTS + 3 * i + 1] = p[1]; E[IX_PTS + 3 * i + 2] = p[2]; });
  return E;
}
export const ixPack = E => { const u = new Uint8Array(E.buffer, E.byteOffset, E.byteLength); let t = ''; for (let i = 0; i < u.length; i += 0x8000) t += String.fromCharCode.apply(null, u.subarray(i, i + 0x8000)); return btoa(t); };
export const ixUnpack = s => { const t = atob(s), u = new Uint8Array(t.length); for (let i = 0; i < t.length; i++) u[i] = t.charCodeAt(i); return new Float32Array(u.buffer); };
function qrot(q, x, y, z, o) {
  const [qx, qy, qz, qw] = q, tx = 2 * (qy * z - qz * y), ty = 2 * (qz * x - qx * z), tz = 2 * (qx * y - qy * x);
  o[0] = x + qw * tx + qy * tz - qz * ty; o[1] = y + qw * ty + qz * tx - qx * tz; o[2] = z + qw * tz + qx * ty - qy * tx;
}
