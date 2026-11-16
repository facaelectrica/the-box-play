// The body as plain numbers, read from the game's own body file (body/soma23_box.xml): 23 bodies, each joint three
// hinges turned in the order x, y, z. A pose is the game's 73 numbers: hips position 3, hips quaternion (w x y z) 4,
// 66 hinges. No physics engine is needed here; test.mjs checks this file against MuJoCo's own numbers.
export const M = {   // 3x3 matrices as 9 numbers, row by row
  I: () => [1, 0, 0, 0, 1, 0, 0, 0, 1],
  mul: (a, b) => { const o = new Array(9); for (let r = 0; r < 3; r++) for (let c = 0; c < 3; c++) o[3 * r + c] = a[3 * r] * b[c] + a[3 * r + 1] * b[3 + c] + a[3 * r + 2] * b[6 + c]; return o; },
  T: a => [a[0], a[3], a[6], a[1], a[4], a[7], a[2], a[5], a[8]],
  vec: (a, v) => [a[0] * v[0] + a[1] * v[1] + a[2] * v[2], a[3] * v[0] + a[4] * v[1] + a[5] * v[2], a[6] * v[0] + a[7] * v[1] + a[8] * v[2]],
  cols: (x, y, z) => [x[0], y[0], z[0], x[1], y[1], z[1], x[2], y[2], z[2]],
  // hinges x, y, z -> rotation Rx * Ry * Rz, and back (the answer nearest to `near`, so angles do not jump)
  euler: (a, b, c) => { const ca = Math.cos(a), sa = Math.sin(a), cb = Math.cos(b), sb = Math.sin(b), cc = Math.cos(c), sc = Math.sin(c);
    return [cb * cc, -cb * sc, sb, sa * sb * cc + ca * sc, -sa * sb * sc + ca * cc, -sa * cb, -ca * sb * cc + sa * sc, ca * sb * sc + sa * cc, ca * cb]; },
  toEuler: (R, near) => { const b = Math.asin(Math.max(-1, Math.min(1, R[2]))), a = Math.atan2(-R[5], R[8]), c = Math.atan2(-R[1], R[0]);
    if (!near) return [a, b, c];
    const wr = (v, n) => v + 2 * Math.PI * Math.round((n - v) / (2 * Math.PI)), s1 = [wr(a, near[0]), wr(b, near[1]), wr(c, near[2])], s2 = [wr(a + Math.PI, near[0]), wr(Math.PI - b, near[1]), wr(c + Math.PI, near[2])];
    const d = s => Math.abs(s[0] - near[0]) + Math.abs(s[1] - near[1]) + Math.abs(s[2] - near[2]), o = d(s2) < d(s1) ? s2 : s1;
    for (let k = 0; k < 3; k++) { if (o[k] > Math.PI) o[k] -= 2 * Math.PI; else if (o[k] < -Math.PI) o[k] += 2 * Math.PI; }
    return o; },
  fromQuat: (w, x, y, z) => { const n = Math.hypot(w, x, y, z) || 1; w /= n; x /= n; y /= n; z /= n;
    return [1 - 2 * (y * y + z * z), 2 * (x * y - w * z), 2 * (x * z + w * y), 2 * (x * y + w * z), 1 - 2 * (x * x + z * z), 2 * (y * z - w * x), 2 * (x * z - w * y), 2 * (y * z + w * x), 1 - 2 * (x * x + y * y)]; },
  toQuat: R => { const t = R[0] + R[4] + R[8]; let w, x, y, z;
    if (t > 0) { const s = Math.sqrt(t + 1) * 2; w = s / 4; x = (R[7] - R[5]) / s; y = (R[2] - R[6]) / s; z = (R[3] - R[1]) / s; }
    else if (R[0] > R[4] && R[0] > R[8]) { const s = Math.sqrt(1 + R[0] - R[4] - R[8]) * 2; w = (R[7] - R[5]) / s; x = s / 4; y = (R[1] + R[3]) / s; z = (R[2] + R[6]) / s; }
    else if (R[4] > R[8]) { const s = Math.sqrt(1 + R[4] - R[0] - R[8]) * 2; w = (R[2] - R[6]) / s; x = (R[1] + R[3]) / s; y = s / 4; z = (R[5] + R[7]) / s; }
    else { const s = Math.sqrt(1 + R[8] - R[0] - R[4]) * 2; w = (R[3] - R[1]) / s; x = (R[2] + R[6]) / s; y = (R[5] + R[7]) / s; z = s / 4; }
    return [w, x, y, z]; },
  // a part of a rotation (k = 0..1), by its axis and angle
  part: (R, k) => { const [w, x, y, z] = M.toQuat(R), s = Math.hypot(x, y, z); if (s < 1e-9) return M.I();
    const sg = w < 0 ? -1 : 1, ang = 2 * Math.atan2(s, Math.abs(w)) * k, h = Math.sin(ang / 2) / s * sg; return M.fromQuat(Math.cos(ang / 2), x * h, y * h, z * h); },
};
export const V = {
  sub: (a, b) => [a[0] - b[0], a[1] - b[1], a[2] - b[2]], add: (a, b) => [a[0] + b[0], a[1] + b[1], a[2] + b[2]], mul: (a, k) => [a[0] * k, a[1] * k, a[2] * k],
  dot: (a, b) => a[0] * b[0] + a[1] * b[1] + a[2] * b[2], cross: (a, b) => [a[1] * b[2] - a[2] * b[1], a[2] * b[0] - a[0] * b[2], a[0] * b[1] - a[1] * b[0]],
  len: a => Math.hypot(a[0], a[1], a[2]), norm: a => { const l = Math.hypot(a[0], a[1], a[2]) || 1; return [a[0] / l, a[1] / l, a[2] / l]; },
  perp: (a, n) => { const d = a[0] * n[0] + a[1] * n[1] + a[2] * n[2]; return [a[0] - n[0] * d, a[1] - n[1] * d, a[2] - n[2] * d]; },   // a without its part along unit n
  mid: (a, b) => [(a[0] + b[0]) / 2, (a[1] + b[1]) / 2, (a[2] + b[2]) / 2],
};

// the body file's text -> { names, parent, off, id }: bodies in the file's order, which is also the order of the hinges
export function readBody(xml) {
  const names = [], parent = [], off = [], stack = [], re = /<body\s+name='([^']+)'\s+pos='([^']+)'|<\/body>/g; let k;
  while ((k = re.exec(xml))) { if (k[1] === undefined) { stack.pop(); continue; }
    names.push(k[1]); parent.push(stack.length ? stack[stack.length - 1] : -1); off.push(k[2].trim().split(/\s+/).map(Number)); stack.push(names.length - 1); }
  const id = {}; names.forEach((n, i) => { id[n] = i; });
  if (names.length !== 23 || names[0] !== 'Hips') throw new Error('unexpected body file');
  return { names, parent, off, id, nq: 7 + 3 * (names.length - 1) };
}
// pose (73 numbers at q[o...]) -> positions and rotations of every body
export function fk(S, q, o = 0) {
  const n = S.names.length, p = new Array(n), R = new Array(n);
  p[0] = [q[o], q[o + 1], q[o + 2]]; R[0] = M.fromQuat(q[o + 3], q[o + 4], q[o + 5], q[o + 6]);
  for (let b = 1; b < n; b++) { const a = S.parent[b], h = o + 7 + 3 * (b - 1);
    p[b] = V.add(p[a], M.vec(R[a], S.off[b])); R[b] = M.mul(R[a], M.euler(q[h], q[h + 1], q[h + 2])); }
  return { p, R };
}
// the game's standing pose (sim/brain.js BRAIN_STAND): arms down from the T pose
export function standPose(S) { const q = new Float64Array(S.nq); q[3] = 1; const set = (n, k, v) => { q[7 + 3 * (S.id[n] - 1) + k] = v; };
  set('RightArm', 1, -1.25); set('LeftArm', 1, 1.25); set('RightArm', 2, -0.2); set('LeftArm', 2, 0.2); return q; }
