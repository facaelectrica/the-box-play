// analytic two-bone IK (shared by the editor and this test): set the upper bone's three hinges and the lower bone's
// bend so the end reaches T; the elbow/knee bends toward the pole. env: { THREE, m, W, pose, fkRaw, clampA, acts }
export function makeTwoBone(env) {
  const { THREE, m, W } = env, V = (a) => new THREE.Vector3(...a);
  const restOff = n => { const b = W.bodyId[n]; return V([m.body_pos[3 * b], m.body_pos[3 * b + 1], m.body_pos[3 * b + 2]]); };
  const parentName = n => { const p = m.body_parentid[W.bodyId[n]]; return W.bodyNames.find(k => W.bodyId[k] === p) || null; };
  const act = n => env.acts.find(a => a.name === n);
  return function twoBone(LIMB, end, Tarr) {
    const [up, lo, en, natural, flex] = LIMB[end], par = parentName(up), pose = env.pose();
    const R = env.fkRaw(pose, [par, up, lo, en]);
    const S = V(R[up].p), E0 = V(R[lo].p), H = V(R[en].p), T = V(Tarr);
    const r1 = restOff(lo), r2 = restOff(en), L1 = r1.length(), L2 = r2.length();
    const D = T.clone().sub(S), dl = D.length() || 1e-4, d = Math.max(Math.abs(L1 - L2) + 1e-3, Math.min(L1 + L2 - 1e-3, dl)), u = D.clone().divideScalar(dl);
    const proj = a => a.clone().sub(u.clone().multiplyScalar(a.dot(u)));
    // candidate bend directions: the way it already bends, its natural way, and turns of those around the reach line;
    // each with both twists of the upper bone. Pick the one that breaks human limits least, then the closest to now.
    const cands = [], base = [proj(E0.clone().sub(S)), proj(V(natural))].filter(v => v.length() > 1e-4).map(v => v.normalize());
    if (!base.length) base.push(proj(V([1, 0, 0])).normalize());
    for (const b0 of base) for (let k = 0; k < 8; k++) cands.push(b0.clone().applyAxisAngle(u, k * Math.PI / 4));
    const ca = (L1 * L1 + d * d - L2 * L2) / (2 * L1 * d), sa = Math.sqrt(Math.max(0, 1 - ca * ca)), Tc = S.clone().addScaledVector(u, d);
    const f = V(flex === 'x' ? [1, 0, 0] : flex === 'y' ? [0, 1, 0] : [0, 0, 1]), r1n = r1.clone().normalize(), r2n = r2.clone().normalize();
    const basis = (x, y) => { const X = x.clone().normalize(), Y = y.clone().sub(X.clone().multiplyScalar(y.dot(X))).normalize(), Z = X.clone().cross(Y); return new THREE.Matrix4().makeBasis(X, Y, Z); };
    const Rr = basis(r1n, f), qPar = R[par] ? R[par].q.clone() : new THREE.Quaternion(), qPi = qPar.clone().invert();
    const limOf = nm => { const A = act(nm); return A && m.jnt_limited[A.j] ? [m.jnt_range[2 * A.j], m.jnt_range[2 * A.j + 1]] : [-Math.PI, Math.PI]; };
    const over = (nm, v) => { const [l, h] = limOf(nm); return Math.max(0, l - v, v - h); };
    let best = null;
    for (const p of cands) {
      const E = S.clone().addScaledVector(u, L1 * ca).addScaledVector(p, L1 * sa), a = E.clone().sub(S).normalize(), b = Tc.clone().sub(E).normalize();
      let n = a.clone().cross(b); if (n.length() < 1e-4) n = a.clone().cross(p); n.normalize();
      for (const s of [1, -1]) {
        const qU = new THREE.Quaternion().setFromRotationMatrix(basis(a, n.clone().multiplyScalar(s)).multiply(Rr.clone().transpose()));
        const bU = b.clone().applyQuaternion(qU.clone().invert()), th = Math.atan2(r2n.clone().cross(bU).dot(f), r2n.dot(bU));
        const e = new THREE.Euler().setFromQuaternion(qPi.clone().multiply(qU), 'XYZ'), ang = { x: e.x, y: e.y, z: e.z };
        const bad = over(lo + '_' + flex, th) + ['x', 'y', 'z'].reduce((t, k) => t + over(up + '_' + k, ang[k]), 0);
        const dist = ['x', 'y', 'z'].reduce((t, k) => t + Math.abs(ang[k] - (pose[up + '_' + k] || 0)), 0) + Math.abs(th - (pose[lo + '_' + flex] || 0));
        if (!best || bad < best.bad - 1e-4 || (Math.abs(bad - best.bad) <= 1e-4 && dist < best.dist)) best = { ang, th, bad, dist };
      }
    }
    for (const k of ['x', 'y', 'z']) { const A = act(up + '_' + k); if (A) pose[up + '_' + k] = env.clampA(A, best.ang[k]); }
    for (const k of ['x', 'y', 'z']) { const A = act(lo + '_' + k); if (A) pose[lo + '_' + k] = env.clampA(A, k === flex ? best.th : 0); }
    return best.bad;
  };
}
