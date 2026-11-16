// The reverse of the solver, for tests and for the Record page's "Test with a library move": a pose -> the 33 points
// the camera tracker would report for it (MediaPipe's axes, centred on the hips).
import { M, V, fk } from './skel.js?v=1';
import { LM } from './solve.js?v=7';
export function fakePoints(S, q, o = 0) {
  const K = fk(S, q, o), id = S.id, at = (n, v) => V.add(K.p[id[n]], M.vec(K.R[id[n]], v || [0, 0, 0])), G = new Array(33).fill(null);
  G[LM.nose] = at('Head', [0, -0.10, 0.09]); G[LM.lEar] = at('Head', [0.075, 0, 0.09]); G[LM.rEar] = at('Head', [-0.075, 0, 0.09]);
  for (const [s, p, side] of [['Right', 'r', -1], ['Left', 'l', 1]]) {
    G[LM[p + 'Sh']] = at(s + 'Arm'); G[LM[p + 'El']] = at(s + 'ForeArm'); G[LM[p + 'Wr']] = at(s + 'Hand');
    G[LM[p + 'Ix']] = at(s + 'Hand', [side * 0.09, -0.03, 0]); G[LM[p + 'Pk']] = at(s + 'Hand', [side * 0.08, 0.03, 0]);
    G[LM[p + 'Hip']] = at(s + 'Leg'); G[LM[p + 'Kn']] = at(s + 'Shin'); G[LM[p + 'An']] = at(s + 'Foot');
    G[LM[p + 'He']] = at(s + 'Foot', [0, 0.05, -0.012]); G[LM[p + 'To']] = at(s + 'Foot', [0, -0.139, -0.105]); }   // heel to toe tilts 26 degrees down on a flat foot, as the tracker reports it
  const c = V.mid(G[LM.lHip], G[LM.rHip]);
  return G.map(g => { const p = g ? V.sub(g, c) : [0, 0, 0]; return { x: p[0], y: -p[2], z: p[1], visibility: g ? 1 : 0 }; });
}
