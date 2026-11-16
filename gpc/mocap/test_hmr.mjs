// Checks the rig -> body mapping in JS against the Python reference (tools/mocap/rt.py): node gpc/mocap/test_hmr.mjs
import fs from 'fs'; import path from 'path'; import { fileURLToPath } from 'url';
import { readBody, M } from './skel.js'; import { bodyRotations } from './hmr.js'; import { makeSolver } from './solve.js';
const here = path.dirname(fileURLToPath(import.meta.url)), S = readBody(fs.readFileSync(path.join(here, '../../body/soma23_box.xml'), 'utf8'));
const map = JSON.parse(fs.readFileSync(path.join(here, 'mhr_map.json'))), cases = JSON.parse(fs.readFileSync(path.join(here, '../../tools/mocap/mhr_cases.json')));
let wRot = 0, wHinge = 0;
for (const c of cases) { const W = bodyRotations(map, S, c.state);
  for (const n in c.W) for (let k = 0; k < 9; k++) wRot = Math.max(wRot, Math.abs(W[S.id[n]][k] - c.W[n][k]));
  // the solver, with holding off, must give the same pose as the reference (same rotations; hinge angles may differ by a full turn)
  const sol = makeSolver(S, { travel: false }), q = sol.solveRot(W, 0);
  for (let b = 1; b < 23; b++) { const h = 7 + 3 * (b - 1), A = M.euler(q[h], q[h + 1], q[h + 2]), B = M.euler(c.q[h], c.q[h + 1], c.q[h + 2]), R = M.mul(M.T(A), B); wHinge = Math.max(wHinge, Math.acos(Math.max(-1, Math.min(1, (R[0] + R[4] + R[8] - 1) / 2))) * 180 / Math.PI); } }
console.log(`${cases.length} reference poses: body rotations differ by at most ${wRot.toExponential(1)}; joint rotations of the solved pose differ by at most ${wHinge.toFixed(4)} deg`);
