// Checks for the camera recording maths (run: node gpc/mocap/test.mjs [clip ids...]). No camera is involved:
// library moves are turned into the points a tracker would report, solved back into poses, and compared.
import fs from 'fs'; import path from 'path'; import { fileURLToPath } from 'url';
import { readBody, fk, M } from './skel.js'; import { makeSolver, toMove } from './solve.js'; import { fakePoints } from './synth.js';
import { encodeQ } from './encq.js'; import { decodeQ } from '../libq.js';
const here = path.dirname(fileURLToPath(import.meta.url)), S = readBody(fs.readFileSync(path.join(here, '../../body/soma23_box.xml'), 'utf8'));
const clipDir = path.join(here, '../clips'), all = fs.readdirSync(clipDir).filter(f => f.endsWith('.q'));
const di = process.argv.indexOf('--dump'), args = process.argv.slice(2).filter((a, i) => !a.startsWith('--') && i + 2 !== di + 1), pick = args.length ? args.map(a => a + '.q') : all.filter((_, i) => i % 25 === 0);
const load = async f => { const b = fs.readFileSync(path.join(clipDir, f)); return decodeQ(b.buffer.slice(b.byteOffset, b.byteOffset + b.byteLength)); };
if (di > 0) {   // poses and this file's body positions, for the MuJoCo comparison (test_fk.py)
  const d = await load(pick[0]), fr = [0, 10, 50, d.T - 1].map(t => ({ q: Array.from(d.q.slice(t * 73, t * 73 + 73)), p: fk(S, d.q, t * 73).p }));
  fs.writeFileSync(process.argv[di + 1], JSON.stringify({ names: S.names, fr })); process.exit(0); }
const LIMBS = ['RightForeArm', 'RightHand', 'LeftForeArm', 'LeftHand', 'RightShin', 'RightFoot', 'LeftShin', 'LeftFoot', 'Head'];
let trav = 0, travLen = 0, zErr = 0; let n = 0, sum = 0, worst = 0, worstAt = '', sumL = 0, nL = 0, jump = 0, encWorst = 0, wraps = 0, big = 0; const per = {};
for (const f of pick) {
  const d = await load(f), sol = makeSolver(S, { smooth: 0 }), frames = []; let prev = null;
  for (let t = 0; t < d.T; t++) {
    const o = t * 73, out = sol.solve(fakePoints(S, d.q, o), t / 30); frames.push({ t: t / 30, q: out });
    const A = fk(S, d.q, o), B = fk(S, out), ca = [0, 1, 2].map(k => (A.p[S.id.RightLeg][k] + A.p[S.id.LeftLeg][k]) / 2), cb = [0, 1, 2].map(k => (B.p[S.id.RightLeg][k] + B.p[S.id.LeftLeg][k]) / 2);
    for (let b = 1; b < 23; b++) { const e = Math.hypot(...[0, 1, 2].map(k => (A.p[b][k] - ca[k]) - (B.p[b][k] - cb[k]))) * 100;
      n++; sum += e; per[S.names[b]] = Math.max(per[S.names[b]] || 0, e); if (e > worst) { worst = e; worstAt = f + ' frame ' + t + ' ' + S.names[b]; }
      if (LIMBS.includes(S.names[b])) { sumL += e; nL++; } }
    { const e = Math.hypot((out[0] - frames[0].q[0]) - (d.q[o] - d.q[0]), (out[1] - frames[0].q[1]) - (d.q[o + 1] - d.q[1])) * 100; if (e > trav) { trav = e; travLen = Math.hypot(d.q[o] - d.q[0], d.q[o + 1] - d.q[1]) * 100; } zErr = Math.max(zErr, Math.abs(out[2] - d.q[o + 2]) * 100); }
    if (prev) for (let c = 7; c < 73; c++) { let dd = Math.abs(out[c] - prev[c]); if (dd > Math.PI) { dd = Math.abs(dd - 2 * Math.PI); wraps++; } jump = Math.max(jump, dd); }; prev = out;
  }
  const mv = toMove(frames, 30, 0); for (let i = 0; i < mv.T; i++) { const A = fk(S, frames[i].q), B = fk(S, mv.q, i * 73); for (let b = 0; b < 23; b++) big = Math.max(big, Math.hypot(...[0, 1, 2].map(k => A.p[b][k] - B.p[b][k])) * 100); }
  const back = await decodeQ(await encodeQ(mv.q, mv.T));
  if (back.T !== mv.T) throw new Error('frame count changed'); for (let i = 0; i < mv.T * 73; i++) encWorst = Math.max(encWorst, Math.abs(back.q[i] - mv.q[i]) * (i % 73 < 3 ? 100 : i % 73 < 7 ? 1 : 180 / Math.PI));
}
console.log(`${pick.length} moves, ${n / 22} frames`);
console.log(`joint position after the round trip (centred between the hip joints): mean ${(sum / n).toFixed(2)} cm, elbows/wrists/knees/ankles/head mean ${(sumL / nL).toFixed(2)} cm, worst ${worst.toFixed(1)} cm (${worstAt})`);
console.log('worst per joint (cm): ' + Object.entries(per).map(([k, v]) => k + ' ' + v.toFixed(1)).join(', '));
console.log(`hips travel across the floor, against the library move: worst ${trav.toFixed(1)} cm off (at a point ${travLen.toFixed(0)} cm from the start); hips height worst ${zErr.toFixed(1)} cm off`);
console.log(`largest one-frame hinge change: ${(jump * 180 / Math.PI).toFixed(0)} deg; passes through +-180: ${wraps}`);
console.log(`finished move against the solver's own frames: worst ${big.toFixed(3)} cm`);
console.log(`.q written and read back: worst difference ${encWorst.toFixed(4)} (cm for position, deg for hinges)`);
