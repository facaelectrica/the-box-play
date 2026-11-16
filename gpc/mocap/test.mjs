// Checks for the camera recording maths (run: node gpc/mocap/test.mjs [clip ids...]). No camera is involved:
// library moves are turned into the points a tracker would report, solved back into poses, and compared.
import fs from 'fs'; import path from 'path'; import { fileURLToPath } from 'url';
import { readBody, fk, M } from './skel.js'; import { makeSolver, toMove } from './solve.js'; import { fakePoints } from './synth.js';
import { encodeQ } from './encq.js'; import { decodeQ } from '../libq.js';
const here = path.dirname(fileURLToPath(import.meta.url)), S = readBody(fs.readFileSync(path.join(here, '../../body/soma23_box.xml'), 'utf8'));
const clipDir = path.join(here, '../clips'), all = fs.readdirSync(clipDir).filter(f => f.endsWith('.q'));
const di = process.argv.indexOf('--dump'), args = process.argv.slice(2).filter((a, i) => !a.startsWith('--') && (di < 0 || i + 2 !== di + 1)), pick = args.length ? args.map(a => a + '.q') : all.filter((_, i) => i % 25 === 0);
const load = async f => { const b = fs.readFileSync(path.join(clipDir, f)); return decodeQ(b.buffer.slice(b.byteOffset, b.byteOffset + b.byteLength)); };
if (di > 0) {   // poses and this file's body positions, for the MuJoCo comparison (test_fk.py)
  const d = await load(pick[0]), fr = [0, 10, 50, d.T - 1].map(t => ({ q: Array.from(d.q.slice(t * 73, t * 73 + 73)), p: fk(S, d.q, t * 73).p }));
  fs.writeFileSync(process.argv[di + 1], JSON.stringify({ names: S.names, fr })); process.exit(0); }
const LIMBS = ['RightForeArm', 'RightHand', 'LeftForeArm', 'LeftHand', 'RightShin', 'RightFoot', 'LeftShin', 'LeftFoot', 'Head'];
const TW = ['RightFoot', 'LeftFoot', 'RightLeg', 'RightShin', 'LeftLeg', 'LeftShin', 'RightArm', 'RightForeArm', 'LeftArm', 'LeftForeArm'], rot = {}; const usePic = !process.argv.includes('--nopic'); let onN = 0, missed = 0, offN = 0, falseHold = 0, slide = 0, heldN = 0, trav = 0, travLen = 0, zErr = 0; let n = 0, sum = 0, worst = 0, worstAt = '', sumL = 0, nL = 0, jump = 0, encWorst = 0, wraps = 0, big = 0; const per = {};
const S2 = JSON.parse(JSON.stringify(S)); for (const n of ['RightShin', 'RightFoot', 'LeftShin', 'LeftFoot']) S2.off[S2.id[n]] = S2.off[S2.id[n]].map(v => v * 0.89); for (const n of ['RightLeg', 'LeftLeg']) S2.off[S2.id[n]][0] *= 1.15;
const other = process.argv.includes('--other'), SP = other ? S2 : S; if (other) console.log('performer with legs 11% shorter and hips 15% wider than his');
for (const f of pick) {
  const d = await load(f), sol = makeSolver(S, { smooth: 0 }), frames = []; let prev = null, pk = null, ph = []; const truePrev = [null, null]; let floorZ = 1e9; for (let tt = 0; tt < d.T; tt++) { const Kt = fk(SP, d.q, tt * 73); for (const nm of ['RightFoot', 'LeftFoot']) floorZ = Math.min(floorZ, Kt.p[S.id[nm]][2] - 0.075); for (const nm of ['RightToeBase', 'LeftToeBase']) floorZ = Math.min(floorZ, Kt.p[S.id[nm]][2] - 0.025); }
  for (let t = 0; t < d.T; t++) {
    const o = t * 73, fp = fakePoints(SP, d.q, o), Kp = fk(SP, d.q, o), mid = [0, 1, 2].map(k => (Kp.p[S.id.RightLeg][k] + Kp.p[S.id.LeftLeg][k]) / 2);
    const img = usePic ? fp.map(p => ({ x: 0.5 + (p.x + mid[0]) / 4, y: 0.5 + (p.y - mid[2]) / 4 + (p.z + mid[1]) / 12 })) : null, out = sol.solve(fp, t / 30, img, 1);   // a camera in front of him and a little above: depth shows as height in the picture
    { const c = sol.contacts; [['RightFoot', 'RightToeBase'], ['LeftFoot', 'LeftToeBase']].forEach(([f, tb], i) => { const zf = Kp.p[S.id[f]][2] - 0.075, zt = Kp.p[S.id[tb]][2] - 0.025, pvt = zt < zf ? Kp.p[S.id[tb]] : Kp.p[S.id[f]], pp = truePrev[i];
      if (pp) { const sp = Math.hypot(pvt[0] - pp[0], pvt[1] - pp[1]) * 30, on = Math.min(zf, zt) - floorZ < 0.04 && sp < 0.25, off = Math.min(zf, zt) - floorZ > 0.08 || sp > 0.6, got = c[i].g > 0.5; if (on) { onN++; if (!got) missed++; } if (off) { offN++; if (got) falseHold++; } } truePrev[i] = pvt; }); } frames.push({ t: t / 30, q: out });
    { const c = sol.contacts; c.forEach((k, i) => { const p = pk && pk[i]; if (k.g === 1 && k.pt && p && p.g === 1 && p.pt && p.pt[2] === k.pt[2]) { slide = Math.max(slide, Math.hypot(k.pt[0] - p.pt[0], k.pt[1] - p.pt[1]) * 100); heldN++; } }); pk = c; }
    const A = fk(S, d.q, o), B = fk(S, out), ca = [0, 1, 2].map(k => (A.p[S.id.RightLeg][k] + A.p[S.id.LeftLeg][k]) / 2), cb = [0, 1, 2].map(k => (B.p[S.id.RightLeg][k] + B.p[S.id.LeftLeg][k]) / 2);
    for (let b = 1; b < 23; b++) { const e = Math.hypot(...[0, 1, 2].map(k => (A.p[b][k] - ca[k]) - (B.p[b][k] - cb[k]))) * 100;
      n++; sum += e; per[S.names[b]] = Math.max(per[S.names[b]] || 0, e); if (e > worst) { worst = e; worstAt = f + ' frame ' + t + ' ' + S.names[b]; }
      if (LIMBS.includes(S.names[b])) { sumL += e; nL++; } }
    { const e = Math.hypot((out[0] - frames[0].q[0]) - (d.q[o] - d.q[0]), (out[1] - frames[0].q[1]) - (d.q[o + 1] - d.q[1])) * 100; if (e > trav) { trav = e; travLen = Math.hypot(d.q[o] - d.q[0], d.q[o + 1] - d.q[1]) * 100; } zErr = Math.max(zErr, Math.abs(out[2] - d.q[o + 2]) * 100); }
    for (const nm of TW) { const b = S.id[nm], R = M.mul(M.T(A.R[b]), B.R[b]), ang = Math.acos(Math.max(-1, Math.min(1, (R[0] + R[4] + R[8] - 1) / 2))) * 180 / Math.PI; (rot[nm] = rot[nm] || []).push(ang); }
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
console.log('bone rotation against the library move, deg (median / 95% / worst): ' + TW.map(nm => { const a = rot[nm].sort((x, y) => x - y); return `${nm} ${a[a.length >> 1].toFixed(0)}/${a[Math.floor(a.length * 0.95)].toFixed(0)}/${a.at(-1).toFixed(0)}`; }).join(', '));
console.log(`foot contact (${usePic ? 'height and motion in the picture' : 'height only'}): planted feet not held ${(100 * missed / onN).toFixed(1)}% of ${onN}; lifted or moving feet held ${(100 * falseHold / offN).toFixed(1)}% of ${offN}`);
console.log(`firmly held feet: ${heldN} foot-frames, worst slide of the held point in one frame ${slide.toFixed(2)} cm`);
console.log(`largest one-frame hinge change: ${(jump * 180 / Math.PI).toFixed(0)} deg; passes through +-180: ${wraps}`);
console.log(`finished move against the solver's own frames: worst ${big.toFixed(3)} cm`);
console.log(`.q written and read back: worst difference ${encWorst.toFixed(4)} (cm for position, deg for hinges)`);
