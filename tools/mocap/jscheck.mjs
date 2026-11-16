// The page's own tracker code (crop, network through onnxruntime-web, rig maths, mapping) on frames of the footage,
// against the offline Python result for the same frames and boxes.
import puppeteer from 'puppeteer-core';
const b = await puppeteer.launch({ executablePath: '/usr/bin/google-chrome', headless: 'new', protocolTimeout: 1200000, args: ['--no-sandbox', '--no-proxy-server', '--use-gl=angle', '--use-angle=swiftshader', '--enable-unsafe-swiftshader'] });
const p = await b.newPage(), errs = []; p.on('pageerror', e => errs.push(e.message)); p.on('console', m => { if (m.type() === 'error' || m.type() === 'warning') errs.push(m.text().slice(0, 200)); });
await p.goto('http://localhost:8765/gpc/record.html', { waitUntil: 'domcontentloaded' }); await p.waitForFunction('window.__rec');
const res = await p.evaluate(async () => { const R = window.__rec, t0 = performance.now(); await R.loadTracker('hmr'); const H = R.st.hmr, loadS = (performance.now() - t0) / 1000, ref = await (await fetch('/_ft/ref_frames.json')).json(), ang = (A, B) => { let tr = 0; for (let r = 0; r < 3; r++) for (let c = 0; c < 3; c++) tr += A[3 * r + c] * B[3 * r + c]; return Math.acos(Math.max(-1, Math.min(1, (tr - 1) / 2))) * 180 / Math.PI; };
  const per = {}, dp = [], ts = [];
  for (const f of ref) { const img = new Image(); img.src = '/_ft/fr/' + String(f.i + 1).padStart(5, '0') + '.jpg'; await img.decode(); img.videoWidth = img.naturalWidth; img.videoHeight = img.naturalHeight;
    const a = performance.now(), r = await H.step(img, f.box); ts.push(performance.now() - a); let m = 0; for (let k = 0; k < 204; k++) m = Math.max(m, Math.abs(r.params[k] - f.params[k])); dp.push(m);
    const W = R.bodyRotations(H.map, R.S, r.state); for (const n in f.W) (per[n] = per[n] || []).push(ang(W[R.S.id[n]], f.W[n])); }
  const med = a => a.slice().sort((x, y) => x - y)[a.length >> 1], out = {}; for (const n in per) out[n] = [+med(per[n]).toFixed(2), +Math.max(...per[n]).toFixed(2)];
  return { gpu: H.gpu, loadS: +loadS.toFixed(1), frames: ref.length, msPerFrame: Math.round(med(ts)), paramDiff: { median: +med(dp).toFixed(4), max: +Math.max(...dp).toFixed(4) }, rotDiffDeg_median_max: out }; });
console.log(JSON.stringify(res)); console.log('page problems:', errs.length ? errs.slice(0, 4).join(' | ') : 'none'); await b.close();
