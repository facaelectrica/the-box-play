// Offline render: tracked rig states -> the page's own mapping, solver and figure -> one picture per frame.
// node render.mjs <name> <outdir> [travel 0|1]
import puppeteer from 'puppeteer-core'; import fs from 'fs';
const [name, outdir, travel = '1', only = '', zoom = ''] = process.argv.slice(2), pick = only ? new Set(only.split(',').map(Number)) : null; fs.mkdirSync(outdir, { recursive: true });
const b = await puppeteer.launch({ executablePath: '/usr/bin/google-chrome', headless: 'new', args: ['--no-sandbox', '--no-proxy-server', '--use-gl=angle', '--use-angle=swiftshader', '--enable-unsafe-swiftshader', '--window-size=1500,900'] });
const p = await b.newPage(); await p.setViewport({ width: 1500, height: 900 }); const errs = []; p.on('pageerror', e => errs.push(e.message));
await p.goto('http://localhost:8765/gpc/record.html', { waitUntil: 'domcontentloaded' }); await p.waitForFunction('window.__rec'); await new Promise(r => setTimeout(r, 2500));
const n = await p.evaluate(async (name, travel, zoom) => { const get = async f => new Float32Array(await (await fetch('/_ft/' + f)).arrayBuffer()), R = window.__rec;
  window.__T = { st: await get(name + '.states.f32'), feet: await get(name + '.feet.f32'), ppm: await get(name + '.ppm.f32'), map: await (await fetch('mocap/mhr_map.json')).json(), q: [] };
  document.getElementById('mirror').click(); R.sol.opt.travel = travel === '1'; window.__zoom = !!zoom; return window.__T.st.length / 1016; }, name, travel, zoom);
const el = await p.$('#c3'), box = await el.boundingBox(); let log = '';
for (let i = 0; i < n; i++) {
  await p.evaluate(i => { const T = window.__T, R = window.__rec, W = R.bodyRotations(T.map, R.S, T.st.subarray(i * 1016, (i + 1) * 1016)); if (i === 5) R.sol.calibrateRot(W);
    const q = R.sol.solveRot(W, i / 30, [[T.feet[4 * i], T.feet[4 * i + 1]], [T.feet[4 * i + 2], T.feet[4 * i + 3]]], T.ppm[i]); T.q.push(Array.from(q)); R.drawPose(q, 0, R.sol.held); if (window.__zoom) R.view([q[0], q[1] - 2.3, 1.3], [q[0], q[1], 1.25]); R.render(); }, i);
  if (!pick || pick.has(i)) await p.screenshot({ path: `${outdir}/${String(i + 1).padStart(5, '0')}.jpg`, type: 'jpeg', quality: 90, clip: { x: box.x, y: box.y, width: box.width, height: box.height } });
}
fs.writeFileSync(`${outdir}/q.json`, JSON.stringify(await p.evaluate(() => window.__T.q))); console.log(n, 'frames rendered; page errors:', errs.length ? errs.slice(0, 3).join(' | ') : 'none'); await b.close();
