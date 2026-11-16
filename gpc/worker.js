// NVIDIA's SOMA GPC prior, the big part, in the background: his body (343 numbers) -> 8 picks -> 40 codes.
// Purpose-built engine (engine.js + gemv.wasm), one core. The page's small part turns the latest picks into joint
// commands every frame.
import { makeEngine } from './engine.js?v=10';
const here = new URL('./', import.meta.url).href; let eng = null;
async function get(name) {
  const r = await fetch(here + name), rd = r.body.getReader(), parts = []; let got = 0;
  for (;;) { const { done, value } = await rd.read(); if (done) break; parts.push(value); got += value.length; postMessage({ load: name, got }); }
  const b = new Uint8Array(got); let o = 0; for (const p of parts) { b.set(p, o); o += p.length; } return b;
}
onmessage = async e => {
  const m = e.data;
  try {
    if (m.init) {
      const wasm = await get('gemv.wasm'), manifest = JSON.parse(new TextDecoder().decode(await get('live/manifest.json')));
      const layers = await get('live/layers.bin'), head = await get('live/head.bin'), f32 = await get('live/f32.bin'), emb = await get('live/emb_q8.bin'), embScale = await get('live/emb_scale.bin');
      postMessage({ stage: 'building' });
      let wasmMt = null; if (m.threads > 1) try { wasmMt = await get('gemv_mt.wasm'); } catch (_) {}
      eng = await makeEngine({ wasm, wasmMt, helperURL: here + 'helper.js?v=10', threads: m.threads, layers, head, f32, manifest, emb, embScale }); eng.decide(new Float32Array(343)); postMessage({ ready: true, threads: eng.threads });
    } else if (m.obs) { const t0 = performance.now(); const { codes } = eng.decide(m.obs, null, null, m.want, m.beta); postMessage({ codes, ms: performance.now() - t0 }); }
  } catch (err) { postMessage({ error: String(err && err.message || err) }); }
};
