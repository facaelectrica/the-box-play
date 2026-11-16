// NVIDIA's SOMA GPC prior, the big part: his body (343 numbers) -> 8 picks from 59,049 "movement words" each.
// Runs here in the background; the page's small part turns the latest picks into joint commands every frame.
import * as ort from './ort/ort.all.min.mjs';
const here = new URL('./', import.meta.url).href, L = 6, NP = 9, D = 1024, V = 59049;
ort.env.wasm.wasmPaths = here + 'ort/';
let S, H, C, emb, escale, pos; const PB = new Float64Array(V);
async function get(name, mb) {
  const r = await fetch(here + 'live/' + name), n = +r.headers.get('content-length') || mb * 1048576, rd = r.body.getReader(), parts = []; let got = 0;
  for (;;) { const { done, value } = await rd.read(); if (done) break; parts.push(value); got += value.length; postMessage({ load: name, got, of: n }); }
  const b = new Uint8Array(got); let o = 0; for (const p of parts) { b.set(p, o); o += p.length; } return b;
}
function pick(lg) {   // NVIDIA's rule: keep the top moves holding 90% of the probability, draw among them
  let mx = -1e9; for (let i = 0; i < V; i++) if (lg[i] > mx) mx = lg[i];
  let tot = 0; for (let i = 0; i < V; i++) { const e = Math.exp(lg[i] - mx); PB[i] = e; tot += e; }
  let lo = 0, hi = 1; for (let it = 0; it < 22; it++) { const th = (lo + hi) / 2; let m = 0; for (let i = 0; i < V; i++) if (PB[i] >= th) m += PB[i]; if (m >= 0.9 * tot) lo = th; else hi = th; }
  let m = 0; for (let i = 0; i < V; i++) if (PB[i] >= lo) m += PB[i]; let u = Math.random() * m, c = 0, last = 0;
  for (let i = 0; i < V; i++) if (PB[i] >= lo) { c += PB[i]; last = i; if (c >= u) return i; } return last;
}
async function decide(obs) {
  let x = (await C.run({ obs: new ort.Tensor('float32', obs, [1, 343]) })).x.data;
  let Kc = new ort.Tensor('float32', new Float32Array(L * NP * D), [L, NP, D]), Vc = new ort.Tensor('float32', new Float32Array(L * NP * D), [L, NP, D]); const toks = [];
  for (let k = 0; k < 8; k++) {
    const add = new Float32Array(NP).fill(-1e9), hot = new Float32Array(NP); for (let j = 0; j <= k; j++) add[j] = 0; hot[k] = 1;
    const o = await S.run({ x: new ort.Tensor('float32', x, [1, D]), Kc, Vc, add: new ort.Tensor('float32', add, [NP]), hot: new ort.Tensor('float32', hot, [NP]) });
    Kc = o.K; Vc = o.V; const t = pick((await H.run({ h: o.h })).logits.data); toks.push(t);
    if (k < 7) { x = new Float32Array(D); const s = escale[t], a = t * D, p = (k + 1) * D; for (let i = 0; i < D; i++) x[i] = emb[a + i] * s + pos[p + i]; }
  }
  const codes = new Float32Array(40); for (let k = 0; k < 8; k++) { let t = toks[k]; for (let i = 0; i < 5; i++) { codes[5 * k + i] = t % 9 - 4; t = Math.floor(t / 9); } }
  return codes;
}
onmessage = async e => {
  const m = e.data;
  if (m.init) {
    try {
      ort.env.wasm.numThreads = m.threads;
      const so = { executionProviders: ['wasm'], graphOptimizationLevel: 'all' };
      const [c, s, h, eb, es, ps] = [await get('ctx.onnx', 5.4), await get('step.onnx', 72.6), await get('head.onnx', 57.9), await get('emb_q8.bin', 57.7), await get('emb_scale.bin', 0.2), await get('pos.bin', 0.04)];
      postMessage({ stage: 'building' });
      C = await ort.InferenceSession.create(c, so); S = await ort.InferenceSession.create(s, so); H = await ort.InferenceSession.create(h, so);
      emb = new Int8Array(eb.buffer); escale = new Float32Array(es.buffer); pos = new Float32Array(ps.buffer);
      await decide(new Float32Array(343)); postMessage({ ready: true, threads: ort.env.wasm.numThreads });
    } catch (err) { postMessage({ error: String(err && err.message || err) }); }
  } else if (m.obs) { const t0 = performance.now(); try { const codes = await decide(m.obs); postMessage({ codes, ms: performance.now() - t0, id: m.id }); } catch (err) { postMessage({ error: String(err && err.message || err) }); } }
};
