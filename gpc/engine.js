// NVIDIA's SOMA GPC prior, big part, purpose-built engine: 8-bit weights (a scale per row), 11-bit activations, one
// WebAssembly SIMD kernel for every matrix (k/gemv.c), the small float steps (attention over <= 9 positions, layer
// norm, GELU, the dice) in plain JS. his body (343) -> 8 picks -> 40 codes.
const L = 6, D = 1024, H = 4, HD = 256, V = 59049, BASE = 1 << 20;
// threads > 1 (needs a cross-origin-isolated page): the same kernel in helper workers on one shared memory; every matrix's
// rows are split between them (helper.js). wasmMt = the shared-memory build of the kernel.
export async function makeEngine({ wasm, wasmMt, helperURL, threads = 1, layers, head, f32, manifest, emb, embScale }) {
  let gemv, memory, par = null;
  if (threads > 1 && wasmMt && typeof SharedArrayBuffer !== 'undefined') {
    memory = new WebAssembly.Memory({ initial: 2560, maximum: 4096, shared: true });
    const mod = await WebAssembly.compile(wasmMt); gemv = (await WebAssembly.instantiate(mod, { env: { memory } })).exports.gemv;
    const ctl = new Int32Array(new SharedArrayBuffer(64)), ws = [];   // 0 generation, 1 helpers done, 2 ptr, 3 x, 4 y, 5 rows, 6 cols, 7 block rows, 8 blocks, 9 next block
    await Promise.all(Array.from({ length: threads - 1 }, (_, i) => new Promise((ok, no) => { const w = new Worker(helperURL, { type: 'module' }); w.onmessage = () => ok(); w.onerror = e => no(new Error('helper: ' + (e.message || e))); w.postMessage({ mod, memory, ctl, idx: i + 1 }); ws.push(w); })));
    const one = gemv;
    // rows go out in small blocks that each core takes as it gets free (a phone's cores are not equally fast: with
    // equal thirds the slowest core set the pace - live 8: 3 cores gave only 1.5x)
    par = (ptr, X, Y, rows, cols) => {
      const blk = Math.max(2, (Math.ceil(rows / (threads * 6)) + 1) & ~1), nb = Math.ceil(rows / blk);
      ctl[2] = ptr; ctl[3] = X; ctl[4] = Y; ctl[5] = rows; ctl[6] = cols; ctl[7] = blk; ctl[8] = nb; Atomics.store(ctl, 9, 0); Atomics.store(ctl, 1, 0); Atomics.add(ctl, 0, 1); Atomics.notify(ctl, 0);
      for (;;) { const b = Atomics.add(ctl, 9, 1); if (b >= nb) break; const r0 = b * blk; one(ptr + r0 * cols, X, Y + 4 * r0, Math.min(blk, rows - r0), cols); }
      let d; while ((d = Atomics.load(ctl, 1)) < threads - 1) Atomics.wait(ctl, 1, d, 2);
    };
    gemv = par;
  } else { const inst = (await WebAssembly.instantiate(wasm)).instance; gemv = inst.exports.gemv; memory = inst.exports.memory; threads = 1; }
  const mem = new Uint8Array(memory.buffer), off = { 'layers.bin': BASE, 'head.bin': BASE + layers.length };
  mem.set(layers, off['layers.bin']); mem.set(head, off['head.bin']);
  const X = BASE + layers.length + head.length + 64 & ~15, Y = X + 4096 * 2 + 64;      // scratch: x (int16), y (int32, up to V)
  const x16 = new Int16Array(memory.buffer, X, 4096), y32 = new Int32Array(memory.buffer, Y, V);
  const Fa = new Float32Array(f32.buffer, f32.byteOffset, f32.byteLength >> 2), f = n => { const [o, k] = manifest.f[n]; return Fa.subarray(o, o + k); };
  const mats = {}; for (const [n, m] of Object.entries(manifest.mats)) mats[n] = { ptr: off[m.file] + m.off, rows: m.rows, cols: m.cols, s: f(n + '.s'), b: f(n + '.b') };
  const E = new Int8Array(emb.buffer, emb.byteOffset, emb.byteLength), ES = new Float32Array(embScale.buffer, embScale.byteOffset, embScale.byteLength >> 2), pos = f('pos');
  function lin(name, x, out) {
    const M = mats[name], n = M.cols; let mx = 1e-20; for (let i = 0; i < n; i++) { const a = Math.abs(x[i]); if (a > mx) mx = a; }
    const sx = mx / 2047, inv = 1 / sx; for (let i = 0; i < n; i++) x16[i] = Math.round(x[i] * inv);
    gemv(M.ptr, X, Y, M.rows, n); const s = M.s, b = M.b; for (let r = 0; r < M.rows; r++) out[r] = y32[r] * s[r] * sx + b[r]; return out;
  }
  const erf = x => { const s = x < 0 ? -1 : 1, a = Math.abs(x), t = 1 / (1 + 0.3275911 * a); return s * (1 - (((((1.061405429 * t - 1.453152027) * t) + 1.421413741) * t - 0.284496736) * t + 0.254829592) * t * Math.exp(-a * a)); };
  const gelu = v => { for (let i = 0; i < v.length; i++) v[i] = 0.5 * v[i] * (1 + erf(v[i] * Math.SQRT1_2)); return v; };
  function norm(v, w, b) { let m = 0; for (let i = 0; i < D; i++) m += v[i]; m /= D; let s = 0; for (let i = 0; i < D; i++) { const d = v[i] - m; s += d * d; } const k = 1 / Math.sqrt(s / D + 1e-5); for (let i = 0; i < D; i++) v[i] = (v[i] - m) * k * w[i] + b[i]; return v; }
  const LN = []; for (let i = 0; i < L; i++) LN.push([f(`L${i}.norm1.w`), f(`L${i}.norm1.b`), f(`L${i}.norm2.w`), f(`L${i}.norm2.b`)]);
  const cm = f('ctx.mean'), ci = f('ctx.istd'), w0 = f('ctx.w0'), b0 = f('ctx.b0'), w1 = f('ctx.w1'), b1 = f('ctx.b1');
  const qkv = new Float32Array(3 * D), o = new Float32Array(D), t1 = new Float32Array(D), h4 = new Float32Array(4 * D), logits = new Float32Array(V), PB = new Float64Array(V);
  const Kc = [], Vc = []; for (let i = 0; i < L; i++) { Kc.push(new Float32Array(9 * D)); Vc.push(new Float32Array(9 * D)); }
  function context(obs) {
    const a = new Float32Array(343), hdn = new Float32Array(D), x = new Float32Array(D);
    for (let i = 0; i < 343; i++) a[i] = Math.max(-5, Math.min(5, (obs[i] - cm[i]) * ci[i]));
    for (let r = 0; r < D; r++) { let s = b0[r]; const w = r * 343; for (let c = 0; c < 343; c++) s += w0[w + c] * a[c]; hdn[r] = s; } gelu(hdn);
    for (let r = 0; r < D; r++) { let s = b1[r]; const w = r * D; for (let c = 0; c < D; c++) s += w1[w + c] * hdn[c]; x[r] = s + pos[r]; } return x;
  }
  function step(x, n) {   // position n (0..8) through the 6 layers, in place
    const sc = new Float64Array(9);
    for (let i = 0; i < L; i++) {
      lin(`L${i}.in`, x, qkv); Kc[i].set(qkv.subarray(D, 2 * D), n * D); Vc[i].set(qkv.subarray(2 * D), n * D); o.fill(0);
      for (let hh = 0; hh < H; hh++) {
        const a = hh * HD; let mx = -1e30;
        for (let p = 0; p <= n; p++) { let s = 0; const kp = p * D + a; for (let j = 0; j < HD; j++) s += Kc[i][kp + j] * qkv[a + j]; sc[p] = s / 16; if (sc[p] > mx) mx = sc[p]; }
        let z = 0; for (let p = 0; p <= n; p++) { sc[p] = Math.exp(sc[p] - mx); z += sc[p]; }
        for (let p = 0; p <= n; p++) { const w = sc[p] / z, vp = p * D + a; for (let j = 0; j < HD; j++) o[a + j] += w * Vc[i][vp + j]; }
      }
      lin(`L${i}.out`, o, t1); for (let j = 0; j < D; j++) x[j] += t1[j]; norm(x, LN[i][0], LN[i][1]);
      lin(`L${i}.f1`, x, h4); gelu(h4); lin(`L${i}.f2`, h4, t1); for (let j = 0; j < D; j++) x[j] += t1[j]; norm(x, LN[i][2], LN[i][3]);
    }
    return x;
  }
  function pick(lg) {   // NVIDIA's rule: keep the top moves holding 90% of the probability, draw among them
    let mx = -1e9; for (let i = 0; i < V; i++) if (lg[i] > mx) mx = lg[i];
    let tot = 0; for (let i = 0; i < V; i++) { const e = Math.exp(lg[i] - mx); PB[i] = e; tot += e; }
    let lo = 0, hi = 1; for (let it = 0; it < 22; it++) { const th = (lo + hi) / 2; let m = 0; for (let i = 0; i < V; i++) if (PB[i] >= th) m += PB[i]; if (m >= 0.9 * tot) lo = th; else hi = th; }
    let m = 0; for (let i = 0; i < V; i++) if (PB[i] >= lo) m += PB[i]; let u = Math.random() * m, c = 0, last = 0;
    for (let i = 0; i < V; i++) if (PB[i] >= lo) { c += PB[i]; last = i; if (c >= u) return i; } return last;
  }
  // forced: use these picks instead of the dice (tests); onLogits(k, logits) to inspect
  // want (40 codes 0..8) + beta: tilt the dice toward the words that lead to a wanted pose (bonus = -beta x squared distance)
  function decide(obs, forced, onLogits, want, beta) {
    let x = context(obs); const toks = [];
    for (let k = 0; k < 8; k++) {
      step(x, k); lin('head', x, logits);
      if (want && beta > 0) { const w0 = want[5 * k], w1 = want[5 * k + 1], w2 = want[5 * k + 2], w3 = want[5 * k + 3], w4 = want[5 * k + 4]; let t = 0;
        for (let d4 = 0; d4 < 9; d4++) for (let d3 = 0; d3 < 9; d3++) for (let d2 = 0; d2 < 9; d2++) for (let d1 = 0; d1 < 9; d1++) { const base = (d4 - w4) ** 2 + (d3 - w3) ** 2 + (d2 - w2) ** 2 + (d1 - w1) ** 2; for (let d0 = 0; d0 < 9; d0++) logits[t++] -= beta * (base + (d0 - w0) ** 2); } }
      if (onLogits) onLogits(k, logits);
      const t = forced ? forced[k] : pick(logits); toks.push(t);
      if (k < 7) { const s = ES[t], a = t * D, p = (k + 1) * D; for (let i = 0; i < D; i++) x[i] = E[a + i] * s + pos[p + i]; }
    }
    const codes = new Float32Array(40); for (let k = 0; k < 8; k++) { let t = toks[k]; for (let i = 0; i < 5; i++) { codes[5 * k + i] = t % 9 - 4; t = Math.floor(t / 9); } }
    return { codes, toks };
  }
  return { decide, threads };
}
