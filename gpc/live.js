// Puts NVIDIA's GPC prior in the page brain's seat. Big part in a background worker (a new set of picks whenever it
// finishes one), small part (decoder, 3 M numbers) here every control tick on his current body.
export async function attachGPC(brain, ort, say) {
  const here = new URL('./', import.meta.url).href, iso = self.crossOriginIsolated, cores = navigator.hardwareConcurrency || 2;
  const threads = iso ? Math.max(1, Math.min(4, cores - 1)) : 1;
  const w = new Worker(here + 'worker.js', { type: 'module' }), tracker = brain.session, st = { codes: null, busy: false, ms: 0, picks: 0, ticks: 0, sinceT: performance.now(), since: 0, rate: 0, err: null, loaded: {}, ready: false, threads };
  const TOTAL = 194.3e6;
  w.onmessage = e => {
    const m = e.data;
    if (m.load) { st.loaded[m.load] = m.got; const g = Object.values(st.loaded).reduce((a, b) => a + b, 0); say(`NVIDIA brain: downloading ${(g / 1e6).toFixed(0)} of ${(TOTAL / 1e6).toFixed(0)} MB (once; kept by the phone after)`); }
    else if (m.stage) say('NVIDIA brain: starting up…');
    else if (m.ready) { st.ready = true; say('NVIDIA brain: ready'); }
    else if (m.codes) { st.codes = m.codes; st.busy = false; st.ms += (m.ms - st.ms) * (st.picks ? 0.2 : 1); st.picks++; }
    else if (m.error) { st.err = m.error; st.busy = false; say('NVIDIA brain failed: ' + m.error.slice(0, 120)); }
  };
  w.postMessage({ init: true, threads });
  const dec = await ort.InferenceSession.create(new Uint8Array(await (await fetch(here + 'live/dec.onnx')).arrayBuffer()), { executionProviders: ['wasm'], graphOptimizationLevel: 'all' });
  brain.meta.getup = true;   // never the tracker's "gave up, go limp": this brain keeps control on the floor too
  let last = 0;
  brain.session = { run: async feeds => {
    const o = feeds.obs.data.subarray(0, 343);
    if (st.ready && !st.busy) { st.busy = true; w.postMessage({ obs: new Float32Array(o) }); }
    if (!st.codes) return tracker.run(feeds);   // until the first picks arrive he stands with the page's own tracker
    st.ticks++;
    const now = performance.now(); if (now - last > 1000) { st.rate = st.picks - st.since; st.since = st.picks; last = now; say(`NVIDIA brain live · ${st.rate} picks a second · ${st.ms.toFixed(0)} ms each · ${threads} core${threads > 1 ? 's' : ''}${iso ? '' : ' (several-core mode refused)'}`); }
    return dec.run({ obs: new ort.Tensor('float32', new Float32Array(o), [1, 343]), codes: new ort.Tensor('float32', st.codes, [1, 40]) });
  } };
  return st;
}
