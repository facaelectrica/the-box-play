// a helper core for the GPC engine: takes blocks of each matrix's rows as it gets free, on the shared memory (engine.js)
onmessage = async e => {
  const { mod, memory, ctl } = e.data, gemv = (await WebAssembly.instantiate(mod, { env: { memory } })).exports.gemv;
  postMessage('ready'); let gen = Atomics.load(ctl, 0);
  for (;;) {
    Atomics.wait(ctl, 0, gen); gen = Atomics.load(ctl, 0);
    const ptr = ctl[2], X = ctl[3], Y = ctl[4], rows = ctl[5], cols = ctl[6], blk = ctl[7], nb = ctl[8];
    for (;;) { const b = Atomics.add(ctl, 9, 1); if (b >= nb) break; const r0 = b * blk; gemv(ptr + r0 * cols, X, Y + 4 * r0, Math.min(blk, rows - r0), cols); }
    Atomics.add(ctl, 1, 1); Atomics.notify(ctl, 1);
  }
};
