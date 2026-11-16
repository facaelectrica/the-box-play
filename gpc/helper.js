// a helper core for the GPC engine: runs its share of each matrix's rows on the shared memory (see engine.js)
onmessage = async e => {
  const { mod, memory, ctl, idx } = e.data, gemv = (await WebAssembly.instantiate(mod, { env: { memory } })).exports.gemv;
  postMessage('ready'); let gen = Atomics.load(ctl, 0);
  for (;;) {
    Atomics.wait(ctl, 0, gen); gen = Atomics.load(ctl, 0);
    const ptr = ctl[2], X = ctl[3], Y = ctl[4], rows = ctl[5], cols = ctl[6], parts = ctl[7], per = (Math.floor(rows / parts) + 1) & ~1;
    const r0 = Math.min(rows, per * idx), r1 = idx === parts - 1 ? rows : Math.min(rows, per * (idx + 1));
    if (r1 > r0) gemv(ptr + r0 * cols, X, Y + 4 * r0, r1 - r0, cols);
    Atomics.add(ctl, 1, 1); Atomics.notify(ctl, 1);
  }
};
