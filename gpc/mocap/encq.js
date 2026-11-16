// Write a move in the library's .q format (the reverse of libq.js decodeQ): gzip of a small header and int16 frame
// steps. Steps are taken between rounded values, so rounding never adds up over a move.
export function packQ(q, T, C = 73) {
  const sc = new Float32Array(C); for (let c = 0; c < C; c++) sc[c] = c < 3 ? 2000 : c < 7 ? 16000 : 3000;
  let o = 12 + 5 * C; o += (4 - (o % 4)) % 4;
  const buf = new ArrayBuffer(o + 2 * T * C), v = new DataView(buf);
  v.setUint32(0, 0x00315142, true); v.setUint32(4, T, true); v.setUint16(8, C, true);
  for (let c = 0; c < C; c++) v.setFloat32(12 + 4 * c, sc[c], true);
  for (let c = 0; c < C; c++) { let prev = 0;
    for (let t = 0; t < T; t++) { const x = Math.round(q[t * C + c] * sc[c]), dlt = x - prev; if (dlt > 32767 || dlt < -32768) throw new Error('step too large for the format at frame ' + t + ', channel ' + c);
      v.setInt16(o + 2 * (c * T + t), dlt, true); prev = x; } }
  return buf;
}
export async function encodeQ(q, T, C = 73) {
  return await new Response(new Blob([packQ(q, T, C)]).stream().pipeThrough(new CompressionStream('gzip'))).arrayBuffer();
}
