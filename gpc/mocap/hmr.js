// InstantHMR as the tracker: a network that reads the pose of a 127-joint body rig (Meta's MHR) from one picture of the
// person, as joint rotations. That is what a skeleton needs. A tracker that gives joint positions (MediaPipe) leaves
// the turn of each bone around its own length unknown, and guessing it is what made elbows and hands flip.
//   picture crop -> network (WebGPU) -> 204 rig parameters -> the rig's joint rotations (rigState, plain maths)
//   -> each of his 23 bodies takes the world rotation of one rig joint (mhr_map.json, built by tools/mocap/rt.py)
import { M } from './skel.js?v=1';

export const HMR_SIZE = 288;
// the rig's 70 picture points that the page draws and the solver reads (instanthmr/skeleton.py)
export const KP = { lAnkle: 13, rAnkle: 14, lToe: 15, lHeel: 17, rToe: 18, rHeel: 20 };
export const KP_EDGES = [[0, 69], [69, 5], [69, 6], [5, 9], [6, 10], [9, 10], [5, 7], [7, 62], [6, 8], [8, 41], [9, 11], [11, 13], [13, 17], [13, 15], [10, 12], [12, 14], [14, 20], [14, 18],
  [62, 42], [62, 46], [62, 50], [62, 54], [62, 58], [41, 21], [41, 25], [41, 29], [41, 33], [41, 37]];

// The rig's forward kinematics, rotations only: 204 parameters -> 127 x 8 numbers with each joint's world quaternion
// (x y z w) at [3..6], the layout of InstantHMR's own skeleton state. rig (mhr_rig.json, from tools/mocap): parent,
// pre = each joint's rest rotation in its parent, pt = [joint * 7 + 3..5, parameter, weight] for the joint's x y z turns.
// Checked against InstantHMR's graph on real frames: the same to 6e-5.
export function rigState(rig, params, out = new Float64Array(127 * 8)) {
  const e = rig._e || (rig._e = new Float64Array(127 * 3)); e.fill(0); for (const [k, c, w] of rig.pt) e[3 * ((c / 7) | 0) + (c % 7) - 3] += params[k] * w;
  const mul = (a, ao, b, bo, o, oo) => { const x1 = a[ao], y1 = a[ao + 1], z1 = a[ao + 2], w1 = a[ao + 3], x2 = b[bo], y2 = b[bo + 1], z2 = b[bo + 2], w2 = b[bo + 3];
    o[oo] = w1 * x2 + x1 * w2 + y1 * z2 - z1 * y2; o[oo + 1] = w1 * y2 - x1 * z2 + y1 * w2 + z1 * x2; o[oo + 2] = w1 * z2 + x1 * y2 - y1 * x2 + z1 * w2; o[oo + 3] = w1 * w2 - x1 * x2 - y1 * y2 - z1 * z2; };
  const qe = [0, 0, 0, 1], loc = [0, 0, 0, 1];
  for (let j = 0; j < 127; j++) { const cr = Math.cos(e[3 * j] / 2), sr = Math.sin(e[3 * j] / 2), cp = Math.cos(e[3 * j + 1] / 2), sp = Math.sin(e[3 * j + 1] / 2), cy = Math.cos(e[3 * j + 2] / 2), sy = Math.sin(e[3 * j + 2] / 2);
    qe[0] = sr * cp * cy - cr * sp * sy; qe[1] = cr * sp * cy + sr * cp * sy; qe[2] = cr * cp * sy - sr * sp * cy; qe[3] = cr * cp * cy + sr * sp * sy; mul(rig.pre, 4 * j, qe, 0, loc, 0);
    const p = rig.parent[j]; if (p < 0) { out[8 * j + 3] = loc[0]; out[8 * j + 4] = loc[1]; out[8 * j + 5] = loc[2]; out[8 * j + 6] = loc[3]; } else mul(out, 8 * p + 3, loc, 0, out, 8 * j + 3); }
  return out;
}

// rig joint states (127 x [position 3, quaternion x y z w, scale]) -> world rotation of each of his bodies, game axes
export function bodyRotations(map, S, state) {
  const W = new Array(S.names.length).fill(null);
  for (const n in map.joint) { const o = 8 * map.joint[n]; W[S.id[n]] = M.mul(M.mul(map.C, M.fromQuat(state[o + 6], state[o + 3], state[o + 4], state[o + 5])), map.T[n]); }
  const a = W[S.id.Neck1], b = W[S.id.Head]; W[S.id.Neck2] = M.mul(a, M.part(M.mul(M.T(a), b), 0.5));   // the rig has one neck joint, he has two
  return W;
}

// The person's box for the next frame, from picture points (pixels): the points' extent, widened to what a person
// detector would give (the head above the nose, the feet below the ankles).
export function boxOf(pts, w, h) {
  let x1 = 1e9, y1 = 1e9, x2 = -1e9, y2 = -1e9; for (const p of pts) { x1 = Math.min(x1, p[0]); x2 = Math.max(x2, p[0]); y1 = Math.min(y1, p[1]); y2 = Math.max(y2, p[1]); }
  const bw = x2 - x1, bh = y2 - y1; if (!(bw > 8 && bh > 8) || !Number.isFinite(bw + bh)) return null;
  return [x1 - 0.12 * bw, y1 - 0.12 * bh, x2 + 0.12 * bw, y2 + 0.06 * bh];
}

// Loads the network and the rig. ort: onnxruntime-web (WebGPU build). say(text): progress for the user.
export async function loadHMR(ort, modelUrl, here, say = () => {}) {
  const get = async (url, label) => {   // the model is large: keep it in the browser's cache between visits
    let cache = null; try { cache = await caches.open('record-models'); const hit = await cache.match(url); if (hit) return hit.arrayBuffer(); } catch (_) {}
    const r = await fetch(url); if (!r.ok) throw new Error(label + ' could not be downloaded (' + r.status + ').');
    const total = +r.headers.get('content-length') || 0, rd = r.body.getReader(), parts = []; let got = 0;
    for (;;) { const { done, value } = await rd.read(); if (done) break; parts.push(value); got += value.length; say(total ? `Downloading ${label}: ${Math.round(100 * got / total)}% of ${Math.round(total / 1e6)} MB` : `Downloading ${label}: ${Math.round(got / 1e6)} MB`); }
    const buf = new Uint8Array(got); let o = 0; for (const p of parts) { buf.set(p, o); o += p.length; }
    try { if (cache) await cache.put(url, new Response(buf, { headers: { 'content-type': 'application/octet-stream' } })); } catch (_) {}
    return buf.buffer;
  };
  const jget = async f => (await fetch(here + f)).json(), map = await jget('mhr_map.json?v=2'), rig = await jget('mhr_rig.json?v=1'), netBuf = await get(modelUrl, 'the tracker');
  say('Preparing the tracker on the GPU. The first time takes about 10 seconds.');
  let net, gpu = true; // one picture at a time: with the input size fixed the runtime folds away a third of the graph (29 -> 25 ms on the user's GPU, and steadier)
  const fixed = { graphOptimizationLevel: 'all', freeDimensionOverrides: { batch: 1 } };
  try { net = await ort.InferenceSession.create(netBuf, { executionProviders: ['webgpu'], ...fixed }); }
  catch (e) { console.warn('No WebGPU for the tracker; using the CPU, which is slow', e); gpu = false; net = await ort.InferenceSession.create(netBuf, { executionProviders: ['wasm'], ...fixed }); }
  const S = HMR_SIZE, cv = Object.assign(document.createElement('canvas'), { width: S, height: S }), g = cv.getContext('2d', { willReadFrequently: true }), x = new Float32Array(3 * S * S);
  const MEAN = [0.485, 0.456, 0.406], STD = [0.229, 0.224, 0.225], ms = { crop: 0, net: 0 };
  // One frame: the crop around `box` (x1, y1, x2, y2 in the frame's pixels) -> the rig's joints and picture points.
  async function step(video, box) {
    const t0 = performance.now(), w = video.videoWidth, h = video.videoHeight, f = 1.05 * Math.hypot(h, w), cx = (box[0] + box[2]) / 2, cy = (box[1] + box[3]) / 2, size = Math.max(box[2] - box[0], box[3] - box[1]), side = size * 1.2;
    g.fillStyle = '#000'; g.fillRect(0, 0, S, S); g.drawImage(video, cx - side / 2, cy - side / 2, side, side, 0, 0, S, S);
    const px = g.getImageData(0, 0, S, S).data, n = S * S; for (let i = 0; i < n; i++) { x[i] = (px[4 * i] / 255 - MEAN[0]) / STD[0]; x[n + i] = (px[4 * i + 1] / 255 - MEAN[1]) / STD[1]; x[2 * n + i] = (px[4 * i + 2] / 255 - MEAN[2]) / STD[2]; }
    const t1 = performance.now(), r = await net.run({ image: new ort.Tensor('float32', x, [1, 3, S, S]), cliff_cond: new ort.Tensor('float32', new Float32Array([Math.atan((cx - w / 2) / f), Math.atan((cy - h / 2) / f), size / f]), [1, 3]) }), t2 = performance.now();
    const params = Float32Array.from(r.mhr_params.data), j2 = r.joints_2d.data, pts = new Array(70);
    for (let k = 0; k < 70; k++) pts[k] = [(j2[2 * k] + 1) * 0.5 * side + cx - side / 2, (j2[2 * k + 1] + 1) * 0.5 * side + cy - side / 2];
    ms.crop += 0.1 * (t1 - t0 - ms.crop); ms.net += 0.1 * (t2 - t1 - ms.net);
    return { params, state: rigState(rig, params), pts, cam: Array.from(r.cam_trans.data), focal: f };
  }
  // Following him from frame to frame. The box around him comes from the tracker's own picture points of the frame
  // before, eased so the crop does not shake: a different box alone changes the pose by several degrees, so a box
  // from another tracker must never be mixed in. To find him, start from the whole picture and let the points pull
  // the box in.
  let box = null;
  async function track(video) {
    const w = video.videoWidth, h = video.videoHeight;
    if (!box) { box = [w * 0.1, 0, w * 0.9, h]; for (let k = 0; k < 3; k++) { const r = await step(video, box), b = boxOf(r.pts, w, h); if (b) box = b; } }
    const r = await step(video, box), nb = boxOf(r.pts, w, h);
    if (!nb || nb[3] - nb[1] < 40) box = null; else for (let k = 0; k < 4; k++) box[k] += 0.5 * (nb[k] - box[k]);
    return r;
  }
  return { map, rig, step, track, lose: () => { box = null; }, ms, gpu };
}
