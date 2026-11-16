import sys, math, cv2, numpy as np, onnxruntime as ort
from rt import *
NET = ort.InferenceSession('instanthmr-medium.onnx', providers=['CPUExecutionProvider'])
MEAN, STD = np.array([0.485, 0.456, 0.406], np.float32), np.array([0.229, 0.224, 0.225], np.float32)
def predict(rgb, box, s=288):
    h, w = rgb.shape[:2]; f = 1.05 * math.hypot(h, w); x1, y1, x2, y2 = box; cx, cy, size = (x1 + x2) / 2, (y1 + y2) / 2, max(x2 - x1, y2 - y1)
    cliff = np.array([[math.atan((cx - w / 2) / f), math.atan((cy - h / 2) / f), size / f]], np.float32); side = size * 1.2
    ix1, iy1 = math.floor(cx - side / 2), math.floor(cy - side / 2); ix2, iy2 = math.ceil(cx - side / 2 + side), math.ceil(cy - side / 2 + side); sx, sy = (ix2 - ix1) / s, (iy2 - iy1) / s
    m = np.array([[sx, 0, ix1 + 0.5 * sx - 0.5], [0, sy, iy1 + 0.5 * sy - 0.5]]); crop = cv2.warpAffine(rgb, m, (s, s), flags=cv2.INTER_LINEAR | cv2.WARP_INVERSE_MAP, borderValue=(0, 0, 0))
    x = np.ascontiguousarray(((crop.astype(np.float32) / 255 - MEAN) / STD).transpose(2, 0, 1))[None]
    mp, sp, ct, j2 = NET.run(['mhr_params', 'shape_params', 'cam_trans', 'joints_2d'], {'image': x, 'cliff_cond': cliff})
    return mp[0], ct[0], (j2[0] + 1) * 0.5 * np.array([ix2 - ix1, iy2 - iy1]) + np.array([ix1, iy1]), crop
def draw(canvas, pts, edges, col, ox, oy, sc, ax):
    for a, b in edges: cv2.line(canvas, (int(ox + pts[a][ax[0]] * sc * ax[2]), int(oy - pts[a][ax[1]] * sc)), (int(ox + pts[b][ax[0]] * sc * ax[2]), int(oy - pts[b][ax[1]] * sc)), col, 2, cv2.LINE_AA)
if __name__ == '__main__':
    img = cv2.imread(sys.argv[1]); x0, y0, x1, y1 = map(int, sys.argv[2:6]); pane = img[y0:y1, x0:x1][:, ::-1].copy()      # the page shows the camera mirrored: flip it back
    box = list(map(float, sys.argv[6:10])); rgb = cv2.cvtColor(pane, cv2.COLOR_BGR2RGB)
    import time; t = time.time(); mp, ct, j2, crop = predict(rgb, box); dt = time.time() - t
    q, W, R, P = retarget(mp); p, Rs = soma_fk(q); deg = lambda n: np.round(np.degrees(q[7 + 3 * (ID[n] - 1): 10 + 3 * (ID[n] - 1)])).astype(int).tolist()
    print('network on the CPU here: %.0f ms; cam_trans' % (dt * 1000), np.round(ct, 2), '| hips quaternion', np.round(q[3:7], 2))
    for n in ['RightArm', 'RightForeArm', 'RightHand', 'LeftArm', 'LeftForeArm', 'LeftHand', 'RightLeg', 'RightShin', 'RightFoot', 'LeftLeg', 'LeftShin', 'LeftFoot', 'Spine1', 'Chest', 'Head']: print(n, deg(n), end=' | ')
    print()
    S = 420; out = np.full((S, S * 3, 3), 30, np.uint8); big = cv2.resize(cv2.cvtColor(crop, cv2.COLOR_RGB2BGR), (S, S)); out[:, :S] = big
    sedges = [(parent[b], b) for b in range(1, 23)]; medges = [(MAP[names[parent[b]]] if names[parent[b]] in MAP else 1, MAP[names[b]]) for b in range(1, 23) if names[b] in MAP]
    pc = p - p[0]; Pc = P - P[1]
    for k, (ax, lab) in enumerate([((0, 2, -1), 'front (as the camera sees him)'), ((1, 2, 1), 'side (his front to the left)')]):
        ox = S + S // 2 + k * S; draw(out, Pc, medges, (150, 150, 150), ox, S * 0.55, 190, ax); draw(out, pc, sedges, (60, 160, 255), ox, S * 0.55, 190, ax); cv2.putText(out, lab, (S + k * S + 8, 18), cv2.FONT_HERSHEY_SIMPLEX, 0.45, (200, 200, 200), 1)
    cv2.putText(out, 'grey: InstantHMR rig   orange: his body', (S + 8, S - 10), cv2.FONT_HERSHEY_SIMPLEX, 0.45, (200, 200, 200), 1); cv2.imwrite(sys.argv[10], out)
