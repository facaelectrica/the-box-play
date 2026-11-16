"""Run InstantHMR over a folder of frames, following the person with its own picture points. Saves the raw outputs."""
import sys, glob, math, time, cv2, numpy as np, onnxruntime as ort
MEAN, STD = np.array([0.485, 0.456, 0.406], np.float32), np.array([0.229, 0.224, 0.225], np.float32)
def box_of(pts):
    x1, y1 = pts.min(0); x2, y2 = pts.max(0); bw, bh = x2 - x1, y2 - y1
    return None if not (bw > 8 and bh > 8) else np.array([x1 - 0.12 * bw, y1 - 0.12 * bh, x2 + 0.12 * bw, y2 + 0.06 * bh])
class Net:
    def __init__(self, path):
        so = ort.SessionOptions(); so.intra_op_num_threads = 8; self.s = ort.InferenceSession(path, so, providers=['CPUExecutionProvider']); m = self.s.get_modelmeta().custom_metadata_map
        self.crops = int(m.get('crops', 0)); self.scales = tuple(float(v) for v in m['crop_scales'].split(',')) if self.crops else (1.0,); self.S = int(m.get('image_size', 288))
    def crop(self, rgb, box, scale, f):
        h, w = rgb.shape[:2]; x1, y1, x2, y2 = box; cx, cy, size = (x1 + x2) / 2, (y1 + y2) / 2, max(x2 - x1, y2 - y1) * scale; side = size * 1.2; s = self.S
        ix1, iy1 = math.floor(cx - side / 2), math.floor(cy - side / 2); ix2, iy2 = math.ceil(cx - side / 2 + side), math.ceil(cy - side / 2 + side); sx, sy = (ix2 - ix1) / s, (iy2 - iy1) / s
        m = np.array([[sx, 0, ix1 + 0.5 * sx - 0.5], [0, sy, iy1 + 0.5 * sy - 0.5]]); c = cv2.warpAffine(rgb, m, (s, s), flags=cv2.INTER_LINEAR | cv2.WARP_INVERSE_MAP, borderValue=(0, 0, 0))
        return c, np.array([math.atan((cx - w / 2) / f), math.atan((cy - h / 2) / f), size / f], np.float32), (ix1, iy1, ix2 - ix1, iy2 - iy1)
    def run(self, rgb, box):
        h, w = rgb.shape[:2]; f = 1.05 * math.hypot(h, w); names = ['mhr_params', 'shape_params', 'cam_trans', 'joints_2d']
        if self.crops:   # network + kinematic fit: 8-bit crops, the box enlarged to a ground-truth-like box
            c = (box[:2] + box[2:]) / 2; half = (box[2:] - box[:2]) / 0.866 / 2; gt = np.concatenate([c - half, c + half]); cr, cl, sq = zip(*(self.crop(rgb, gt, s, f) for s in self.scales))
            mp, sp, ct, j2 = self.s.run(names, {'image': np.stack(cr), 'cliff_cond': np.stack(cl), 'crop_box': np.asarray(sq, np.float32), 'cam': np.array([[f, w / 2, h / 2]], np.float32)}); return mp[0], ct[0], j2[0]
        cr, cl, sq = self.crop(rgb, box, 1.0, f); x = np.ascontiguousarray(((cr.astype(np.float32) / 255 - MEAN) / STD).transpose(2, 0, 1))[None]
        mp, sp, ct, j2 = self.s.run(names, {'image': x, 'cliff_cond': cl[None]}); return mp[0], ct[0], (j2[0] + 1) * 0.5 * np.array(sq[2:]) + np.array(sq[:2])
if __name__ == '__main__':
    frames = sorted(glob.glob(sys.argv[1] + '/*.jpg')); net = Net(sys.argv[2]); out = sys.argv[3]; step = int(sys.argv[4]) if len(sys.argv) > 4 else 1; frames = frames[::step]
    P, C, J, B = [], [], [], []; box = None; t0 = time.time()
    for i, f in enumerate(frames):
        rgb = cv2.cvtColor(cv2.imread(f), cv2.COLOR_BGR2RGB); h, w = rgb.shape[:2]
        if box is None:   # find him: start from the whole picture and let the picture points pull the box in
            box = np.array([w * 0.1, 0, w * 0.9, h], float)
            for _ in range(3): mp, ct, j2 = net.run(rgb, box); box = box_of(j2) if box_of(j2) is not None else box
        mp, ct, j2 = net.run(rgb, box); P.append(mp); C.append(ct); J.append(j2); B.append(box.copy()); nb = box_of(j2); box = nb if nb is not None else None
    np.savez(out, params=np.array(P), cam=np.array(C), j2=np.array(J), box=np.array(B), step=step); print(out, len(P), 'frames, %.0f ms a frame on this CPU' % ((time.time() - t0) / len(P) * 1000))
