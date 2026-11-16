import sys, cv2, numpy as np
name, rdir, out = sys.argv[1], sys.argv[2], sys.argv[3]; d = np.load(name); J = d['j2']; B = d['box']; n = len(J); H = 540
EDGES = [(0, 69), (69, 5), (69, 6), (5, 9), (6, 10), (9, 10), (5, 7), (7, 62), (6, 8), (8, 41), (9, 11), (11, 13), (13, 17), (13, 15), (10, 12), (12, 14), (14, 20), (14, 18), (62, 42), (62, 46), (62, 50), (62, 54), (62, 58), (41, 21), (41, 25), (41, 29), (41, 33), (41, 37)]
def frame(i, overlay=True):
    im = cv2.imread('fr/%05d.jpg' % (i + 1))
    if overlay:
        for a, b in EDGES: cv2.line(im, tuple(J[i][a].astype(int)), tuple(J[i][b].astype(int)), (60, 170, 255), 2, cv2.LINE_AA)
    im = cv2.resize(im, (int(1280 * H / 720), H)); fg = cv2.imread('%s/%05d.jpg' % (rdir, i + 1)); fg = cv2.resize(fg, (int(fg.shape[1] * H / fg.shape[0]), H))
    o = np.hstack([im, fg]); cv2.putText(o, '%d  %.1fs' % (i, i / 30), (10, 24), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (255, 255, 255), 2); return o
if out.endswith('.mp4'):
    f0 = frame(0); vw = cv2.VideoWriter('tmp.avi', cv2.VideoWriter_fourcc(*'MJPG'), 30, (f0.shape[1], f0.shape[0]))
    for i in range(n): vw.write(frame(i))
    vw.release(); import subprocess; subprocess.run(['ffmpeg', '-v', 'error', '-y', '-i', 'tmp.avi', '-c:v', 'libx264', '-pix_fmt', 'yuv420p', '-crf', '23', out], check=True)
else:
    idx = [int(v) for v in sys.argv[4].split(',')]; rows = []
    for i in idx:   # zoom on him: crop the video around his box, and the middle of the figure
        im = cv2.imread('fr/%05d.jpg' % (i + 1)); x1, y1, x2, y2 = B[i]; cx, cy, s = (x1 + x2) / 2, (y1 + y2) / 2, max(x2 - x1, y2 - y1) * 1.05; M = np.array([[1, 0, -(cx - s / 2)], [0, 1, -(cy - s / 2)]]) * (H / s)
        a = cv2.warpAffine(im, M, (H, H)); fg = cv2.imread('%s/%05d.jpg' % (rdir, i + 1)); fh, fw = fg.shape[:2]; fg = cv2.resize(fg, (int(fw * H / fh), H)); c = fg.shape[1] // 2; b = fg[:, max(0, c - H // 2): c + H // 2]
        o = np.hstack([a, b]); cv2.putText(o, '%d' % i, (8, 24), cv2.FONT_HERSHEY_SIMPLEX, 0.8, (255, 255, 255), 2); rows.append(o)
    per = int(sys.argv[5]) if len(sys.argv) > 5 else 3; grid = [np.hstack(rows[k:k + per]) for k in range(0, len(rows), per)]; w = max(g.shape[1] for g in grid); grid = [np.pad(g, ((0, 0), (0, w - g.shape[1]), (0, 0))) for g in grid]; cv2.imwrite(out, np.vstack(grid))
