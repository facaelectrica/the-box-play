"""Measure a tracked take: how much each body flickers from frame to frame, and where the wrists flip."""
import sys, numpy as np
from rt import *
d = np.load(sys.argv[1]); P = d['params']; n = len(P); names_ = ['Hips', 'Chest', 'Head', 'RightArm', 'RightForeArm', 'RightHand', 'LeftArm', 'LeftForeArm', 'LeftHand', 'RightLeg', 'RightShin', 'RightFoot', 'LeftLeg', 'LeftShin', 'LeftFoot']
Ws = []; 
for i in range(n): q, W, R, Pp = retarget(P[i]); Ws.append({k: W[k] for k in names_})
ang = lambda A, B: np.degrees(np.arccos(np.clip((np.trace(A.T @ B) - 1) / 2, -1, 1)))
print('frames', n)
# flicker: the turn between consecutive frames that reverses on the next frame (acceleration), and plain frame-to-frame turn
for k in names_:
    step = np.array([ang(Ws[i][k], Ws[i + 1][k]) for i in range(n - 1)]); 
    # jitter: rotation from frame i-1 to i+1 compared with the two single steps: if he turns steadily, step(i-1,i+1) ~ step1 + step2; flicker makes it much smaller
    two = np.array([ang(Ws[i - 1][k], Ws[i + 1][k]) for i in range(1, n - 1)]); jit = (step[:-1] + step[1:] - two) / 2
    print('%-13s step median %5.1f  95%% %5.1f  max %5.0f deg | back-and-forth part median %4.1f  95%% %5.1f | steps over 45 deg: %d' % (k, np.median(step), np.percentile(step, 95), step.max(), np.median(jit), np.percentile(jit, 95), (step > 45).sum()))
# wrist relative to forearm: flips
for s in ('Right', 'Left'):
    rel = [Ws[i][s + 'ForeArm'].T @ Ws[i][s + 'Hand'] for i in range(n)]; st = np.array([ang(rel[i], rel[i + 1]) for i in range(n - 1)]); idx = np.where(st > 45)[0]
    print(s, 'hand against its forearm: frame-to-frame median %.1f, 95%% %.1f deg; jumps over 45 deg at frames' % (np.median(st), np.percentile(st, 95)), idx[:40].tolist(), '(%d)' % len(idx))
    tw = [Ws[i][s + 'Arm'].T @ Ws[i][s + 'ForeArm'] for i in range(n)]; st2 = np.array([ang(tw[i], tw[i + 1]) for i in range(n - 1)]); print(s, 'forearm against upper arm: jumps over 45 deg:', np.where(st2 > 45)[0][:30].tolist())
