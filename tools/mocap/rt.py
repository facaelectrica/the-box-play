"""MHR (InstantHMR's rig) -> the game's SOMA body, rotation to rotation. Prototype in numpy."""
import re, numpy as np, onnxruntime as ort
par = np.load('mhr_parents.npy'); SK = ort.InferenceSession('mhr-skeleton.onnx', providers=['CPUExecutionProvider'])
state = lambda p: SK.run(None, {'mhr_params': np.asarray(p, np.float32).reshape(1, 204)})[0][0]
C = np.array([[1, 0, 0], [0, 0, -1], [0, 1, 0]], float)          # MHR axes (x his left, y up, z his front) -> game (x his left, y his back, z up)
def qmat(q):
    x, y, z, w = q / np.linalg.norm(q)
    return np.array([[1-2*(y*y+z*z), 2*(x*y-w*z), 2*(x*z+w*y)], [2*(x*y+w*z), 1-2*(x*x+z*z), 2*(y*z-w*x)], [2*(x*z-w*y), 2*(y*z+w*x), 1-2*(x*x+y*y)]])
def frames(st):
    """global rotation (game axes) and position (metres, game axes) of every MHR joint"""
    R = np.stack([C @ qmat(st[j, 3:7]) for j in range(127)])   # columns: the joint's own axes, in the game's world axes
    P = (st[:, :3] * 0.01) @ C.T
    return R, P
norm = lambda v: v / (np.linalg.norm(v) or 1)
perp = lambda a, n: a - n * (a @ n)
def basis(d, f): u = norm(d); w = norm(perp(f, u)); return np.stack([u, np.cross(w, u), w], 1)
aim = lambda d, f, d0, f0: basis(d, f) @ basis(d0, f0).T
# SOMA body
xml = open('/home/claude/the-box-play/body/soma23_box.xml').read(); names, parent, off, stack = [], [], [], []
for k in re.finditer(r"<body\s+name='([^']+)'\s+pos='([^']+)'|</body>", xml):
    if k.group(1) is None: stack.pop(); continue
    names.append(k.group(1)); parent.append(stack[-1] if stack else -1); off.append(np.array(k.group(2).split(), float)); stack.append(len(names) - 1)
ID = {n: i for i, n in enumerate(names)}
X0, Z0 = np.array([1., 0, 0]), np.array([0, 0, 1.])
# which MHR joint gives each SOMA body its world rotation
MAP = dict(Hips=1, Spine1=34, Spine2=35, Chest=37, Neck1=110, Head=113, RightShoulder=38, RightArm=39, RightForeArm=41, RightHand=42,
           LeftShoulder=74, LeftArm=75, LeftForeArm=77, LeftHand=78, RightLeg=18, RightShin=19, RightFoot=22, RightToeBase=23, LeftLeg=2, LeftShin=3, LeftFoot=6, LeftToeBase=7)
st0 = state(np.zeros(204)); R0m, P0 = frames(st0)
def rest_offsets():
    """for each mapped body: the SOMA world rotation that reproduces the MHR rest pose, from two matching directions"""
    A = {n: np.eye(3) for n in names}; info = {}
    def hinge(j, expect):   # MHR's local z axis of a hinge joint, in the game's axes, signed like the expected axis
        a = R0m[j][:, 2]; return a if a @ expect > 0 else -a
    for side, s, sh, up, el, tw, wr, hip, kn, an, heel, ball, tip in (('Right', -1, 38, 39, 40, 41, 42, 18, 19, 20, 22, 23, 24), ('Left', 1, 74, 75, 76, 77, 78, 2, 3, 4, 6, 7, 8)):
        fingers = [c for c in range(127) if par[c] == wr]; bases = sorted(fingers, key=lambda c: np.linalg.norm(P0[c] - P0[wr]))
        d_up, d_fo = P0[el] - P0[up], P0[tw] - P0[el]
        swing = aim(d_up, np.cross(d_up, [0, -1, 0]), off[ID[side + 'ForeArm']], np.cross(off[ID[side + 'ForeArm']], [0, -1, 0]))   # T pose arm lowered to the A pose around the front axis
        hz = hinge(el, swing @ Z0)
        A[side + 'Shoulder'] = aim(P0[up] - P0[sh], Z0, off[ID[side + 'Arm']], Z0)
        A[side + 'Arm'] = aim(d_up, hz, off[ID[side + 'ForeArm']], Z0); A[side + 'ForeArm'] = aim(d_fo, hz, off[ID[side + 'Hand']], Z0)
        info[side + ' elbow axis vs lowered T pose (deg)'] = round(float(np.degrees(np.arccos(np.clip(norm(perp(hz, norm(d_up))) @ norm(perp(swing @ Z0, norm(d_up))), -1, 1)))), 1)
        info[side + ' finger chains'] = fingers
        # hand: direction wrist -> the long fingers' knuckles, knuckle line from the little-finger side to the index side
        ch = lambda c: [c] + (ch(next(k for k in range(127) if par[k] == c)) if any(par[k] == c for k in range(127)) else [])
        chains = [ch(c) for c in fingers]; info[side + ' chains'] = [len(c) for c in chains]
        A['_' + side] = (wr, chains)
        mid, ix, pk = chains[2][0], chains[1][0], chains[4][1]           # middle and index knuckles, little-finger knuckle
        dirh = P0[mid] - P0[wr]; A[side + 'Hand'] = aim(dirh, np.cross(dirh, P0[ix] - P0[pk]) * -s, [s, 0, 0], Z0)
        d_th, d_sh = P0[kn] - P0[hip], P0[an] - P0[kn]; hx = hinge(kn, X0)
        A[side + 'Leg'] = aim(d_th, hx, off[ID[side + 'Shin']], X0); A[side + 'Shin'] = aim(d_sh, hx, off[ID[side + 'Foot']], X0)
        info[side + ' knee axis (game axes)'] = np.round(hx, 2).tolist(); info[side + ' elbow axis'] = np.round(hz, 2).tolist()
        fwd = P0[tip] - P0[heel]; A[side + 'Foot'] = aim(perp(fwd, Z0), Z0, [0, -1, 0], Z0); A[side + 'ToeBase'] = A[side + 'Foot']
        info[side + ' foot: heel->toe'] = np.round(norm(fwd), 2).tolist()
    return A, info
A0, INFO = rest_offsets()
def euler_xyz(R, near=None):
    b = np.arcsin(np.clip(R[0, 2], -1, 1)); a = np.arctan2(-R[1, 2], R[2, 2]); c = np.arctan2(-R[0, 1], R[0, 0]); return np.array([a, b, c])
def eul(a, b, c):
    ca, sa, cb, sb, cc, sc = np.cos(a), np.sin(a), np.cos(b), np.sin(b), np.cos(c), np.sin(c)
    return np.array([[cb*cc, -cb*sc, sb], [sa*sb*cc+ca*sc, -sa*sb*sc+ca*cc, -sa*cb], [-ca*sb*cc+sa*sc, ca*sb*sc+sa*cc, ca*cb]])
def mquat(R):
    w = np.sqrt(max(0, 1 + R[0, 0] + R[1, 1] + R[2, 2])) / 2
    if w > 1e-6: return np.array([w, (R[2, 1] - R[1, 2]) / (4 * w), (R[0, 2] - R[2, 0]) / (4 * w), (R[1, 0] - R[0, 1]) / (4 * w)])
    x = np.sqrt(max(0, 1 + R[0, 0] - R[1, 1] - R[2, 2])) / 2; return np.array([0, x, (R[0, 1] + R[1, 0]) / (4 * x), (R[0, 2] + R[2, 0]) / (4 * x)])
def slerp_mid(Ra, Rb):   # half way between two rotations
    D = Ra.T @ Rb; ang = np.arccos(np.clip((np.trace(D) - 1) / 2, -1, 1))
    if ang < 1e-6: return Ra
    ax = np.array([D[2, 1] - D[1, 2], D[0, 2] - D[2, 0], D[1, 0] - D[0, 1]]) / (2 * np.sin(ang)); K = np.array([[0, -ax[2], ax[1]], [ax[2], 0, -ax[0]], [-ax[1], ax[0], 0]]); h = ang / 2
    return Ra @ (np.eye(3) + np.sin(h) * K + (1 - np.cos(h)) * K @ K)
def retarget(mhr_params):
    """mhr_params (204) -> the game's pose (73: hips position 0, hips quaternion w x y z, 66 hinges) and each body's world rotation"""
    R, P = frames(state(mhr_params)); W = {}
    for n, j in MAP.items(): W[n] = R[j] @ R0m[j].T @ A0[n]          # how the joint has turned since the rig's rest pose, applied to the body posed like that rest pose
    W['Neck2'] = slerp_mid(W['Neck1'], W['Head'])
    q = np.zeros(73); q[3:7] = mquat(W['Hips'])
    for b in range(1, 23): q[7 + 3 * (b - 1): 10 + 3 * (b - 1)] = euler_xyz(W[names[parent[b]]].T @ W[names[b]])
    return q, W, R, P
def soma_fk(q):
    p = [q[:3].copy()]; w, x, y, z = q[3:7]; R = [qmat(np.array([x, y, z, w]))]
    for b in range(1, 23): a = parent[b]; p.append(p[a] + R[a] @ off[b]); R.append(R[a] @ eul(*q[7 + 3 * (b - 1): 10 + 3 * (b - 1)]))
    return np.array(p), R
if __name__ == '__main__':
    A, info = rest_offsets()
    for k, v in info.items(): print(k, v)
    for s in ('Right', 'Left'):
        wr, chains = A['_' + s]
        for c in chains: print(s, 'finger chain', c, 'base pos', np.round(P0[c[1] if len(c) > 1 else c[0]] - P0[wr], 3).tolist(), 'tip', np.round(P0[c[-1]] - P0[wr], 3).tolist())
    print('rest: R wrist', np.round(P0[42], 3), 'R elbow', np.round(P0[40], 3), 'R shoulder', np.round(P0[39], 3), 'head', np.round(P0[113], 3), 'eye', np.round(P0[122], 3))
