"""Move toolkit for the light scene: read and write library moves (.q), forward kinematics on the box body."""
import os; os.environ.setdefault("MUJOCO_GL", "egl")
import gzip, struct, numpy as np, mujoco, os

REPO = '/home/claude/the-box-play'
XML = REPO + '/body/soma23_box.xml'
M = mujoco.MjModel.from_xml_path(XML)
Dd = mujoco.MjData(M)
HINGES = [M.joint(j).name for j in range(M.njnt) if M.jnt_type[j] == mujoco.mjtJoint.mjJNT_HINGE]
QADR = {M.joint(j).name: M.jnt_qposadr[j] for j in range(M.njnt)}
BODY = {M.body(b).name: b for b in range(M.nbody)}


def decode(path):
    raw = gzip.open(path).read()
    magic, T, C = struct.unpack_from('<IIH', raw, 0)
    assert magic == 0x00315142, 'not a move'
    sc = np.frombuffer(raw, np.float32, C, 12)
    md = np.frombuffer(raw, np.uint8, C, 12 + 4 * C)
    o = 12 + 5 * C; o += (4 - o % 4) % 4
    A = np.frombuffer(raw, np.int16, T * C, o).reshape(C, T).astype(np.float64)
    q = np.empty((T, C))
    for c in range(C):
        q[:, c] = (A[c] if md[c] else np.cumsum(A[c])) / sc[c]
    return q


def encode(q, path):
    """Same layout as tools/pack.py: steps of 0.5 mm (root position), 1/16000 (quaternion), 1/3000 rad (hinges)."""
    T, C = q.shape
    sc = np.array([2000.0] * 3 + [16000.0] * 4 + [3000.0] * (C - 7), np.float32)
    md = np.zeros(C, np.uint8)
    cols = []
    for c in range(C):
        ints = np.round(q[:, c] * sc[c]).astype(np.int64)
        d = np.diff(ints, prepend=0)
        if np.abs(d).max() > 32767:
            md[c] = 1; d = ints
            if np.abs(ints).max() > 32767: raise ValueError('channel %d out of range' % c)
        cols.append(d.astype(np.int16))
    head = struct.pack('<IIHH', 0x00315142, T, C, 0) + sc.tobytes() + md.tobytes()
    head += b'\0' * ((4 - len(head) % 4) % 4)
    body = b''.join(c.tobytes() for c in cols)
    with gzip.open(path, 'wb', compresslevel=9) as f: f.write(head + body)


def fk(qpos):
    Dd.qpos[:] = qpos; mujoco.mj_kinematics(M, Dd)
    return Dd


def bpos(qpos, name):
    return fk(qpos).xpos[BODY[name]].copy()


def lib(id):
    for d in (REPO + '/gpc/clips/', '/home/claude/the-box-library/moves/'):
        p = d + id + '.q'
        if os.path.exists(p): return decode(p)
    raise FileNotFoundError(id)
