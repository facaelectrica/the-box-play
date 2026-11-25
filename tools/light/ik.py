"""Inverse kinematics on the box body: damped least squares over chosen hinges, inside the game's joint limits."""
import json, numpy as np, mujoco
from mv import M, Dd, BODY, QADR, REPO

LIM = json.load(open(REPO + '/body/clips/kin/limits_gpc.json'))
DOF = {M.joint(j).name: M.jnt_dofadr[j] for j in range(M.njnt)}


def lim(name, margin=3.0):
    lo, hi = LIM.get(name, [-180, 180])
    return np.radians(lo + margin), np.radians(hi - margin)


def clamp_limits(q, names, margin=3.0):
    for n in names:
        lo, hi = lim(n, margin); a = QADR[n]; q[a] = min(hi, max(lo, q[a]))
    return q


def chain(*bodies):
    return [b + '_' + ax for b in bodies for ax in 'xyz']


def solve(q0, tasks, joints, iters=60, damp=0.03, reg=0.02, step=1.0):
    """tasks: dicts with body, and pos (world target, m) and/or axis (local axis 3) + dir (world unit target), weight.
    Returns qpos with only `joints` changed. reg pulls toward q0."""
    q = q0.copy(); ja = np.array([QADR[n] for n in joints]); jd = np.array([DOF[n] for n in joints])
    jp = np.zeros((3, M.nv)); jr = np.zeros((3, M.nv))
    for _ in range(iters):
        Dd.qpos[:] = q; mujoco.mj_kinematics(M, Dd); mujoco.mj_comPos(M, Dd)
        rows, errs = [], []
        for t in tasks:
            b = BODY[t['body']]; w = t.get('weight', 1.0)
            mujoco.mj_jacBody(M, Dd, jp, jr, b)
            if 'pos' in t:
                p = Dd.xpos[b] if 'offset' not in t else Dd.xpos[b] + Dd.xmat[b].reshape(3, 3) @ t['offset']
                if 'offset' in t: mujoco.mj_jac(M, Dd, jp, jr, p, b)
                rows.append(w * jp[:, jd]); errs.append(w * (np.array(t['pos']) - p))
            if 'axis' in t:
                R = Dd.xmat[b].reshape(3, 3); a = R @ np.array(t['axis']); d = np.array(t['dir']); d = d / np.linalg.norm(d)
                e = np.cross(a, d)   # rotation that takes a toward d
                ww = t.get('aweight', w)
                rows.append(ww * jr[:, jd]); errs.append(ww * e)
        J = np.vstack(rows); e = np.concatenate(errs)
        dq0 = q0[ja] - q[ja]
        A = J.T @ J + (damp + reg) * np.eye(len(ja))
        dq = np.linalg.solve(A, J.T @ e + reg * dq0)
        q[ja] += step * dq
        clamp_limits(q, joints)
        env_project(q, {n.rsplit('_', 1)[0] for n in joints})
        if np.linalg.norm(e) < 1e-4: break
    return q


def head_frame(q):
    """Head position and axes in the world: forward, up, left (forward is the head's -y, up its z, left its x)."""
    Dd.qpos[:] = q; mujoco.mj_kinematics(M, Dd)
    b = BODY['Head']; R = Dd.xmat[b].reshape(3, 3)
    return Dd.xpos[b].copy(), -R[:, 1], R[:, 2], R[:, 0]


# ---------- the game's joint envelopes (sim/anat.js): hinge triples people reach, from 17,162 recorded moves ----------
from scipy.ndimage import distance_transform_edt
ENV = {}
for _b, _e in LIM.get('_env', {}).items():
    n = _e['n']; occ = np.zeros(n[0] * n[1] * n[2], np.uint8); p = 0; val = 0
    for r in _e['rle'].split(','):
        k = int(r)
        if val: occ[p:p + k] = 1
        p += k; val ^= 1
    occ = occ.reshape(n)
    ENV[_b] = dict(h=_e['h'], mg=_e['mg'], lo=np.array(_e['lo'], float), n=np.array(n), D=distance_transform_edt(occ == 0), qa=[QADR[_b + a] for a in ('_x', '_y', '_z')])


def env_depth(b, q):
    """degrees outside the envelope (<= 0 inside), and the unit way out in hinge space; as AnatGuard.depth"""
    E = ENV[b]; a = np.degrees(q[E['qa']]); u = (a - E['lo']) / E['h']; c = np.clip(u, 0, E['n'] - 1.000001)
    ex = (u - c) * E['h']; i = np.floor(c).astype(int); f = c - i; D = E['D']
    def at(d0, d1, d2): return D[i[0] + d0, i[1] + d1, i[2] + d2]
    x00 = at(0, 0, 0) + (at(0, 0, 1) - at(0, 0, 0)) * f[2]; x01 = at(0, 1, 0) + (at(0, 1, 1) - at(0, 1, 0)) * f[2]
    x10 = at(1, 0, 0) + (at(1, 0, 1) - at(1, 0, 0)) * f[2]; x11 = at(1, 1, 0) + (at(1, 1, 1) - at(1, 1, 0)) * f[2]
    y0 = x00 + (x01 - x00) * f[1]; y1 = x10 + (x11 - x10) * f[1]; v = (y0 + (y1 - y0) * f[0]) * E['h']
    g = np.array([y1 - y0, (1 - f[0]) * (x01 - x00) + f[0] * (x11 - x10),
                  (1 - f[0]) * ((1 - f[1]) * (at(0, 0, 1) - at(0, 0, 0)) + f[1] * (at(0, 1, 1) - at(0, 1, 0))) + f[0] * ((1 - f[1]) * (at(1, 0, 1) - at(1, 0, 0)) + f[1] * (at(1, 1, 1) - at(1, 1, 0)))])
    dep = v - E['mg']; e = np.linalg.norm(ex)
    if e > 0: dep += e; g = g + ex / e
    gn = np.linalg.norm(g); return dep, (g / gn if gn > 1e-9 else g)


def env_project(q, bodies, inside=-1.0, iters=40):
    """walk each ball joint back inside its envelope (to `inside` deg below the edge)"""
    for b in bodies:
        if b not in ENV: continue
        for _ in range(iters):
            dep, g = env_depth(b, q)
            if dep <= inside: break
            q[ENV[b]['qa']] -= np.radians(min(4.0, dep - inside + 0.5)) * g
    return q
