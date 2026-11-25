"""Makes the light scene's moves by hand, on top of real BONES sitting and stand-up takes.

Every move keeps a real person's hips and legs; the edits are joint offsets (head, neck, spine, shoulders) and arm
targets solved with IK, eased in and out over time. Output: .q files in the game's library format."""
import sys, numpy as np, mujoco
sys.path.insert(0, '/home/claude/work')
from mv import M, Dd, BODY, QADR, lib, encode, fk, HINGES
from ik import solve, chain, head_frame, LIM, lim

FPS = 30
BASE = 'sitting_legs_bend_arms_front_loop_R_001__A029'
STAND = 'sitting_legs_bend_arms_front_stop_R_003__A040'
DOWN = 'sitting_legs_bend_arms_front_start_R_001__A052'


# ---------- time curves ----------
def ss(x):
    x = np.clip(x, 0, 1); return x * x * (3 - 2 * x)


def ramp(t, t0, t1):
    """0 before t0, eased to 1 at t1"""
    return ss((t - t0) / max(1e-6, t1 - t0))


def bump(t, t0, t1, t2, t3):
    """eased up t0->t1, held, eased down t2->t3"""
    return ramp(t, t0, t1) * (1 - ramp(t, t2, t3))


def snap(t, t0, dur, over=0.12):
    """a fast startled move: reaches 1 in dur with a small overshoot, settles (critically damped spring feel)"""
    x = np.clip((t - t0) / dur, 0, None)
    return np.where(x <= 0, 0, 1 - np.exp(-5 * x) * (1 + 5 * x) + over * np.sin(np.pi * np.clip(x, 0, 1.6) / 1.6) * np.exp(-2 * x))


# ---------- pose offsets (radians, added to the base take) ----------
SLEEP = {'Spine2_x': 0.10, 'Chest_x': 0.22, 'Neck1_x': 0.55, 'Neck2_x': 0.12, 'Head_x': 0.35, 'Neck1_z': 0.06, 'Head_y': 0.10,
         'RightShoulder_z': -0.06, 'LeftShoulder_z': 0.06}


def offs(q, d, w=1.0):
    for k, v in d.items(): q[QADR[k]] += w * v


def lowest(q):
    Dd.qpos[:] = q; mujoco.mj_kinematics(M, Dd); lo = 9
    for g in range(M.ngeom):
        if M.geom_bodyid[g] == 0: continue
        t = M.geom_type[g]; R = Dd.geom_xmat[g].reshape(3, 3); s = M.geom_size[g]
        dz = (abs(R[2, 0]) * s[0] + abs(R[2, 1]) * s[1] + abs(R[2, 2]) * s[2]) if t == 6 else (abs(R[2, 2]) * s[1] + s[0]) if t == 3 else s[0]
        lo = min(lo, Dd.geom_xpos[g, 2] - dz)
    return lo


def floor_of(q):
    return np.percentile([lowest(x) for x in q[::3]], 2)


def yaw_of(qw):
    w, x, y, z = qw; R = np.zeros(9); mujoco.mju_quat2Mat(R, np.array([w, x, y, z])); R = R.reshape(3, 3)
    return np.arctan2(R[1, 0], R[0, 0])


def align(take, ref):
    """turn and shift take (T x 73) so its first frame's hips sit where ref's hips are, facing the same way"""
    out = take.copy(); dy = yaw_of(ref[3:7]) - yaw_of(take[0, 3:7]); c, s = np.cos(dy), np.sin(dy)
    qz = np.array([np.cos(dy / 2), 0, 0, np.sin(dy / 2)])
    p0 = take[0, :2].copy()
    for t in range(len(out)):
        dx, dyy = take[t, 0] - p0[0], take[t, 1] - p0[1]
        out[t, 0] = ref[0] + c * dx - s * dyy; out[t, 1] = ref[1] + s * dx + c * dyy
        r = np.zeros(4); mujoco.mju_mulQuat(r, qz, take[t, 3:7]); out[t, 3:7] = r
    return out


def blend_into(a, b, n):
    """last n frames of a cross-fade into the first n of b (b already aligned); returns joined"""
    n = min(n, len(a), len(b)); mid = []
    for i in range(n):
        w = ss((i + 1) / (n + 1)); x = a[len(a) - n + i].copy(); y = b[i]
        x[:3] = (1 - w) * x[:3] + w * y[:3]; x[7:] = (1 - w) * x[7:] + w * y[7:]
        r = np.zeros(4); qa, qb = x[3:7].copy(), y[3:7].copy()
        if np.dot(qa, qb) < 0: qb = -qb
        r = (1 - w) * qa + w * qb; x[3:7] = r / np.linalg.norm(r); mid.append(x)
    return np.vstack([a[:len(a) - n], np.array(mid), b[n:]])


def resample(q, speed):
    T = len(q); n = int(round((T - 1) / speed)) + 1; src = np.linspace(0, T - 1, n); out = np.empty((n, q.shape[1]))
    for i, s in enumerate(src):
        a = int(np.floor(s)); b = min(T - 1, a + 1); w = s - a; out[i] = (1 - w) * q[a] + w * q[b]
        out[i, 3:7] /= np.linalg.norm(out[i, 3:7])
    return out


# ---------- arms: hands to the face ----------
RARM = chain('RightShoulder', 'RightArm', 'RightForeArm', 'RightHand')
LARM = chain('LeftShoulder', 'LeftArm', 'LeftForeArm', 'LeftHand')


def cover_targets(q, spread=0.045, fwd=0.09, down=0.07):
    """wrists in front of the jaw, fingers up over the eyes, thumbs out along the cheeks"""
    h, f, u, l = head_frame(q)
    c = h + f * fwd - u * down
    return [dict(body='RightHand', pos=c - l * spread, weight=1.0),
            dict(body='RightHand', axis=[-1, 0, 0], dir=u * 0.9 + l * 0.25 + f * 0.1, weight=0.35),
            dict(body='RightHand', axis=[0, -1, 0], dir=-l * 0.8 - u * 0.5, weight=0.2),
            dict(body='LeftHand', pos=c + l * spread, weight=1.0),
            dict(body='LeftHand', axis=[1, 0, 0], dir=u * 0.9 - l * 0.25 + f * 0.1, weight=0.35),
            dict(body='LeftHand', axis=[0, -1, 0], dir=l * 0.8 - u * 0.5, weight=0.2)]


def rub_targets(q, phase):
    """right hand rubbing the right eye: knuckles at the eye, a small circle"""
    h, f, u, l = head_frame(q)
    c = h + f * 0.085 + u * 0.005 - l * 0.03 + 0.012 * (np.cos(phase) * l + np.sin(phase) * u)
    return [dict(body='RightHand', pos=c - u * 0.07, weight=1.0),
            dict(body='RightHand', axis=[-1, 0, 0], dir=u * 0.8 + l * 0.5, weight=0.3)]


def arm_ik(q, tasks, joints):
    return solve(q, tasks, joints, iters=80, damp=0.02, reg=0.004)


def hand_track(q, w, targets_fn, joints, hands):
    """hands travel in a straight-ish line from where the take has them to the targets (w: 0..1 per frame); each
    frame's IK starts from the frame before, so the arms move smoothly"""
    prev = None
    for i in range(len(q)):
        if w[i] <= 0.001: prev = None; continue
        tg = targets_fn(i)
        base = {}
        for h in hands: base[h] = fk(q[i]).xpos[BODY[h]].copy()
        tasks = []
        for x in tg:
            x = dict(x)
            if 'pos' in x: x['pos'] = (1 - w[i]) * base[x['body']] + w[i] * np.array(x['pos'])
            else: x['weight'] = x.get('weight', 1) * w[i]
            tasks.append(x)
        seed = q[i].copy()
        if prev is not None:
            for n in joints: seed[QADR[n]] = prev[QADR[n]]
        sol = solve(seed, tasks, joints, iters=60, damp=0.02, reg=0.01)
        q[i] = mix_joints(q[i], sol, joints, 1.0); prev = sol


def mix_joints(a, b, joints, w):
    out = a.copy()
    for n in joints: out[QADR[n]] = (1 - w) * a[QADR[n]] + w * b[QADR[n]]
    return out


def base_frames(n, start=0):
    B = lib(BASE); idx = (start + np.arange(n)) % len(B); return B[idx].copy()


def breathe(q, t, amp=1.0, f=0.22):
    s = np.sin(2 * np.pi * f * t)
    q[:, QADR['Spine2_x']] -= 0.018 * amp * s; q[:, QADR['Chest_x']] -= 0.022 * amp * s
    q[:, QADR['RightShoulder_z']] -= 0.015 * amp * s; q[:, QADR['LeftShoulder_z']] += 0.015 * amp * s
    q[:, QADR['Neck1_x']] += 0.012 * amp * s


def sleep_pose(q, t, w):
    for i in range(len(q)): offs(q[i], SLEEP, float(w[i]) if np.ndim(w) else w)


# ---------- the moves ----------
def m_sleep(dur=14.0, start=60):
    n = int(dur * FPS); t = np.arange(n) / FPS; q = base_frames(n, start)
    sleep_pose(q, t, 1.0); breathe(q, t, 1.0)
    # head sways a little with the breath, very slowly
    q[:, QADR['Neck1_z']] += 0.03 * np.sin(2 * np.pi * 0.05 * t)
    # loop: the last 1.5 s fade into the first frame, so the next pick of this move starts where it ended
    k = int(1.5 * FPS)
    for i in range(k):
        w = ss((i + 1) / k); a = q[n - k + i]; a[7:] = (1 - w) * a[7:] + w * q[0, 7:]
    return q


def m_fall_asleep(dur=7.0, start=200):
    n = int(dur * FPS); t = np.arange(n) / FPS; q = base_frames(n, start)
    w = ramp(t, 1.0, 5.5)
    # the head nods down, comes back up once (fighting it), then drops
    nod = bump(t, 2.0, 2.8, 3.0, 3.4) * 0.6
    ww = np.clip(w + 0.0, 0, 1); ww = np.maximum(ww - 0.35 * bump(t, 3.0, 3.3, 3.5, 4.2), 0)
    sleep_pose(q, t, np.maximum(ww, nod * 0.8)); breathe(q, t, ramp(t, 1, 5))
    return q


IDLE = 'idle_to_idle_right_R_002__A032'
ARMS = [j for j in RARM + LARM if 'ForeArm_x' not in j and 'ForeArm_y' not in j]   # real people hardly use these two
RARM1 = [j for j in ARMS if j.startswith('Right')]


def standing_after(q, S, dur, blend=18):
    """q (sitting part) -> stand-up take S -> a calm standing take for dur seconds; returns (joined, index where standing starts)"""
    S = align(S, q[-1]); j = blend_into(q, S, blend)
    I = lib(IDLE); n = int(dur * FPS); I = np.vstack([I] * (n // len(I) + 1))[:n + 15]
    I = align(I, j[-1]); k = len(j) - 15
    return blend_into(j, I, 15), k


def m_wake(start=60):
    """asleep; stirs, a deep breath, head rises slowly, glances up at the lamp; stands up (real take); standing, rubs
    his eye, rolls his shoulders, then is still"""
    dur = 7.5; n = int(dur * FPS); t = np.arange(n) / FPS; q = base_frames(n, start)
    awake = ramp(t, 1.6, 4.4)
    sleep_pose(q, t, 1 - awake); breathe(q, t, 1.0 - 0.4 * awake)
    db = bump(t, 2.2, 3.4, 3.6, 4.8)
    for k, v in {'Chest_x': -0.12, 'Spine2_x': -0.06, 'RightShoulder_z': 0.05, 'LeftShoulder_z': -0.05, 'Neck1_x': -0.08}.items():
        q[:, QADR[k]] += v * db
    up = bump(t, 5.0, 5.8, 6.3, 7.0)
    for k, v in {'Neck1_x': -0.35, 'Head_x': -0.3}.items(): q[:, QADR[k]] += v * up
    j, k0 = standing_after(q, lib(STAND), 6.0)
    t2 = (np.arange(len(j)) - k0) / FPS
    wr = bump(t2, 0.6, 1.2, 2.4, 3.0)
    hand_track(j, wr, lambda i: rub_targets(j[i], 2 * np.pi * 2.2 * t2[i]), RARM1, ['RightHand'])
    sh = bump(t2, 3.3, 3.8, 4.2, 4.8)   # shoulders roll back once
    for k, v in {'RightShoulder_z': 0.12, 'LeftShoulder_z': -0.12, 'RightShoulder_x': -0.08, 'LeftShoulder_x': -0.08, 'Chest_x': -0.06}.items(): j[:, QADR[k]] += v * sh
    return j


def m_startle(start=60):
    """asleep; the light jolts him awake; he scrambles up (real take, 1.3x); standing, his hands fly to his face, head
    down and turned away from the lamp; holds, trembling; peeks over his hands; squints up; lowers them"""
    dur = 0.75; n = int(dur * FPS); t = np.arange(n) / FPS; q = base_frames(n, start)
    jolt = snap(t, 0.08, 0.22)
    sleep_pose(q, t, 1 - np.clip(jolt, 0, 1)); breathe(q, t, 1 - np.clip(jolt, 0, 1))
    for k, v in {'Neck1_x': -0.28, 'Head_x': -0.22, 'Chest_x': -0.20, 'Spine2_x': -0.10, 'RightShoulder_z': 0.10, 'LeftShoulder_z': -0.10}.items():
        q[:, QADR[k]] += v * jolt
    j, k0 = standing_after(q, resample(lib(STAND), 1.3), 6.0, blend=8)
    t2 = (np.arange(len(j)) - k0) / FPS
    hold = 1 - ramp(t2, 3.6, 4.6)
    curl = ramp(t2, 0.0, 0.35) * hold
    C = {'Chest_x': 0.16, 'Spine2_x': 0.06, 'Neck1_x': 0.30, 'Head_x': 0.20, 'Neck1_z': 0.25, 'Head_z': 0.15, 'Head_y': 0.08,
         'RightShoulder_z': 0.08, 'LeftShoulder_z': -0.08, 'RightLeg_x': -0.08, 'LeftLeg_x': -0.08, 'RightShin_x': 0.14, 'LeftShin_x': 0.14}
    for k, v in C.items(): j[:, QADR[k]] += v * curl
    rng = np.random.default_rng(7); N = len(j)
    def noise(a):
        x = np.cumsum(rng.normal(0, 1, N)); x -= np.convolve(x, np.ones(9) / 9, 'same'); return a * x / (np.abs(x).max() + 1e-9)
    tr = ramp(t2, 0.4, 0.7) * hold
    for k, a in {'Neck1_z': 0.02, 'Neck1_x': 0.015, 'RightShoulder_z': 0.015, 'LeftShoulder_z': 0.015, 'Chest_x': 0.01}.items():
        j[:, QADR[k]] += tr * noise(a)
    cov = np.clip(snap(t2, -0.05, 0.30, over=0.05), 0, 1) * (1 - ramp(t2, 3.7, 4.8))
    peek = bump(t2, 2.6, 3.1, 3.5, 4.0)
    hand_track(j, cov, lambda i: cover_targets(j[i], down=0.07 + 0.09 * peek[i], fwd=0.09 + 0.03 * peek[i]), ARMS, ['RightHand', 'LeftHand'])
    sq = bump(t2, 3.9, 4.4, 4.8, 5.4)
    for k, v in {'Neck1_x': -0.18, 'Head_x': -0.14}.items(): j[:, QADR[k]] += v * sq
    return j


def check(q, name):
    bad = []
    for j in HINGES:
        lo, hi = np.radians(LIM.get(j, [-180, 180])); v = q[:, QADR[j]]
        if v.min() < lo - 1e-3 or v.max() > hi + 1e-3: bad.append((j, round(np.degrees(v.min()), 1), round(np.degrees(v.max()), 1), LIM.get(j)))
    fl = floor_of(q)
    hh = np.array([fk(x).xpos[BODY['Head'], 2] for x in q]) - fl
    hips = np.array([x[2] for x in q]) - fl
    print(f'{name}: {len(q)} frames {len(q)/FPS:.1f}s, head {hh.min():.2f}-{hh.max():.2f} m, hips {hips.min():.2f}-{hips.max():.2f} m, joints outside limits: {bad or "none"}')
    return bad


def m_sit_down_sleep():
    """a real person sits down on the floor (BONES take), then nods off: ends in the sleep pose"""
    Dn = lib(DOWN); fa = m_fall_asleep(8.0, 200)
    fa = align(fa, Dn[-1])
    return blend_into(Dn, fa, 20)


MOVES = {'light_sit_down_sleep_001': m_sit_down_sleep, 'light_sleep_sit_001': m_sleep, 'light_fall_asleep_001': m_fall_asleep, 'light_wake_slow_001': m_wake, 'light_startle_cover_001': m_startle}

if __name__ == '__main__':
    import os
    out = sys.argv[1] if len(sys.argv) > 1 else '/home/claude/work/out'
    os.makedirs(out, exist_ok=True)
    only = sys.argv[2:] or list(MOVES)
    for name in only:
        q = MOVES[name]()
        for j in HINGES:
            lo, hi = np.radians(LIM.get(j, [-180, 180])); q[:, QADR[j]] = np.clip(q[:, QADR[j]], lo, hi)
        check(q, name)
        from ik import ENV, env_depth
        w = {}
        for f in range(0, len(q), 3):
            for b in ENV:
                dd, _ = env_depth(b, q[f])
                if dd > 1 and dd > w.get(b, (0,))[0]: w[b] = (round(float(dd), 1), f)
        print('   envelope (deg outside, frame):', w); np.save(f'{out}/{name}.npy', q); encode(q, f'{out}/{name}.q')
