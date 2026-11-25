"""Draw moves with MuJoCo (EGL): contact sheets and mp4 previews."""
import os
os.environ.setdefault('MUJOCO_GL', 'egl')
import numpy as np, mujoco, imageio
from mv import M, Dd, BODY

_R = {}


def renderer(w, h):
    if (w, h) not in _R: _R[(w, h)] = mujoco.Renderer(M, h, w)
    return _R[(w, h)]


def frame(qpos, az=90, el=-8, dist=2.6, w=320, h=320, look=None):
    """az 90 = looking along -y (at a figure facing +y); the camera follows the hips unless look is given."""
    r = renderer(w, h)
    Dd.qpos[:] = qpos; mujoco.mj_forward(M, Dd)
    cam = mujoco.MjvCamera(); cam.azimuth = az; cam.elevation = el; cam.distance = dist
    p = Dd.xpos[BODY['Hips']].copy() if look is None else np.array(look)
    if look is None: p[2] = 0.55
    cam.lookat[:] = p
    r.update_scene(Dd, cam)
    return r.render()


def sheet(q, frames, path, views=((90, -8),), **kw):
    rows = []
    for az, el in views:
        rows.append(np.concatenate([frame(q[f], az, el, **kw) for f in frames], 1))
    imageio.imwrite(path, np.concatenate(rows, 0))


def video(q, path, step=1, views=((90, -8), (35, -12)), **kw):
    W = imageio.get_writer(path, fps=30 // step, codec='libx264', quality=7, macro_block_size=16)
    for f in range(0, len(q), step):
        W.append_data(np.concatenate([frame(q[f], az, el, **kw) for az, el in views], 1))
    W.close()
