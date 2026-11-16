# Compares skel.js body positions with MuJoCo's for the same poses (run after: node test.mjs --dump /tmp/fk.json)
import json, sys, numpy as np, mujoco
m = mujoco.MjModel.from_xml_path(sys.argv[2]); d = mujoco.MjData(m); J = json.load(open(sys.argv[1])); worst = 0
print('nq', m.nq, 'bodies', m.nbody)
for f in J['fr']:
    d.qpos[:] = f['q']; mujoco.mj_kinematics(m, d)
    for n, p in zip(J['names'], f['p']):
        worst = max(worst, float(np.abs(d.body(n).xpos - np.array(p)).max()))
print('worst difference %.6f mm' % (worst * 1000))
