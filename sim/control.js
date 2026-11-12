// The Box - the controller slot: control(obs) -> joint targets (radians, one per actuator).
//
// Phase 1 (this file): a hand-written stand. A pose held by the servos, and the ankles as real muscles: each foot's
// servo target is set so the servo makes the torque that keeps the weight over the feet (gravity's pull on the body
// plus a spring and damper on where the weight is and where it is going). The servos are the only thing acting.
// Phase 2: a trained policy (ONNX Runtime Web) returns the same array from the same obs.

// the standing pose, as offsets from the body file's rest pose (a T-pose): arms down by the sides, soft knees
export const STAND = {
  RightArm_y: -1.25, LeftArm_y: 1.25,
  RightForeArm_z: 0.15, LeftForeArm_z: -0.15,
};

export const BAL = {
  kp: 2.2,      // weight-over-feet spring, x (m g) per metre of error
  kd: 0.9,      // damper, x (m g) per m/s
  sign: 1,
  ahead: 0.035, // where the weight should sit: this far in front of the ankles (m)
};

export function makeController(W) {
  const m = W.m, d = W.d, mj = W.mj;
  const id = W.bodyId, H = id.Hips;
  const act = W.act, nu = m.nu;
  const byName = {}; act.forEach((a, i) => byName[a.name] = i);
  const out = new Float64Array(nu);
  const g = 9.81, M = W.mass;
  const ankles = ['RightFoot', 'LeftFoot'];
  let prevCom = null, prevT = 0;
  return function control(obs) {
    for (let i = 0; i < nu; i++) out[i] = STAND[act[i].name] || 0;
    // where the weight is and where it's going
    const com = [d.subtree_com[3 * H], d.subtree_com[3 * H + 1], d.subtree_com[3 * H + 2]];
    let vel = [0, 0];
    if (prevCom && d.time > prevT) { const dt = d.time - prevT; vel = [(com[0] - prevCom[0]) / dt, (com[1] - prevCom[1]) / dt]; }
    prevCom = com; prevT = d.time;
    // the ankles' middle, and the spot the weight should be over (a little in front of them: forward is -y)
    let ax = 0, ay = 0;
    for (const f of ankles) { ax += d.xpos[3 * id[f]] / 2; ay += d.xpos[3 * id[f] + 1] / 2; }
    const tx = ax, ty = ay - BAL.ahead;
    // torque the floor must give the body about x (pitch) and y (roll), shared by the two ankles
    const eY = com[1] - ty, eX = com[0] - tx;
    const TX = M * g * (-(com[1] - ay) - BAL.kp * eY - BAL.kd * vel[1]);   // + about x tips the body forward (-y)
    const TY = M * g * ((com[0] - ax) + BAL.kp * eX + BAL.kd * vel[0]);   // + about y tips the body to +x
    for (const f of ankles) {
      for (const [ax_, T] of [['_x', TX], ['_y', TY]]) {
        const i = byName[f + ax_]; if (i === undefined) continue;
        const a = act[i], j = a.joint;
        const axis = [d.xaxis[3 * j], d.xaxis[3 * j + 1], d.xaxis[3 * j + 2]];
        // the joint pushes +tau along its axis on the foot, -tau on the shin; the body needs -T about the world axis
        const comp = ax_ === '_x' ? axis[0] : axis[1];
        if (Math.abs(comp) < 0.3) continue;
        const tau = BAL.sign * (T / 2) / comp;
        const q = d.qpos[a.qadr], qd = d.qvel[a.dof];
        out[i] = q + (tau + a.kd * qd) / a.kp;
      }
    }
    return out;
  };
}
