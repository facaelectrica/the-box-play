// Test poses for look mode (user, 12:43: "a sitting pose, a star pose, a bending to pick up something pose ... so we
// can test how it looks in the extremes"). Joint values are SOMA hinge angles in radians (`<Body>_<x|y|z>`); a pose
// gives the right side and the middle, and the left side is mirrored (left = -right on every axis, as in BRAIN_STAND).
// root: the whole body's lean, degrees forward (pitch) and to his left (roll). Drawing only: physics is paused in look mode.
//
// Signs (T-pose frame: x = his left, -y = forward, z = up):
//   spine/neck _x > 0 bends forward; Right*Leg_x < 0 brings the thigh forward; Right*Shin_x > 0 bends the knee;
//   RightArm_y < 0 lowers the arm from the T-pose, > 0 raises it; RightArm_z > 0 swings it forward;
//   RightForeArm_z > 0 bends the elbow forward; RightLeg_y > 0 spreads the leg outward; RightFoot_x < 0 points the toes up.

function mirror(right) {
  const out = {};
  for (const [k, v] of Object.entries(right)) {
    out[k] = v;
    if (k.startsWith('Right')) out['Left' + k.slice(5)] = -v;
  }
  return out;
}

export const POSES = [
  { id: 'stand', label: 'Standing', root: [0, 0], joints: null },   // null = the brain's standing pose
  { id: 'sit', label: 'Sitting (on an unseen chair)', root: [0, 0], joints: mirror({
      RightLeg_x: -1.57, RightShin_x: 1.57, RightFoot_x: 0.0,
      RightArm_y: -1.2, RightArm_z: 0.35, RightForeArm_z: 0.9,
      Spine1_x: 0.05, Neck1_x: 0.1 }) },
  { id: 'star', label: 'Star', root: [0, 0], joints: mirror({
      RightArm_y: 0.7, RightLeg_y: 0.42, RightFoot_y: -0.2, Neck1_x: -0.1 }) },
  { id: 'pick', label: 'Bending to pick something up', root: [62, 0], joints: mirror({
      RightLeg_x: -1.32, RightShin_x: 0.45, RightFoot_x: -0.22,
      Spine1_x: 0.25, Spine2_x: 0.25, Chest_x: 0.2, Neck1_x: -0.25, Head_x: -0.2,
      RightArm_y: -1.25, RightArm_z: 1.2, RightForeArm_z: 0.2 }) },
  { id: 'squat', label: 'Deep squat', root: [20, 0], joints: mirror({
      RightLeg_x: -2.1, RightLeg_y: 0.25, RightShin_x: 2.3, RightFoot_x: -0.55,
      Spine1_x: 0.15, Spine2_x: 0.1, Neck1_x: -0.3,
      RightArm_y: -0.2, RightArm_z: 1.3 }) },
  { id: 'lunge', label: 'Lunge', root: [0, 0], joints: {
      RightLeg_x: -1.35, RightShin_x: 1.45, RightFoot_x: 0.0,
      LeftLeg_x: -0.35, LeftShin_x: -1.2, LeftFoot_x: -0.5, LeftToeBase_x: 0.6,
      RightArm_y: -1.2, LeftArm_y: 1.2, RightArm_z: 0.5, LeftArm_z: -0.5, RightForeArm_z: 0.5, LeftForeArm_z: -0.5 } },
  { id: 'reach', label: 'Reaching up, twisting', root: [0, 0], joints: {
      RightArm_y: 1.45, LeftArm_y: -1.3, RightForeArm_z: 0.15, LeftForeArm_z: -0.3,
      Spine1_z: 0.25, Spine2_z: 0.25, Chest_z: 0.2, Spine1_y: -0.1, Spine2_y: -0.1, Chest_y: -0.1, Neck1_x: -0.35 } },
];

// put the body in a pose: joints, then the whole body's lean, then feet (or whatever is lowest) on the floor
export function applyPose(W, mj, pose, standPose) {
  const d = W.d;
  W.standAt(pose.joints || standPose);
  const [pitch, roll] = (pose.root || [0, 0]).map(a => a * Math.PI / 180);
  // forward lean = rotation about +x (up goes to -y); roll = about -y
  const qx = [Math.cos(pitch / 2), Math.sin(pitch / 2), 0, 0], qy = [Math.cos(roll / 2), 0, -Math.sin(roll / 2), 0];
  const m = (a, b) => [a[0] * b[0] - a[1] * b[1] - a[2] * b[2] - a[3] * b[3], a[0] * b[1] + a[1] * b[0] + a[2] * b[3] - a[3] * b[2],
                       a[0] * b[2] - a[1] * b[3] + a[2] * b[0] + a[3] * b[1], a[0] * b[3] + a[1] * b[2] - a[2] * b[1] + a[3] * b[0]];
  const q = m(qy, qx);
  d.qpos[3] = q[0]; d.qpos[4] = q[1]; d.qpos[5] = q[2]; d.qpos[6] = q[3];
  mj.mj_forward(W.m, d);
  d.qpos[2] += (-W.box.h / 2 + 0.002) - W.lowestZ();
  mj.mj_forward(W.m, d);
}
