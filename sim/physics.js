// The Box - physics layer (MuJoCo). Shared by the page and the Node tests.
//
// Rules (claude/plan-humanoid-body-nov12.md):
//  - the body comes from an MJCF file (body/soma23_humanoid.xml); swapping bodies = swapping that file + GAINS below
//  - real units: metres, kilograms, seconds, g = 9.81. The box is sized around the person, never the person to the box
//  - one call turns the phone's tilt into gravity: setGravity(vec)
//  - joints are PD position servos with torque limits (what trained policies output), as ProtoMotions sets them up:
//    kp/kd from the MJCF joints' stiffness/damping, the joints' own passive springs zeroed, gear 1
//  - nothing pushes the body except its actuators and contacts (and your finger, when you touch him)
//
// MuJoCo frame: z up, x to the right of the screen, y into the box (away from you). The glass is the plane y = 0.

export const BODY_FILE = 'body/soma23_humanoid.xml';

// torque limits by joint (ProtoMotions robot_configs/soma23.py, override_control_info)
export const EFFORT = [
  [/^(Spine|Chest|Neck|Head)/, 300],
  [/(Shoulder|Arm|Hand)_/, 150],
  [/(Leg|Shin|Foot|ToeBase)_/, 300],
];

export const PHYS = {
  dt: 1 / 480,          // physics step (ProtoMotions uses 960 Hz with implicitfast; 480 holds the PD gains stable too)
  controlHz: 30,        // how often control(obs) is asked for new targets (ProtoMotions soma23: 30 Hz; a controller
                        // can ask for more with control.hz)
  friction: 1.0,        // the body's sliding friction (ProtoMotions default_robot_friction)
  wallFriction: 0.9,
  T: 0.5,               // wall slab half-thickness
};

// walls: thick static slabs, sized once (huge in their plane) and moved when the box changes shape
const WALLS = ['glass', 'back', 'left', 'right', 'floor', 'ceil'];
function wallXML() {
  const T = PHYS.T, L = 20, rgba = 'rgba="0 0 0 0"';
  // each wall is a mocap body (moved by us, never by the physics), so the box can change shape while he is in it
  const g = (n, sx, sy, sz) => `<body name="wall_${n}" mocap="true"><geom name="wall_${n}" type="box" size="${sx} ${sy} ${sz}" ${rgba} friction="${PHYS.wallFriction} 0.005 0.0001" contype="1" conaffinity="7" condim="3"/></body>`;
  return [g('glass', L, T, L), g('back', L, T, L), g('left', T, L, L), g('right', T, L, L), g('floor', L, L, T), g('ceil', L, L, T)].join('\n');
}

// the scene: the body file as it is, the walls added, real gravity, the step
export function sceneXML(bodyXml) {
  let x = bodyXml;
  // drop the file's own light and floor-less extras we don't use; keep everything about the body
  x = x.replace(/<light[^>]*\/>/g, '');
  x = x.replace('</worldbody>', wallXML() + '\n</worldbody>');
  // the neck's capsule sits inside the collarbones' at rest: let them overlap (they'd shove each other forever)
  x = x.replace('<contact>', `<contact>
      <exclude body1='Neck1' body2='RightShoulder'/>
      <exclude body1='Neck1' body2='LeftShoulder'/>`);
  const opt = `<option timestep="${PHYS.dt}" gravity="0 0 -9.81" integrator="implicitfast"/>`;
  x = x.replace(/<compiler([^>]*)\/>/, m => m + '\n  ' + opt);
  return x;
}

// After compiling: the motors become PD position servos (exactly ProtoMotions' _configure_actuators_for_pd)
export function makeServos(mj, m) {
  const nu = m.nu, info = [];
  for (let a = 0; a < nu; a++) {
    const j = m.actuator_trnid[2 * a];
    const name = mj.mj_id2name(m, mj.mjtObj.mjOBJ_JOINT.value, j);
    const dof = m.jnt_dofadr[j], qadr = m.jnt_qposadr[j];
    const kp = m.jnt_stiffness[j], kd = m.dof_damping[dof];
    let effort = 300; for (const [re, e] of EFFORT) if (re.test(name)) { effort = e; break; }
    // force = kp * (ctrl - q) - kd * qd, limited to +-effort
    m.actuator_gear[6 * a] = 1;
    m.actuator_gainprm[10 * a] = kp;
    m.actuator_biastype[a] = 1;   // affine
    m.actuator_biasprm[10 * a] = 0; m.actuator_biasprm[10 * a + 1] = -kp; m.actuator_biasprm[10 * a + 2] = -kd;
    m.actuator_forcerange[2 * a] = -effort; m.actuator_forcerange[2 * a + 1] = effort;
    m.actuator_forcelimited[a] = 1; m.actuator_ctrllimited[a] = 0;
    info.push({ name, joint: j, dof, qadr, kp, kd, effort });
  }
  // the joints' passive springs and damping go: the servos are the only muscles
  for (let j = 0; j < m.njnt; j++) m.jnt_stiffness[j] = 0;
  for (let v = 0; v < m.nv; v++) m.dof_damping[v] = 0;
  return info;
}

export class BoxWorld {
  constructor(mj, bodyXml) {
    this.mj = mj;
    this.m = mj.MjModel.from_xml_string(sceneXML(bodyXml));
    this.d = new mj.MjData(this.m);
    const m = this.m;
    this.act = makeServos(mj, m);
    const ob = mj.mjtObj;
    this.wall = {}; this.wallMocap = {};
    for (const n of WALLS) {
      this.wall[n] = mj.mj_name2id(m, ob.mjOBJ_GEOM.value, 'wall_' + n);
      this.wallMocap[n] = m.body_mocapid[mj.mj_name2id(m, ob.mjOBJ_BODY.value, 'wall_' + n)];
    }
    // the figure's bodies: everything that isn't a wall
    this.bodyNames = []; this.bodyId = {};
    for (let b = 1; b < m.nbody; b++) { const n = mj.mj_id2name(m, ob.mjOBJ_BODY.value, b); if (!n.startsWith('wall_')) { this.bodyNames.push(n); this.bodyId[n] = b; } }
    this.isFig = new Uint8Array(m.nbody); for (const n of this.bodyNames) this.isFig[this.bodyId[n]] = 1;
    // the body's own geoms get its friction
    for (let g = 0; g < m.ngeom; g++) if (this.isFig[m.geom_bodyid[g]]) m.geom_friction[3 * g] = PHYS.friction;
    this.mass = 0; for (const n of this.bodyNames) this.mass += m.body_mass[this.bodyId[n]];
    this.ctrl = new Float64Array(m.nu);
    this.box = { w: 1, h: 2.4, D: 1.3 };
    this.tControl = 0;
  }
  // the inner box, metres: w wide, h high, D deep. Glass at y = 0, back at y = D, floor at z = -h/2.
  setBox(w, h, D) {
    const T = PHYS.T;
    this.box = { w, h, D };
    this.wallAt = { glass: [0, -T, 0], back: [0, D + T, 0], left: [-w / 2 - T, D / 2, 0], right: [w / 2 + T, D / 2, 0],
      floor: [0, D / 2, -h / 2 - T], ceil: [0, D / 2, h / 2 + T] };
    this.placeWalls();
  }
  placeWalls() {
    const mp = this.d.mocap_pos, far = 18;
    for (const n of WALLS) {
      const k = this.wallMocap[n], p = this.wallAt[n].slice();
      // open floor: every wall but the floor goes far away (the floor slab is 40 m across, centred under him)
      if (this.open && n !== 'floor') { if (n === 'ceil') p[2] += far; else if (n === 'glass') p[1] -= far; else if (n === 'back') p[1] += far; else p[0] += Math.sign(p[0]) * far; }
      mp[3 * k] = p[0]; mp[3 * k + 1] = p[1]; mp[3 * k + 2] = p[2];
    }
  }
  // open floor (true) or the box (false); he stays where he is
  setOpen(on) { this.open = !!on; this.placeWalls(); }
  setGravity(v) { const g = this.m.opt.gravity; g[0] = v[0]; g[1] = v[1]; g[2] = v[2]; }
  // stand him up: on the floor, mid-depth, facing the glass (the body's forward is -y: toward you)
  standAt(q) {
    const m = this.m, d = this.d, mj = this.mj;
    mj.mj_resetData(m, d);
    this.tControl = 0;     // the clock restarts: ask the controller again right away
    this.placeWalls();
    const z0 = -this.box.h / 2 + 0.95;   // ProtoMotions soma23 default_root_height
    d.qpos[0] = 0; d.qpos[1] = this.box.D / 2; d.qpos[2] = z0;
    d.qpos[3] = 1; d.qpos[4] = 0; d.qpos[5] = 0; d.qpos[6] = 0;
    if (q) for (const a of this.act) d.qpos[a.qadr] = q[a.name] || 0;
    for (const a of this.act) this.ctrl[this.act.indexOf(a)] = d.qpos[a.qadr];
    mj.mj_forward(m, d);
    // feet on the floor: lift or lower him so his lowest point just touches it
    d.qpos[2] += (-this.box.h / 2 + 0.002) - this.lowestZ();
    mj.mj_forward(m, d);
  }
  // the lowest point of the body (boxes by their corners, capsules by their ends, spheres by their bottoms)
  lowestZ() {
    const m = this.m, d = this.d; let lo = 1e9;
    for (let g = 0; g < m.ngeom; g++) {
      if (!this.isFig[m.geom_bodyid[g]]) continue;
      const t = m.geom_type[g], c = 3 * g, R = d.geom_xmat.subarray(9 * g, 9 * g + 9), sz = m.geom_size;
      const z = d.geom_xpos[c + 2];
      if (t === 6) {        // box: z of each corner
        let dz = 0; for (let k = 0; k < 3; k++) dz += Math.abs(R[6 + k]) * sz[c + k];
        lo = Math.min(lo, z - dz);
      } else if (t === 3) { // capsule: along its local z, half-length size[1]
        lo = Math.min(lo, z - Math.abs(R[8]) * sz[c + 1] - sz[c]);
      } else lo = Math.min(lo, z - sz[c]);
    }
    return lo;
  }
  // what a controller sees (kept small and named; a trained policy builds its own vector from this)
  obs() {
    const m = this.m, d = this.d;
    return { t: d.time, qpos: d.qpos, qvel: d.qvel, xpos: d.xpos, xquat: d.xquat, cvel: d.cvel, gravity: m.opt.gravity,
      act: this.act, box: this.box, ncon: d.ncon };
  }
  // one physics step with the last targets; control(obs) is called at controlHz
  step(control) {
    const d = this.d, mj = this.mj, m = this.m;
    if (d.time >= this.tControl) {
      const tg = control(this.obs());
      if (tg) for (let i = 0; i < m.nu; i++) this.ctrl[i] = tg[i];
      this.tControl = d.time + 1 / (control.hz || PHYS.controlHz) - 1e-9;
    }
    d.ctrl.set(this.ctrl);
    mj.mj_step(m, d);
  }
}
