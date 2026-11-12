# body/soma23_humanoid.xml

The SOMA 23-body humanoid, copied unchanged from NVIDIA ProtoMotions
(https://github.com/NVlabs/ProtoMotions, `protomotions/data/assets/mjcf/soma23_humanoid.xml`,
commit 13de0cccd06f53ce5c9cf9ec7d1932c7da73610f, Nov 9 2026).

Copyright (c) 2025-2026 NVIDIA CORPORATION & AFFILIATES. Licensed under the Apache License, Version 2.0
(https://www.apache.org/licenses/LICENSE-2.0).

How The Box uses it (sim/physics.js), the same way ProtoMotions' MuJoCo backend does:
- the motors become PD position servos: kp/kd = each joint's `stiffness`/`damping`, gear 1, the joints' passive
  springs zeroed; torque limits 300 N·m (spine, neck, head, legs) and 150 N·m (shoulders, arms, hands)
- implicitfast integrator; the box's walls are added as mocap bodies, nothing in the body is changed
- note: with its density-1000 shapes the body weighs 42.5 kg (as shipped; kept so NVIDIA's trained models match)
