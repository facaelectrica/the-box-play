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

# body/soma_tracker.onnx, body/soma_tracker.json (v15+)

Licensed by NVIDIA Corporation under the NVIDIA Open Model License (`body/soma_tracker_LICENSE.pdf`).

NVIDIA ProtoMotions' pretrained SOMA motion tracker (`data/pretrained_models/motion_tracker/soma-bones/
inference_last_lab.ckpt`, the IsaacLab fine-tune; PPO, 6 x 1024 MLP). Only the actor network is used, exported to
ONNX with float16 weights by `tools/brain/export_onnx.py` (no other change); `soma_tracker.json` holds the PD gains
and action scale from the same checkpoint's `resolved_configs.json`. ProtoMotions source code is Apache-2.0; the
model was trained on BONES-SEED motion data, whose own terms are not granted by the model licence.

# body/doll.glb (v15.6+)

The figure's look: the female doll from "Ball Joint Doll Rigged Basemesh (male ver)" supplied by the project's owner
(source and licence to be recorded here). Converted by `tools/look/doll_build.py` (parts fitted onto the SOMA body,
triangles reduced); drawing only.

## Figure models (Options > Figure)
- `bodychan.glb` (Body-chan) and `bodykun.glb` (Body-kun): built from "BodyChan - BodyKun | Posable Drawing Figures" by vinchau
  (https://blendswap.com/blend/23521), CC0. Parts moved onto the SOMA skeleton by `tools/look/bodychan_build.py`.
- `hbm_female.glb`: built from Blender Studio's "Human Base Meshes" bundle v1.4.1 (realistic female), CC0
  (https://www.blender.org/download/demo-files/). Cut into parts at the SOMA joints by `tools/look/hbm_build.py`.
