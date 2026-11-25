# Light scene tools (Nov 24)

- `gen.py`: makes the light scene's moves (`gpc/clips/light_*.q`) by hand on top of real BONES takes.
  `python3 gen.py <outdir> [names]`. Needs the library moves it uses (`mv.lib` looks in `gpc/clips/`, then a clone of
  `the-box-library` at `/home/claude/the-box-library`), MuJoCo 4.0.1 for Python, numpy, scipy.
- `mv.py`: read and write `.q` moves, forward kinematics on `body/soma23_box.xml`.
- `ik.py`: damped least-squares IK inside `limits_gpc.json`, and the game's joint envelopes (a port of `sim/anat.js`;
  same numbers as the page).
- `rend.py`: contact sheets and videos with MuJoCo (EGL).
- `harness/`: the real game page in headless Chrome, stepped as `GpcDrive.update` steps it. `h.py` serves the repo and
  maps the CDN to local npm packages (`three@0.160.0`, `@mujoco/mujoco@4.0.1`, `onnxruntime-web@1.31.1` in
  `/home/claude/harness/node_modules`). `tscene.py slow fast off` runs the scene and saves pictures; `tprobe.py`,
  `tsit.py`, `tsit2.py` compare the reference with his body (hands to head, joint commands).

Paths are the cloud session's (`/home/claude/...`); change `REPO` in `mv.py` and `h.py` elsewhere.
