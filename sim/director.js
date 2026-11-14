// The director (v15.35, user Nov 13 20:14): when he starts losing his balance, what he copies switches from the calm
// idle to a real person's recovery from a push in that direction (Inria push-recovery recordings, body/clips/react_*.bin,
// made by tools/push/make_react_clips.py: real steps, planted feet, arms, knees, torso), then back to the idle where
// he ended up (brain.playReact). The brain was trained on exactly these recordings with their pushes (Stage 2).
// Trigger: his hips move faster than `vTrig` m/s horizontally (calm idle sway is ~0.02 m/s). Direction: that motion
// relative to where he faces (0 = forward, +90 = to his left). Strong (> `vStep`): a recording with a real step;
// weaker: one where the person absorbs it in place. No recordings of pushes toward his back yet: those keep the idle.
// v15.37 (A5): the switch in and out is eased over brain.blendT (0.4 s) instead of a one-frame jump (brain.applyBlend);
// push test, 56 pushes: falls 2 -> 1, steps 74 -> 100, biggest target jump per frame 123 -> 4.5 cm. vTrig/lead re-tuned:
// 0.30 m/s and +4 frames stay best (0.22/0.38 m/s, 0/+8 frames no better).
export class Director {
  constructor(brain, clips, opt = {}) {
    this.brain = brain; this.clips = clips;          // [{buf: Float32Array, ang, kind, push_frame}]
    this.vTrig = opt.vTrig ?? 0.30; this.vStep = opt.vStep ?? 0.45; this.cool = 0; this.coolT = opt.coolT ?? 0.8;
    this.lead = opt.lead ?? 4;                        // frames after the recorded push onset to start from
    this.last = null; this.n = 0;
    // v15.38 (A5, user 00:57-00:59 "arms more relaxed", "weight on one leg", knees): idle recordings to rotate through
    // while he is calm, every `every` s (random in the range), eased in by brain.switchIdle; the first one soon after load
    this.idles = opt.idles || []; this.every = opt.every || [25, 45]; this.ki = -1; this.idleT = opt.firstIdle ?? 2; this.settleIdle = opt.settleIdle ?? 1.5;
  }
  nextIdle() {
    const B = this.brain; if (!this.idles.length) return;
    this.ki = (this.ki + 1) % this.idles.length; const c = this.idles[this.ki];
    B.switchIdle(c.buf); this.idleNow = c.file;
    this.cool = (B.blendT || 0.4) + this.settleIdle;   // the weight shift into the new idle is not a stumble
    this.idleT = this.every[0] + Math.random() * (this.every[1] - this.every[0]);
  }
  update(dt) {
    const B = this.brain;
    if (!B || !this.clips.length || B.down) return;
    if (B.react) { this.cool = this.coolT; return; }
    if (this.idles.length && (this.idleT -= dt) <= 0 && this.cool <= 0 && Math.hypot(B.vel[0], B.vel[1]) < 0.15) this.nextIdle();
    if ((this.cool -= dt) > 0) return;
    const vx = B.vel[0], vy = B.vel[1], v = Math.hypot(vx, vy);
    if (v < this.vTrig) return;
    const r = new Float64Array(4); for (let i = 0; i < 4; i++) r[i] = B.rot[i];
    const t = [1 - 2 * (r[1] * r[1] + r[2] * r[2]), 2 * (r[0] * r[1] + r[3] * r[2])];   // his x axis (= his left), xy
    const yaw = Math.atan2(t[1], t[0]), f = [Math.sin(yaw), -Math.cos(yaw)], l = [Math.cos(yaw), Math.sin(yaw)];
    const ang = Math.atan2(vx * l[0] + vy * l[1], vx * f[0] + vy * f[1]) * 180 / Math.PI;
    if (Math.abs(ang) > 112) return;                  // toward his back: no recordings yet
    const want = Math.round(ang / 45) * 45, kind = v > this.vStep ? 'step' : 'inplace';
    let pool = this.clips.filter(c => c.ang === want && c.kind === kind);
    if (!pool.length) pool = this.clips.filter(c => c.ang === want);
    if (!pool.length) return;
    const c = pool[(this.n++) % pool.length];
    B.playReact(c.buf, (c.push_frame + this.lead) / 30);
    this.last = { ang: Math.round(ang), v: +v.toFixed(2), clip: c.file, kind: c.kind };
    this.cool = this.coolT;
  }
}
