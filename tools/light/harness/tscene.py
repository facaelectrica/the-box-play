import asyncio, time, sys, json, base64
sys.path.insert(0, '.')
from h import *
SHOT = '/home/claude/harness/shots/'
os.makedirs(SHOT, exist_ok=True)
TICK = "() => { window.__tick = () => { window.__scene.frame(1/30); window.__scene.gaze(1/30); }; }"
STATE = """() => { const g = window.__gpc, S = g.signs; return { t: +window.__box.W.d.time.toFixed(2), order: g.active && g.active.name, rec: g.rec && (g.rec.file || (g.rec.pending ? 'pending' : g.rec.failed ? 'FAILED ' + g.rec.failed : null)),
  head: +(S.head||0).toFixed(2), hips: +(S.hips||0).toFixed(2), light: S.light, rate: S.lightRate && +S.lightRate.toFixed(2), wall: g.rec && g.rec.wall, lookW: +(window.__brainNet.lookW||0).toFixed(2), handHead: (() => { const P = window.__brainNet.pos, d = (i, j) => Math.hypot(P[3*i]-P[3*j], P[3*i+1]-P[3*j+1], P[3*i+2]-P[3*j+2]); return [+d(10,6).toFixed(2), +d(14,6).toFixed(2)]; })() }; }"""

async def shot(pg, name):
    u = await pg.evaluate('() => window.__scene.draw()')
    open(SHOT + name + '.png', 'wb').write(base64.b64decode(u.split(',')[1]))

async def run(pg, secs, tag, level=None, ramp=None, every=1.0, shots=()):
    """ramp: (from, to, over seconds) applied at the start"""
    t = 0.0; log = []
    while t < secs - 1e-6:
        dt = min(every, secs - t)
        if ramp:
            a, b, T = ramp; n = max(1, int(dt * 30))
            for i in range(n):
                tt = t + (i + 1) * dt / n; lv = a + (b - a) * min(1, tt / T) if T > 0 else b
                await pg.evaluate(f'() => {{ window.__scene.SCN.level = {lv}; }}'); await step(pg, dt / n)
        else:
            if level is not None: await pg.evaluate(f'() => {{ window.__scene.SCN.level = {level}; }}')
            await step(pg, dt)
        t += dt
        s = await pg.evaluate(STATE); log.append(s)
        print(tag, s, flush=True)
        for (ts, nm) in shots:
            if abs(t - ts) < 1e-6: await shot(pg, nm)
    return log

async def scene(pg):
    await pg.evaluate('() => window.__scene.setScene(false)'); await pg.wait_for_timeout(100)
    await pg.evaluate('() => window.__scene.setScene(true)'); await pg.wait_for_timeout(500)

async def main(which):
    s = serve()
    pw, br, pg = await open_game('?light')
    await pg.wait_for_function('window.__scene && window.__scene.SCN.on', timeout=60000)
    await pg.wait_for_timeout(1500)
    await freeze(pg); await pg.evaluate(TICK)
    if 'slow' in which:
        await scene(pg)
        await run(pg, 4, 'sleep', level=0, shots=[(2, 'slow_0_asleep')])
        await run(pg, 4, 'slide', ramp=(0, 1, 4), shots=[(2, 'slow_1_half')])
        await run(pg, 26, 'lit', level=1, shots=[(3, 'slow_2'), (6, 'slow_3'), (10, 'slow_4'), (13, 'slow_5'), (15, 'slow_6'), (18, 'slow_7'), (24, 'slow_8')])
    if 'fast' in which:
        await scene(pg)
        await run(pg, 3, 'sleep', level=0)
        await run(pg, 18, 'tap', level=1, every=0.5 , shots=[(0.5, 'fast_0'), (1.5, 'fast_1'), (3.0, 'fast_2'), (4.0, 'fast_3'), (5.0, 'fast_4'), (6.5, 'fast_5'), (8.0, 'fast_6'), (10.0, 'fast_7'), (13.0, 'fast_8'), (17.0, 'fast_9')])
    if 'off' in which:
        await run(pg, 30, 'off', level=0, shots=[(5, 'off_0'), (12, 'off_1'), (16, 'off_2'), (22, 'off_3'), (29, 'off_4')])
    await br.close(); await pw.stop()
if __name__ == "__main__": asyncio.run(main(sys.argv[1:] or ["slow"]))
