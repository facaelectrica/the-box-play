import asyncio, sys
sys.path.insert(0, '.')
from h import *
from tscene import TICK, STATE, scene
PROBE = """() => { const B = window.__brainNet, R = B.lastRef, P = B.pos, d = (A, i, j) => Math.hypot(A[3*i]-A[3*j], A[3*i+1]-A[3*j+1], A[3*i+2]-A[3*j+2]);
  const g = window.__gpc, re = B.react;
  return { t: +window.__box.W.d.time.toFixed(2), rec: g.rec && g.rec.file, reactT: re && +re.t.toFixed(2), refHandHeadR: +d(R.pos,10,6).toFixed(2), refHandHeadL: +d(R.pos,14,6).toFixed(2), realHandHeadR: +d(P,10,6).toFixed(2), realHandHeadL: +d(P,14,6).toFixed(2),
    refHeadZ: +R.pos[20].toFixed(2), realHeadZ: +P[20].toFixed(2), refHandZ: +R.pos[32].toFixed(2), realHandZ: +P[32].toFixed(2) }; }"""
async def main():
    s = serve(); pw, br, pg = await open_game(sys.argv[1] if len(sys.argv) > 1 else '?light')
    await pg.wait_for_function('window.__scene && window.__scene.SCN.on', timeout=60000); await pg.wait_for_timeout(1500)
    await freeze(pg); await pg.evaluate(TICK)
    await step(pg, 2); print(await pg.evaluate(PROBE))
    await pg.evaluate('() => { window.__scene.SCN.level = 1; }')
    for i in range(16): await step(pg, 0.25); print(await pg.evaluate(PROBE))
    await br.close(); await pw.stop()
if __name__ == "__main__": asyncio.run(main())
