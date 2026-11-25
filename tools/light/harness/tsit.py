import asyncio, sys, json
sys.path.insert(0, '.')
from h import *
from tscene import TICK
PROBE = __import__("tprobe").PROBE
MOVE = sys.argv[1]; N = int(sys.argv[2]) if len(sys.argv) > 2 else 20
async def main():
    s = serve(); pw, br, pg = await open_game(os.environ.get('Q', '?light'))
    await pg.wait_for_timeout(1500)
    await freeze(pg); await pg.evaluate(TICK); await step(pg, 2)
    await pg.evaluate("""(id) => { const g = window.__gpc; const O = g.ordersFile; const o = { name: 'Test', on: true, look: false, do: { type: 'recording', moves: [id] }, start: { all: [{ s: 'head', c: '>', v: 99 }], for: 1 }, stop: { all: [{ s: 'head', c: '>', v: 99 }], for: 1 }, steps: [{ pose: {}, time: 1 }] };
      g.apply({ version: 3, saved: 'x', orders: [o].concat(O.orders) }); g.begin('Test'); }""", MOVE)
    for i in range(N):
        await step(pg, 0.25); r = await pg.evaluate(PROBE); print(r['t'], r['rec'], 'ref', r['refHandHeadR'], r['refHandHeadL'], 'real', r['realHandHeadR'], r['realHandHeadL'], flush=True)
    await br.close(); await pw.stop()
asyncio.run(main())
