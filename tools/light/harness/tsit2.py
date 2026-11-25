import asyncio, sys, json
sys.path.insert(0, '.')
from h import *
from tscene import TICK
MOVE = sys.argv[1]
P = """() => { const W = window.__box.W, B = window.__brainNet, m = W.m, d = W.d, re = B.react; if (!re) return { noReact: true, rec: window.__gpc.rec, active: window.__gpc.active && window.__gpc.active.name }; const f = Math.min(re.n - 1, Math.round(re.t * 30)), q = re.buf.subarray(f * 376 + 299, f * 376 + 372), cmp = {};
  for (let a = 0; a < m.nu; a++) { const j = m.actuator_trnid[2 * a], nm = W.act[a].name; if (!/^Left(Shoulder|Arm|ForeArm|Hand)/.test(nm)) continue; const qa = m.jnt_qposadr[j];
    cmp[nm] = [+(q[qa] * 57.3).toFixed(0), +(B.targets[a] * 57.3).toFixed(0), +(d.qpos[qa] * 57.3).toFixed(0)]; }
  const con = []; for (let i = 0; i < d.ncon; i++) { const c = d.contact.get ? d.contact.get(i) : d.contact[i]; if (!c) break; const b1 = m.geom_bodyid[c.geom1], b2 = m.geom_bodyid[c.geom2]; con.push([b1, b2, +c.dist.toFixed(3)]); }
  const A = W.__anat; return { cmp, ncon: d.ncon, con: con.slice(0, 20), anat: A ? A.violations(d.qpos, 1).map(x => [x.joint, +x.depth.toFixed(1)]) : null, anatOn: A && A.on }; }"""
async def main():
    s = serve(); pw, br, pg = await open_game('?light')
    await pg.wait_for_function('window.__scene && window.__scene.SCN.on', timeout=60000); await pg.wait_for_timeout(1500)
    await freeze(pg); await pg.evaluate(TICK); await step(pg, 2)
    await pg.evaluate("""(id) => { const g = window.__gpc; const O = g.ordersFile; const o = { name: 'Test', on: true, look: false, do: { type: 'recording', moves: [id] }, start: { all: [{ s: 'head', c: '>', v: 99 }], for: 1 }, stop: { all: [{ s: 'head', c: '>', v: 99 }], for: 1 }, steps: [{ pose: {}, time: 1 }] };
      g.apply({ version: 3, saved: 'x', orders: [o].concat(O.orders) }); g.begin('Test'); }""", MOVE)
    for i in range(10): await step(pg, 0.25)
    print(json.dumps(await pg.evaluate(P)))
    await br.close(); await pw.stop()
asyncio.run(main())
