"""Headless checker: the real game page (gpc/live.html) in Chrome, stepped exactly as GpcDrive.update steps it."""
import asyncio, json, os, sys, threading, functools, http.server, socketserver
from playwright.async_api import async_playwright

REPO = os.environ.get('REPO', '/home/claude/the-box-play')
NM = '/home/claude/harness/node_modules/'
PORT = int(os.environ.get('PORT', 8765))


class H(http.server.SimpleHTTPRequestHandler):
    def log_message(self, *a): pass
    def end_headers(self):
        self.send_header('Cross-Origin-Opener-Policy', 'same-origin'); self.send_header('Cross-Origin-Embedder-Policy', 'require-corp')
        self.send_header('Cache-Control', 'no-store'); super().end_headers()


def serve():
    socketserver.TCPServer.allow_reuse_address = True
    s = socketserver.ThreadingTCPServer(('127.0.0.1', PORT), functools.partial(H, directory=REPO))
    threading.Thread(target=s.serve_forever, daemon=True).start(); return s


async def cdn(route):
    u = route.request.url.split('?')[0]; p = u.split('cdn.jsdelivr.net/npm/')[1]
    pkg, rest = (p.split('/', 2)[0] + '/' + p.split('/', 2)[1], p.split('/', 2)[2]) if p.startswith('@') else (p.split('/', 1)[0], p.split('/', 1)[1])
    name = pkg.rsplit('@', 1)[0]; f = NM + name + '/' + rest
    if not os.path.exists(f): await route.fulfill(status=404, body='missing ' + f); return
    ct = 'application/javascript' if f.endswith(('.js', '.mjs')) else 'application/wasm' if f.endswith('.wasm') else 'font/woff2' if f.endswith('.woff2') else 'application/octet-stream'
    await route.fulfill(path=f, headers={'content-type': ct, 'Cross-Origin-Resource-Policy': 'cross-origin', 'Access-Control-Allow-Origin': '*'})

STEP_JS = """async (secs) => {
  const W = window.__box.W, B = window.__brainNet, d = W.d, n = Math.round(secs * 480), out = [];
  for (let i = 0; i < n; i++) {
    d.xfrc_applied.fill(0);
    if (B.due()) { B.down = false; B.lost = 0; await B.think(); if (B.down) { B.down = false; B.setTone(1); } if (window.__tick) window.__tick(); }
    W.step(B.control);
  }
  return true; }"""


async def open_game(query='', w=412, h=915):
    pw = await async_playwright().start()
    br = await pw.chromium.launch(executable_path='/usr/bin/google-chrome', args=['--use-gl=angle', '--use-angle=swiftshader', '--enable-unsafe-swiftshader', '--no-sandbox'])
    pg = await br.new_page(viewport={'width': w, 'height': h})
    await pg.route('https://cdn.jsdelivr.net/**', cdn)
    pg.on('console', lambda m: print('[page]', m.text[:300]) if m.type in ('error', 'warning') and 'GPU stall' not in m.text else None)
    pg.on('pageerror', lambda e: print('[pageerror]', str(e)[:500]))
    await pg.goto(f'http://127.0.0.1:{PORT}/gpc/live.html{query}')
    await pg.wait_for_function('document.getElementById("go") && !document.getElementById("go").disabled', timeout=600000)
    await pg.click('#go')
    await pg.wait_for_function('window.__gpc && window.__gpc.ordersReady && window.__box', timeout=600000)
    return pw, br, pg


async def freeze(pg):
    """stop the page's own frame loop: from now on only step() moves time"""
    await pg.evaluate('() => { window.__rafSaved = window.requestAnimationFrame; window.requestAnimationFrame = () => 0; }')
    await pg.wait_for_timeout(300)


async def step(pg, secs):
    return await pg.evaluate(STEP_JS, secs)
