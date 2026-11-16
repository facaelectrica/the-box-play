# The user's Female_v2.fbx -> body/bodychan_v2.glb in the game's figure format: one node per body (named as the body),
# its pieces as child meshes in the body's own frame (metres, the body's joint at the origin, rest frames aligned).
import json, struct, numpy as np, sys
sys.path.insert(0, __import__('os').path.dirname(__file__)); from fig_from_fbx_load import pieces
skel = json.load(open('/home/claude/gutest/skel.json')); W = {}
for b in skel: W[b['name']] = (W[b['parent']] if b['parent'] else np.zeros(3)) + np.array(b['pos'])
SPHERE = {'Sphere003': 'Spine1', 'Sphere004': 'Spine2', 'Sphere001': 'Hips', 'Sphere005': 'Hips', 'Sphere002': 'LeftShin', 'Sphere006': 'RightShin', 'Sphere007': 'Neck2'}
groups = {}
for p in pieces():
    nd = p['node']
    if nd in SPHERE: body, mat = SPHERE[nd], 'joint'
    else: body, mat = nd.split('_mesh')[0], 'shell'
    assert body in W, nd
    P = p['P'] / 100.0 - W[body]   # FBX (cm, z up, same axes as the physics) -> metres in the body's frame
    # smooth shading: each corner gets the average normal of the faces around its point that turn less than 50 deg from
    # its own face (the FBX came with one normal per face on some pieces, which showed as facets); sharper edges stay sharp
    Ix = p['I'].reshape(-1, 3); fn = np.cross(P[Ix[:, 1]] - P[Ix[:, 0]], P[Ix[:, 2]] - P[Ix[:, 0]]); fu = fn / (np.linalg.norm(fn, axis=1, keepdims=True) + 1e-15)
    _, wid = np.unique(np.round(P, 6), axis=0, return_inverse=True); wid = wid.reshape(-1)
    corners = {}
    for t in range(len(Ix)):
        for j in range(3): corners.setdefault(wid[Ix[t, j]], []).append(t)
    Nn = np.zeros_like(P); cosk = np.cos(np.radians(50))
    for t in range(len(Ix)):
        for j in range(3):
            v = Ix[t, j]; fs = corners[wid[v]]; m = fu[fs] @ fu[t] > cosk; acc_ = fn[fs][m].sum(0); Nn[v] += acc_ / (np.linalg.norm(acc_) + 1e-15)
    p['N'] = Nn / (np.linalg.norm(Nn, axis=1, keepdims=True) + 1e-15)
    # merge corners that share place and normal (the FBX repeats them)
    key = np.round(np.hstack([P, p['N']]), 6); u, inv = np.unique(key, axis=0, return_inverse=True); inv = inv.reshape(-1)
    groups.setdefault(body, []).append((u[:, :3].astype(np.float32), u[:, 3:].astype(np.float32), inv[p['I']].astype(np.uint32), mat, nd))
print({k: [x[4] for x in v] for k, v in groups.items()})
buf = bytearray(); views, accs, meshes, nodes = [], [], [], []
def add(a, target, kind, mm=False):
    a = np.ascontiguousarray(a); o = len(buf); buf.extend(a.tobytes()); buf.extend(b'\0' * ((4 - len(buf) % 4) % 4))
    views.append({'buffer': 0, 'byteOffset': o, 'byteLength': a.nbytes, 'target': target})
    d = {'bufferView': len(views) - 1, 'componentType': 5126 if a.dtype == np.float32 else 5125, 'count': a.shape[0], 'type': kind}
    if mm: d['min'] = a.min(0).tolist(); d['max'] = a.max(0).tolist()
    accs.append(d); return len(accs) - 1
MATS = ['shell', 'joint']; root = []
for body, items in groups.items():
    kids = []   # one mesh node per piece, each directly under the body's node (three.js names the body from the parent)
    for k, (P, N, I, mat, nd) in enumerate(items):
        meshes.append({'name': body + '_' + str(k), 'primitives': [{'attributes': {'POSITION': add(P, 34962, 'VEC3', True), 'NORMAL': add(N, 34962, 'VEC3')}, 'indices': add(I, 34963, 'SCALAR'), 'material': MATS.index(mat)}]})
        nodes.append({'name': body + '_' + str(k), 'mesh': len(meshes) - 1}); kids.append(len(nodes) - 1)
    nodes.append({'name': body, 'children': kids}); root.append(len(nodes) - 1)
nodes.append({'name': 'pieces_on_joints'}); root.append(len(nodes) - 1)   # marker: the pieces already hang on their own joints (no spineByJoint)
G = {'asset': {'version': '2.0', 'generator': 'the-box: Female_v2.fbx (user, Nov 16 17:01)'}, 'scene': 0, 'scenes': [{'nodes': root}], 'nodes': nodes, 'meshes': meshes,
     'materials': [{'name': 'shell', 'pbrMetallicRoughness': {'baseColorFactor': [0.92, 0.9, 0.85, 1], 'metallicFactor': 0, 'roughnessFactor': 0.5}},
                   {'name': 'joint', 'pbrMetallicRoughness': {'baseColorFactor': [0.92, 0.9, 0.85, 1], 'metallicFactor': 0, 'roughnessFactor': 0.5}}],
     'accessors': accs, 'bufferViews': views, 'buffers': [{'byteLength': len(buf)}]}
js = json.dumps(G).encode(); js += b' ' * ((4 - len(js) % 4) % 4)
glb = struct.pack('<III', 0x46546C67, 2, 12 + 8 + len(js) + 8 + len(buf)) + struct.pack('<II', len(js), 0x4E4F534A) + js + struct.pack('<II', len(buf), 0x004E4942) + bytes(buf)
open('/home/claude/the-box-play/body/bodychan_v2.glb', 'wb').write(glb); print(len(glb), 'bytes')
