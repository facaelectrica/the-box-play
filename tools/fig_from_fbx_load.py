import json, numpy as np, base64, os
G=json.load(open('/home/claude/newmodel/f.gltf')); N=G['nodes']; BIN=open('/home/claude/newmodel/f.bin','rb').read()
par={}
for i,n in enumerate(N):
    for c in n.get('children',[]): par[c]=i
def local(n):
    if 'matrix' in n: return np.array(n['matrix']).reshape(4,4).T
    M=np.eye(4); t=n.get('translation',[0,0,0]); r=n.get('rotation',[0,0,0,1]); s=n.get('scale',[1,1,1])
    x,y,z,w=r; R=np.array([[1-2*(y*y+z*z),2*(x*y-z*w),2*(x*z+y*w)],[2*(x*y+z*w),1-2*(x*x+z*z),2*(y*z-x*w)],[2*(x*z-y*w),2*(y*z+x*w),1-2*(x*x+y*y)]])
    M[:3,:3]=R*np.array(s); M[:3,3]=t; return M
def world(i):
    M=local(N[i])
    while i in par: i=par[i]; M=local(N[i])@M
    return M
def acc(k):
    a=G['accessors'][k]; v=G['bufferViews'][a['bufferView']]; n={'SCALAR':1,'VEC2':2,'VEC3':3,'VEC4':4}[a['type']]; dt={5126:np.float32,5123:np.uint16,5125:np.uint32}[a['componentType']]
    x=np.frombuffer(BIN, dt, a['count']*n, v.get('byteOffset',0)+a.get('byteOffset',0)); return x.reshape(-1,n) if n>1 else x
def pieces():
    out=[]
    for i,n in enumerate(N):
        if 'mesh' not in n: continue
        M=world(i)
        for pr in G['meshes'][n['mesh']]['primitives']:
            P=acc(pr['attributes']['POSITION']).astype(np.float64); P=(M[:3,:3]@P.T).T+M[:3,3]
            Nn=acc(pr['attributes']['NORMAL']).astype(np.float64) if 'NORMAL' in pr['attributes'] else None
            if Nn is not None: Nn=(np.linalg.inv(M[:3,:3]).T@Nn.T).T; Nn/=np.linalg.norm(Nn,axis=1,keepdims=True)+1e-12
            I=acc(pr['indices']).astype(np.uint32) if 'indices' in pr else np.arange(len(P),dtype=np.uint32)
            out.append({'node':n.get('name'),'P':P,'N':Nn,'I':I,'mat':G['materials'][pr['material']]['name'] if 'material' in pr else ''})
    return out
