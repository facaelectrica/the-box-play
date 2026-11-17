// Camera-accurate motion blur (user, Nov 17 08:20: "camera accurate motion blur, while keeping 60 fps on mobile").
//
// A film camera's shutter stays open for part of each frame, so whatever moves during that time is smeared along its
// path: the floor when the camera pans, a fast arm on its own. This draws the same thing:
//   1. the picture, as usual, into an off-screen image (linear light, before tone mapping);
//   2. motion: every visible solid piece again, at half resolution, writing how far that point moved on screen since
//      the last frame (its own movement and the camera's) and how far it is from the eye. Each piece is drawn where it
//      is now and where it was a frame ago (its previous placement, the previous camera; skinned pieces: the previous
//      bone placements too);
//   3. tiles of 24 x 24 pixels: the biggest motion in each tile and its neighbours (how far any smear can reach);
//   4. the final picture: each pixel gathers 10 samples along the biggest nearby motion and weighs each one by whether
//      it is in front, behind, and moving far enough to cover this pixel (McGuire et al. 2012, "A reconstruction filter
//      for plausible motion blur"), so a moving arm smears over the still floor behind it, not the floor over the arm.
//      Then tone mapping and sRGB, as the renderer does for the screen.
// Shutter: half a frame (180 degrees, a film camera's usual setting). The smear is capped at 24 pixels each way.
// Pixels where nothing nearby moves skip the gathering, so a still picture costs only the extra passes.

export function makeMotionBlur(THREE, renderer, scene) {
  const gl = renderer.getContext(), ext = renderer.extensions;
  if (!renderer.capabilities.isWebGL2 || !(ext.has('EXT_color_buffer_float') || ext.has('EXT_color_buffer_half_float'))) return null;
  const K = 24, SHUTTER = 0.5, N = 10, FAR = 40;   // tile / biggest smear radius (px), shutter (part of a frame), samples, depth range (m)
  const size = new THREE.Vector2();
  const half = { type: THREE.HalfFloatType, format: THREE.RGBAFormat, depthBuffer: true, stencilBuffer: false };
  const colorRT = new THREE.WebGLRenderTarget(1, 1, { ...half, samples: 4 });
  const velRT = new THREE.WebGLRenderTarget(1, 1, { ...half, minFilter: THREE.NearestFilter, magFilter: THREE.NearestFilter });
  const flat = { type: THREE.HalfFloatType, format: THREE.RGBAFormat, depthBuffer: false, stencilBuffer: false, minFilter: THREE.NearestFilter, magFilter: THREE.NearestFilter };
  const tileRT = new THREE.WebGLRenderTarget(1, 1, flat), nbRT = new THREE.WebGLRenderTarget(1, 1, flat);
  const res = new THREE.Vector2(1, 1);

  // ---- the motion of each solid piece ----
  const prevVP = new THREE.Matrix4(), curVP = new THREE.Matrix4(); let havePrev = false;
  const VEL_VS = /* glsl */`
    #include <common>
    #include <skinning_pars_vertex>
    uniform mat4 prevModel; uniform mat4 prevVP;
    #ifdef USE_SKINNING
      uniform highp sampler2D prevBones;
      mat4 prevBone(const in float i) {
        int size = textureSize(prevBones, 0).x; int j = int(i) * 4; int x = j % size; int y = j / size;
        return mat4(texelFetch(prevBones, ivec2(x, y), 0), texelFetch(prevBones, ivec2(x + 1, y), 0), texelFetch(prevBones, ivec2(x + 2, y), 0), texelFetch(prevBones, ivec2(x + 3, y), 0));
      }
    #endif
    varying vec4 vCur; varying vec4 vPrev;
    void main() {
      vec3 p = position, q = position;
      #ifdef USE_SKINNING
        vec4 sv = bindMatrix * vec4(position, 1.0);
        vec4 a = getBoneMatrix(skinIndex.x) * sv * skinWeight.x + getBoneMatrix(skinIndex.y) * sv * skinWeight.y
               + getBoneMatrix(skinIndex.z) * sv * skinWeight.z + getBoneMatrix(skinIndex.w) * sv * skinWeight.w;
        vec4 b = prevBone(skinIndex.x) * sv * skinWeight.x + prevBone(skinIndex.y) * sv * skinWeight.y
               + prevBone(skinIndex.z) * sv * skinWeight.z + prevBone(skinIndex.w) * sv * skinWeight.w;
        p = (bindMatrixInverse * a).xyz; q = (bindMatrixInverse * b).xyz;
      #endif
      vec4 wp = vec4(p, 1.0), wq = vec4(q, 1.0);
      #ifdef USE_INSTANCING
        wp = instanceMatrix * wp; wq = instanceMatrix * wq;
      #endif
      vCur = projectionMatrix * viewMatrix * modelMatrix * wp;
      vPrev = prevVP * prevModel * wq;
      gl_Position = vCur;
    }`;
  const VEL_FS = /* glsl */`
    uniform vec2 res; uniform float farInv, scale, K;
    varying vec4 vCur; varying vec4 vPrev;
    void main() {
      vec2 v = (vCur.xy / vCur.w - vPrev.xy / vPrev.w) * 0.5 * res * scale;   // half the smear, in screen pixels
      float l = length(v); if (l > K) v *= K / l;
      gl_FragColor = vec4(v, vCur.w * farInv, 1.0);
    }`;
  const velShared = { res: { value: res }, farInv: { value: 1 / FAR }, scale: { value: SHUTTER * 0.5 }, K: { value: K }, prevVP: { value: prevVP } };
  function velMatFor(o, src) {
    const u = o.userData;
    if (!u._mb) {
      u._mb = { prev: o.matrixWorld.clone(), mats: new Map() };
      if (o.isSkinnedMesh) u._mb.bones = null;
    }
    let m = u._mb.mats.get(src);
    if (!m) {
      const uni = { ...velShared, prevModel: { value: u._mb.prev } };
      if (o.isSkinnedMesh) uni.prevBones = { value: null };
      m = new THREE.ShaderMaterial({ vertexShader: VEL_VS, fragmentShader: VEL_FS, uniforms: uni, side: src.side });
      u._mb.mats.set(src, m);
    }
    if (o.isSkinnedMesh) m.uniforms.prevBones.value = prevBonesOf(o);
    return m;
  }
  // a skinned piece's previous bone placements: a copy of its skeleton's bone texture from the frame before
  const boneCopies = new Map();
  function prevBonesOf(o) {
    const sk = o.skeleton; if (!sk || !sk.boneTexture) return sk && sk.boneTexture || null;
    let c = boneCopies.get(sk);
    if (!c) { const src = sk.boneTexture.image; c = new THREE.DataTexture(new Float32Array(src.data), src.width, src.height, THREE.RGBAFormat, THREE.FloatType); c.needsUpdate = true; boneCopies.set(sk, c); }
    return c;
  }

  // ---- full-screen passes ----
  const quadCam = new THREE.OrthographicCamera(-1, 1, 1, -1, 0, 1), quadScene = new THREE.Scene();
  const quad = new THREE.Mesh(new THREE.PlaneGeometry(2, 2)); quad.frustumCulled = false; quadScene.add(quad);
  const QUAD_VS = 'varying vec2 vUv; void main() { vUv = uv; gl_Position = vec4(position.xy, 0.0, 1.0); }';
  const tileMat = new THREE.ShaderMaterial({ vertexShader: QUAD_VS, uniforms: { tVel: { value: velRT.texture }, step: { value: 12 } },
    fragmentShader: /* glsl */`
      uniform highp sampler2D tVel; uniform int step;
      void main() {
        ivec2 o = ivec2(gl_FragCoord.xy) * step, n = textureSize(tVel, 0); vec2 best = vec2(0.0); float bl = 0.0;
        for (int y = 0; y < 16; y++) { if (y >= step) break;
          for (int x = 0; x < 16; x++) { if (x >= step) break;
            ivec2 c = min(o + ivec2(x, y), n - 1); vec2 v = texelFetch(tVel, c, 0).xy; float l = dot(v, v);
            if (l > bl) { bl = l; best = v; } } }
        gl_FragColor = vec4(best, 0.0, 1.0);
      }` });
  const nbMat = new THREE.ShaderMaterial({ vertexShader: QUAD_VS, uniforms: { tTile: { value: tileRT.texture } },
    fragmentShader: /* glsl */`
      uniform highp sampler2D tTile;
      void main() {
        ivec2 c = ivec2(gl_FragCoord.xy), n = textureSize(tTile, 0); vec2 best = vec2(0.0); float bl = 0.0;
        for (int y = -1; y <= 1; y++) for (int x = -1; x <= 1; x++) {
          vec2 v = texelFetch(tTile, clamp(c + ivec2(x, y), ivec2(0), n - 1), 0).xy; float l = dot(v, v);
          if (l > bl) { bl = l; best = v; } }
        gl_FragColor = vec4(best, 0.0, 1.0);
      }` });
  const outMat = new THREE.ShaderMaterial({ vertexShader: QUAD_VS, toneMapped: true,
    uniforms: { tColor: { value: colorRT.texture }, tVel: { value: velRT.texture }, tNb: { value: nbRT.texture }, res: { value: res }, softZ: { value: 0.03 / FAR } },
    fragmentShader: /* glsl */`
      uniform sampler2D tColor; uniform highp sampler2D tVel; uniform highp sampler2D tNb; uniform vec2 res; uniform float softZ;
      varying vec2 vUv;
      float soft(float a, float b) { return clamp(1.0 - (a - b) / softZ, 0.0, 1.0); }   // 1 when depth a is in front of (nearer than) b
      float cone(float d, float l) { return clamp(1.0 - d / l, 0.0, 1.0); }
      float cyl(float d, float l) { return 1.0 - smoothstep(0.95 * l, 1.05 * l, d); }
      void main() {
        vec4 c = texture2D(tColor, vUv);
        vec2 vn = texture2D(tNb, vUv).xy;
        if (dot(vn, vn) < 0.25) { gl_FragColor = c; }
        else {
          vec4 vx = texture2D(tVel, vUv); float lx = max(length(vx.xy), 0.5), zx = vx.z;
          float w = 1.0 / lx; vec4 sum = c * w;
          float j = fract(sin(dot(gl_FragCoord.xy, vec2(12.9898, 78.233))) * 43758.5453) - 0.5;
          for (int i = 0; i < ${N}; i++) {
            float t = mix(-1.0, 1.0, (float(i) + j + 1.0) / float(${N + 1}));
            vec2 d = vn * t, uy = vUv + d / res; float dl = length(d);
            vec4 vy = texture2D(tVel, uy); float ly = max(length(vy.xy), 0.5), zy = vy.z;
            float a = soft(zy, zx) * cone(dl, ly) + soft(zx, zy) * cone(dl, lx) + cyl(dl, ly) * cyl(dl, lx) * 2.0;
            sum += texture2D(tColor, uy) * a; w += a;
          }
          gl_FragColor = sum / w;
        }
        #include <tonemapping_fragment>
        #include <colorspace_fragment>
      }` });
  const pass = (mat, rt) => { quad.material = mat; renderer.setRenderTarget(rt); renderer.render(quadScene, quadCam); };

  function resize() {
    renderer.getDrawingBufferSize(size);
    const w = Math.max(1, size.x), h = Math.max(1, size.y);
    if (w === res.x && h === res.y) return;
    res.set(w, h);
    colorRT.setSize(w, h);
    const hw = Math.ceil(w / 2), hh = Math.ceil(h / 2); velRT.setSize(hw, hh);
    tileRT.setSize(Math.ceil(hw / (K / 2)), Math.ceil(hh / (K / 2))); nbRT.setSize(tileRT.width, tileRT.height);
    tileMat.uniforms.step.value = K / 2;
  }

  // what the motion pass hides (see-through, lines, overlays) and what it swaps (solid pieces)
  const hidden = [], swapped = [];
  const solid = m => m && m.visible !== false && !m.transparent && !m.wireframe && m.colorWrite !== false && m.depthTest !== false;
  const clear = new THREE.Color(), velClear = new THREE.Color(0, 0, 1);

  function render(camera) {
    resize();
    camera.updateMatrixWorld();
    curVP.multiplyMatrices(camera.projectionMatrix, camera.matrixWorldInverse);
    if (!havePrev) { prevVP.copy(curVP); havePrev = true; }
    // 1. the picture
    renderer.setRenderTarget(colorRT); renderer.render(scene, camera);
    // 2. motion (no shadows redrawn, no background, no fog)
    scene.traverseVisible(o => {
      if (o === scene) return;
      if (o.isMesh) {
        const mats = Array.isArray(o.material) ? o.material : [o.material];
        if (mats.every(solid)) { swapped.push(o, o.material); o.material = Array.isArray(o.material) ? mats.map(m => velMatFor(o, m)) : velMatFor(o, o.material); return; }
      }
      if (o.isMesh || o.isLine || o.isPoints || o.isSprite) { hidden.push(o); o.visible = false; }
    });
    const bg = scene.background, fog = scene.fog, auto = renderer.shadowMap.autoUpdate, ca = renderer.getClearAlpha();
    renderer.getClearColor(clear);
    scene.background = null; scene.fog = null; renderer.shadowMap.autoUpdate = false; renderer.setClearColor(velClear, 1);
    try { renderer.setRenderTarget(velRT); renderer.clear(); renderer.render(scene, camera); }
    finally {
      scene.background = bg; scene.fog = fog; renderer.shadowMap.autoUpdate = auto; renderer.setClearColor(clear, ca);
      for (let i = 0; i < swapped.length; i += 2) swapped[i].material = swapped[i + 1];
      for (const o of hidden) o.visible = true;
    }
    // the placements of this frame become "the frame before" for the next one
    for (let i = 0; i < swapped.length; i += 2) { const o = swapped[i]; o.userData._mb.prev.copy(o.matrixWorld); }
    for (const [sk, c] of boneCopies) if (sk.boneMatrices) { c.image.data.set(sk.boneMatrices); c.needsUpdate = true; }
    swapped.length = 0; hidden.length = 0;
    prevVP.copy(curVP);
    // 3. how far any smear reaches, per tile; 4. the final picture
    pass(tileMat, tileRT); pass(nbMat, nbRT); pass(outMat, null);
  }
  // after a jump (standing up again, box <-> open floor): no smear from where things were before
  function reset() { havePrev = false; scene.traverse(o => { if (o.userData._mb) o.userData._mb.prev.copy(o.matrixWorld); }); }
  return { _rt: { velRT, tileRT, nbRT, colorRT }, render, reset, dispose() { for (const t of [colorRT, velRT, tileRT, nbRT]) t.dispose(); } };
}
