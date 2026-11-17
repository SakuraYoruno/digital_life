"""WebGL 版记忆星图：莫比乌斯环 + 六万点粒子 + Bloom 后处理。

canvas 2D 那版只能做"点 + 假辉光"，质感差了一个量级——那版保留作为 `--engine canvas`
的降级路径；默认走这版：three.js（r128 UMD）+ UnrealBloomPass，输出仍是**单文件 HTML**，
依赖在生成时内联进去，离线也能打开。

three 的几个库首次生成时会从 unpkg 下载并缓存在 `<data_dir>/_libs`，之后不再联网。
"""

from __future__ import annotations

import json
import math
import time
import urllib.request
from pathlib import Path
from typing import Any

from .memory import MemoryStore
from .persistence import StateStore
from .view import build_payload

_CDN = "https://unpkg.com/three@0.128.0/"
_LIBS = {
    "three.min.js": "build/three.min.js",
    "CopyShader.js": "examples/js/shaders/CopyShader.js",
    "LuminosityHighPassShader.js": "examples/js/shaders/LuminosityHighPassShader.js",
    "EffectComposer.js": "examples/js/postprocessing/EffectComposer.js",
    "RenderPass.js": "examples/js/postprocessing/RenderPass.js",
    "ShaderPass.js": "examples/js/postprocessing/ShaderPass.js",
    "UnrealBloomPass.js": "examples/js/postprocessing/UnrealBloomPass.js",
}
_ORDER = list(_LIBS)

_HTML = """<!doctype html>
<html lang="zh-CN">
<head>
<meta charset="utf-8">
<title>意识海 · 记忆星图</title>
<style>
  :root { --ink:#eaf1ff; --dim:#7d8cb0; }
  * { box-sizing:border-box; }
  html,body { margin:0; height:100%; background:#02030a; overflow:hidden;
              color:var(--ink); font:14px/1.6 "Segoe UI","Microsoft YaHei",sans-serif; }
  canvas { display:block; }
  .hud { position:fixed; left:34px; top:26px; pointer-events:none; }
  .hud h1 { margin:0; font-size:28px; letter-spacing:9px; font-weight:600;
            text-shadow:0 0 26px rgba(120,180,255,.6); }
  .hud .sub { color:var(--dim); font-size:12px; letter-spacing:3px; margin-top:6px; }
  .stats { position:fixed; right:34px; top:26px; text-align:right; font-size:12px;
           color:var(--dim); letter-spacing:1px; }
  .stats div { margin-bottom:4px; }
  .stats b { color:var(--ink); font-size:17px; font-weight:500; margin-left:8px; }
  .legend { position:fixed; left:34px; bottom:26px; font-size:12px; color:var(--dim); }
  .legend span { display:inline-flex; align-items:center; margin-right:16px; }
  .legend i { width:9px; height:9px; border-radius:50%; margin-right:7px;
              display:inline-block; box-shadow:0 0 10px currentColor; }
  .label { position:fixed; transform:translate(-50%,-50%); font-size:12px; letter-spacing:1px;
           color:#d6e4ff; text-shadow:0 0 8px rgba(0,0,0,.95), 0 0 18px rgba(90,140,255,.6);
           pointer-events:none; white-space:nowrap; opacity:.92; }
  .card { position:fixed; width:300px; padding:15px 17px; border-radius:12px;
          background:linear-gradient(160deg, rgba(14,19,36,.94), rgba(9,12,24,.94));
          border:1px solid rgba(120,160,255,.2); box-shadow:0 12px 48px rgba(0,0,0,.7);
          backdrop-filter:blur(8px); opacity:0; transform:translateY(6px);
          transition:opacity .18s, transform .18s; pointer-events:none; }
  .card.on { opacity:1; transform:translateY(0); }
  .card .who { font-size:12px; color:#8fb0ff; letter-spacing:2px; }
  .card .txt { margin:9px 0 11px; font-size:13px; line-height:1.75; color:#dbe4f7; }
  .card .meta { font-size:11px; color:var(--dim); display:flex; justify-content:space-between; }
  .card .bar { height:2px; background:rgba(140,170,255,.14); margin-top:11px; border-radius:2px; }
  .card .bar i { display:block; height:100%; border-radius:2px;
                 background:linear-gradient(90deg,#4d8cff,#b06bff); }
</style>
</head>
<body>
<div class="hud"><h1>意识海</h1><div class="sub">记忆星图 · 每一颗星都是一段记忆</div></div>
<div class="stats" id="stats"></div>
<div class="legend" id="legend"></div>
<div class="card" id="card">
  <div class="who" id="who"></div><div class="txt" id="txt"></div>
  <div class="meta"><span id="kind"></span><span id="when"></span></div>
  <div class="bar"><i id="imp"></i></div>
</div>
__LIBS__
<script>
const DATA = __DATA__;
const W = () => innerWidth, H = () => innerHeight;

const renderer = new THREE.WebGLRenderer({ antialias: true, alpha: false });
renderer.setPixelRatio(Math.min(2, devicePixelRatio || 1));
renderer.setSize(W(), H());
renderer.setClearColor(0x02030a, 1);
document.body.appendChild(renderer.domElement);

const scene = new THREE.Scene();
scene.fog = new THREE.FogExp2(0x02030a, 0.055);
const camera = new THREE.PerspectiveCamera(42, W() / H(), 0.1, 200);
camera.position.set(0, 2.35, 3.9);
camera.lookAt(0, 0, 0);

// —— 莫比乌斯环粒子带 ——
const STARS = 34000;
const pos = new Float32Array(STARS * 3);
const col = new Float32Array(STARS * 3);
const R = 1.0, WIDTH = 0.40;
const c1 = new THREE.Color('#5b8cff'), c2 = new THREE.Color('#a06bff'), c3 = new THREE.Color('#3ee0c0');
for (let i = 0; i < STARS; i++) {
  const u = Math.random() * Math.PI * 2;
  const v = (Math.random() - 0.5) * WIDTH * 2;
  const half = u / 2;
  const r = R + v * Math.cos(half);
  pos[i * 3] = r * Math.cos(u) + (Math.random() - 0.5) * 0.012;
  pos[i * 3 + 1] = v * Math.sin(half) + (Math.random() - 0.5) * 0.012;
  pos[i * 3 + 2] = r * Math.sin(u) + (Math.random() - 0.5) * 0.012;
  const t = (Math.sin(u * 1.0 + 0.4) + 1) / 2;
  const c = t < 0.5 ? c1.clone().lerp(c2, t * 2) : c2.clone().lerp(c3, (t - 0.5) * 2);
  const f = 0.55 + Math.random() * 0.75;
  col[i * 3] = c.r * f; col[i * 3 + 1] = c.g * f; col[i * 3 + 2] = c.b * f;
}
const bandGeo = new THREE.BufferGeometry();
bandGeo.setAttribute('position', new THREE.BufferAttribute(pos, 3));
bandGeo.setAttribute('color', new THREE.BufferAttribute(col, 3));
const bandMat = new THREE.PointsMaterial({
  size: 0.0085, vertexColors: true, transparent: true, opacity: 0.62,
  blending: THREE.AdditiveBlending, depthWrite: false, sizeAttenuation: true,
});
const band = new THREE.Points(bandGeo, bandMat);
band.rotation.x = 0.62;
scene.add(band);

// —— 中心恒星（贴图 sprite + 光束）——
function glowTexture(inner, outer) {
  const c = document.createElement('canvas'); c.width = c.height = 256;
  const g = c.getContext('2d');
  const grad = g.createRadialGradient(128, 128, 0, 128, 128, 128);
  grad.addColorStop(0, 'rgba(255,255,255,0.95)');
  grad.addColorStop(0.12, inner);
  grad.addColorStop(0.55, outer);
  grad.addColorStop(1, 'rgba(0,0,0,0)');
  g.fillStyle = grad; g.fillRect(0, 0, 256, 256);
  return new THREE.CanvasTexture(c);
}
const coreColor = DATA.core.color;
const coreTex = glowTexture(`rgba(${coreColor},0.85)`, `rgba(${coreColor},0.16)`);
const core = new THREE.Sprite(new THREE.SpriteMaterial({
  map: coreTex, blending: THREE.AdditiveBlending, depthWrite: false, transparent: true,
}));
core.scale.set(0.62, 0.62, 1);
scene.add(core);

// 四向光束
const beamMat = new THREE.LineBasicMaterial({
  color: 0xbcd4ff, transparent: true, opacity: 0.07, blending: THREE.AdditiveBlending,
});
for (let i = 0; i < 4; i++) {
  const g = new THREE.BufferGeometry().setFromPoints([
    new THREE.Vector3(0, 0, 0),
    new THREE.Vector3(Math.cos(i * Math.PI / 2) * 4.2, Math.sin(i * Math.PI / 2) * 4.2, 0),
  ]);
  scene.add(new THREE.Line(g, beamMat));
}

// —— 记忆结点 ——
const nodeTex = (color) => glowTexture(`rgba(${color},0.95)`, `rgba(${color},0.25)`);
const nodes = [];
DATA.nodes.forEach((n, i) => {
  const half = n.u / 2;
  const r = R + n.v * Math.cos(half);
  const p = new THREE.Vector3(r * Math.cos(n.u), n.v * Math.sin(half), r * Math.sin(n.u));
  const sprite = new THREE.Sprite(new THREE.SpriteMaterial({
    map: nodeTex(n.color), blending: THREE.AdditiveBlending, depthWrite: false, transparent: true,
  }));
  const scale = 0.16 + n.imp * 0.22;
  sprite.scale.set(scale, scale, 1);
  sprite.position.copy(p);
  scene.add(sprite);

  const el = document.createElement('div');
  el.className = 'label';
  el.textContent = n.who;
  document.body.appendChild(el);
  nodes.push({ node: n, sprite, el, world: p });
});

// —— Bloom ——
const composer = new THREE.EffectComposer(renderer);
composer.addPass(new THREE.RenderPass(scene, camera));
const bloom = new THREE.UnrealBloomPass(new THREE.Vector2(W(), H()), 0.72, 0.62, 0.42);
composer.addPass(bloom);

// —— 交互：拖拽绕轨 ——
let yaw = 0.55, pitch = 0.30, dragging = false, lx = 0, ly = 0, auto = true;
const el = renderer.domElement;
el.addEventListener('mousedown', (e) => { dragging = true; auto = false; lx = e.clientX; ly = e.clientY; });
addEventListener('mouseup', () => { dragging = false; });
addEventListener('mousemove', (e) => {
  if (!dragging) return;
  yaw -= (e.clientX - lx) * 0.006;
  pitch = Math.max(-1.1, Math.min(1.1, pitch + (e.clientY - ly) * 0.004));
  lx = e.clientX; ly = e.clientY;
});
el.addEventListener('wheel', (e) => {
  const d = camera.position.length();
  const nd = Math.max(1.6, Math.min(7, d + e.deltaY * 0.0016));
  camera.position.setLength(nd); e.preventDefault();
}, { passive: false });

addEventListener('resize', () => {
  camera.aspect = W() / H(); camera.updateProjectionMatrix();
  renderer.setSize(W(), H()); composer.setSize(W(), H());
});

// —— 悬停卡片 ——
const card = document.getElementById('card');
let hovered = null;
el.addEventListener('mousemove', (e) => {
  if (dragging) { hovered = null; card.classList.remove('on'); return; }
  let best = null, bestD = 1e9;
  for (const item of nodes) {
    const p = item.world.clone().project(camera);
    const sx = (p.x * 0.5 + 0.5) * W(), sy = (-p.y * 0.5 + 0.5) * H();
    const d = Math.hypot(sx - e.clientX, sy - e.clientY);
    if (d < 26 && d < bestD) { best = item; bestD = d; }
  }
  hovered = best;
  if (!best) { card.classList.remove('on'); return; }
  const n = best.node;
  document.getElementById('who').textContent = n.who;
  document.getElementById('txt').textContent = n.text;
  document.getElementById('kind').textContent = n.kind + ' · 重要性 ' + n.imp.toFixed(2);
  document.getElementById('when').textContent = n.when;
  document.getElementById('imp').style.width = Math.round(n.imp * 100) + '%';
  card.style.left = Math.min(W() - 330, e.clientX + 18) + 'px';
  card.style.top = Math.max(20, Math.min(H() - 190, e.clientY - 20)) + 'px';
  card.classList.add('on');
});

// —— HUD ——
document.getElementById('stats').innerHTML =
  '<div>核心恒星<b>1</b></div><div>记忆<b>' + DATA.nodes.length + '</b></div>' +
  '<div>关系<b>' + DATA.rings.length + '</b></div><div>情绪<b>' + DATA.core.mood + '</b></div>';
document.getElementById('legend').innerHTML = DATA.rings.map(r =>
  `<span style="color:rgb(${r.color})"><i style="background:rgb(${r.color})"></i>${r.who} · ${r.count}</span>`).join('') +
  `<span style="color:rgb(${coreColor})"><i style="background:rgb(${coreColor})"></i>她 PAD ${DATA.core.pad}</span>`;

// —— 渲染循环 ——
const clock = new THREE.Clock();
const tmp = new THREE.Vector3();
function tick() {
  const t = clock.getElapsedTime();
  if (auto) yaw += 0.0009;
  const dist = camera.position.length();
  camera.position.set(
    Math.sin(yaw) * Math.cos(pitch) * dist,
    Math.sin(pitch) * dist,
    Math.cos(yaw) * Math.cos(pitch) * dist
  );
  camera.lookAt(0, 0, 0);

  core.scale.setScalar(0.60 + 0.03 * Math.sin(t * 0.9));
  bandMat.size = 0.0074 + 0.0004 * Math.sin(t * 0.7);

  for (const item of nodes) {
    tmp.copy(item.world).project(camera);
    const sx = (tmp.x * 0.5 + 0.5) * W(), sy = (-tmp.y * 0.5 + 0.5) * H();
    const visible = tmp.z < 1;
    item.el.style.display = visible ? 'block' : 'none';
    item.el.style.left = sx + 'px';
    item.el.style.top = (sy - 16) + 'px';
    item.el.style.opacity = visible ? 0.55 + 0.35 * item.node.imp : 0;
  }
  composer.render();
  requestAnimationFrame(tick);
}
tick();
</script>
</body>
</html>
"""


def _ensure_libs(cache_dir: Path) -> dict[str, str]:
    cache_dir.mkdir(parents=True, exist_ok=True)
    blobs: dict[str, str] = {}
    for name in _ORDER:
        path = cache_dir / name
        if not path.exists() or path.stat().st_size < 200:
            url = _CDN + _LIBS[name]
            with urllib.request.urlopen(url, timeout=30) as resp:
                path.write_bytes(resp.read())
        blobs[name] = path.read_text(encoding="utf-8", errors="ignore")
    return blobs


def render_webgl_html(payload: dict[str, Any], *, cache_dir: Path) -> str:
    libs = _ensure_libs(cache_dir)
    blocks = "\n".join(f"<script>{libs[name]}</script>" for name in _ORDER)
    data = json.dumps(payload, ensure_ascii=False).replace("</", "<\\/")
    return _HTML.replace("__LIBS__", blocks).replace("__DATA__", data)


def export_webgl_map(
    mem_path: str | Path,
    state_path: str | Path,
    out_path: str | Path,
    *,
    cache_dir: str | Path | None = None,
) -> Path:
    payload = build_payload(mem_path, state_path)
    cache = Path(cache_dir) if cache_dir else Path(out_path).parent / "_libs"
    html = render_webgl_html(payload, cache_dir=cache)
    target = Path(out_path)
    target.write_text(html, encoding="utf-8")
    return target


__all__ = ["export_webgl_map", "render_webgl_html"]
