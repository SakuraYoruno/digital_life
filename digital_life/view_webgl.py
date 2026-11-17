"""WebGL 记忆星图：莫比乌斯环 + 体积粒子 + Bloom + 诊断面板。

参考视觉：环带斜着扫过整个画幅、边缘有强泛光、中心恒星带十字光束、
四周浮着等宽字体的"诊断面板"（记忆写入/索引状态/异常）。

数据全部来自 `memory.db` / `state.db`；three.js 与后处理库在生成时内联，
输出单文件 HTML，离线可开。
"""

from __future__ import annotations

import json
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
  :root { --ink:#eaf1ff; --dim:#8291b4; --red:#ff3b46; }
  * { box-sizing:border-box; }
  html,body { margin:0; height:100%; background:#02030a; overflow:hidden; color:var(--ink);
              font:13px/1.6 "Segoe UI","Microsoft YaHei",sans-serif; }
  canvas { display:block; }
  .hud { position:fixed; left:38px; top:30px; pointer-events:none; z-index:5; }
  .hud h1 { margin:0; font-size:30px; letter-spacing:10px; font-weight:600;
            text-shadow:0 0 28px rgba(140,190,255,.65); }
  .hud .sub { color:var(--dim); font-size:12px; letter-spacing:4px; margin-top:8px; }
  .hud .note { color:#5d6b8c; font-size:11px; margin-top:16px; letter-spacing:1px; }
  .stats { position:fixed; right:38px; top:30px; text-align:right; font-size:12px;
           color:var(--dim); letter-spacing:1.5px; z-index:5; }
  .stats div { margin-bottom:5px; }
  .stats b { color:var(--ink); font-size:18px; font-weight:500; margin-left:8px; }
  .legend { position:fixed; left:38px; bottom:26px; font-size:11px; color:var(--dim);
            letter-spacing:1px; z-index:5; }
  .legend span { display:inline-flex; align-items:center; margin-right:16px; }
  .legend i { width:8px; height:8px; border-radius:50%; margin-right:7px; display:inline-block;
              box-shadow:0 0 10px currentColor; }
  .panel { position:fixed; width:248px; padding:9px 11px; background:rgba(6,8,14,.82);
           border:1px solid rgba(90,110,160,.30); border-left:2px solid var(--red);
           box-shadow:0 0 26px rgba(0,0,0,.75); font:10.5px/1.55 Consolas,"Cascadia Mono",monospace;
           color:#9fb0d0; letter-spacing:.2px; pointer-events:none; z-index:4; }
  .panel .t { color:#ff6b73; letter-spacing:2px; font-size:10px; margin-bottom:5px; }
  .panel .row { white-space:nowrap; overflow:hidden; text-overflow:ellipsis; }
  .panel.ok { border-left-color:#3ee0c0; }
  .panel.ok .t { color:#4fe3c4; }
  .card { position:fixed; width:300px; padding:15px 17px; border-radius:12px;
          background:linear-gradient(160deg, rgba(14,19,36,.94), rgba(9,12,24,.94));
          border:1px solid rgba(120,160,255,.2); box-shadow:0 12px 48px rgba(0,0,0,.7);
          backdrop-filter:blur(8px); opacity:0; transform:translateY(6px);
          transition:opacity .18s, transform .18s; pointer-events:none; z-index:6; }
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
<div class="hud">
  <h1>意识海</h1>
  <div class="sub">记忆星图</div>
  <div class="note">每一颗星都是一段记忆<br>只增加，不删除</div>
</div>
<div class="stats" id="stats"></div>
<div class="legend" id="legend"></div>
<div id="panels"></div>
<div class="card" id="card">
  <div class="who" id="who"></div><div class="txt" id="txt"></div>
  <div class="meta"><span id="kind"></span><span id="when"></span></div>
  <div class="bar"><i id="imp"></i></div>
</div>
__LIBS__
<script>
const DATA = __DATA__;
const W = () => innerWidth, H = () => innerHeight;

const renderer = new THREE.WebGLRenderer({ antialias: true });
renderer.setPixelRatio(Math.min(2, devicePixelRatio || 1));
renderer.setSize(W(), H());
renderer.setClearColor(0x02030a, 1);
document.body.appendChild(renderer.domElement);

const scene = new THREE.Scene();
const camera = new THREE.PerspectiveCamera(46, W() / H(), 0.1, 200);

// —— 莫比乌斯带 ——
const R = 1.32, WIDTH = 0.5;
function mobius(u, v) {
  const half = u / 2, r = R + v * Math.cos(half);
  return new THREE.Vector3(r * Math.cos(u), v * Math.sin(half), r * Math.sin(u));
}

function buildBand(count, size, opacity, spread, hueShift) {
  const pos = new Float32Array(count * 3), col = new Float32Array(count * 3);
  const c1 = new THREE.Color('#4f7bff'), c2 = new THREE.Color('#9a5cff'), c3 = new THREE.Color('#54f2c8');
  for (let i = 0; i < count; i++) {
    const u = Math.random() * Math.PI * 2;
    const v = (Math.random() - 0.5) * WIDTH * 2;
    const p = mobius(u, v);
    pos[i*3] = p.x + (Math.random() - 0.5) * spread;
    pos[i*3+1] = p.y + (Math.random() - 0.5) * spread;
    pos[i*3+2] = p.z + (Math.random() - 0.5) * spread;
    const t = (Math.sin(u + hueShift) + 1) / 2;
    const c = t < 0.5 ? c1.clone().lerp(c2, t * 2) : c2.clone().lerp(c3, (t - 0.5) * 2);
    const f = 0.45 + Math.random() * 0.85;
    col[i*3] = c.r * f; col[i*3+1] = c.g * f; col[i*3+2] = c.b * f;
  }
  const g = new THREE.BufferGeometry();
  g.setAttribute('position', new THREE.BufferAttribute(pos, 3));
  g.setAttribute('color', new THREE.BufferAttribute(col, 3));
  return new THREE.Points(g, new THREE.PointsMaterial({
    size, vertexColors: true, transparent: true, opacity,
    blending: THREE.AdditiveBlending, depthWrite: false, sizeAttenuation: true,
  }));
}

const band = buildBand(70000, 0.010, 0.55, 0.02, 0.4);
const bandBright = buildBand(9000, 0.020, 0.85, 0.01, 0.4);
const bandSpark = buildBand(2600, 0.030, 1.0, 0.03, 1.6);
const group = new THREE.Group();
group.add(band, bandBright, bandSpark);
group.rotation.x = 0.66;
group.rotation.z = -0.34;
scene.add(group);

// —— 星野 ——
const skyCount = 2600;
const skyPos = new Float32Array(skyCount * 3), skyCol = new Float32Array(skyCount * 3);
for (let i = 0; i < skyCount; i++) {
  const r = 6 + Math.random() * 10;
  const th = Math.random() * Math.PI * 2, ph = Math.acos(2 * Math.random() - 1);
  skyPos[i*3] = r * Math.sin(ph) * Math.cos(th);
  skyPos[i*3+1] = r * Math.cos(ph);
  skyPos[i*3+2] = r * Math.sin(ph) * Math.sin(th);
  const f = 0.2 + Math.random() * 0.6;
  skyCol[i*3] = 0.7 * f; skyCol[i*3+1] = 0.8 * f; skyCol[i*3+2] = f;
}
const skyGeo = new THREE.BufferGeometry();
skyGeo.setAttribute('position', new THREE.BufferAttribute(skyPos, 3));
skyGeo.setAttribute('color', new THREE.BufferAttribute(skyCol, 3));
scene.add(new THREE.Points(skyGeo, new THREE.PointsMaterial({
  size: 0.05, vertexColors: true, transparent: true, opacity: 0.9,
  blending: THREE.AdditiveBlending, depthWrite: false,
})));

// —— 星云 ——
function glowTex(inner, outer) {
  const c = document.createElement('canvas'); c.width = c.height = 256;
  const g = c.getContext('2d');
  const grad = g.createRadialGradient(128, 128, 0, 128, 128, 128);
  grad.addColorStop(0, inner); grad.addColorStop(0.45, outer); grad.addColorStop(1, 'rgba(0,0,0,0)');
  g.fillStyle = grad; g.fillRect(0, 0, 256, 256);
  return new THREE.CanvasTexture(c);
}
[['rgba(70,110,255,0.5)', 'rgba(40,60,160,0.12)'],
 ['rgba(140,70,255,0.42)', 'rgba(70,30,140,0.10)'],
 ['rgba(60,220,190,0.32)', 'rgba(20,90,90,0.08)']].forEach(([inner, outer], i) => {
  const s = new THREE.Sprite(new THREE.SpriteMaterial({
    map: glowTex(inner, outer), blending: THREE.AdditiveBlending,
    transparent: true, opacity: 0.55, depthWrite: false,
  }));
  const ang = i * 2.1;
  s.position.set(Math.cos(ang) * 2.0, Math.sin(ang) * 0.8, Math.sin(ang) * 1.6);
  s.scale.set(5.5, 3.4, 1);
  scene.add(s);
});

// —— 中心恒星 ——
const coreCol = DATA.core.color;
const core = new THREE.Sprite(new THREE.SpriteMaterial({
  map: glowTex(`rgba(${coreCol},0.95)`, `rgba(${coreCol},0.25)`),
  blending: THREE.AdditiveBlending, transparent: true, depthWrite: false,
}));
core.scale.set(0.85, 0.85, 1);
scene.add(core);

const beamMat = new THREE.LineBasicMaterial({
  color: 0xd6e6ff, transparent: true, opacity: 0.5,
  blending: THREE.AdditiveBlending, depthWrite: false,
});
for (let i = 0; i < 4; i++) {
  const a = i * Math.PI / 2 + 0.15;
  const len = 3.6 + (i % 2) * 1.6;
  scene.add(new THREE.Line(new THREE.BufferGeometry().setFromPoints([
    new THREE.Vector3(0, 0, 0),
    new THREE.Vector3(Math.cos(a) * len, Math.sin(a) * len, 0),
  ]), beamMat));
}

// —— 记忆结点 ——
const nodes = [];
DATA.nodes.forEach((n) => {
  const p = mobius(n.u, n.v);
  p.applyEuler(new THREE.Euler(0.62, 0, -0.22));
  const sprite = new THREE.Sprite(new THREE.SpriteMaterial({
    map: glowTex(`rgba(${n.color},0.95)`, `rgba(${n.color},0.3)`),
    blending: THREE.AdditiveBlending, transparent: true, depthWrite: false,
  }));
  const s = 0.14 + n.imp * 0.26;
  sprite.scale.set(s, s, 1);
  sprite.position.copy(p);
  scene.add(sprite);
  const el = document.createElement('div');
  el.textContent = n.who;
  el.style.cssText = 'position:fixed;transform:translate(-50%,-50%);font-size:11px;' +
    'color:#d6e4ff;text-shadow:0 0 8px #000,0 0 16px rgba(90,140,255,.7);pointer-events:none;' +
    'white-space:nowrap;opacity:.9;z-index:3';
  document.body.appendChild(el);
  nodes.push({ node: n, sprite, el, world: p });
});

// —— 后处理 ——
const composer = new THREE.EffectComposer(renderer);
composer.addPass(new THREE.RenderPass(scene, camera));
const bloom = new THREE.UnrealBloomPass(new THREE.Vector2(W(), H()), 1.15, 0.85, 0.12);
composer.addPass(bloom);

// —— 相机 ——
let yaw = 0.12, pitch = 0.74, dist = 3.45, dragging = false, lx = 0, ly = 0, auto = true;
function applyCamera() {
  camera.position.set(
    Math.sin(yaw) * Math.cos(pitch) * dist,
    Math.sin(pitch) * dist,
    Math.cos(yaw) * Math.cos(pitch) * dist
  );
  camera.lookAt(0, 0, 0);
}
applyCamera();

const el = renderer.domElement;
el.addEventListener('mousedown', (e) => { dragging = true; auto = false; lx = e.clientX; ly = e.clientY; });
addEventListener('mouseup', () => { dragging = false; });
addEventListener('mousemove', (e) => {
  if (!dragging) return;
  yaw -= (e.clientX - lx) * 0.005;
  pitch = Math.max(-1.2, Math.min(1.2, pitch + (e.clientY - ly) * 0.004));
  lx = e.clientX; ly = e.clientY;
  applyCamera();
});
el.addEventListener('wheel', (e) => {
  dist = Math.max(1.8, Math.min(8, dist + e.deltaY * 0.0015));
  applyCamera(); e.preventDefault();
}, { passive: false });
addEventListener('resize', () => {
  camera.aspect = W() / H(); camera.updateProjectionMatrix();
  renderer.setSize(W(), H()); composer.setSize(W(), H());
});

// —— 悬停卡片 ——
const card = document.getElementById('card');
el.addEventListener('mousemove', (e) => {
  if (dragging) { card.classList.remove('on'); return; }
  let best = null, bestD = 1e9;
  for (const it of nodes) {
    const p = it.world.clone().project(camera);
    const sx = (p.x * 0.5 + 0.5) * W(), sy = (-p.y * 0.5 + 0.5) * H();
    const d = Math.hypot(sx - e.clientX, sy - e.clientY);
    if (d < 28 && d < bestD) { best = it; bestD = d; }
  }
  if (!best) { card.classList.remove('on'); return; }
  const n = best.node;
  document.getElementById('who').textContent = n.who;
  document.getElementById('txt').textContent = n.text;
  document.getElementById('kind').textContent = n.kind + ' · 重要性 ' + n.imp.toFixed(2);
  document.getElementById('when').textContent = n.when;
  document.getElementById('imp').style.width = Math.round(n.imp * 100) + '%';
  card.style.left = Math.min(W() - 330, e.clientX + 18) + 'px';
  card.style.top = Math.max(20, Math.min(H() - 190, e.clientY - 30)) + 'px';
  card.classList.add('on');
});

// —— HUD ——
document.getElementById('stats').innerHTML =
  '<div>核心恒星<b>1</b></div>' +
  '<div>觉星<b>' + DATA.counts.episodic + '</b></div>' +
  '<div>尘星<b>' + DATA.counts.knowledge + '</b></div>' +
  '<div>关系<b>' + DATA.rings.length + '</b></div>' +
  '<div>情绪<b>' + DATA.core.mood + '</b></div>';

document.getElementById('legend').innerHTML = DATA.rings.map(r =>
  `<span style="color:rgb(${r.color})"><i style="background:rgb(${r.color})"></i>${r.who} · ${r.count}</span>`).join('') +
  `<span style="color:rgb(${coreCol})"><i style="background:rgb(${coreCol})"></i>她 PAD ${DATA.core.pad}</span>`;

const spots = [
  { left: '38px', top: '196px' },
  { left: '38px', top: 'calc(50% + 30px)' },
  { right: '38px', top: '150px' },
  { right: '38px', top: 'calc(50% + 110px)' },
];
DATA.panels.forEach((panel, i) => {
  const spot = spots[i % spots.length];
  const div = document.createElement('div');
  div.className = 'panel' + (panel.ok ? ' ok' : '');
  div.style.cssText = Object.entries(spot).map(([k, v]) => `${k}:${v}`).join(';');
  div.innerHTML = `<div class="t">${panel.title}</div>` +
    panel.rows.map(r => `<div class="row">${r}</div>`).join('');
  document.getElementById('panels').appendChild(div);
});

// —— 循环 ——
const clock = new THREE.Clock();
const tmp = new THREE.Vector3();
function tick() {
  const t = clock.getElapsedTime();
  if (auto) group.rotation.y += 0.0011;
  core.scale.setScalar(0.82 + 0.04 * Math.sin(t * 1.1));
  for (const it of nodes) {
    tmp.copy(it.world).project(camera);
    const sx = (tmp.x * 0.5 + 0.5) * W(), sy = (-tmp.y * 0.5 + 0.5) * H();
    it.el.style.display = tmp.z < 1 ? 'block' : 'none';
    it.el.style.left = sx + 'px';
    it.el.style.top = (sy - 14) + 'px';
    it.el.style.opacity = 0.45 + 0.4 * it.node.imp;
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
            with urllib.request.urlopen(_CDN + _LIBS[name], timeout=30) as resp:
                path.write_bytes(resp.read())
        blobs[name] = path.read_text(encoding="utf-8", errors="ignore")
    return blobs


def _panels(mem_path: str | Path, state: Any) -> list[dict[str, Any]]:
    store = MemoryStore(mem_path)
    rows = list(
        store._conn.execute(
            "SELECT uid, kind, text, importance, ts FROM memories ORDER BY ts DESC LIMIT 3"
        )
    )
    total = store._conn.execute("SELECT COUNT(*) FROM memories").fetchone()[0]
    store.close()

    panels = []
    for row in rows[:2]:
        stamp = time.strftime("%m-%d %H:%M", time.localtime(float(row["ts"])))
        text = str(row["text"]).replace("\n", " ")
        panels.append({
            "title": "MEM-WRITE / " + str(row["kind"]).upper(),
            "rows": [f"{stamp}  {row['uid']}", f"imp {float(row['importance']):.2f}",
                     text[:34]],
            "ok": True,
        })
    panels.append({
        "title": "MEMORY INDEX / 索引状态",
        "rows": [f"committed  {total} 条", "vector    bge-m3 1024d",
                 "index     sqlite-vec", "status    OK"],
        "ok": True,
    })
    return panels


def build_webgl_payload(mem_path: str | Path, state_path: str | Path) -> dict[str, Any]:
    payload = build_payload(mem_path, state_path)
    counts = {"episodic": 0, "knowledge": 0}
    for node in payload["nodes"]:
        counts["knowledge" if node["kind"] == "知识" else "episodic"] += 1
    payload["counts"] = counts
    state_store = StateStore(state_path)
    state = state_store.load()
    state_store.close()
    payload["panels"] = _panels(mem_path, state)
    return payload


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
    payload = build_webgl_payload(mem_path, state_path)
    cache = Path(cache_dir) if cache_dir else Path(out_path).parent / "_libs"
    html = render_webgl_html(payload, cache_dir=cache)
    target = Path(out_path)
    target.write_text(html, encoding="utf-8")
    return target


__all__ = ["build_webgl_payload", "export_webgl_map", "render_webgl_html"]
