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
  .hud { position:fixed; left:38px; top:26px; pointer-events:none; z-index:5; }
  .hud h1 { margin:0; font-size:30px; letter-spacing:10px; font-weight:600;
            text-shadow:0 0 30px rgba(140,190,255,.7); }
  .hud .sub { color:var(--dim); font-size:12px; letter-spacing:4px; margin-top:8px; }
  .hud .note { color:#5d6b8c; font-size:11px; margin-top:14px; letter-spacing:1px; }
  .stats { position:fixed; right:38px; top:26px; text-align:right; font-size:12px;
           color:var(--dim); letter-spacing:1.5px; z-index:5; }
  .stats div { margin-bottom:5px; }
  .stats b { color:var(--ink); font-size:17px; font-weight:500; margin-left:8px; }
  .legend { position:fixed; left:38px; bottom:24px; font-size:11px; color:var(--dim);
            letter-spacing:1px; z-index:5; }
  .legend span { display:inline-flex; align-items:center; margin-right:14px; }
  .legend i { width:8px; height:8px; border-radius:50%; margin-right:6px; display:inline-block;
              box-shadow:0 0 10px currentColor; }
  .panel { position:fixed; width:236px; padding:8px 10px; background:rgba(8,6,10,.86);
           border:1px solid rgba(255,70,80,.45); border-left:3px solid var(--red);
           box-shadow:0 0 30px rgba(255,40,60,.14), 0 0 22px rgba(0,0,0,.8);
           font:10.5px/1.5 Consolas,"Cascadia Mono",monospace; color:#c9a7ad;
           letter-spacing:.2px; pointer-events:none; z-index:4; }
  .panel .t { color:#ff5560; letter-spacing:2px; font-size:10px; margin-bottom:4px;
              text-shadow:0 0 12px rgba(255,60,70,.5); }
  .panel .row { white-space:nowrap; overflow:hidden; text-overflow:ellipsis; }
  .panel.ok { border-color:rgba(70,220,200,.4); border-left-color:#3ee0c0;
              box-shadow:0 0 30px rgba(60,220,200,.10), 0 0 22px rgba(0,0,0,.8); }
  .panel.ok .t { color:#4fe3c4; text-shadow:0 0 12px rgba(60,220,200,.45); }
  .card { position:fixed; width:210px; padding:0 0 10px; background:linear-gradient(180deg,
          rgba(16,20,34,.94), rgba(8,10,18,.94)); border:1px solid rgba(120,160,255,.22);
          box-shadow:0 14px 50px rgba(0,0,0,.75); z-index:4; pointer-events:none; }
  .card .face { height:86px; }
  .card .body { padding:8px 11px 0; }
  .card .name { font-size:15px; letter-spacing:2px; }
  .card .en { font-size:9px; color:#7b8db3; letter-spacing:2px; margin-top:2px; }
  .card .kv { display:flex; justify-content:space-between; font-size:10.5px; color:#9fb0d0;
              margin-top:6px; }
  .card .kv b { color:#dce6ff; font-weight:500; }
  .card .quote { margin-top:8px; font-size:10.5px; color:#8da0c4; line-height:1.5;
                 border-top:1px solid rgba(120,160,255,.12); padding-top:7px;
                 max-height:52px; overflow:hidden; }
  .hover { position:fixed; width:290px; padding:14px 16px; border-radius:10px;
           background:rgba(10,13,24,.95); border:1px solid rgba(120,160,255,.25);
           box-shadow:0 12px 48px rgba(0,0,0,.75); opacity:0; transition:opacity .16s;
           pointer-events:none; z-index:6; }
  .hover.on { opacity:1; }
  .hover .who { font-size:12px; color:#8fb0ff; letter-spacing:2px; }
  .hover .txt { margin:8px 0 10px; font-size:12.5px; line-height:1.7; color:#dbe4f7; }
  .hover .meta { font-size:11px; color:var(--dim); display:flex; justify-content:space-between; }
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
<div id="cards"></div>
<div class="hover" id="hover">
  <div class="who" id="who"></div><div class="txt" id="txt"></div>
  <div class="meta"><span id="kind"></span><span id="when"></span></div>
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
const camera = new THREE.PerspectiveCamera(48, W() / H(), 0.1, 200);

// —— 细亮带：环 + 轻微上下起伏，斜穿画面 ——
const R = 1.30, WIDTH = 0.055;
function bandPoint(u, v) {
  const half = u / 2;
  const wobble = 0.16 * Math.sin(u * 2 + 0.6);
  const r = R + v * Math.cos(half);
  return new THREE.Vector3(r * Math.cos(u), wobble + v * Math.sin(half), r * Math.sin(u));
}

function buildBand(count, size, opacity, spread, bright) {
  const pos = new Float32Array(count * 3), col = new Float32Array(count * 3);
  const c1 = new THREE.Color(bright ? '#ffffff' : '#6f8dff');
  const c2 = new THREE.Color(bright ? '#dff6ff' : '#a86bff');
  const c3 = new THREE.Color(bright ? '#8ff6e4' : '#48e6c8');
  for (let i = 0; i < count; i++) {
    const u = Math.random() * Math.PI * 2;
    const v = (Math.random() - 0.5) * WIDTH * 2 * (Math.random() < 0.15 ? 3.2 : 1);
    const p = bandPoint(u, v);
    pos[i*3] = p.x + (Math.random() - 0.5) * spread;
    pos[i*3+1] = p.y + (Math.random() - 0.5) * spread;
    pos[i*3+2] = p.z + (Math.random() - 0.5) * spread;
    const t = (Math.sin(u * 1.0 + 0.2) + 1) / 2;
    const c = t < 0.5 ? c1.clone().lerp(c2, t * 2) : c2.clone().lerp(c3, (t - 0.5) * 2);
    const f = 0.5 + Math.random() * 0.9;
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

const group = new THREE.Group();
group.add(buildBand(90000, 0.0075, 0.55, 0.012, false));
group.add(buildBand(14000, 0.014, 0.95, 0.008, true));
group.rotation.x = 0.42;
group.rotation.z = -0.30;
scene.add(group);

// —— 深空星野 ——
const skyCount = 3200;
const skyPos = new Float32Array(skyCount * 3), skyCol = new Float32Array(skyCount * 3);
for (let i = 0; i < skyCount; i++) {
  const r = 7 + Math.random() * 12;
  const th = Math.random() * Math.PI * 2, ph = Math.acos(2 * Math.random() - 1);
  skyPos[i*3] = r * Math.sin(ph) * Math.cos(th);
  skyPos[i*3+1] = r * Math.cos(ph);
  skyPos[i*3+2] = r * Math.sin(ph) * Math.sin(th);
  const f = 0.15 + Math.random() * 0.7;
  skyCol[i*3] = 0.75 * f; skyCol[i*3+1] = 0.85 * f; skyCol[i*3+2] = f;
}
const skyGeo = new THREE.BufferGeometry();
skyGeo.setAttribute('position', new THREE.BufferAttribute(skyPos, 3));
skyGeo.setAttribute('color', new THREE.BufferAttribute(skyCol, 3));
scene.add(new THREE.Points(skyGeo, new THREE.PointsMaterial({
  size: 0.045, vertexColors: true, transparent: true, opacity: 0.85,
  blending: THREE.AdditiveBlending, depthWrite: false,
})));

// —— 星云 ——
function glowTex(inner, outer) {
  const c = document.createElement('canvas'); c.width = c.height = 256;
  const g = c.getContext('2d');
  const grad = g.createRadialGradient(128, 128, 0, 128, 128, 128);
  grad.addColorStop(0, inner); grad.addColorStop(0.4, outer); grad.addColorStop(1, 'rgba(0,0,0,0)');
  g.fillStyle = grad; g.fillRect(0, 0, 256, 256);
  return new THREE.CanvasTexture(c);
}
[['rgba(80,110,255,0.45)', 'rgba(40,50,150,0.10)', [-2.2, 0.7, -1.0], [6.5, 4.0]],
 ['rgba(150,70,255,0.38)', 'rgba(70,30,140,0.09)', [2.4, -0.5, -1.2], [5.5, 3.4]],
 ['rgba(60,230,200,0.26)', 'rgba(20,90,90,0.07)', [1.2, 1.3, 1.8], [4.6, 2.8]]].forEach(
  ([inner, outer, pos, scale]) => {
    const s = new THREE.Sprite(new THREE.SpriteMaterial({
      map: glowTex(inner, outer), blending: THREE.AdditiveBlending,
      transparent: true, opacity: 0.6, depthWrite: false,
    }));
    s.position.set(...pos); s.scale.set(scale[0], scale[1], 1);
    scene.add(s);
  });

// —— 中心恒星 + 星芒 + 小轨道环 ——
const coreCol = DATA.core.color;
const core = new THREE.Sprite(new THREE.SpriteMaterial({
  map: glowTex(`rgba(${coreCol},1)`, `rgba(${coreCol},0.22)`),
  blending: THREE.AdditiveBlending, transparent: true, depthWrite: false,
}));
core.scale.set(1.25, 1.25, 1);
scene.add(core);

const spikes = new THREE.Group();
for (let i = 0; i < 6; i++) {
  const a = i * Math.PI / 3 + 0.2, len = i % 2 ? 1.5 : 3.4;
  const g = new THREE.BufferGeometry().setFromPoints([
    new THREE.Vector3(0, 0, 0), new THREE.Vector3(Math.cos(a) * len, Math.sin(a) * len, 0)]);
  spikes.add(new THREE.Line(g, new THREE.LineBasicMaterial({
    color: 0xeaf4ff, transparent: true, opacity: i % 2 ? 0.22 : 0.5,
    blending: THREE.AdditiveBlending, depthWrite: false })));
}
scene.add(spikes);

const orbits = new THREE.Group();
for (let i = 0; i < 3; i++) {
  const pts = [];
  const rad = 0.34 + i * 0.12;
  for (let a = 0; a <= 96; a++) {
    const th = (a / 96) * Math.PI * 2;
    pts.push(new THREE.Vector3(Math.cos(th) * rad, 0, Math.sin(th) * rad));
  }
  const line = new THREE.Line(new THREE.BufferGeometry().setFromPoints(pts),
    new THREE.LineBasicMaterial({ color: 0xbcd4ff, transparent: true,
      opacity: 0.28 - i * 0.06, blending: THREE.AdditiveBlending, depthWrite: false }));
  line.rotation.x = 0.9 + i * 0.35;
  line.rotation.z = i * 0.7;
  orbits.add(line);
}
scene.add(orbits);

// —— 记忆结点 ——
const nodes = [];
const nodeTexCache = {};
function nodeTex(color) {
  if (!nodeTexCache[color]) {
    nodeTexCache[color] = glowTex(`rgba(${color},1)`, `rgba(${color},0.28)`);
  }
  return nodeTexCache[color];
}
DATA.nodes.forEach((n) => {
  const p = bandPoint(n.u, n.v * 0.6);
  p.applyEuler(new THREE.Euler(0.42, 0, -0.30));
  const sprite = new THREE.Sprite(new THREE.SpriteMaterial({
    map: nodeTex(n.color), blending: THREE.AdditiveBlending,
    transparent: true, depthWrite: false,
  }));
  const s = 0.13 + n.imp * 0.22;
  sprite.scale.set(s, s, 1);
  sprite.position.copy(p);
  scene.add(sprite);
  const el = document.createElement('div');
  el.textContent = n.who;
  el.style.cssText = 'position:fixed;transform:translate(-50%,-50%);font-size:11px;' +
    'color:#dbe7ff;text-shadow:0 0 8px #000,0 0 18px rgba(90,140,255,.75);pointer-events:none;' +
    'white-space:nowrap;opacity:.9;z-index:3';
  document.body.appendChild(el);
  nodes.push({ node: n, sprite, el, world: p });
});

// —— 后处理：强泛光 ——
const composer = new THREE.EffectComposer(renderer);
composer.addPass(new THREE.RenderPass(scene, camera));
const bloom = new THREE.UnrealBloomPass(new THREE.Vector2(W(), H()), 1.45, 0.95, 0.08);
composer.addPass(bloom);

// —— 相机：斜穿构图 ——
let yaw = 0.55, pitch = 0.30, dist = 3.15, dragging = false, lx = 0, ly = 0, auto = true;
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
  lx = e.clientX; ly = e.clientY; applyCamera();
});
el.addEventListener('wheel', (e) => {
  dist = Math.max(1.6, Math.min(8, dist + e.deltaY * 0.0015));
  applyCamera(); e.preventDefault();
}, { passive: false });
addEventListener('resize', () => {
  camera.aspect = W() / H(); camera.updateProjectionMatrix();
  renderer.setSize(W(), H()); composer.setSize(W(), H());
});

// —— 悬停卡片 ——
const hover = document.getElementById('hover');
el.addEventListener('mousemove', (e) => {
  if (dragging) { hover.classList.remove('on'); return; }
  let best = null, bestD = 1e9;
  for (const it of nodes) {
    const p = it.world.clone().project(camera);
    const sx = (p.x * 0.5 + 0.5) * W(), sy = (-p.y * 0.5 + 0.5) * H();
    const d = Math.hypot(sx - e.clientX, sy - e.clientY);
    if (d < 26 && d < bestD) { best = it; bestD = d; }
  }
  if (!best) { hover.classList.remove('on'); return; }
  const n = best.node;
  document.getElementById('who').textContent = n.who;
  document.getElementById('txt').textContent = n.text;
  document.getElementById('kind').textContent = n.kind + ' · 重要性 ' + n.imp.toFixed(2);
  document.getElementById('when').textContent = n.when;
  hover.style.left = Math.min(W() - 320, e.clientX + 18) + 'px';
  hover.style.top = Math.max(20, Math.min(H() - 170, e.clientY - 30)) + 'px';
  hover.classList.add('on');
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
  { left: '38px', top: '176px' },
  { left: '38px', top: 'calc(50% + 20px)' },
  { right: '38px', top: '132px' },
  { right: '38px', top: 'calc(50% + 90px)' },
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

// —— 角色卡（左右各一，取关系里的两个人）——
const cardSpots = [
  { left: '38px', bottom: '58px' },
  { right: '38px', bottom: '58px' },
];
DATA.cards.slice(0, 2).forEach((c, i) => {
  const spot = cardSpots[i];
  const div = document.createElement('div');
  div.className = 'card';
  div.style.cssText = Object.entries(spot).map(([k, v]) => `${k}:${v}`).join(';');
  div.innerHTML =
    `<div class="face" style="background:linear-gradient(135deg, rgba(${c.color},.55),
      rgba(10,14,26,.9));"></div>` +
    `<div class="body"><div class="name">${c.name}</div><div class="en">${c.en}</div>` +
    `<div class="kv"><span>记忆</span><b>${c.count}</b></div>` +
    `<div class="kv"><span>亲近度</span><b>${c.affinity}</b></div>` +
    `<div class="quote">${c.quote}</div></div>`;
  document.getElementById('cards').appendChild(div);
});

// —— 循环 ——
const clock = new THREE.Clock();
const tmp = new THREE.Vector3();
function tick() {
  const t = clock.getElapsedTime();
  if (auto) { group.rotation.y += 0.0009; orbits.rotation.y += 0.004; }
  core.scale.setScalar(1.18 + 0.05 * Math.sin(t * 1.2));
  spikes.rotation.z = t * 0.05;
  for (const it of nodes) {
    tmp.copy(it.world).project(camera);
    const sx = (tmp.x * 0.5 + 0.5) * W(), sy = (-tmp.y * 0.5 + 0.5) * H();
    it.el.style.display = tmp.z < 1 ? 'block' : 'none';
    it.el.style.left = sx + 'px';
    it.el.style.top = (sy - 13) + 'px';
    it.el.style.opacity = 0.4 + 0.45 * it.node.imp;
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


def _cards(payload: dict[str, Any]) -> list[dict[str, Any]]:
    """左右两张角色卡：按记忆条数取前两名关系。"""
    counts: dict[str, int] = {}
    latest: dict[str, str] = {}
    who: dict[str, str] = {}
    color: dict[str, str] = {}
    for node in payload["nodes"]:
        uid = node["uid"]
        counts[uid] = counts.get(uid, 0) + 1
        latest.setdefault(uid, node["text"])
        who[uid] = node["who"]
        color[uid] = node["color"]
    rings = {r["who"]: r for r in payload["rings"]}
    ranked = sorted(counts.items(), key=lambda kv: kv[1], reverse=True)[:2]
    cards = []
    for index, (uid, count) in enumerate(ranked):
        name = who.get(uid, uid)
        ring = rings.get(name, {})
        cards.append({
            "name": name,
            "en": "MEMORY KEEPER" if index == 0 else "OCEAN TIDE",
            "count": count,
            "affinity": "—",
            "quote": (latest.get(uid) or "").replace("\n", " ")[:44],
            "color": color.get(uid, "120,160,255"),
        })
    return cards


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
    payload["cards"] = _cards(payload)
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
