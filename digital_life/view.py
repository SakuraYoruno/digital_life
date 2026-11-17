"""把记忆渲染成一张可交互的星图（单文件 HTML，无外部依赖）。

视觉骨架是**莫比乌斯环**：环面上铺几千颗粒子组成光带，记忆作为更亮的结点钉在环带上，
中心是"她"（颜色随 P 变化，带四向光束）。

数据全部来自本地：`memory.db` 的记忆 + `state.db` 的 PAD/关系。
记忆在环上的位置由 embedding 的 PCA 前两维决定——语义相近的记忆会落在环上相邻处。
"""

from __future__ import annotations

import json
import math
import time
from pathlib import Path
from typing import Any

from .memory import MemoryStore
from .persistence import StateStore

_KIND_LABEL = {
    "episodic": "情景",
    "semantic": "语义",
    "reflection": "反思",
    "knowledge": "知识",
}

_HTML = """<!doctype html>
<html lang="zh-CN">
<head>
<meta charset="utf-8">
<title>意识海 · 记忆星图</title>
<style>
  :root { --bg:#03040a; --ink:#eaf1ff; --dim:#7d8cb0; --line:rgba(120,160,255,.18); }
  * { box-sizing: border-box; }
  body { margin:0; background:radial-gradient(ellipse at 50% 45%, #0a0f22 0%, #03040a 60%);
         color:var(--ink); font:14px/1.6 "Segoe UI","Microsoft YaHei",system-ui,sans-serif;
         overflow:hidden; }
  canvas { display:block; position:fixed; inset:0; cursor:grab; }
  canvas:active { cursor:grabbing; }
  .hud { position:fixed; left:34px; top:26px; pointer-events:none; }
  .hud h1 { margin:0; font-size:28px; letter-spacing:8px; font-weight:600;
            text-shadow:0 0 24px rgba(120,180,255,.55); }
  .hud .sub { color:var(--dim); font-size:12px; letter-spacing:3px; margin-top:6px; }
  .stats { position:fixed; right:34px; top:26px; text-align:right; font-size:12px;
           color:var(--dim); letter-spacing:1px; }
  .stats div { margin-bottom:4px; }
  .stats b { color:var(--ink); font-size:17px; font-weight:500; margin-left:8px; }
  .legend { position:fixed; left:34px; bottom:26px; font-size:12px; color:var(--dim); }
  .legend span { display:inline-flex; align-items:center; margin-right:16px; }
  .legend i { width:9px; height:9px; border-radius:50%; margin-right:7px; display:inline-block;
              box-shadow:0 0 10px currentColor; }
  .card { position:fixed; width:300px; padding:15px 17px; border-radius:12px;
          background:linear-gradient(160deg, rgba(14,19,36,.94), rgba(9,12,24,.94));
          border:1px solid var(--line); box-shadow:0 12px 48px rgba(0,0,0,.7);
          backdrop-filter:blur(8px); opacity:0; transform:translateY(6px);
          transition:opacity .18s, transform .18s; pointer-events:none; }
  .card.on { opacity:1; transform:translateY(0); }
  .card .who { font-size:12px; color:#8fb0ff; letter-spacing:2px; }
  .card .txt { margin:9px 0 11px; font-size:13px; line-height:1.75; color:#dbe4f7; }
  .card .meta { font-size:11px; color:var(--dim); display:flex; justify-content:space-between; }
  .card .bar { height:2px; background:rgba(140,170,255,.14); margin-top:11px; border-radius:2px; }
  .card .bar i { display:block; height:100%; border-radius:2px;
                 background:linear-gradient(90deg,#4d8cff,#b06bff); }
  .empty { position:fixed; inset:0; display:grid; place-content:center; text-align:center;
           color:var(--dim); letter-spacing:2px; }
</style>
</head>
<body>
<canvas id="sky"></canvas>
<div class="hud">
  <h1>意识海</h1>
  <div class="sub">记忆星图 · 每一颗星都是一段记忆</div>
</div>
<div class="stats" id="stats"></div>
<div class="legend" id="legend"></div>
<div class="card" id="card">
  <div class="who" id="who"></div>
  <div class="txt" id="txt"></div>
  <div class="meta"><span id="kind"></span><span id="when"></span></div>
  <div class="bar"><i id="imp"></i></div>
</div>
<script>
const DATA = __DATA__;
const cv = document.getElementById('sky');
const ctx = cv.getContext('2d', { alpha: false });
let W = 0, H = 0, DPR = Math.min(2, window.devicePixelRatio || 1);

function resize() {
  W = window.innerWidth; H = window.innerHeight;
  cv.width = Math.floor(W * DPR); cv.height = Math.floor(H * DPR);
  ctx.setTransform(DPR, 0, 0, DPR, 0, 0);
}
window.addEventListener('resize', resize);

// —— 莫比乌斯环参数方程 ——
function mobius(u, v, R) {
  const half = u / 2;
  const r = R + v * Math.cos(half);
  return { x: r * Math.cos(u), y: v * Math.sin(half), z: r * Math.sin(u) };
}

const CAM = 3.0, FOV = 2.4;
const view = { yaw: 0.9, pitch: 0.30, dragging: false, lx: 0, ly: 0, spin: 0.00035 };

function project(p) {
  const cy = Math.cos(view.yaw), sy = Math.sin(view.yaw);
  const cp = Math.cos(view.pitch), sp = Math.sin(view.pitch);
  let x = p.x * cy - p.z * sy;
  let z = p.x * sy + p.z * cy;
  let y = p.y * cp - z * sp;
  z = p.y * sp + z * cp;
  const d = CAM + z;
  const k = FOV / Math.max(0.22, d);
  return { sx: W / 2 + x * k * Math.min(W, H) * 0.62,
           sy: H / 2 + y * k * Math.min(W, H) * 0.62, k, depth: d };
}

// —— 环面粒子（一次性生成，逐帧只做投影）——
const BAND = [];
(function buildBand() {
  const N = 6400;
  for (let i = 0; i < N; i++) {
    const u = (i / N) * Math.PI * 2 * 3;
    const v = (Math.random() - 0.5) * 0.34;
    const p = mobius(u % (Math.PI * 2), v, 1.0);
    BAND.push({
      x: p.x + (Math.random() - 0.5) * 0.02,
      y: p.y + (Math.random() - 0.5) * 0.02,
      z: p.z + (Math.random() - 0.5) * 0.02,
      s: 0.6 + Math.random() * 1.5,
      ph: Math.random() * Math.PI * 2,
      hue: (i / N) * 3,
    });
  }
})();

const STARS = [];
for (let i = 0; i < 220; i++) {
  STARS.push({ x: ((i * 73) % 1000) / 1000, y: ((i * 197) % 1000) / 1000,
               r: 0.6 + (i % 3) * 0.5, ph: i * 0.9 });
}

function bandColor(hue, alpha) {
  const h = 200 + 55 * Math.sin(hue + 0.3);
  return `hsla(${h}, 92%, ${66 + 6 * Math.sin(hue * 0.5)}%, ${alpha})`;
}

function drawStars(t) {
  for (const s of STARS) {
    const a = 0.15 + 0.35 * (0.5 + 0.5 * Math.sin(t * 0.0012 + s.ph));
    ctx.fillStyle = `rgba(190,215,255,${a})`;
    ctx.fillRect(s.x * W, s.y * H, s.r, s.r);
  }
}

function drawRibbon(t) {
  ctx.globalCompositeOperation = 'lighter';
  const N = 260;
  for (let i = 0; i < N; i++) {
    const u0 = (i / N) * Math.PI * 2;
    const u1 = ((i + 1) / N) * Math.PI * 2;
    const a = project(mobius(u0, 0, 1.0));
    const b = project(mobius(u1, 0, 1.0));
    const h = 200 + 55 * Math.sin(u0 + 0.3);          // 蓝 → 紫 → 青，沿环慢变
    const w = Math.max(2, 9 * (a.k + b.k));
    ctx.strokeStyle = `hsla(${h},92%,68%,0.085)`;
    ctx.lineWidth = w;
    ctx.beginPath(); ctx.moveTo(a.sx, a.sy); ctx.lineTo(b.sx, b.sy); ctx.stroke();
  }
  ctx.globalCompositeOperation = 'source-over';
}

function drawBand(t) {
  ctx.globalCompositeOperation = 'lighter';
  for (const p of BAND) {
    const q = project(p);
    if (q.k < 0.05) continue;
    const twinkle = 0.55 + 0.45 * Math.sin(t * 0.0016 + p.ph);
    const alpha = Math.min(0.85, (0.10 + 0.34 * twinkle) * q.k * 1.6);
    const size = p.s * q.k * 1.9;
    ctx.fillStyle = bandColor(p.hue, alpha);
    ctx.fillRect(q.sx, q.sy, size, size);
  }
  ctx.globalCompositeOperation = 'source-over';
}

function drawHaze() {
  ctx.globalCompositeOperation = 'lighter';
  const spots = [[0.24, 0.46, 260, '80,120,255'], [0.76, 0.44, 300, '120,90,255'],
                 [0.52, 0.72, 240, '60,180,190']];
  for (const [fx, fy, r, c] of spots) {
    const g = ctx.createRadialGradient(W * fx, H * fy, 0, W * fx, H * fy, r);
    g.addColorStop(0, `rgba(${c},0.10)`);
    g.addColorStop(1, 'rgba(0,0,0,0)');
    ctx.fillStyle = g;
    ctx.fillRect(W * fx - r, H * fy - r, r * 2, r * 2);
  }
  ctx.globalCompositeOperation = 'source-over';
}

function drawCore(t) {
  const c = project({ x: 0, y: 0, z: 0 });
  const pulse = 1 + 0.05 * Math.sin(t * 0.0014);
  const R = 60 * pulse * c.k * 2.2;
  const col = DATA.core.color;

  ctx.globalCompositeOperation = 'lighter';
  const g = ctx.createRadialGradient(c.sx, c.sy, 0, c.sx, c.sy, R);
  g.addColorStop(0, 'rgba(255,255,255,0.95)');
  g.addColorStop(0.18, `rgba(${col},0.55)`);
  g.addColorStop(0.55, `rgba(${col},0.12)`);
  g.addColorStop(1, 'rgba(0,0,0,0)');
  ctx.fillStyle = g;
  ctx.beginPath(); ctx.arc(c.sx, c.sy, R, 0, Math.PI * 2); ctx.fill();

  for (let i = 0; i < 4; i++) {
    const ang = view.pitch * 0.6 + i * Math.PI / 2 + Math.sin(t * 0.0004) * 0.05;
    const len = R * (2.6 + 0.2 * Math.sin(t * 0.0011 + i));
    const lg = ctx.createLinearGradient(c.sx, c.sy,
      c.sx + Math.cos(ang) * len, c.sy + Math.sin(ang) * len);
    lg.addColorStop(0, 'rgba(255,255,255,0.55)');
    lg.addColorStop(1, 'rgba(120,170,255,0)');
    ctx.strokeStyle = lg; ctx.lineWidth = 1.4;
    ctx.beginPath(); ctx.moveTo(c.sx, c.sy);
    ctx.lineTo(c.sx + Math.cos(ang) * len, c.sy + Math.sin(ang) * len); ctx.stroke();
  }
  ctx.globalCompositeOperation = 'source-over';
}

let HITS = [];
function drawNodes(t) {
  HITS = [];
  ctx.globalCompositeOperation = 'lighter';
  for (const n of DATA.nodes) {
    const p = mobius(n.u, n.v, 1.0);
    const q = project(p);
    const pulse = 0.8 + 0.2 * Math.sin(t * 0.002 + n.idx);
    const r = (5 + n.imp * 9) * q.k * 2.0 * pulse;
    const g = ctx.createRadialGradient(q.sx, q.sy, 0, q.sx, q.sy, r * 3.4);
    g.addColorStop(0, 'rgba(255,255,255,0.95)');
    g.addColorStop(0.2, `rgba(${n.color},0.7)`);
    g.addColorStop(1, 'rgba(0,0,0,0)');
    ctx.fillStyle = g;
    ctx.beginPath(); ctx.arc(q.sx, q.sy, r * 3.4, 0, Math.PI * 2); ctx.fill();
    HITS.push({ node: n, sx: q.sx, sy: q.sy, r: Math.max(16, r * 2.6) });
  }
  ctx.globalCompositeOperation = 'source-over';

  ctx.font = '12px "Microsoft YaHei",sans-serif';
  for (const h of HITS) {
    const a = Math.min(0.95, 0.45 + h.node.imp * 0.5);
    ctx.lineWidth = 3;
    ctx.strokeStyle = 'rgba(3,4,10,0.85)';
    ctx.strokeText(h.node.who, h.sx + 14, h.sy - 11);
    ctx.fillStyle = `rgba(215,232,255,${a})`;
    ctx.fillText(h.node.who, h.sx + 14, h.sy - 11);
  }
}

function frame(t) {
  if (!view.dragging) view.yaw += view.spin;
  ctx.fillStyle = '#03040a';
  ctx.fillRect(0, 0, W, H);
  drawStars(t);
  drawHaze();
  drawRibbon(t);
  drawBand(t);
  drawCore(t);
  drawNodes(t);
  requestAnimationFrame(frame);
}

// —— 交互 ——
const card = document.getElementById('card');
cv.addEventListener('mousemove', (e) => {
  if (view.dragging) {
    view.yaw += (e.clientX - view.lx) * 0.006;
    view.pitch = Math.max(-1.3, Math.min(1.3, view.pitch + (e.clientY - view.ly) * 0.004));
    view.lx = e.clientX; view.ly = e.clientY;
    return;
  }
  let best = null, bestD = 1e9;
  for (const h of HITS) {
    const d = Math.hypot(h.sx - e.clientX, h.sy - e.clientY);
    if (d < h.r && d < bestD) { best = h; bestD = d; }
  }
  if (!best) { card.classList.remove('on'); return; }
  const n = best.node;
  document.getElementById('who').textContent = n.who;
  document.getElementById('txt').textContent = n.text;
  document.getElementById('kind').textContent = n.kind + ' · 重要性 ' + n.imp.toFixed(2);
  document.getElementById('when').textContent = n.when;
  document.getElementById('imp').style.width = Math.round(n.imp * 100) + '%';
  card.style.left = Math.min(W - 330, best.sx + 18) + 'px';
  card.style.top = Math.max(20, Math.min(H - 190, best.sy - 30)) + 'px';
  card.classList.add('on');
});
cv.addEventListener('mousedown', (e) => {
  view.dragging = true; view.lx = e.clientX; view.ly = e.clientY;
  card.classList.remove('on');
});
window.addEventListener('mouseup', () => { view.dragging = false; });
cv.addEventListener('wheel', (e) => {
  view.pitch = Math.max(-1.3, Math.min(1.3, view.pitch - e.deltaY * 0.0008));
  e.preventDefault();
}, { passive: false });

document.getElementById('stats').innerHTML =
  '<div>核心恒星<b>1</b></div><div>记忆<b>' + DATA.nodes.length + '</b></div>' +
  '<div>关系<b>' + DATA.rings.length + '</b></div><div>情绪<b>' + DATA.core.mood + '</b></div>';
document.getElementById('legend').innerHTML = DATA.rings.map(r =>
  `<span style="color:rgb(${r.color})"><i style="background:rgb(${r.color})"></i>${r.who} · ${r.count}</span>`).join('') +
  `<span style="color:rgb(${DATA.core.color})"><i style="background:rgb(${DATA.core.color})"></i>她 PAD ${DATA.core.pad}</span>`;

if (!DATA.nodes.length) {
  document.body.insertAdjacentHTML('beforeend',
    '<div class="empty">还没有记忆 · 先聊几句再生成</div>');
}

resize();
requestAnimationFrame(frame);
</script>
</body>
</html>
"""


def _hex_to_rgb(seed: str) -> tuple[int, int, int]:
    digest = sum(ord(ch) * (i + 7) for i, ch in enumerate(seed))
    palette = [
        (120, 168, 255), (168, 132, 255), (96, 220, 210), (255, 156, 128),
        (255, 128, 190), (140, 220, 140), (240, 200, 120),
    ]
    return palette[digest % len(palette)]


def _pad_color(p: float, a: float) -> tuple[int, int, int]:
    warm = (255, 190, 120)
    cool = (110, 140, 255)
    t = max(0.0, min(1.0, (p + 1) / 2))
    r = int(cool[0] + (warm[0] - cool[0]) * t)
    g = int(cool[1] + (warm[1] - cool[1]) * t)
    b = int(cool[2] + (warm[2] - cool[2]) * t)
    boost = 0.75 + 0.35 * max(0.0, min(1.0, a))
    return (min(255, int(r * boost)), min(255, int(g * boost)), min(255, int(b * boost)))


def _embedding_space(vectors: list[list[float]]) -> list[tuple[float, float]]:
    """把 embedding 压成环上的 (角度, 带宽偏移)。

    语义相近 → 角度相近，所以"同一个话题"的记忆会在环上挨着。
    没有向量时用索引均匀撒开。
    """
    n = len(vectors)
    if n == 0:
        return []
    try:
        import numpy as np

        mat = np.array(vectors, dtype="float32")
        mat = mat - mat.mean(axis=0, keepdims=True)
        u, s, _vt = np.linalg.svd(mat, full_matrices=False)
        coords = u[:, :2] * s[:2]
        out = []
        for row in coords:
            ang = math.atan2(float(row[0]), float(row[1])) % math.tau
            radius = min(1.0, math.hypot(float(row[0]), float(row[1])))
            out.append((ang, 0.10 + 0.10 * radius))
        return out
    except Exception:
        return [((i / max(1, n)) * math.tau, 0.12) for i in range(n)]


def build_payload(mem_path: str | Path, state_path: str | Path) -> dict[str, Any]:
    store = MemoryStore(mem_path)
    rows = list(
        store._conn.execute(
            "SELECT id, uid, kind, text, importance, ts, category, embedding "
            "FROM memories WHERE text != '' ORDER BY ts"
        )
    )
    state_store = StateStore(state_path)
    state = state_store.load()
    store.close()
    state_store.close()

    relations = {}
    core = {"pad": "0.00 / 0.00 / 0.00", "mood": "—", "color": "140,170,255"}
    if state is not None:
        relations = dict(state.relations)
        rgb = _pad_color(state.P, state.A)
        core = {
            "pad": f"{state.P:+.2f} / {state.A:.2f} / {state.D:+.2f}",
            "mood": state.mood,
            "color": f"{rgb[0]},{rgb[1]},{rgb[2]}",
        }

    names: dict[str, str] = {}
    for uid, rel in relations.items():
        names[uid] = getattr(rel.persona, "nickname", "") or uid

    vectors = []
    for row in rows:
        blob = row["embedding"]
        if blob:
            import array

            arr = array.array("f")
            arr.frombytes(blob)
            vectors.append(list(arr))
    space = _embedding_space(vectors) if len(vectors) == len(rows) else []

    nodes = []
    uids = sorted({str(row["uid"]) for row in rows})
    for index, row in enumerate(rows):
        uid = str(row["uid"])
        who = names.get(uid) or uid
        if uid == "__knowledge__":
            who, color = "知识", (150, 200, 255)
        elif uid.endswith("__knowledge__"):
            who, color = "群知识", (110, 210, 190)
        else:
            color = _hex_to_rgb(uid)
        if index < len(space):
            u, v = space[index]
        else:
            u, v = (index / max(1, len(rows))) * math.tau, 0.12
        nodes.append(
            {
                "u": round(u, 5),
                "v": round(v, 4),
                "idx": index,
                "color": f"{color[0]},{color[1]},{color[2]}",
                "who": who,
                "kind": _KIND_LABEL.get(str(row["kind"]), str(row["kind"])),
                "text": str(row["text"])[:220],
                "imp": float(row["importance"] or 0.5),
                "when": time.strftime("%m-%d %H:%M", time.localtime(float(row["ts"]))),
                "uid": uid,
            }
        )

    counts: dict[str, int] = {}
    for node in nodes:
        counts[node["uid"]] = counts.get(node["uid"], 0) + 1
    rings = []
    for uid in uids:
        if uid == "__knowledge__":
            who, rgb = "知识", (150, 200, 255)
        elif uid.endswith("__knowledge__"):
            who, rgb = "群知识", (110, 210, 190)
        else:
            who, rgb = names.get(uid) or uid, _hex_to_rgb(uid)
        rings.append(
            {"who": who, "count": counts[uid], "color": f"{rgb[0]},{rgb[1]},{rgb[2]}"}
        )

    return {"core": core, "nodes": nodes, "rings": rings}


def render_html(payload: dict[str, Any]) -> str:
    data = json.dumps(payload, ensure_ascii=False).replace("</", "<\\/")
    return _HTML.replace("__DATA__", data)


def export_map(
    mem_path: str | Path,
    state_path: str | Path,
    out_path: str | Path,
) -> Path:
    payload = build_payload(mem_path, state_path)
    target = Path(out_path)
    target.write_text(render_html(payload), encoding="utf-8")
    return target


__all__ = ["build_payload", "render_html", "export_map"]
