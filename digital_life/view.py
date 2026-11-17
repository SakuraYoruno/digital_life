"""把记忆渲染成一张可交互的星图（单文件 HTML，无外部依赖）。

数据全部来自本地：`memory.db` 的记忆 + `state.db` 的 PAD/关系。
每条记忆是一颗星：位置由 embedding 的 PCA 三维投影决定（长相近的记忆会聚成星座），
亮度/大小由 importance 决定，颜色由所属的人（uid）决定；中心那颗是"她"，
颜色跟着 PAD 的愉悦度走。

生成后是一个自包含 HTML：直接双击就能看，不需要起服务、不联网。
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
  :root { --bg:#05060c; --ink:#e8eefc; --dim:#7b88a8; }
  * { box-sizing: border-box; }
  body { margin:0; background:var(--bg); color:var(--ink);
         font:14px/1.6 "Segoe UI","Microsoft YaHei",system-ui,sans-serif; overflow:hidden; }
  canvas { display:block; position:fixed; inset:0; cursor:crosshair; }
  .hud { position:fixed; left:28px; top:24px; pointer-events:none; }
  .hud h1 { margin:0; font-size:26px; letter-spacing:6px; font-weight:600; }
  .hud .sub { color:var(--dim); font-size:12px; letter-spacing:2px; margin-top:4px; }
  .stats { position:fixed; right:28px; top:24px; text-align:right; font-size:12px; color:var(--dim); }
  .stats b { color:var(--ink); font-size:18px; font-weight:500; margin-left:6px; }
  .legend { position:fixed; left:28px; bottom:24px; font-size:12px; color:var(--dim); }
  .legend span { display:inline-flex; align-items:center; margin-right:14px; }
  .legend i { width:8px; height:8px; border-radius:50%; margin-right:6px; display:inline-block; }
  .card { position:fixed; width:300px; padding:14px 16px; border-radius:10px;
          background:rgba(12,16,28,.92); border:1px solid rgba(120,160,255,.25);
          box-shadow:0 8px 40px rgba(0,0,0,.6); backdrop-filter:blur(6px);
          opacity:0; transition:opacity .18s; pointer-events:none; }
  .card.on { opacity:1; }
  .card .who { font-size:12px; color:#8fb0ff; letter-spacing:1px; }
  .card .txt { margin:8px 0 10px; font-size:13px; line-height:1.7; color:#dbe4f7; }
  .card .meta { font-size:11px; color:var(--dim); display:flex; justify-content:space-between; }
  .card .bar { height:2px; background:rgba(140,170,255,.15); margin-top:10px; border-radius:2px; }
  .card .bar i { display:block; height:100%; background:linear-gradient(90deg,#5b8cff,#a06bff); border-radius:2px; }
  .empty { position:fixed; inset:0; display:grid; place-content:center; text-align:center; color:var(--dim); }
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
const cv = document.getElementById('sky'), ctx = cv.getContext('2d');
let W = 0, H = 0, DPR = Math.min(2, window.devicePixelRatio || 1);
function resize() {
  W = window.innerWidth; H = window.innerHeight;
  cv.width = Math.floor(W * DPR); cv.height = Math.floor(H * DPR);
  ctx.setTransform(DPR, 0, 0, DPR, 0, 0);
}
window.addEventListener('resize', resize);

const state = { yaw: 0.6, pitch: -0.32, spin: 0.00035, dragging: false, lx: 0, ly: 0 };
const CAM = 3.2, FOV = 2.1;

function project(p) {
  const cy = Math.cos(state.yaw), sy = Math.sin(state.yaw);
  const cp = Math.cos(state.pitch), sp = Math.sin(state.pitch);
  let x = p.x * cy - p.z * sy;
  let z = p.x * sy + p.z * cy;
  let y = p.y * cp - z * sp;
  z = p.y * sp + z * cp;
  const d = CAM + z;
  const k = FOV / Math.max(0.25, d);
  const scale = k * Math.min(W, H) * 0.95;
  return { sx: W / 2 + x * scale, sy: H / 2 + y * scale, depth: d, k };
}

function drawSky(t) {
  ctx.fillStyle = 'rgba(5,6,12,0.28)';
  ctx.fillRect(0, 0, W, H);
  for (const s of DATA.stars) {
    const a = 0.10 + 0.22 * ((Math.sin(t * 0.001 + s.p) + 1) / 2);
    ctx.fillStyle = `rgba(150,180,255,${a})`;
    ctx.fillRect(s.x * W, s.y * H, s.r, s.r);
  }
}

function drawOrbits() {
  for (const ring of DATA.rings) {
    ctx.beginPath();
    ctx.strokeStyle = `rgba(${ring.color},0.10)`;
    ctx.lineWidth = 1;
    for (let i = 0; i <= 72; i++) {
      const a = (i / 72) * Math.PI * 2;
      const p = project({ x: Math.cos(a) * ring.r, y: 0, z: Math.sin(a) * ring.r });
      i ? ctx.lineTo(p.sx, p.sy) : ctx.moveTo(p.sx, p.sy);
    }
    ctx.stroke();
  }
}

function drawCore(t) {
  const p = project({ x: 0, y: 0, z: 0 });
  const pulse = 1 + 0.06 * Math.sin(t * 0.0016);
  const r = 46 * pulse * p.k;
  const g = ctx.createRadialGradient(p.sx, p.sy, 0, p.sx, p.sy, r * 2.6);
  const c = DATA.core.color;
  g.addColorStop(0, `rgba(${c},0.95)`);
  g.addColorStop(0.25, `rgba(${c},0.35)`);
  g.addColorStop(1, 'rgba(0,0,0,0)');
  ctx.fillStyle = g;
  ctx.beginPath(); ctx.arc(p.sx, p.sy, r * 2.6, 0, Math.PI * 2); ctx.fill();
  ctx.fillStyle = 'rgba(255,255,255,0.92)';
  ctx.beginPath(); ctx.arc(p.sx, p.sy, r * 0.30, 0, Math.PI * 2); ctx.fill();
}

function drawNodes(t) {
  const hits = [];
  for (const n of DATA.nodes) {
    const p = project(n);
    const glow = Math.max(0.35, 1.25 - p.depth * 0.28);
    const r = (2.6 + n.imp * 5.0) * glow;
    const g = ctx.createRadialGradient(p.sx, p.sy, 0, p.sx, p.sy, r * 4);
    g.addColorStop(0, `rgba(${n.color},0.95)`);
    g.addColorStop(0.35, `rgba(${n.color},0.28)`);
    g.addColorStop(1, 'rgba(0,0,0,0)');
    ctx.fillStyle = g;
    ctx.beginPath(); ctx.arc(p.sx, p.sy, r * 4, 0, Math.PI * 2); ctx.fill();
    ctx.fillStyle = `rgba(255,255,255,${0.65 + n.imp * 0.35})`;
    ctx.beginPath(); ctx.arc(p.sx, p.sy, r * 0.5, 0, Math.PI * 2); ctx.fill();
    if (n.imp >= 0.65) {
      ctx.font = '11px "Microsoft YaHei",sans-serif';
      ctx.fillStyle = `rgba(200,220,255,${0.45 + 0.25 * n.imp})`;
      ctx.fillText(n.who, p.sx + r * 2.2, p.sy - r * 1.4);
    }
    hits.push({ node: n, sx: p.sx, sy: p.sy, r: Math.max(12, r * 3) });
  }
  return hits;
}

function drawLinks(hits, t) {
  ctx.lineWidth = 1;
  for (const [a, b] of DATA.links) {
    const A = hits[a], B = hits[b];
    if (!A || !B) continue;
    const alpha = 0.12 + 0.10 * Math.sin(t * 0.001 + a);
    ctx.strokeStyle = `rgba(120,160,255,${alpha})`;
    ctx.beginPath(); ctx.moveTo(A.sx, A.sy); ctx.lineTo(B.sx, B.sy); ctx.stroke();
  }
}

let HITS = [];
function frame(t) {
  if (!state.dragging) state.yaw += state.spin;
  drawSky(t);
  drawOrbits();
  drawCore(t);
  HITS = drawNodes(t);
  drawLinks(HITS, t);
  requestAnimationFrame(frame);
}

// ---- 交互 ----
const card = document.getElementById('card');
cv.addEventListener('mousemove', (e) => {
  if (state.dragging) {
    state.yaw += (e.clientX - state.lx) * 0.006;
    state.pitch = Math.max(-1.2, Math.min(1.2, state.pitch - (e.clientY - state.ly) * 0.004));
    state.lx = e.clientX; state.ly = e.clientY;
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
  card.style.left = Math.min(W - 320, best.sx + 18) + 'px';
  card.style.top = Math.min(H - 180, best.sy - 20) + 'px';
  card.classList.add('on');
});
cv.addEventListener('mousedown', (e) => {
  state.dragging = true; state.lx = e.clientX; state.ly = e.clientY;
  card.classList.remove('on');
});
window.addEventListener('mouseup', () => { state.dragging = false; });
cv.addEventListener('wheel', (e) => {
  state.pitch = Math.max(-1.2, Math.min(1.2, state.pitch + e.deltaY * 0.0006));
  e.preventDefault();
}, { passive: false });

// ---- 侧栏 ----
document.getElementById('stats').innerHTML =
  '核心恒星<b>1</b><br>记忆<b>' + DATA.nodes.length + '</b><br>关系<b>' + DATA.rings.length + '</b>' +
  '<br>情绪<b>' + DATA.core.mood + '</b>';
document.getElementById('legend').innerHTML = DATA.rings.map(r =>
  `<span><i style="background:rgb(${r.color})"></i>${r.who} · ${r.count}</span>`).join('') +
  `<span><i style="background:rgb(${DATA.core.color})"></i>她（PAD ${DATA.core.pad}）</span>`;

if (!DATA.nodes.length) {
  document.body.insertAdjacentHTML('beforeend',
    '<div class="empty">还没有记忆。<br>先聊几句，再生成一次。</div>');
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
    """P（愉悦）决定冷暖，A 决定亮度。"""
    warm = (255, 190, 120)
    cool = (110, 140, 255)
    t = max(0.0, min(1.0, (p + 1) / 2))
    r = int(cool[0] + (warm[0] - cool[0]) * t)
    g = int(cool[1] + (warm[1] - cool[1]) * t)
    b = int(cool[2] + (warm[2] - cool[2]) * t)
    boost = 0.75 + 0.35 * max(0.0, min(1.0, a))
    return (min(255, int(r * boost)), min(255, int(g * boost)), min(255, int(b * boost)))


def _project_3d(vectors: list[list[float]]) -> list[tuple[float, float, float]]:
    """PCA 降到三维；没有 numpy 或向量缺失时退化成按索引排布。"""
    n = len(vectors)
    if n == 0:
        return []
    try:
        import numpy as np

        mat = np.array(vectors, dtype="float32")
        if mat.shape[1] < 3:
            raise ValueError("dim too small")
        mat = mat - mat.mean(axis=0, keepdims=True)
        u, s, _vt = np.linalg.svd(mat, full_matrices=False)
        coords = u[:, :3] * s[:3]
        scale = float(np.abs(coords).max()) or 1.0
        coords = coords / scale
        # 铺开到半径为 1 的球壳附近，避免全部挤在中心
        out = []
        for row in coords:
            norm = float(np.linalg.norm(row)) or 1.0
            radius = 0.30 + 0.42 * min(1.0, norm)
            out.append(tuple(float(v) / norm * radius for v in row))
        return out
    except Exception:
        out = []
        for i in range(n):
            ang = i / max(1, n) * math.tau
            out.append((math.cos(ang) * 0.8, math.sin(ang * 2) * 0.35, math.sin(ang) * 0.8))
        return out


def build_payload(mem_path: str | Path, state_path: str | Path) -> dict[str, Any]:
    store = MemoryStore(mem_path)
    try:
        rows = list(
            store._conn.execute(
                "SELECT id, uid, kind, text, importance, ts, category, embedding "
                "FROM memories WHERE text != '' ORDER BY ts"
            )
        )
    finally:
        pass
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
        label = getattr(rel.persona, "nickname", "") or uid
        names[uid] = label

    # 每条记忆一个坐标：同一个人用同一色，embedding 决定它在哪
    weights = [1.0 if row["kind"] == "episodic" else 0.7 for row in rows]
    vectors = []
    for row in rows:
        blob = row["embedding"]
        if blob:
            import array

            arr = array.array("f")
            arr.frombytes(blob)
            vectors.append(list(arr))
    coords = _project_3d(vectors) if len(vectors) == len(rows) else []
    nodes = []
    uids = sorted({str(row["uid"]) for row in rows})
    for index, row in enumerate(rows):
        uid = str(row["uid"])
        who = names.get(uid) or ("群 · " + uid.split(":")[0] if ":" in uid else uid)
        if uid == "__knowledge__":
            who = "知识"
            color = (150, 200, 255)
        else:
            color = _hex_to_rgb(uid)
        if index < len(coords):
            x, y, z = coords[index]
        else:
            ang = index / max(1, len(rows)) * math.tau
            x, y, z = math.cos(ang) * 0.8, math.sin(ang * 2) * 0.3, math.sin(ang) * 0.8
        nodes.append(
            {
                "x": round(x, 4),
                "y": round(y, 4),
                "z": round(z, 4),
                "color": f"{color[0]},{color[1]},{color[2]}",
                "who": who,
                "kind": _KIND_LABEL.get(str(row["kind"]), str(row["kind"])),
                "text": str(row["text"])[:220],
                "imp": float(row["importance"] or 0.5),
                "when": time.strftime("%m-%d %H:%M", time.localtime(float(row["ts"]))),
                "uid": uid,
            }
        )

    rings = []
    counts = {}
    for node in nodes:
        counts[node["uid"]] = counts.get(node["uid"], 0) + 1
    for i, uid in enumerate(uids):
        rgb = _hex_to_rgb(uid) if uid != "__knowledge__" else (150, 200, 255)
        rings.append(
            {
                "who": names.get(uid) or ("知识" if uid == "__knowledge__" else uid),
                "count": counts[uid],
                "r": 1.05 + 0.16 * i,
                "color": f"{rgb[0]},{rgb[1]},{rgb[2]}",
            }
        )

    stars = []
    for i in range(140):
        stars.append(
            {
                "x": ((i * 37) % 100) / 100,
                "y": ((i * 61) % 100) / 100,
                "r": 1 + (i % 2),
                "p": i * 0.7,
            }
        )
    links: list[list[int]] = []
    by_uid: dict[str, list[int]] = {}
    for index, node in enumerate(nodes):
        by_uid.setdefault(node["uid"], []).append(index)
    for indexes in by_uid.values():
        for a, b in zip(indexes, indexes[1:]):
            links.append([a, b])
    return {
        "core": core, "nodes": nodes, "rings": rings, "stars": stars, "links": links,
    }


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
