"""SVG 版记忆星图：照样图精确描形，不做三维。

为什么换成 SVG：三维场景里"形状对不对"受相机、比例、透视三重影响，
每调一次都要重新猜；SVG 是直接画坐标——椭圆的扁度、带子的走向、光晕的扩散半径
都是一个数字，改完就是那个样子。

结构：
- 背景：深空渐变 + 星云（径向渐变）+ 星点（JS 生成）
- 环带：椭圆路径 + 三层描边（大模糊/中模糊/实线）叠出体积光，
  再沿路径撒 4200 颗雪点做出"星尘"质感
- 中心：径向渐变恒星 + 六道星芒 + 三个小轨道环
- 悬浮层：等宽字体诊断面板、左右角色卡、右上统计
"""

from __future__ import annotations

import json
import time
from pathlib import Path
from typing import Any

from .memory import MemoryStore
from .persistence import StateStore
from .view import build_payload
from .view_webgl import _cards, _panels

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
  svg { position:fixed; inset:0; width:100%; height:100%; display:block; }
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
  .card { position:fixed; width:212px; padding:0 0 10px; background:linear-gradient(180deg,
          rgba(16,20,34,.94), rgba(8,10,18,.94)); border:1px solid rgba(120,160,255,.22);
          box-shadow:0 14px 50px rgba(0,0,0,.75); z-index:4; pointer-events:none; }
  .card .face { height:88px; }
  .card .body { padding:8px 11px 0; }
  .card .name { font-size:15px; letter-spacing:2px; }
  .card .en { font-size:9px; color:#7b8db3; letter-spacing:2px; margin-top:2px; }
  .card .kv { display:flex; justify-content:space-between; font-size:10.5px; color:#9fb0d0;
              margin-top:6px; }
  .card .kv b { color:#dce6ff; font-weight:500; }
  .card .quote { margin-top:8px; font-size:10.5px; color:#8da0c4; line-height:1.5;
                 border-top:1px solid rgba(120,160,255,.12); padding-top:7px;
                 max-height:54px; overflow:hidden; }
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
<svg id="sky" viewBox="0 0 1920 1080" preserveAspectRatio="xMidYMid slice">
  <defs>
    <radialGradient id="bg" cx="50%" cy="46%" r="72%">
      <stop offset="0%" stop-color="#0a1024"/>
      <stop offset="55%" stop-color="#05070f"/>
      <stop offset="100%" stop-color="#010207"/>
    </radialGradient>
    <radialGradient id="nebBlue" cx="50%" cy="50%" r="50%">
      <stop offset="0%" stop-color="rgba(70,110,255,.55)"/>
      <stop offset="55%" stop-color="rgba(40,60,160,.16)"/>
      <stop offset="100%" stop-color="rgba(0,0,0,0)"/>
    </radialGradient>
    <radialGradient id="nebPurple" cx="50%" cy="50%" r="50%">
      <stop offset="0%" stop-color="rgba(150,70,255,.42)"/>
      <stop offset="60%" stop-color="rgba(70,30,140,.12)"/>
      <stop offset="100%" stop-color="rgba(0,0,0,0)"/>
    </radialGradient>
    <radialGradient id="coreGlow" cx="50%" cy="50%" r="50%">
      <stop offset="0%" stop-color="#ffffff"/>
      <stop offset="22%" stop-color="rgba(255,235,250,.95)"/>
      <stop offset="52%" stop-color="rgba(200,190,255,.35)"/>
      <stop offset="100%" stop-color="rgba(90,110,255,0)"/>
    </radialGradient>
    <linearGradient id="bandNear" x1="0%" y1="50%" x2="100%" y2="50%">
      <stop offset="0%" stop-color="#cfe0ff" stop-opacity=".55"/>
      <stop offset="35%" stop-color="#ffffff" stop-opacity=".95"/>
      <stop offset="72%" stop-color="#b9fff0" stop-opacity=".85"/>
      <stop offset="100%" stop-color="#e6e9ff" stop-opacity=".45"/>
    </linearGradient>
    <linearGradient id="bandFar" x1="0%" y1="50%" x2="100%" y2="50%">
      <stop offset="0%" stop-color="#7f8cff" stop-opacity=".18"/>
      <stop offset="50%" stop-color="#a4b6ff" stop-opacity=".30"/>
      <stop offset="100%" stop-color="#7ff2d8" stop-opacity=".16"/>
    </linearGradient>
    <filter id="blurXL" x="-40%" y="-40%" width="180%" height="180%">
      <feGaussianBlur stdDeviation="26"/>
    </filter>
    <filter id="blurL" x="-30%" y="-30%" width="160%" height="160%">
      <feGaussianBlur stdDeviation="11"/>
    </filter>
    <filter id="blurM" x="-20%" y="-20%" width="140%" height="140%">
      <feGaussianBlur stdDeviation="4"/>
    </filter>
    <filter id="blurS" x="-20%" y="-20%" width="140%" height="140%">
      <feGaussianBlur stdDeviation="1.6"/>
    </filter>
  </defs>

  <rect width="1920" height="1080" fill="url(#bg)"/>
  <ellipse cx="360" cy="235" rx="330" ry="200" fill="url(#nebBlue)"/>
  <ellipse cx="1620" cy="640" rx="330" ry="200" fill="url(#nebPurple)"/>
  <ellipse cx="1290" cy="300" rx="220" ry="130" fill="url(#nebPurple)" opacity=".55"/>
  <g id="stars"></g>

  <!-- 环带：远侧（细、暗）+ 近侧（粗、亮） -->
  <g id="bandFar">
    <ellipse cx="960" cy="540" rx="860" ry="232" transform="rotate(-11 960 540)"
             fill="none" stroke="url(#bandFar)" stroke-width="16" filter="url(#blurL)"/>
  </g>
  <g id="bandNear">
    <ellipse cx="960" cy="556" rx="852" ry="222" transform="rotate(-11 960 556)"
             fill="none" stroke="url(#bandNear)" stroke-width="46" filter="url(#blurXL)" opacity=".55"/>
    <ellipse cx="960" cy="556" rx="852" ry="222" transform="rotate(-11 960 556)"
             fill="none" stroke="url(#bandNear)" stroke-width="14" filter="url(#blurM)" opacity=".95"/>
    <ellipse cx="960" cy="556" rx="852" ry="222" transform="rotate(-11 960 556)"
             fill="none" stroke="#ffffff" stroke-width="2.4" opacity=".85" filter="url(#blurS)"/>
    <!-- 右下侧更厚的一段亮带 -->
    <path d="M 1180 760 A 852 222 0 0 0 1780 470" transform="rotate(-11 960 556)"
          fill="none" stroke="#dcfff8" stroke-width="26" opacity=".55" filter="url(#blurL)"/>
    <path d="M 1180 760 A 852 222 0 0 0 1780 470" transform="rotate(-11 960 556)"
          fill="none" stroke="#ffffff" stroke-width="6" opacity=".8" filter="url(#blurM)"/>
  </g>
  <g id="dust"></g>

  <!-- 中心恒星 -->
  <g id="core">
    <circle cx="960" cy="520" r="300" fill="url(#coreGlow)" opacity=".85"/>
    <circle cx="960" cy="520" r="34" fill="#ffffff"/>
    <g stroke="#ffffff" stroke-linecap="round" opacity=".5">
      <line x1="960" y1="520" x2="360" y2="400" stroke-width="1.4" opacity=".42"/>
      <line x1="960" y1="520" x2="1560" y2="640" stroke-width="1.4" opacity=".42"/>
      <line x1="960" y1="520" x2="640" y2="760" stroke-width=".9" opacity=".22"/>
      <line x1="960" y1="520" x2="1280" y2="280" stroke-width=".9" opacity=".22"/>
      <line x1="960" y1="520" x2="960" y2="120" stroke-width="1.6" opacity=".45"/>
      <line x1="960" y1="520" x2="960" y2="920" stroke-width="1.6" opacity=".45"/>
      <line x1="960" y1="520" x2="700" y2="230" stroke-width=".9" opacity=".25"/>
      <line x1="960" y1="520" x2="1230" y2="820" stroke-width=".9" opacity=".25"/>
    </g>
    <g fill="none" stroke="#cfe0ff" opacity=".3">
      <ellipse cx="960" cy="520" rx="120" ry="42" transform="rotate(-12 960 520)"/>
      <ellipse cx="960" cy="520" rx="170" ry="58" transform="rotate(18 960 520)" opacity=".6"/>
      <ellipse cx="960" cy="520" rx="215" ry="74" transform="rotate(-30 960 520)" opacity=".4"/>
    </g>
  </g>
</svg>

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
<script>
const DATA = __DATA__;
const NS = 'http://www.w3.org/2000/svg';

// 伪随机（固定种子，保证每次生成的星图一致）
let seed = 20261117;
function rnd() { seed = (seed * 1103515245 + 12345) & 0x7fffffff; return seed / 0x7fffffff; }
function gauss() { return (rnd() + rnd() + rnd() - 1.5) / 1.5; }

// —— 背景星点 ——
const stars = document.getElementById('stars');
for (let i = 0; i < 900; i++) {
  const c = document.createElementNS(NS, 'circle');
  c.setAttribute('cx', rnd() * 1920);
  c.setAttribute('cy', rnd() * 1080);
  c.setAttribute('r', (0.4 + rnd() * 1.1).toFixed(2));
  c.setAttribute('fill', '#dbe7ff');
  c.setAttribute('opacity', (0.12 + rnd() * 0.6).toFixed(2));
  stars.appendChild(c);
}

// —— 环带星尘：沿椭圆撒点，靠近中心线更密 ——
const CX = 960, CY = 556, RX = 852, RY = 222, ROT = -7 * Math.PI / 180;
const dust = document.getElementById('dust');
for (let i = 0; i < 5200; i++) {
  const th = rnd() * Math.PI * 2;
  const spread = gauss() * 9;                       // 带宽方向的抖动
  const shrink = 1 + gauss() * 0.004;
  let x = Math.cos(th) * (RX * shrink) + spread * Math.cos(th);
  let y = Math.sin(th) * (RY * shrink) + spread * Math.sin(th);
  const cos = Math.cos(ROT), sin = Math.sin(ROT);
  const px = CX + x * cos - y * sin, py = CY + x * sin + y * cos;
  const c = document.createElementNS(NS, 'circle');
  c.setAttribute('cx', px.toFixed(1));
  c.setAttribute('cy', py.toFixed(1));
  c.setAttribute('r', (0.5 + rnd() * 1.7).toFixed(2));
  const warm = th > Math.PI * 0.9 && th < Math.PI * 1.7;
  c.setAttribute('fill', warm ? '#b9fff0' : '#ffffff');
  c.setAttribute('opacity', (0.25 + rnd() * 0.75).toFixed(2));
  dust.appendChild(c);
}

// —— 记忆结点（贴在环带上）——
const hy = document.getElementById('hover');
DATA.nodes.forEach((n) => {
  const th = n.u, shrink = 1 + (n.v - 0.12) * 0.12;
  const x = Math.cos(th) * RX * shrink, y = Math.sin(th) * RY * shrink;
  const cos = Math.cos(ROT), sin = Math.sin(ROT);
  const px = CX + x * cos - y * sin, py = CY + x * sin + y * cos;
  const g = document.createElementNS(NS, 'g');
  const halo = document.createElementNS(NS, 'circle');
  halo.setAttribute('cx', px); halo.setAttribute('cy', py);
  halo.setAttribute('r', 16 + n.imp * 16);
  halo.setAttribute('fill', `rgb(${n.color})`);
  halo.setAttribute('opacity', 0.22);
  halo.setAttribute('filter', 'url(#blurL)');
  const dot = document.createElementNS(NS, 'circle');
  dot.setAttribute('cx', px); dot.setAttribute('cy', py);
  dot.setAttribute('r', 3 + n.imp * 3.4);
  dot.setAttribute('fill', '#ffffff');
  dot.setAttribute('opacity', 0.95);
  const label = document.createElementNS(NS, 'text');
  label.setAttribute('x', px + 14); label.setAttribute('y', py - 10);
  label.setAttribute('fill', '#cddcff'); label.setAttribute('font-size', '12');
  label.setAttribute('opacity', 0.75);
  label.textContent = n.who;
  g.append(halo, dot, label);
  g.style.cursor = 'pointer';
  g.addEventListener('mousemove', (e) => {
    document.getElementById('who').textContent = n.who;
    document.getElementById('txt').textContent = n.text;
    document.getElementById('kind').textContent = n.kind + ' · 重要性 ' + n.imp.toFixed(2);
    document.getElementById('when').textContent = n.when;
    hy.style.left = Math.min(innerWidth - 320, e.clientX + 18) + 'px';
    hy.style.top = Math.max(20, Math.min(innerHeight - 170, e.clientY - 20)) + 'px';
    hy.classList.add('on');
  });
  g.addEventListener('mouseleave', () => hy.classList.remove('on'));
  document.getElementById('sky').appendChild(g);
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
  `<span style="color:rgb(${DATA.core.color})"><i style="background:rgb(${DATA.core.color})"></i>她 PAD ${DATA.core.pad}</span>`;

const spots = [
  { left: '38px', top: '150px' },
  { left: '38px', top: 'calc(50% + 60px)' },
  { right: '38px', top: '150px' },
  { right: '38px', top: 'calc(50% + 60px)' },
];
DATA.panels.forEach((p, i) => {
  const spot = spots[i % spots.length];
  const div = document.createElement('div');
  div.className = 'panel' + (p.ok ? ' ok' : '');
  div.style.cssText = Object.entries(spot).map(([k, v]) => `${k}:${v}`).join(';');
  div.innerHTML = `<div class="t">${p.title}</div>` +
    p.rows.map(r => `<div class="row">${r}</div>`).join('');
  document.getElementById('panels').appendChild(div);
});

const cardSpots = [{ left: '38px', bottom: '56px' }, { right: '38px', bottom: '56px' }];
DATA.cards.slice(0, 2).forEach((c, i) => {
  const spot = cardSpots[i];
  const div = document.createElement('div');
  div.className = 'card';
  div.style.cssText = Object.entries(spot).map(([k, v]) => `${k}:${v}`).join(';');
  div.innerHTML =
    `<div class="face" style="background:linear-gradient(135deg, rgba(${c.color},.5), rgba(10,14,26,.92));"></div>` +
    `<div class="body"><div class="name">${c.name}</div><div class="en">${c.en}</div>` +
    `<div class="kv"><span>记忆</span><b>${c.count}</b></div>` +
    `<div class="kv"><span>亲近度</span><b>${c.affinity}</b></div>` +
    `<div class="quote">${c.quote}</div></div>`;
  document.getElementById('cards').appendChild(div);
});
</script>
</body>
</html>
"""


def _svg_payload(mem_path: str | Path, state_path: str | Path) -> dict[str, Any]:
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


def render_svg_html(payload: dict[str, Any]) -> str:
    data = json.dumps(payload, ensure_ascii=False).replace("</", "<\\/")
    return _HTML.replace("__DATA__", data)


def export_svg_map(
    mem_path: str | Path, state_path: str | Path, out_path: str | Path
) -> Path:
    payload = _svg_payload(mem_path, state_path)
    target = Path(out_path)
    target.write_text(render_svg_html(payload), encoding="utf-8")
    return target


__all__ = ["export_svg_map", "render_svg_html"]
