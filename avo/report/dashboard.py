"""Self-contained HTML dashboard for an evolution run (wandb-style, local).

Pure template generation — no LLM involved anywhere. `collect()` reads the
run directory's structured artifacts (lineage.jsonl, state.json, evals/,
logs/step_*.jsonl, logs/supervisor.jsonl, baselines JSON), `render()` embeds
them as JSON in a static HTML page whose inline vanilla JS draws the charts
(SVG). `--watch` regenerates every N seconds and the page auto-reloads, so it
live-tracks a running evolution.
"""
from __future__ import annotations

import html
import json
import time
from pathlib import Path

from avo.report.report import load_baselines


# ---------------------------------------------------------------------------
# data collection (pure file reads)
# ---------------------------------------------------------------------------

def _read_jsonl(path: Path) -> list[dict]:
    if not path.exists():
        return []
    out = []
    for line in path.read_text().splitlines():
        if line.strip():
            try:
                out.append(json.loads(line))
            except json.JSONDecodeError:
                pass
    return out


def _step_stats(run_dir: Path) -> list[dict]:
    steps = []
    for f in sorted((run_dir / "logs").glob("step_*.jsonl")):
        recs = _read_jsonl(f)
        tools: dict[str, int] = {}
        turns = evals = 0
        for r in recs:
            if r.get("kind") == "assistant":
                turns += 1
            elif r.get("kind") == "tool":
                name = r["payload"].get("name", "?")
                tools[name] = tools.get(name, 0) + 1
                if name in ("evaluate", "submit"):
                    evals += 1
        ts = [r["ts"] for r in recs if "ts" in r]
        steps.append({"step": int(f.stem.split("_")[1]), "turns": turns,
                      "evals": evals, "tools": tools,
                      "kb": sum(v for k, v in tools.items() if k.startswith("kb_")),
                      "t0": min(ts) if ts else None,
                      "t1": max(ts) if ts else None})
    return steps


def collect(run_dir: Path, baselines_path: Path | None = None) -> dict:
    run_dir = Path(run_dir)
    lineage = _read_jsonl(run_dir / "lineage.jsonl")
    state = {}
    if (run_dir / "state.json").exists():
        state = json.loads((run_dir / "state.json").read_text())
    summary = {}
    if (run_dir / "summary.json").exists():
        summary = json.loads((run_dir / "summary.json").read_text())

    # per-version per-config metrics from the eval cache
    for e in lineage:
        cfgs = []
        eval_file = run_dir / "evals" / f"{e.get('eval_hash', '')}.json"
        if eval_file.exists():
            try:
                cfgs = json.loads(eval_file.read_text()).get("configs", [])
            except json.JSONDecodeError:
                pass
        e["configs"] = cfgs

    steps = _step_stats(run_dir)
    committed_by_step = {e["step"]: e for e in lineage if e["version"] != "v0000"}
    supervisor = _read_jsonl(run_dir / "logs" / "supervisor.jsonl")
    for s in steps:
        e = committed_by_step.get(s["step"])
        s["outcome"] = "committed" if e else "failed"
        s["version"] = e["version"] if e else None
        s["score"] = e["score"] if e else None
        # a reflection belongs to the step that started right after it fired
        s["reflected"] = any(s["t0"] and r["ts"] <= s["t0"]
                             and (not steps[steps.index(s) - 1]["t1"]
                                  or r["ts"] >= (steps[steps.index(s) - 1]["t1"] or 0))
                             for r in supervisor) if steps.index(s) > 0 else \
            any(s["t0"] and r["ts"] <= s["t0"] for r in supervisor)

    baselines = load_baselines(baselines_path, run_dir.parent / "baselines")
    config_meta = {}
    if (run_dir / "config.yaml").exists():
        try:
            config_meta = json.loads((run_dir / "config.yaml").read_text())
        except json.JSONDecodeError:
            pass

    return {
        "run": run_dir.name,
        "generated_at": time.strftime("%Y-%m-%d %H:%M:%S"),
        "running": not (run_dir / "summary.json").exists(),
        "lineage": lineage,
        "state": state,
        "summary": summary,
        "steps": steps,
        "supervisor": supervisor,
        "baselines": baselines.get("geomeans", {}),
        "baselines_per_config": baselines.get("per_config", {}),
        "budgets": config_meta.get("budgets", {}),
        "model": (config_meta.get("llm") or {}).get("model", ""),
    }


# ---------------------------------------------------------------------------
# rendering
# ---------------------------------------------------------------------------

def render(data: dict, refresh_s: int | None = None) -> str:
    meta_refresh = (f'<meta http-equiv="refresh" content="{refresh_s}">'
                    if refresh_s else "")
    return (_TEMPLATE
            .replace("__TITLE__", html.escape(f"AVO · {data['run']}"))
            .replace("__META_REFRESH__", meta_refresh)
            .replace("__DATA__", json.dumps(data).replace("</", "<\\/")))


def build(run_dir: Path, baselines_path: Path | None = None,
          refresh_s: int | None = None) -> Path:
    run_dir = Path(run_dir)
    out = run_dir / "report" / "dashboard.html"
    out.parent.mkdir(exist_ok=True)
    out.write_text(render(collect(run_dir, baselines_path), refresh_s))
    return out


def watch(run_dir: Path, baselines_path: Path | None = None,
          interval_s: int = 30) -> None:
    print(f"[avo] regenerating dashboard every {interval_s}s — Ctrl-C to stop")
    while True:
        out = build(run_dir, baselines_path, refresh_s=interval_s)
        print(f"[avo] {time.strftime('%H:%M:%S')} wrote {out}")
        time.sleep(interval_s)


_TEMPLATE = r"""<!doctype html>
<html lang="en">
<head>
<meta charset="utf-8">
<meta name="viewport" content="width=device-width, initial-scale=1">
__META_REFRESH__
<title>__TITLE__</title>
<style>
  :root {
    color-scheme: light;
    --page: #f9f9f7; --surface: #fcfcfb;
    --ink: #0b0b0b; --ink-2: #52514e; --muted: #898781;
    --grid: #e1e0d9; --axis: #c3c2b7; --ring: rgba(11,11,11,.10);
    --s1: #2a78d6; --s2: #eb6834; --s3: #1baf7a; --s4: #eda100;
    --good: #0ca30c; --critical: #d03b3b; --good-text: #006300;
  }
  @media (prefers-color-scheme: dark) {
    :root:not([data-theme="light"]) {
      color-scheme: dark;
      --page: #0d0d0d; --surface: #1a1a19;
      --ink: #ffffff; --ink-2: #c3c2b7; --muted: #898781;
      --grid: #2c2c2a; --axis: #383835; --ring: rgba(255,255,255,.10);
      --s1: #3987e5; --s2: #d95926; --s3: #199e70; --s4: #c98500;
      --good-text: #0ca30c;
    }
  }
  :root[data-theme="dark"] {
    color-scheme: dark;
    --page: #0d0d0d; --surface: #1a1a19;
    --ink: #ffffff; --ink-2: #c3c2b7; --muted: #898781;
    --grid: #2c2c2a; --axis: #383835; --ring: rgba(255,255,255,.10);
    --s1: #3987e5; --s2: #d95926; --s3: #199e70; --s4: #c98500;
    --good-text: #0ca30c;
  }
  * { box-sizing: border-box; margin: 0; }
  body { background: var(--page); color: var(--ink);
         font: 14px/1.45 system-ui, -apple-system, "Segoe UI", sans-serif;
         padding: 20px; }
  .wrap { max-width: 1180px; margin: 0 auto; display: grid; gap: 14px; }
  header { display: flex; align-items: baseline; gap: 12px; flex-wrap: wrap; }
  header h1 { font-size: 17px; font-weight: 650; }
  header .sub { color: var(--muted); font-size: 12px; }
  .pill { font-size: 11px; padding: 2px 9px; border-radius: 99px;
          border: 1px solid var(--ring); color: var(--ink-2); }
  .pill.live::before { content: "●"; color: var(--good); margin-right: 5px; }
  button.tgl { margin-left: auto; background: var(--surface); color: var(--ink-2);
          border: 1px solid var(--ring); border-radius: 7px; padding: 4px 10px;
          font: inherit; font-size: 12px; cursor: pointer; }
  .tiles { display: grid; grid-template-columns: repeat(auto-fit, minmax(150px, 1fr)); gap: 10px; }
  .tile { background: var(--surface); border: 1px solid var(--ring);
          border-radius: 10px; padding: 12px 14px; }
  .tile .k { font-size: 11px; color: var(--muted); text-transform: uppercase;
             letter-spacing: .04em; }
  .tile .v { font-size: 24px; font-weight: 650; margin-top: 2px; }
  .tile .d { font-size: 12px; color: var(--ink-2); margin-top: 1px; }
  .tile .d.up { color: var(--good-text); }
  .card { background: var(--surface); border: 1px solid var(--ring);
          border-radius: 10px; padding: 14px 16px; overflow-x: auto; }
  .card h2 { font-size: 13px; font-weight: 650; color: var(--ink-2); margin-bottom: 8px; }
  .row2 { display: grid; grid-template-columns: 1fr 1fr; gap: 14px; }
  @media (max-width: 900px) { .row2 { grid-template-columns: 1fr; } }
  svg { display: block; width: 100%; height: auto; }
  svg text { font: 11px system-ui, sans-serif; fill: var(--muted);
             font-variant-numeric: tabular-nums; }
  .legend { display: flex; gap: 16px; font-size: 12px; color: var(--ink-2);
            margin: 6px 2px 0; flex-wrap: wrap; }
  .legend .sw { display: inline-block; width: 10px; height: 10px;
                border-radius: 3px; margin-right: 5px; vertical-align: -1px; }
  #tip { position: fixed; pointer-events: none; background: var(--surface);
         border: 1px solid var(--ring); border-radius: 8px; padding: 8px 10px;
         font-size: 12px; color: var(--ink); box-shadow: 0 4px 14px rgba(0,0,0,.18);
         opacity: 0; transition: opacity .08s; max-width: 340px; z-index: 9; }
  #tip .t { color: var(--muted); font-size: 11px; }
  table { border-collapse: collapse; width: 100%; font-size: 13px; }
  th { text-align: left; color: var(--muted); font-weight: 550; font-size: 11px;
       text-transform: uppercase; letter-spacing: .04em; }
  th, td { padding: 6px 10px 6px 0; border-bottom: 1px solid var(--grid); }
  td.num, th.num { text-align: right; font-variant-numeric: tabular-nums; }
  td .delta { color: var(--good-text); font-size: 12px; }
  .steps { display: flex; gap: 4px; flex-wrap: wrap; }
  .stepbx { width: 26px; height: 26px; border-radius: 6px; border: 1px solid var(--ring);
            display: grid; place-items: center; font-size: 11px; color: var(--ink-2);
            position: relative; cursor: default; }
  .stepbx.committed { background: var(--good); color: #fff; border-color: transparent; }
  .stepbx .zap { position: absolute; top: -7px; right: -5px; font-size: 10px; }
  footer { color: var(--muted); font-size: 11px; }
</style>
</head>
<body>
<div class="wrap">
  <header>
    <h1>AVO kernel evolution</h1>
    <span class="sub" id="runname"></span>
    <span class="pill" id="status"></span>
    <button class="tgl" id="theme">◐ theme</button>
  </header>
  <div class="tiles" id="tiles"></div>
  <div class="card">
    <h2>Score per committed version — geomean TFLOPS, baselines dashed
        <button class="tgl" id="logscale" style="float:right">log scale</button></h2>
    <div id="mainchart"></div>
  </div>
  <div class="row2">
    <div class="card">
      <h2>Latest kernel vs best baseline — per benchmark config (TFLOPS)</h2>
      <div id="cfgchart"></div>
      <div class="legend" id="cfglegend"></div>
    </div>
    <div class="card">
      <h2>Variation steps <span style="color:var(--muted);font-weight:400">
          (✓ committed · ⚡ supervisor guidance · hover for detail)</span></h2>
      <div class="steps" id="steps"></div>
      <h2 style="margin-top:14px">Supervisor interventions</h2>
      <div id="sup" style="font-size:12px;color:var(--ink-2)"></div>
    </div>
  </div>
  <div class="card">
    <h2>Lineage</h2>
    <table id="lin"><thead><tr>
      <th>version</th><th>step</th><th class="num">score</th>
      <th class="num">Δ vs seed</th><th>change</th></tr></thead><tbody></tbody></table>
  </div>
  <footer id="foot"></footer>
</div>
<div id="tip"></div>
<script>
const D = __DATA__;
const $ = s => document.querySelector(s);
const el = (t, a = {}, txt) => { const n = document.createElementNS(
  t === "svg" || a._svg ? "http://www.w3.org/2000/svg" : "http://www.w3.org/1999/xhtml", t);
  delete a._svg; for (const k in a) n.setAttribute(k, a[k]);
  if (txt != null) n.textContent = txt; return n; };
const fmt = (x, d = 2) => x == null ? "–" : (+x).toLocaleString("en", {maximumFractionDigits: d});
const css = v => getComputedStyle(document.documentElement).getPropertyValue(v).trim();

/* theme toggle: auto -> light -> dark */
{ const order = ["", "light", "dark"]; let i = 0;
  $("#theme").onclick = () => { i = (i + 1) % 3;
    if (order[i]) document.documentElement.setAttribute("data-theme", order[i]);
    else document.documentElement.removeAttribute("data-theme");
    draw(); }; }

/* tooltip */
const tip = $("#tip");
const showTip = (ev, html_) => { tip.innerHTML = html_; tip.style.opacity = 1;
  const pad = 14, w = tip.offsetWidth;
  tip.style.left = Math.min(ev.clientX + pad, innerWidth - w - 8) + "px";
  tip.style.top = (ev.clientY + pad) + "px"; };
const hideTip = () => tip.style.opacity = 0;

const lin = D.lineage, seed = lin[0], best = lin.reduce((a, b) => b.score > a.score ? b : a, lin[0] || {score: 0});
const bl = Object.entries(D.baselines || {}).sort((a, b) => b[1] - a[1]);
const bestBl = bl.length ? bl[0] : null;
let logScale = false;
$("#logscale").onclick = () => { logScale = !logScale; draw(); };

function tiles() {
  const t = $("#tiles"); t.innerHTML = "";
  const spend = D.state.usd ?? D.summary.usd;
  const toks = (D.state.input_tokens ?? 0) + (D.state.output_tokens ?? 0);
  const items = [
    ["best score", fmt(best?.score) + " TF",
     seed && best ? "+" + fmt((best.score / seed.score - 1) * 100, 1) + "% vs seed" : "", true],
    ["versions", lin.length ? lin.length - 1 : 0, "of " + (D.budgets.max_versions ?? "?") + " max"],
    ["steps", D.state.steps_done ?? 0, (D.state.stagnation ?? 0) + " stagnating"],
    ["spend", "$" + fmt(spend, 2), fmt(toks / 1e6, 2) + "M tokens · " + (D.model || "")],
    bestBl ? ["vs " + bestBl[0], fmt(best?.score / bestBl[1] * 100, 1) + "%",
              "baseline " + fmt(bestBl[1], 1) + " TF"] : null,
  ].filter(Boolean);
  for (const [k, v, d, up] of items) {
    const c = el("div", {class: "tile"});
    c.append(el("div", {class: "k"}, k), el("div", {class: "v"}, String(v)));
    if (d) c.append(el("div", {class: "d" + (up ? " up" : "")}, d));
    t.append(c);
  }
}

function mainChart() {
  const host = $("#mainchart"); host.innerHTML = "";
  if (!lin.length) { host.textContent = "no committed versions yet"; return; }
  const W = 1120, H = 320, L = 52, R = 120, T = 14, B = 34;
  const xs = lin.map((_, i) => i);
  const maxY = Math.max(...lin.map(e => e.score), ...(bl.map(b => b[1])), 1) * 1.08;
  const minY = logScale ? Math.max(0.5, Math.min(...lin.map(e => e.score)) * 0.8) : 0;
  const sx = i => L + (xs.length === 1 ? (W - L - R) / 2 : i * (W - L - R) / (xs.length - 1));
  const sy = v => { if (logScale) { const a = Math.log10(minY), b = Math.log10(maxY);
      return T + (H - T - B) * (1 - (Math.log10(Math.max(v, minY)) - a) / (b - a)); }
    return T + (H - T - B) * (1 - (v - minY) / (maxY - minY)); };
  const svg = el("svg", {viewBox: `0 0 ${W} ${H}`, _svg: 1});
  const ticks = logScale ? [1, 2, 5, 10, 20, 50, 100].filter(v => v >= minY && v <= maxY)
    : Array.from({length: 5}, (_, i) => minY + (maxY - minY) * i / 4);
  for (const v of ticks) {
    svg.append(el("line", {x1: L, x2: W - R, y1: sy(v), y2: sy(v),
      stroke: css("--grid"), "stroke-width": 1, _svg: 1}));
    svg.append(el("text", {x: L - 8, y: sy(v) + 4, "text-anchor": "end", _svg: 1}, fmt(v, v < 10 ? 1 : 0)));
  }
  svg.append(el("line", {x1: L, x2: W - R, y1: sy(minY), y2: sy(minY), stroke: css("--axis"), _svg: 1}));
  for (const [name, v] of bl) {
    svg.append(el("line", {x1: L, x2: W - R, y1: sy(v), y2: sy(v), stroke: css("--muted"),
      "stroke-dasharray": "5 4", "stroke-width": 1.2, opacity: .75, _svg: 1}));
    svg.append(el("text", {x: W - R + 6, y: sy(v) + 4, _svg: 1}, `${name} ${fmt(v, 1)}`));
  }
  const path = lin.map((e, i) => (i ? "L" : "M") + sx(i) + " " + sy(e.score)).join(" ");
  svg.append(el("path", {d: path, fill: "none", stroke: css("--s1"), "stroke-width": 2,
    "stroke-linejoin": "round", _svg: 1}));
  lin.forEach((e, i) => {
    svg.append(el("circle", {cx: sx(i), cy: sy(e.score), r: 4.5, fill: css("--s1"),
      stroke: css("--surface"), "stroke-width": 2, _svg: 1}));
    svg.append(el("text", {x: sx(i), y: H - B + 16, "text-anchor": "middle", _svg: 1}, e.version));
  });
  const last = lin[lin.length - 1];
  svg.append(el("text", {x: sx(lin.length - 1), y: sy(last.score) - 10,
    "text-anchor": "middle", fill: css("--ink"), "font-weight": 600, _svg: 1}, fmt(last.score)));
  const overlay = el("rect", {x: L, y: T, width: W - L - R, height: H - T - B,
    fill: "transparent", _svg: 1});
  overlay.addEventListener("mousemove", ev => {
    const bb = svg.getBoundingClientRect();
    const px = (ev.clientX - bb.left) * W / bb.width;
    let bi = 0, bd = 1e9;
    lin.forEach((_, i) => { const d = Math.abs(sx(i) - px); if (d < bd) { bd = d; bi = i; } });
    const e = lin[bi];
    showTip(ev, `<b>${e.version}</b> · step ${e.step} · ${fmt(e.score)} TFLOPS` +
      (seed && e !== seed ? ` <span class="t">(+${fmt((e.score / seed.score - 1) * 100, 1)}% vs seed)</span>` : "") +
      `<div class="t">${(e.message || "").split("\n")[0].slice(0, 120)}</div>`);
  });
  overlay.addEventListener("mouseleave", hideTip);
  svg.append(overlay);
  host.append(svg);
}

function cfgChart() {
  const host = $("#cfgchart"); host.innerHTML = ""; $("#cfglegend").innerHTML = "";
  const latest = [...lin].reverse().find(e => (e.configs || []).length);
  if (!latest) { host.textContent = "no per-config data yet"; return; }
  const cfgs = latest.configs;
  const blPer = bestBl ? (D.baselines_per_config[bestBl[0]] || {}) : {};
  const key = c => `s${c.seqlen}_b${c.batch}_${c.causal ? "causal" : "full"}`;
  const label = c => `${c.seqlen / 1024}k${c.causal ? " ⧄" : ""}`;
  const W = 560, H = 260, L = 46, T = 12, B = 30;
  const maxY = Math.max(...cfgs.map(c => c.tflops), ...cfgs.map(c => blPer[key(c)] || 0), 1) * 1.1;
  const svg = el("svg", {viewBox: `0 0 ${W} ${H}`, _svg: 1});
  const sy = v => T + (H - T - B) * (1 - v / maxY);
  for (let i = 0; i <= 4; i++) { const v = maxY * i / 4;
    svg.append(el("line", {x1: L, x2: W - 8, y1: sy(v), y2: sy(v), stroke: css("--grid"), _svg: 1}));
    svg.append(el("text", {x: L - 6, y: sy(v) + 4, "text-anchor": "end", _svg: 1}, fmt(v, 0)));
  }
  const groupW = (W - L - 8) / cfgs.length, barW = Math.min(26, groupW / 2 - 4);
  cfgs.forEach((c, i) => {
    const x0 = L + i * groupW + groupW / 2;
    const bars = [[css("--s1"), c.tflops, "kernel " + latest.version],
                  [css("--s2"), blPer[key(c)], bestBl ? bestBl[0] : "baseline"]];
    bars.forEach(([color, v, name], j) => {
      if (v == null) return;
      const x = x0 + (j - 1) * barW + (j ? 2 : -2);
      const r = el("path", {d: `M${x} ${sy(0)} V${sy(v) + 4} Q${x} ${sy(v)} ${x + 4} ${sy(v)}` +
        ` H${x + barW - 4} Q${x + barW} ${sy(v)} ${x + barW} ${sy(v) + 4} V${sy(0)} Z`,
        fill: color, _svg: 1});
      r.addEventListener("mousemove", ev => showTip(ev,
        `<b>${label(c)} ${c.causal ? "causal" : "full"}</b> B=${c.batch}<br>${name}: ${fmt(v)} TFLOPS`));
      r.addEventListener("mouseleave", hideTip);
      svg.append(r);
      if (!j) svg.append(el("text", {x: x + barW / 2, y: sy(v) - 5,
        "text-anchor": "middle", fill: css("--ink"), _svg: 1}, fmt(v, 1)));
    });
    svg.append(el("text", {x: x0, y: H - B + 16, "text-anchor": "middle", _svg: 1}, label(c)));
  });
  host.append(svg);
  const lg = $("#cfglegend");
  [[css("--s1"), "kernel " + latest.version], [css("--s2"), bestBl ? bestBl[0] : "baseline"]]
    .forEach(([c, n]) => { const s = el("span");
      s.append(el("span", {class: "sw", style: `background:${c}`}), n); lg.append(s); });
}

function stepsRow() {
  const host = $("#steps"); host.innerHTML = "";
  for (const s of D.steps) {
    const b = el("div", {class: "stepbx " + s.outcome}, s.outcome === "committed" ? "✓" : s.step);
    if (s.reflected) b.append(el("span", {class: "zap"}, "⚡"));
    b.addEventListener("mousemove", ev => showTip(ev,
      `<b>step ${s.step}</b> · ${s.outcome}${s.version ? " → " + s.version + " (" + fmt(s.score) + ")" : ""}` +
      `<div class="t">${s.turns} turns · ${s.evals} evals · ${s.kb} kb lookups` +
      (s.reflected ? " · supervisor guided" : "") + "</div>"));
    b.addEventListener("mouseleave", hideTip);
    host.append(b);
  }
  const sup = $("#sup"); sup.innerHTML = "";
  if (!D.supervisor.length) sup.textContent = "none yet";
  for (const r of D.supervisor) {
    const d = el("div", {style: "margin-bottom:8px"});
    d.append(el("div", {style: "color:var(--muted);font-size:11px"}, "⚡ " + r.reason));
    d.append(el("div", {}, (r.guidance || "").split("\n")[0].slice(0, 180) + "…"));
    sup.append(d);
  }
}

function table() {
  const tb = $("#lin tbody"); tb.innerHTML = "";
  for (const e of [...lin].reverse()) {
    const tr = el("tr");
    const dlt = seed && e.version !== "v0000"
      ? "+" + fmt((e.score / seed.score - 1) * 100, 1) + "%" : "";
    tr.append(el("td", {}, e.version), el("td", {}, String(e.step)),
      el("td", {class: "num"}, fmt(e.score)),
      Object.assign(el("td", {class: "num"}), {innerHTML: dlt ? `<span class="delta">${dlt}</span>` : "–"}),
      el("td", {}, (e.message || "").split("\n")[0].slice(0, 110)));
    tb.append(tr);
  }
}

function draw() {
  $("#runname").textContent = D.run;
  const st = $("#status");
  st.textContent = D.running ? "running" : "finished";
  st.className = "pill" + (D.running ? " live" : "");
  tiles(); mainChart(); cfgChart(); stepsRow(); table();
  $("#foot").textContent = `generated ${D.generated_at}` +
    (D.running ? " · auto-refreshes while `avo dashboard --watch` is running" : "") +
    ` · ${D.state.elapsed_s ? fmt(D.state.elapsed_s / 3600, 1) + "h elapsed" : ""}`;
}
draw();
</script>
</body>
</html>
"""
