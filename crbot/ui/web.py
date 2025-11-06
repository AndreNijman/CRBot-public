from __future__ import annotations

from flask import Flask, Response, jsonify

from crbot.training.status import TrainingStatus


def create_status_app(status: TrainingStatus) -> Flask:
    app = Flask(__name__)

    index_html = """<!doctype html>
<html>
<head>
  <meta charset="utf-8">
  <title>CRBot status</title>
  <meta name="viewport" content="width=device-width,initial-scale=1">
  <style>
    html,body{background:#0b0b0b;color:#eaeaea;font-family:system-ui,Segoe UI,Roboto,Helvetica,Arial,sans-serif;margin:0;padding:0}
    .wrap{max-width:980px;margin:36px auto;padding:0 16px}
    h1{font-size:24px;margin:0 0 12px}
    .grid{display:grid;grid-template-columns:repeat(auto-fit,minmax(240px,1fr));gap:12px}
    .card{background:#151515;border:1px solid #222;border-radius:12px;padding:14px}
    .k{opacity:.7;font-size:12px;margin-bottom:6px}
    .v{font-size:15px;white-space:pre-wrap;word-break:break-word}
    .yes{color:#90ee90} .none{color:#ffadad}
    table{width:100%;border-collapse:collapse;margin-top:6px}
    th,td{border:1px solid #2a2a2a;padding:6px 8px;text-align:center;font-size:14px}
    th{background:#101010}
    footer{opacity:.6;font-size:12px;margin-top:12px}
    code{background:#0f0f0f;border:1px solid #222;border-radius:6px;padding:2px 6px}
  </style>
</head>
<body>
  <div class="wrap">
    <h1>CRBot status</h1>

    <div class="grid">
      <div class="card"><div class="k">episode</div><div id="episode" class="v">0</div></div>
      <div class="card"><div class="k">step</div><div id="step" class="v">0</div></div>
      <div class="card"><div class="k">epsilon</div><div id="epsilon" class="v">1.0</div></div>
      <div class="card"><div class="k">elixir</div><div id="elixir" class="v">none</div></div>
      <div class="card"><div class="k">win screen</div><div id="win" class="v none">none</div></div>
      <div class="card"><div class="k">play again btn</div><div id="play" class="v none">none</div></div>
      <div class="card" style="grid-column:1/-1"><div class="k">player hand</div><div id="hand" class="v">none</div></div>
      <div class="card" style="grid-column:1/-1"><div class="k">enemy troops</div><div id="enemy" class="v">none</div></div>

      <div class="card" style="grid-column:1/-1">
        <div class="k">tower OCR (all numbers found in big boxes)</div>
        <table>
          <thead>
            <tr><th></th><th>Princess L</th><th>King</th><th>Princess R</th></tr>
          </thead>
          <tbody>
            <tr><th>Enemy</th><td id="e_pl">[]</td><td id="e_k">[]</td><td id="e_pr">[]</td></tr>
            <tr><th>Ally</th><td id="a_pl">[]</td><td id="a_k">[]</td><td id="a_pr">[]</td></tr>
          </tbody>
        </table>
      </div>

      <div class="card" style="grid-column:1/-1"><div class="k">last action</div><div id="last_action" class="v">none</div></div>
    </div>

    <footer>Auto-refreshing. Raw JSON at <code>/status</code>. Crops saved to <code>debug/</code>.</footer>
  </div>

  <script>
    const $ = id => document.getElementById(id);
    const fmtList = x => (!x || x.length===0) ? "[]" : "[" + x.join(", ") + "]";
    const fmt = x => (x===null || x===undefined || x==="" ? "none" : String(x));
    async function tick(){
      try{
        const res = await fetch("/status", {cache:"no-store"});
        if(res.ok){
          const j = await res.json();
          $("episode").textContent = fmt(j.episode);
          $("step").textContent = fmt(j.step);
          $("epsilon").textContent = fmt(j.epsilon);
          $("elixir").textContent = fmt(j.elixir);
          $("hand").textContent = (j.hand && j.hand.length) ? j.hand.join(", ") : "none";
          $("enemy").textContent = (j.enemy && j.enemy.length) ? j.enemy.join(", ") : "none";
          $("win").textContent = j.win ? "yes" : "none";
          $("win").className = "v " + (j.win ? "yes" : "none");
          $("play").textContent = j.play_again ? "yes" : "none";
          $("play").className = "v " + (j.play_again ? "yes" : "none");
          $("last_action").textContent = fmt(j.last_action);

          const t = j.tower_ocr || {ally:{},enemy:{}};
          $("e_pl").textContent = fmtList(t.enemy?.princess_left || []);
          $("e_k").textContent  = fmtList(t.enemy?.king || []);
          $("e_pr").textContent = fmtList(t.enemy?.princess_right || []);
          $("a_pl").textContent = fmtList(t.ally?.princess_left || []);
          $("a_k").textContent  = fmtList(t.ally?.king || []);
          $("a_pr").textContent = fmtList(t.ally?.princess_right || []);
        }
      }catch(e){}
      setTimeout(tick, 60);
    }
    tick();
  </script>
</body>
</html>"""

    @app.get("/")
    def index() -> Response:
        return Response(index_html, mimetype="text/html")

    @app.get("/status")
    def status_endpoint():
        return jsonify(status.get())

    return app
