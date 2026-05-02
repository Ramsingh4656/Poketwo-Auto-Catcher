"""
web.py — Flask dashboard for monitoring and controlling the selfbot.
"""

import os, threading, asyncio, logging, time
from flask import Flask, render_template_string, jsonify, request
from bot import CNN_CONFIDENCE_THRESHOLD, MIN_DELAY, MAX_DELAY, DISTRACTION_CHANCE, CATCH_CHANNEL_ID

logger = logging.getLogger("web")

app = Flask(__name__)

# References set by main.py after bot is created
_bot_ref = None
_bot_thread = None
_bot_loop = None
_bot_token = None
_bot_running = False


def set_bot_ref(bot, token):
    global _bot_ref, _bot_token
    _bot_ref = bot
    _bot_token = token


def _start_bot_thread():
    global _bot_thread, _bot_loop, _bot_running
    if _bot_running:
        return False

    def _run():
        global _bot_loop, _bot_running
        _bot_loop = asyncio.new_event_loop()
        asyncio.set_event_loop(_bot_loop)
        _bot_running = True
        try:
            _bot_loop.run_until_complete(_bot_ref.start(_bot_token))
        except Exception as e:
            logger.error("Bot stopped: %s", e)
        finally:
            _bot_running = False
            try:
                _bot_loop.close()
            except Exception:
                pass

    _bot_thread = threading.Thread(target=_run, daemon=True)
    _bot_thread.start()
    return True


def _stop_bot():
    global _bot_running
    if not _bot_running or not _bot_loop:
        return False
    try:
        asyncio.run_coroutine_threadsafe(_bot_ref.close(), _bot_loop)
    except Exception as e:
        logger.error("Error stopping bot: %s", e)
    _bot_running = False
    return True


# ── HTML Template ──────────────────────────────────────────────────────────────
DASHBOARD_HTML = """
<!DOCTYPE html>
<html lang="en">
<head>
<meta charset="UTF-8">
<meta name="viewport" content="width=device-width, initial-scale=1.0">
<title>PokeCatcher Dashboard</title>
<meta name="description" content="Pokemon auto-catcher selfbot dashboard">
<link href="https://fonts.googleapis.com/css2?family=Inter:wght@300;400;500;600;700&display=swap" rel="stylesheet">
<style>
  :root {
    --bg-primary: #0f0f1a;
    --bg-secondary: #1a1a2e;
    --bg-card: #16213e;
    --bg-card-hover: #1c2a4a;
    --accent: #e94560;
    --accent-glow: rgba(233, 69, 96, 0.3);
    --accent2: #0f3460;
    --text-primary: #eaeaea;
    --text-secondary: #a0a0b8;
    --text-muted: #6c6c80;
    --success: #00d26a;
    --warning: #ffc107;
    --border: rgba(255,255,255,0.06);
    --glass: rgba(255,255,255,0.03);
  }
  * { margin:0; padding:0; box-sizing:border-box; }
  body {
    font-family: 'Inter', sans-serif;
    background: var(--bg-primary);
    color: var(--text-primary);
    min-height: 100vh;
    overflow-x: hidden;
  }
  .bg-anim {
    position: fixed; inset:0; z-index:0; pointer-events:none;
    background:
      radial-gradient(ellipse at 20% 50%, rgba(233,69,96,0.08) 0%, transparent 50%),
      radial-gradient(ellipse at 80% 20%, rgba(15,52,96,0.12) 0%, transparent 50%);
  }
  .container { position:relative; z-index:1; max-width:960px; margin:0 auto; padding:24px 16px; }

  header {
    display:flex; align-items:center; justify-content:space-between;
    padding:20px 28px; margin-bottom:28px;
    background: var(--bg-secondary);
    border:1px solid var(--border);
    border-radius:16px;
    backdrop-filter: blur(12px);
  }
  header h1 { font-size:1.5rem; font-weight:700; letter-spacing:-0.5px; }
  header h1 span { color:var(--accent); }
  .status-badge {
    display:inline-flex; align-items:center; gap:6px;
    padding:6px 14px; border-radius:20px; font-size:0.8rem; font-weight:600;
    background: rgba(0,210,106,0.12); color:var(--success); border:1px solid rgba(0,210,106,0.2);
  }
  .status-badge.offline {
    background:rgba(255,193,7,0.12); color:var(--warning); border-color:rgba(255,193,7,0.2);
  }
  .status-dot { width:8px; height:8px; border-radius:50%; background:currentColor;
    animation: pulse 2s infinite; }
  @keyframes pulse { 0%,100%{opacity:1} 50%{opacity:0.4} }

  .grid { display:grid; grid-template-columns:repeat(auto-fit, minmax(200px,1fr)); gap:16px; margin-bottom:24px; }
  .card {
    background:var(--bg-card); border:1px solid var(--border); border-radius:14px;
    padding:20px 22px; transition:all 0.25s ease;
  }
  .card:hover { background:var(--bg-card-hover); transform:translateY(-2px);
    box-shadow:0 8px 30px rgba(0,0,0,0.3); }
  .card-label { font-size:0.75rem; text-transform:uppercase; letter-spacing:1px;
    color:var(--text-muted); margin-bottom:6px; }
  .card-value { font-size:1.8rem; font-weight:700; }
  .card-value.accent { color:var(--accent); }
  .card-value.success { color:var(--success); }

  .controls { display:flex; gap:12px; margin-bottom:24px; flex-wrap:wrap; }
  .btn {
    padding:10px 24px; border:none; border-radius:10px; font-family:inherit;
    font-size:0.85rem; font-weight:600; cursor:pointer; transition:all 0.2s;
    display:inline-flex; align-items:center; gap:6px;
  }
  .btn-start { background:var(--success); color:#111; }
  .btn-start:hover { filter:brightness(1.15); transform:scale(1.03); }
  .btn-stop { background:var(--accent); color:#fff; }
  .btn-stop:hover { filter:brightness(1.15); transform:scale(1.03); }
  .btn-refresh { background:var(--bg-card); color:var(--text-secondary); border:1px solid var(--border); }
  .btn-refresh:hover { background:var(--bg-card-hover); }

  .log-panel {
    background:var(--bg-card); border:1px solid var(--border); border-radius:14px;
    padding:18px 22px; max-height:420px; overflow-y:auto;
  }
  .log-panel h2 { font-size:0.95rem; font-weight:600; margin-bottom:12px; color:var(--text-secondary); }
  .log-line {
    font-family:'Fira Code','Consolas',monospace; font-size:0.78rem;
    color:var(--text-muted); padding:3px 0; border-bottom:1px solid var(--border);
    line-height:1.6; word-break:break-all;
  }
  .log-line:last-child { border-bottom:none; }

  .model-info {
    display:flex; align-items:center; gap:8px; padding:12px 20px;
    background:var(--bg-card); border:1px solid var(--border); border-radius:10px;
    margin-bottom:24px; font-size:0.85rem; color:var(--text-secondary);
  }
  .model-dot { width:10px; height:10px; border-radius:50%; }
  .model-dot.loaded { background:var(--success); box-shadow:0 0 8px var(--success); }
  .model-dot.unloaded { background:var(--accent); box-shadow:0 0 8px var(--accent); }

  footer { text-align:center; padding:24px 0; font-size:0.75rem; color:var(--text-muted); }
</style>
</head>
<body>
<div class="bg-anim"></div>
<div class="container">
  <header>
    <h1>Poké<span>Catcher</span></h1>
    <div id="statusBadge" class="status-badge offline">
      <div class="status-dot"></div><span id="statusText">Offline</span>
    </div>
  </header>

  <div class="model-info">
    <div id="modelDot" class="model-dot unloaded"></div>
    <span id="modelStatus">Model: checking…</span>
  </div>

  <div class="grid">
    <div class="card"><div class="card-label">Total Caught</div><div class="card-value accent" id="totalCaught">0</div></div>
    <div class="card"><div class="card-label">CNN Catches</div><div class="card-value success" id="cnnCatches">0</div></div>
    <div class="card"><div class="card-label">Hint Catches</div><div class="card-value" id="hintCatches">0</div></div>
    <div class="card"><div class="card-label">Skipped</div><div class="card-value" id="skipped">0</div></div>
    <div class="card"><div class="card-label">Uptime</div><div class="card-value" id="uptime" style="font-size:1.3rem">—</div></div>
  </div>

  <div class="controls">
    <button class="btn btn-start" id="btnStart" onclick="startBot()">▶ Start Bot</button>
    <button class="btn btn-stop" id="btnStop" onclick="stopBot()">■ Stop Bot</button>
    <button class="btn btn-refresh" onclick="refresh()">↻ Refresh</button>
  </div>

  <div class="log-panel" id="logPanel">
    <h2>Live Logs</h2>
    <div id="logContainer"><div class="log-line">Waiting for logs…</div></div>
  </div>

  <footer>PokéCatcher Dashboard · CNN Auto-Catcher · Use at your own risk</footer>
</div>

<script>
function fmtUptime(s){
  if(!s)return '—';
  const h=Math.floor(s/3600),m=Math.floor((s%3600)/60),sec=s%60;
  return (h?h+'h ':'')+(m?m+'m ':'')+(sec+'s');
}

async function refresh(){
  try{
    const r=await fetch('/dashboard/status');
    if(!r.ok) return;
    const d=await r.json();
    document.getElementById('totalCaught').textContent=d.stats.total_caught||0;
    document.getElementById('cnnCatches').textContent=d.stats.cnn_catches||0;
    document.getElementById('hintCatches').textContent=d.stats.hint_catches||0;
    document.getElementById('skipped').textContent=d.stats.skipped||0;
    document.getElementById('uptime').textContent=fmtUptime(d.stats.uptime_seconds);

    const badge=document.getElementById('statusBadge');
    const stxt=document.getElementById('statusText');
    if(d.running){badge.className='status-badge';stxt.textContent='Online';}
    else{badge.className='status-badge offline';stxt.textContent='Offline';}

    const md=document.getElementById('modelDot'),ms=document.getElementById('modelStatus');
    if(d.model_loaded){md.className='model-dot loaded';ms.textContent='Model: Loaded ✓';}
    else{md.className='model-dot unloaded';ms.textContent='Model: Not loaded ✗';}

    const lc=document.getElementById('logContainer');
    if(d.logs&&d.logs.length){
      lc.innerHTML=d.logs.map(l=>'<div class="log-line">'+l.replace(/</g,'&lt;')+'</div>').join('');
      lc.parentElement.scrollTop=lc.parentElement.scrollHeight;
    }
  }catch(e){console.error(e);}
}

async function startBot(){
  await fetch('/dashboard/start',{method:'POST'});
  setTimeout(refresh,1000);
}
async function stopBot(){
  await fetch('/dashboard/stop',{method:'POST'});
  setTimeout(refresh,1000);
}

refresh();
setInterval(refresh,5000);
</script>
</body>
</html>
"""


# ── Routes ─────────────────────────────────────────────────────────────────────
@app.route("/")
def index():
    """Redirect root to dashboard."""
    from flask import redirect
    return redirect("/dashboard")


@app.route("/dashboard")
def dashboard():
    return render_template_string(DASHBOARD_HTML)


@app.route("/dashboard/status")
def status():
    stats = _bot_ref.stats.to_dict() if _bot_ref else {}
    logs = list(_bot_ref.logs) if _bot_ref else []
    model_loaded = _bot_ref.predictor.loaded if _bot_ref else False
    return jsonify({
        "running": _bot_running,
        "model_loaded": model_loaded,
        "stats": stats,
        "logs": logs[-50:],
        "state": _bot_ref.state.name if _bot_ref else "UNKNOWN",
    })


@app.route("/dashboard/start", methods=["POST"])
def start():
    if not _bot_token:
        return jsonify({"error": "No USER_TOKEN configured"}), 400
    ok = _start_bot_thread()
    return jsonify({"started": ok})


@app.route("/dashboard/stop", methods=["POST"])
def stop():
    ok = _stop_bot()
    return jsonify({"stopped": ok})


@app.route("/dashboard/settings")
def settings():
    return jsonify({
        "confidence_threshold": CNN_CONFIDENCE_THRESHOLD,
        "min_delay": MIN_DELAY,
        "max_delay": MAX_DELAY,
        "distraction_chance": DISTRACTION_CHANCE,
        "catch_channel_id": CATCH_CHANNEL_ID,
    })
