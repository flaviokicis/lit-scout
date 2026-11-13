#!/usr/bin/env python3
"""Local browser for the lit-scout eval. Run: ANTHROPIC_API_KEY=... python3 eval/ui.py"""
import json
import sys
import threading
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer
from pathlib import Path
from urllib.parse import parse_qs, urlparse

HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE))
HOST, PORT = "127.0.0.1", 8765


def read_jsonl(path):
    with open(path, encoding="utf-8") as f:
        return [json.loads(line) for line in f if line.strip()]


def list_runs():
    root = HERE / "results"
    return sorted((p.name for p in root.iterdir() if (p / "items.jsonl").exists()), reverse=True) if root.exists() else []


def load_run(run):
    if run not in list_runs():
        raise KeyError(f"unknown run {run!r}")
    d = HERE / "results" / run
    metrics = json.loads((d / "metrics.json").read_text()) if (d / "metrics.json").exists() else {}
    rows = read_jsonl(d / "items.jsonl") if (d / "items.jsonl").exists() else []
    return {"run": run, "metrics": metrics.get("overall", {}).get("all", {}),
            "results": {r["item"]["id"]: {k: v for k, v in r.items() if k != "rendered"} for r in rows}}


def rerun(item_id):
    try:
        from run_eval import classify_item
    except ImportError as e:
        raise RuntimeError(f"run_eval.classify_item is not available yet: {e}")
    item = next((i for i in read_jsonl(HERE / "dataset.jsonl") if i["id"] == item_id), None)
    if item is None:
        raise KeyError(f"unknown item {item_id!r}")
    box = {}

    def work():
        try:
            box["out"] = classify_item(item)
        except Exception as e:  # surfaced to the page
            box["out"] = {"error": f"{type(e).__name__}: {e}"}

    t = threading.Thread(target=work)
    t.start()
    t.join()
    return box["out"]


class Handler(BaseHTTPRequestHandler):
    def log_message(self, fmt, *args):
        sys.stderr.write("%s %s\n" % (self.command, self.path))

    def send(self, code, body, ctype="application/json"):
        data = body.encode() if isinstance(body, str) else body
        self.send_response(code)
        self.send_header("Content-Type", ctype + "; charset=utf-8")
        self.send_header("Content-Length", str(len(data)))
        self.end_headers()
        self.wfile.write(data)

    def do_GET(self):
        url = urlparse(self.path)
        try:
            if url.path == "/":
                self.send(200, PAGE, "text/html")
            elif url.path == "/api/items":
                self.send(200, json.dumps(read_jsonl(HERE / "dataset.jsonl")))
            elif url.path == "/api/runs":
                self.send(200, json.dumps(list_runs()))
            elif url.path == "/api/run":
                runs = list_runs()
                run = parse_qs(url.query).get("run", [runs[0] if runs else ""])[0]
                self.send(200, json.dumps(load_run(run)))
            else:
                self.send(404, json.dumps({"error": "not found"}))
        except Exception as e:
            self.send(500, json.dumps({"error": f"{type(e).__name__}: {e}"}))

    def do_POST(self):
        if urlparse(self.path).path != "/rerun":
            return self.send(404, json.dumps({"error": "not found"}))
        try:
            body = json.loads(self.rfile.read(int(self.headers.get("Content-Length", 0))) or b"{}")
            self.send(200, json.dumps(rerun(body["id"]), default=str))
        except Exception as e:
            self.send(500, json.dumps({"error": f"{type(e).__name__}: {e}"}))


PAGE = r"""<!doctype html>
<html lang="en"><head><meta charset="utf-8"><title>Eval browser</title>
<meta name="viewport" content="width=device-width,initial-scale=1">
<style>
:root{--bg:#fafafa;--panel:#fff;--fg:#1c1c1e;--mute:#6b6b70;--line:#e2e2e6;--idea:#2563eb;--other:#6b7280;--bad:#dc2626;--badbg:#fee2e2;--ok:#16a34a;--user:#e8f0fe;--asst:#f1f1f3;--sel:#eef2ff}
@media(prefers-color-scheme:dark){:root{--bg:#141416;--panel:#1c1c1f;--fg:#e8e8ea;--mute:#9a9aa0;--line:#2e2e33;--idea:#60a5fa;--other:#9ca3af;--bad:#f87171;--badbg:#4a1d1d;--ok:#4ade80;--user:#1e2a44;--asst:#26262a;--sel:#232338}}
*{box-sizing:border-box}
body{margin:0;background:var(--bg);color:var(--fg);font:14px/1.45 -apple-system,system-ui,sans-serif;height:100vh;display:flex;flex-direction:column}
header{display:flex;gap:18px;align-items:center;flex-wrap:wrap;padding:10px 16px;border-bottom:1px solid var(--line);background:var(--panel)}
header h1{font-size:15px;margin:0}
.m{display:flex;gap:10px;align-items:baseline}.m b{font-size:12px;color:var(--mute);font-weight:600}
.stat{font-variant-numeric:tabular-nums}.stat span{color:var(--mute);font-size:12px;margin-right:3px}
select,input,button{font:inherit;color:var(--fg);background:var(--panel);border:1px solid var(--line);border-radius:6px;padding:4px 8px}
button{cursor:pointer}button:hover{border-color:var(--mute)}button:disabled{opacity:.5}
main{flex:1;display:flex;min-height:0}
aside{width:340px;border-right:1px solid var(--line);display:flex;flex-direction:column;background:var(--panel)}
.filters{padding:8px;display:flex;flex-direction:column;gap:6px;border-bottom:1px solid var(--line)}
.filters .row{display:flex;gap:6px;align-items:center}.filters select{flex:1;min-width:0}
#list{overflow:auto;flex:1}
.it{display:flex;gap:6px;align-items:center;padding:6px 10px;border-bottom:1px solid var(--line);cursor:pointer}
.it:hover{background:var(--bg)}.it.sel{background:var(--sel)}
.it .id{flex:1;overflow:hidden;text-overflow:ellipsis;white-space:nowrap;font-size:13px}
.chip{font-size:11px;font-weight:600;padding:1px 7px;border-radius:10px;border:1px solid currentColor}
.chip.LIT{color:var(--idea)}.chip.OTHER{color:var(--other)}.chip.ERR{color:var(--bad)}
.chip.wrong{background:var(--badbg);color:var(--bad)}
.arrow{color:var(--mute);font-size:11px}
#pane{flex:1;overflow:auto;padding:18px 24px}
.meta{display:flex;gap:8px;flex-wrap:wrap;align-items:center;margin-bottom:6px}
.tag{font-size:12px;color:var(--mute);background:var(--asst);padding:1px 8px;border-radius:10px}
.note{color:var(--mute);font-style:italic;margin:6px 0 14px}
h3{font-size:12px;text-transform:uppercase;letter-spacing:.05em;color:var(--mute);margin:18px 0 6px}
.b{padding:8px 12px;border-radius:12px;margin:5px 0;max-width:85%;white-space:pre-wrap;word-wrap:break-word}
.b.user{background:var(--user);margin-left:auto}.b.assistant{background:var(--asst)}
.b small{display:block;color:var(--mute);font-size:11px;margin-bottom:2px}
.prompt{border:2px solid var(--idea);border-radius:10px;padding:10px 14px;white-space:pre-wrap;background:var(--panel)}
pre{background:var(--panel);border:1px solid var(--line);border-radius:8px;padding:10px;white-space:pre-wrap;word-wrap:break-word;margin:4px 0;font:12.5px/1.45 ui-monospace,Menlo,monospace}
.bar{display:flex;gap:8px;margin:14px 0;flex-wrap:wrap}
.err{color:var(--bad)}.empty{color:var(--mute);padding:40px}
</style></head><body>
<header><h1>Eval browser</h1>
<select id="run"></select><span id="count" class="stat"></span>
<div id="metrics" style="display:flex;gap:22px;flex-wrap:wrap"></div></header>
<main><aside><div class="filters">
<input id="q" placeholder="Filter id / text">
<div class="row"><select id="dom"></select><select id="ct"></select></div>
<label><input type="checkbox" id="errs"> only errors (Haiku prediction wrong)</label></div>
<div id="list"></div></aside>
<div id="pane"><div class="empty">Pick an item.</div></div></main>
<script>
const $=s=>document.querySelector(s);
const esc=s=>String(s??'').replace(/[&<>"]/g,c=>({'&':'&amp;','<':'&lt;','>':'&gt;','"':'&quot;'}[c]));
let items=[],run={results:{},metrics:{}},sel=null,reruns={};
const pred=id=>{const l=run.results[id]?.haiku?.label;return l==='IDEA'?'LIT':l||null};
const wrong=i=>{const p=pred(i.id);return p!==null&&p!==i.label};
const pct=x=>x==null?'-':(x*100).toFixed(1)+'%';
const chip=(l,extra='')=>`<span class="chip ${esc(l||'ERR')} ${extra}">${esc(l||'none')}</span>`;
const plain=i=>[...i.context.map(t=>t.role+': '+t.text),'user (newest prompt): '+i.prompt].join('\n\n');

async function j(u,o){const r=await fetch(u,o);const d=await r.json();if(!r.ok)throw new Error(d.error||r.status);return d}
async function init(){
  items=await j('/api/items');
  const runs=await j('/api/runs');
  $('#run').innerHTML=runs.map(r=>`<option>${esc(r)}</option>`).join('');
  const fill=(el,name,k)=>{el.innerHTML=`<option value="">all ${name}</option>`+[...new Set(items.map(i=>i[k]))].sort().map(v=>`<option>${esc(v)}</option>`).join('')};
  fill($('#dom'),'domains','domain');fill($('#ct'),'case types','case_type');
  if(runs.length)await loadRun(runs[0]);else renderList();
}
async function loadRun(r){
  run=await j('/api/run?run='+encodeURIComponent(r));
  const m=run.metrics,row=(n,x)=>x?`<div class="m"><b>${n}</b>`+[['Precision',x.precision],['Recall',x.recall],['F1',x.f1]].map(([k,v])=>`<span class="stat"><span>${k}</span>${v==null?'-':v.toFixed(3)}</span>`).join('')+`<span class="stat"><span>Flagged</span>${x.flagged_pct==null?'-':x.flagged_pct.toFixed(1)+'%'}</span></div>`:'';
  $('#metrics').innerHTML=row('Haiku alone',m.haiku_alone)+row('Haiku then Sonnet',m.haiku_then_sonnet);
  renderList();if(sel)renderPane();
}
function visible(){
  const q=$('#q').value.toLowerCase(),d=$('#dom').value,c=$('#ct').value,e=$('#errs').checked;
  return items.filter(i=>(!d||i.domain===d)&&(!c||i.case_type===c)&&(!e||wrong(i))&&(!q||(i.id+' '+i.prompt+' '+i.note).toLowerCase().includes(q)));
}
function renderList(){
  const v=visible();$('#count').textContent=`${v.length} / ${items.length} items, ${items.filter(wrong).length} wrong`;
  $('#list').innerHTML=v.map(i=>{const p=pred(i.id);return `<div class="it ${sel===i.id?'sel':''}" data-id="${esc(i.id)}"><span class="id">${esc(i.id)}</span>${chip(i.label)}<span class="arrow">pred</span>${p?chip(p,wrong(i)?'wrong':''):'<span class="arrow">n/a</span>'}</div>`}).join('');
}
function block(title,o){
  if(!o)return '';
  return `<h3>${title}</h3>`+(o.error?`<div class="err">error: ${esc(o.error)}</div>`:'')+
   `<div class="meta">${chip(o.label)}${o.query?`<span>query: <b>${esc(o.query)}</b></span>`:''}</div>`+(o.reply??o.raw!=null?`<pre>${esc(o.reply??o.raw)}</pre>`:'');
}
function renderPane(){
  const i=items.find(x=>x.id===sel);if(!i)return;
  const r=run.results[i.id]||{},rr=reruns[i.id];
  $('#pane').innerHTML=`<div class="meta"><b style="font-size:16px">${esc(i.id)}</b>Gold ${chip(i.label)}<span class="tag">${esc(i.case_type)}</span><span class="tag">${esc(i.domain)}</span></div>
  ${i.note?`<div class="note">${esc(i.note)}</div>`:''}
  <div class="bar"><button id="cp">Copy</button><button id="cj">Copy JSON</button><button id="rr">Run Haiku again</button></div>
  <h3>Context</h3>${i.context.length?i.context.map(t=>`<div class="b ${t.role==='user'?'user':'assistant'}"><small>${esc(t.role)}</small>${esc(t.text)}</div>`).join(''):'<div class="tag">no context</div>'}
  <h3>Prompt</h3><div class="prompt">${esc(i.prompt)}</div>
  ${r.haiku?block('Haiku (stored run)',{...r.haiku}):'<h3>Stored run</h3><div class="tag">no result for this item</div>'}
  ${r.sonnet?block('Sonnet confirm (stored run)',r.sonnet)+`<div class="meta">Pipeline label ${chip(r.pipeline_label)}</div>`:''}
  <div id="rrout">${rr?(rr.pending?'<h3>Haiku re-run</h3>Running...':block('Haiku re-run (not saved)',rr)+(rr.usage?`<pre>usage: ${esc(JSON.stringify(rr.usage))}</pre>`:'')):''}</div>`;
  const flash=(b,t)=>{const o=b.textContent;b.textContent=t;setTimeout(()=>b.textContent=o,1000)};
  $('#cp').onclick=e=>navigator.clipboard.writeText(plain(i)).then(()=>flash(e.target,'Copied'));
  $('#cj').onclick=e=>navigator.clipboard.writeText(JSON.stringify(i,null,2)).then(()=>flash(e.target,'Copied'));
  $('#rr').onclick=async()=>{
    const id=i.id;reruns[id]={pending:true};renderPane();
    try{reruns[id]=await j('/rerun',{method:'POST',body:JSON.stringify({id})})}catch(e){reruns[id]={error:e.message}}
    if(sel===id)renderPane();
  };
}
$('#list').onclick=e=>{const el=e.target.closest('.it');if(!el)return;sel=el.dataset.id;renderList();renderPane()};
$('#run').onchange=e=>loadRun(e.target.value);
for(const s of ['#q','#dom','#ct','#errs'])$(s).oninput=renderList;
init().catch(e=>$('#pane').innerHTML=`<div class="err empty">${esc(e.message)}</div>`);
</script></body></html>
"""

if __name__ == "__main__":
    srv = ThreadingHTTPServer((HOST, PORT), Handler)
    print(f"serving http://{HOST}:{PORT}", flush=True)
    srv.serve_forever()
