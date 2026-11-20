"""Review the Commons candidates from collect_commons.py, one photo at a time, in the browser.

For each photo, confirm or correct the suggested class (one of the 6) and J-code, and say whether it looks like a
studio photo (plain or indoor background, item on its own) or an outdoor one (litter where it was found).
Every decision is saved at once in data/commons_candidates/labels.csv, so a review can stop and resume any time.

Rules: several items of the same material can be accepted as one class (leave the J-code empty if they differ);
several materials -> m. A J-code only when it is clear (J7 is a drink bottle <= 0.5 l, J8 > 0.5 l).

If data/commons_candidates/suggestions.csv exists (Claude's first pass: status, class, J-code, setting, reason),
only the photos it suggests accepting are shown, pre-filled; press a to show every photo. A suggestion is never
a label: only what you confirm is written to labels.csv.

Keys:  Enter      accept as pre-filled (class, J-code, studio or outdoor), go to the next photo
       s / o      accept as studio / outdoor, go to the next photo
       1-6, 0     class: cardboard, glass, metal, paper, plastic, trash; 0 = none of them (ambiguous)
       j          edit the J-code (type a code or words from its name, Enter to confirm)
       x          reject (not a usable photo of litter)
       m          reject: several materials (a litter scene kept apart for the detection phase)
       ← →        previous / next photo        u   jump to the first photo not reviewed yet
       a          show every photo / only the suggested ones

With --multiple, the page shows only the photos Claude suggested as several materials (kept for the detection
test set, #10), and Enter confirms them as such.

usage: python review_commons.py            open the review page on http://127.0.0.1:8765
       python review_commons.py --multiple confirm the photos suggested as several materials
       python review_commons.py --export   write the accepted photos to data/commons_large/ for benchmark.py
"""
import argparse
import hashlib
import json
import shutil
import threading
import webbrowser
from datetime import datetime, timezone
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer
from pathlib import Path

import pandas as pd

from common import CLASSES

DIR = Path("data/commons_candidates")
CANDIDATES = DIR / "candidates.csv"
LABELS = DIR / "labels.csv"
SUGGESTIONS = DIR / "suggestions.csv"
EXPORT = Path("data/commons_large")
LABEL_COLUMNS = ["id", "status", "label", "jcode", "setting", "reviewed_at"]
PORT = 8765
FOCUS = ""  # "multiple" with --multiple: only show the photos Claude suggested with that status


def load_labels():
    if LABELS.exists():
        return pd.read_csv(LABELS, dtype=str).fillna("")
    return pd.DataFrame(columns=LABEL_COLUMNS)


SAVING = threading.Lock()  # the server is threaded: two saves at once would each drop the other's row


def save_label(row):
    row = {k: str(row.get(k, "") or "") for k in LABEL_COLUMNS}
    row["reviewed_at"] = datetime.now(timezone.utc).strftime("%Y-%m-%dT%H:%M:%SZ")
    with SAVING:
        labels = load_labels()
        labels = labels[labels.id != row["id"]]
        labels = pd.concat([labels, pd.DataFrame([row])], ignore_index=True)
        tmp = LABELS.with_suffix(".tmp")
        labels.to_csv(tmp, index=False)
        tmp.replace(LABELS)  # never leave a half-written file


def page_data():
    cands = pd.read_csv(CANDIDATES, dtype=str).fillna("")
    jlist = pd.read_csv("data/jlist.csv", dtype=str).fillna("")
    return dict(classes=CLASSES,
                candidates=cands[["id", "title", "source", "license", "artist", "query_category",
                                  "suggested_label", "suggested_jcode"]].to_dict("records"),
                labels={r["id"]: r for r in load_labels().to_dict("records")},
                suggestions=({r["id"]: r for r in pd.read_csv(SUGGESTIONS, dtype=str).fillna("").to_dict("records")}
                             if SUGGESTIONS.exists() else {}),
                jlist=jlist[["jcode", "name", "material"]].to_dict("records"),
                focus=FOCUS)


class Handler(BaseHTTPRequestHandler):
    def send(self, body, ctype, code=200):
        self.send_response(code)
        self.send_header("Content-Type", ctype)
        self.send_header("Content-Length", str(len(body)))
        self.end_headers()
        self.wfile.write(body)

    def do_GET(self):
        if self.path == "/":
            self.send(PAGE.encode(), "text/html; charset=utf-8")
        elif self.path == "/data.json":
            self.send(json.dumps(page_data()).encode(), "application/json")
        elif self.path.startswith("/img/") and self.path[5:-4].isdigit() and self.path.endswith(".jpg"):
            path = DIR / "img" / self.path[5:]
            if path.exists():
                self.send(path.read_bytes(), "image/jpeg")
            else:
                self.send(b"missing", "text/plain", 404)
        else:
            self.send(b"not found", "text/plain", 404)

    def do_POST(self):
        if self.path != "/label":
            return self.send(b"not found", "text/plain", 404)
        row = json.loads(self.rfile.read(int(self.headers["Content-Length"])))
        if not str(row.get("id", "")).isdigit():
            return self.send(b"bad id", "text/plain", 400)
        save_label(row)
        self.send(b"{}", "application/json")

    def log_message(self, *args):  # keep the terminal quiet
        pass


def export():
    """Accepted photos -> data/commons_large/<class>/<id>.jpg and sources.csv, in the format of the small sets."""
    cands = pd.read_csv(CANDIDATES, dtype=str).fillna("").set_index("id")
    labels = load_labels().query("status == 'accepted'")
    if labels.empty:
        return print(f"No accepted photo in {LABELS} yet: nothing to export.")
    rows = []
    for r in labels.itertuples():
        c = cands.loc[r.id]
        dest = EXPORT / (r.label or "none") / f"commons_{r.id}.jpg"
        dest.parent.mkdir(parents=True, exist_ok=True)
        shutil.copyfile(c.path, dest)
        rows.append(dict(path=str(dest), label=r.label, jcode=r.jcode, setting=r.setting, title=c.title,
                         license=c.license, artist=c.artist, source=c.source, image_url=c.image_url,
                         commons_sha1=c.commons_sha1, sha256=hashlib.sha256(dest.read_bytes()).hexdigest(),
                         processing="original"))
    out = pd.DataFrame(rows)
    out.to_csv(EXPORT / "sources.csv", index=False)
    print(f"{len(out)} photos exported to {EXPORT}/sources.csv")
    print(out.groupby(["setting", "label"]).size().unstack(fill_value=0).to_string())


def main():
    parser = argparse.ArgumentParser()
    mode = parser.add_mutually_exclusive_group()
    mode.add_argument("--export", action="store_true", help="write the accepted photos to data/commons_large/")
    mode.add_argument("--multiple", action="store_true", help="only show the photos suggested as several materials")
    args = parser.parse_args()
    if args.export:
        return export()
    global FOCUS
    FOCUS = "multiple" if args.multiple else ""
    server = ThreadingHTTPServer(("127.0.0.1", PORT), Handler)
    url = f"http://127.0.0.1:{PORT}"
    print(f"Review page on {url} (Ctrl+C to stop; labels are saved as you go)")
    webbrowser.open(url)
    try:
        server.serve_forever()
    except KeyboardInterrupt:
        pass


PAGE = r"""<!doctype html>
<html lang="en"><head><meta charset="utf-8"><meta name="viewport" content="width=device-width, initial-scale=1">
<title>Commons review</title>
<style>
:root { --bg:#f6f6f4; --fg:#1d1d1b; --muted:#6b6b66; --card:#fff; --line:#dddcd6; --accent:#2f6fde;
        --ok:#2e8540; --bad:#c0392b; --warn:#b7791f; }
@media (prefers-color-scheme: dark) { :root { --bg:#161615; --fg:#ecebe6; --muted:#9a9993; --card:#21211f;
        --line:#383835; --accent:#6a9cf5; --ok:#5cb46d; --bad:#e2685b; --warn:#e0a84a; } }
* { box-sizing: border-box; }
body { margin:0; background:var(--bg); color:var(--fg); font:15px/1.45 system-ui, sans-serif; }
header { display:flex; gap:16px; align-items:center; padding:10px 16px; border-bottom:1px solid var(--line); }
header b { font-size:16px; }
.bar { flex:1; height:8px; background:var(--line); border-radius:4px; overflow:hidden; }
.bar div { height:100%; background:var(--accent); }
main { display:grid; grid-template-columns:minmax(0,1fr) 360px; gap:16px; padding:16px; height:calc(100vh - 50px); }
.photo { background:var(--card); border:1px solid var(--line); border-radius:8px; display:flex;
         align-items:center; justify-content:center; overflow:hidden; }
.photo img { max-width:100%; max-height:100%; object-fit:contain; }
aside { display:flex; flex-direction:column; gap:12px; overflow:auto; }
.card { background:var(--card); border:1px solid var(--line); border-radius:8px; padding:12px; }
.muted { color:var(--muted); font-size:13px; }
.classes { display:grid; grid-template-columns:repeat(2,1fr); gap:6px; }
button { font:inherit; color:inherit; background:var(--bg); border:1px solid var(--line); border-radius:6px;
         padding:6px 8px; cursor:pointer; text-align:left; }
button.on { border-color:var(--accent); outline:2px solid var(--accent); }
kbd { font:12px ui-monospace, monospace; border:1px solid var(--line); border-radius:4px; padding:0 4px; }
input { width:100%; font:inherit; color:inherit; background:var(--bg); border:1px solid var(--line);
        border-radius:6px; padding:6px 8px; }
.status { font-weight:600; } .accepted { color:var(--ok); } .rejected, .multiple { color:var(--bad); }
.row { display:flex; gap:6px; } .row button { flex:1; text-align:center; }
a { color:var(--accent); }
@media (max-width: 800px) { main { grid-template-columns:1fr; height:auto; } .photo { height:60vh; } }
</style></head><body>
<header><b>Commons review</b><span id="count"></span><div class="bar"><div id="progress"></div></div>
<span class="muted" id="tally"></span></header>
<main>
  <div class="photo"><img id="img" alt=""></div>
  <aside>
    <div class="card"><div id="title"></div><div class="muted" id="meta"></div>
      <div class="muted" id="state"></div></div>
    <div class="card"><div class="muted">Class <kbd>1</kbd>–<kbd>6</kbd>, <kbd>0</kbd> none</div>
      <div class="classes" id="classes"></div></div>
    <div class="card"><div class="muted">J-code <kbd>j</kbd></div>
      <input id="jcode" list="jlist" placeholder="e.g. J175 or 'drinks cans'" autocomplete="off">
      <datalist id="jlist"></datalist><div id="jname" class="muted"></div></div>
    <div class="card"><div class="row">
      <button id="studio"><kbd>s</kbd> accept: studio</button>
      <button id="outdoor"><kbd>o</kbd> accept: outdoor</button></div>
      <div class="row" style="margin-top:6px">
      <button id="reject"><kbd>x</kbd> reject</button>
      <button id="multiple"><kbd>m</kbd> several materials</button></div>
      <div class="row" style="margin-top:6px"><button id="asis"><kbd>Enter</kbd> accept as pre-filled</button></div></div>
    <div class="muted"><kbd>←</kbd> <kbd>→</kbd> previous / next, <kbd>u</kbd> first not reviewed,
      <kbd>a</kbd> <span id="mode"></span>.
      Studio: item on its own, plain or indoor background. Outdoor: litter where it was found.
      Several items of one material: accept them as one class.</div>
  </aside>
</main>
<script>
let D, i = 0, cur = {};
const $ = id => document.getElementById(id);
const names = {};
async function load() {
  D = await (await fetch("/data.json")).json();
  D.jlist.forEach(j => { names[j.jcode] = j.name;
    const o = document.createElement("option"); o.value = j.jcode; o.label = j.name; $("jlist").append(o); });
  D.classes.concat(["(none)"]).forEach((c, k) => { const b = document.createElement("button");
    b.dataset.label = k < 6 ? c : ""; b.innerHTML = `<kbd>${k < 6 ? k + 1 : 0}</kbd> ${c}`;
    b.onclick = () => { cur.label = b.dataset.label; draw(); }; $("classes").append(b); });
  showAll = !Object.keys(D.suggestions).length;
  i = firstTodo(); show();
}
// with suggestions, only show what Claude would accept (and what you already reviewed), unless showAll;
// with --multiple (D.focus), only the photos Claude suggested as several materials
let showAll = false;
const visible = c => showAll || (D.focus ? (D.suggestions[c.id] || {}).status === D.focus
  : !!D.labels[c.id] || (D.suggestions[c.id] || {}).status === "accepted");
const waiting = () => D.candidates.filter(c => !D.labels[c.id] && visible(c)).length;
function step(from, dir) {
  for (let k = from + dir; k >= 0 && k < D.candidates.length; k += dir) if (visible(D.candidates[k])) return k;
  return -1;
}
const firstTodo = () => { const k = D.candidates.findIndex(c => !D.labels[c.id] && visible(c)); return k < 0 ? i : k; };
function show() {
  const c = D.candidates[i], l = D.labels[c.id], s = D.suggestions[c.id];
  cur = l ? { label: l.label, jcode: l.jcode, setting: l.setting }
    : s && s.status === "accepted" ? { label: s.label, jcode: s.jcode, setting: s.setting }
    : { label: c.suggested_label, jcode: c.suggested_jcode.split(" ")[0] || "", setting: "" };
  // a query suggestion with several codes (e.g. "J7 J8") keeps them all as a hint, the first is pre-filled
  $("img").src = `/img/${c.id}.jpg`;
  $("title").innerHTML = `<a href="${c.source}" target="_blank" rel="noopener">${esc(c.title)}</a>`;
  $("meta").textContent = `${c.license} · ${c.artist.slice(0, 80)} · from “${c.query_category}”` +
    (c.suggested_jcode.includes(" ") ? ` · suggested ${c.suggested_jcode}` : "");
  $("state").innerHTML = (l ? `<span class="status ${l.status}">${l.status}</span>${l.setting ? " · " + l.setting : ""}` : "not reviewed") +
    (s ? `<br>Claude: ${esc([s.status, s.label, s.jcode, s.setting].filter(Boolean).join(" · "))} (${esc(s.confidence)}): ${esc(s.reason)}` : "");
  $("mode").textContent = showAll ? "only Claude's suggested photos" : "every photo";
  $("asis").innerHTML = "<kbd>Enter</kbd> " + (prefilled() === "multiple" ? "confirm: several materials" : "accept as pre-filled");
  $("jcode").value = cur.jcode; draw(); tally();
  const next = D.candidates[step(i, 1)]; if (next) new Image().src = `/img/${next.id}.jpg`;
}
function draw() {
  document.querySelectorAll("#classes button").forEach(b => b.classList.toggle("on", b.dataset.label === (cur.label || "")));
  $("jname").textContent = names[cur.jcode] || (cur.jcode ? "unknown code: pick one from the list" : "no J-code");
}
function tally() {
  const ls = Object.values(D.labels), n = D.candidates.length;
  const acc = ls.filter(l => l.status === "accepted");
  $("count").textContent = `${i + 1} / ${n}`;
  $("progress").style.width = `${100 * ls.length / n}%`;
  $("tally").textContent = `${ls.length} reviewed · ${acc.length} accepted (${acc.filter(l => l.setting === "studio").length} studio, ${acc.filter(l => l.setting === "outdoor").length} outdoor)` +
    (showAll ? "" : ` · ${waiting()} suggested to go`);
}
let busy = false;  // a second key press while saving would skip a photo
async function decide(status, setting = "") {
  if (busy) return;
  busy = true;
  try {
    const row = { id: D.candidates[i].id, status, setting, label: status === "accepted" ? cur.label : "", jcode: status === "accepted" ? cur.jcode : "" };
    const r = await fetch("/label", { method: "POST", body: JSON.stringify(row) });
    if (!r.ok) { alert("Not saved: " + await r.text()); return; }
    D.labels[row.id] = row;
    const k = step(i, 1);
    if (k < 0) alert("No more photos to review for now (reload later: Claude may have suggested more).");
    else i = k;
    show();
  } catch (err) { alert("Not saved (is review_commons.py still running?): " + err); }
  finally { busy = false; }
}
// a code ("j175"), or words from a name ("drinks cans"): a single match becomes its code
function findJcode(text) {
  const t = text.trim().toUpperCase();
  if (!t || names[t]) return t;
  const words = text.toLowerCase().split(/\s+/).filter(Boolean);
  const hits = D.jlist.filter(j => words.every(w => j.name.toLowerCase().includes(w)));
  return hits.length === 1 ? hits[0].jcode : t;
}
const esc = s => s.replace(/[&<>"]/g, ch => ({ "&": "&amp;", "<": "&lt;", ">": "&gt;", '"': "&quot;" }[ch]));
$("studio").onclick = () => decide("accepted", "studio");
$("outdoor").onclick = () => decide("accepted", "outdoor");
$("reject").onclick = () => decide("rejected");
$("multiple").onclick = () => decide("multiple");
// Enter keeps the pre-filled decision: several materials if that is what was suggested or saved, else accept
const prefilled = () => { const c = D.candidates[i], l = D.labels[c.id]; return (l || D.suggestions[c.id] || {}).status; };
const asis = () => prefilled() === "multiple" ? decide("multiple") : decide("accepted", cur.setting || "studio");
$("asis").onclick = asis;
$("jcode").addEventListener("change", () => { cur.jcode = findJcode($("jcode").value); $("jcode").value = cur.jcode; draw(); });
$("jcode").addEventListener("keydown", e => { if (e.key === "Enter" || e.key === "Escape") { $("jcode").dispatchEvent(new Event("change")); $("jcode").blur(); } e.stopPropagation(); });
document.addEventListener("keydown", e => {
  if (e.metaKey || e.ctrlKey || e.altKey) return;
  const k = e.key;
  if (k >= "1" && k <= "6") { cur.label = D.classes[+k - 1]; draw(); }
  else if (k === "0") { cur.label = ""; draw(); }
  else if (k === "s") decide("accepted", "studio");
  else if (k === "o") decide("accepted", "outdoor");
  else if (k === "x") decide("rejected");
  else if (k === "m") decide("multiple");
  else if (k === "Enter") asis();
  else if (k === "j") { e.preventDefault(); $("jcode").select(); }
  else if (k === "ArrowLeft" && step(i, -1) >= 0) { i = step(i, -1); show(); }
  else if (k === "ArrowRight" && step(i, 1) >= 0) { i = step(i, 1); show(); }
  else if (k === "u") { i = firstTodo(); show(); }
  else if (k === "a") { showAll = !showAll; if (!visible(D.candidates[i])) i = firstTodo(); show(); }
});
load();
</script></body></html>
"""

if __name__ == "__main__":
    main()
