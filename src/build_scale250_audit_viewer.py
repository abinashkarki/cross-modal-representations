#!/usr/bin/env python3
from __future__ import annotations

import argparse
import json
import os
from datetime import datetime, timezone
from pathlib import Path
from typing import Any


REPO_ROOT = Path(__file__).resolve().parent.parent
DEFAULT_MANIFEST_PATH = REPO_ROOT / "data" / "data_manifest_250.json"
DEFAULT_OUTPUT_PATH = REPO_ROOT / "results" / "scale250_full" / "audit" / "index.html"
SOURCE_ORDER = ("imagenet", "openimages", "unsplash")


HTML_TEMPLATE = """<!doctype html>
<html lang="en">
<head>
  <meta charset="utf-8">
  <meta name="viewport" content="width=device-width, initial-scale=1">
  <title>Scale250 Audit Viewer</title>
  <style>
    :root{--bg:#f5efe3;--panel:#fffaf2;--ink:#202722;--muted:#5f675d;--line:#d9d1c3;--accent:#125949;--accent-soft:#e2f0eb;--pass:#2f6a2a;--review:#8a620c;--fail:#8f3030}
    *{box-sizing:border-box} body{margin:0;font-family:"Avenir Next","Segoe UI",sans-serif;color:var(--ink);background:linear-gradient(180deg,#f8f2e8,#ede3d2)}
    .shell{width:min(1440px,calc(100vw - 24px));margin:12px auto 28px}
    .panel{background:rgba(255,250,242,.92);border:1px solid var(--line);border-radius:20px;box-shadow:0 16px 36px rgba(34,35,20,.08)}
    .hero,.toolbar,.main,.side{padding:16px}
    .hero{margin-bottom:12px}.eyebrow{margin:0 0 6px;font-size:12px;letter-spacing:.18em;text-transform:uppercase;color:var(--accent);font-weight:800}
    h1{margin:0;font-family:"Iowan Old Style","Palatino Linotype",serif;font-size:clamp(30px,4vw,48px);line-height:1}
    .sub{margin:10px 0 0;color:var(--muted);max-width:980px}
    .chips,.meta,.btn-row,.mode-row,.status-row{display:flex;flex-wrap:wrap;gap:10px}
    .chips{margin-top:14px}.chip{display:inline-flex;gap:8px;align-items:center;padding:8px 12px;border-radius:999px;background:#fff;border:1px solid var(--line);font-size:14px}
    .toolbar{display:grid;grid-template-columns:repeat(12,minmax(0,1fr));gap:12px;margin-bottom:12px;position:sticky;top:8px;z-index:10}
    .control{display:flex;flex-direction:column;gap:6px}.control label{font-size:12px;font-weight:800;letter-spacing:.06em;text-transform:uppercase;color:var(--muted)}
    .span-4{grid-column:span 4}.span-3{grid-column:span 3}.span-2{grid-column:span 2}.span-12{grid-column:span 12}
    input,select,textarea,button{font:inherit} input,select,textarea{width:100%;padding:12px 13px;border-radius:14px;border:1px solid var(--line);background:#fff}
    textarea{min-height:110px;resize:vertical}
    button{padding:11px 14px;border-radius:14px;border:1px solid rgba(18,89,73,.18);background:linear-gradient(180deg,#fff,#f7f0e5);cursor:pointer}
    button.active{background:linear-gradient(180deg,#125949,#0d4538);color:#fff;border-color:#125949}
    .pass.active{background:linear-gradient(180deg,#39772f,#28541f);border-color:#28541f}
    .review.active{background:linear-gradient(180deg,#9a6f12,#724f08);border-color:#724f08}
    .fail.active{background:linear-gradient(180deg,#a33a3a,#7d2828);border-color:#7d2828}
    .layout{display:grid;grid-template-columns:minmax(0,2fr) minmax(320px,.9fr);gap:12px}
    .main h2{margin:0;font-family:"Iowan Old Style","Palatino Linotype",serif;font-size:clamp(30px,3vw,42px);line-height:1.02}
    .subtitle{margin:8px 0 12px;color:var(--muted)}
    .section{padding:14px;border:1px solid var(--line);border-radius:18px;background:#fffdf8;margin-top:12px}
    .section-head{display:flex;justify-content:space-between;gap:12px;align-items:baseline;margin-bottom:10px}
    .section-head h3{margin:0;font-size:18px;text-transform:capitalize}.mini{font-size:13px;color:var(--muted)}
    .grid{display:grid;grid-template-columns:repeat(auto-fit,minmax(220px,1fr));gap:12px}
    .card{border:1px solid #e4dccd;border-radius:18px;background:#fff;overflow:hidden;box-shadow:0 10px 22px rgba(20,20,15,.06)}
    .card a{display:block;aspect-ratio:1/1;background:#efe7dc}.card img{width:100%;height:100%;object-fit:cover;display:block}
    .card-body{padding:10px 12px 12px;display:grid;gap:6px}.filename{font-size:14px;font-weight:800;word-break:break-word}
    .small{display:flex;justify-content:space-between;gap:8px;font-size:12px;color:var(--muted)}
    .badge{display:inline-flex;padding:4px 8px;border-radius:999px;background:var(--accent-soft);border:1px solid #cfe5dd;color:var(--accent);font-size:12px;font-weight:800;text-transform:capitalize}
    .side{position:sticky;top:96px}.stats{display:grid;grid-template-columns:repeat(2,minmax(0,1fr));gap:10px;margin-bottom:12px}
    .stat{padding:12px;border-radius:16px;background:#fff;border:1px solid var(--line)} .stat strong{display:block;font-size:22px}
    .hint{margin-top:6px;font-size:12px;color:var(--muted)} .empty{padding:14px;border:1px dashed #cfb7b7;border-radius:16px;background:#fff7f7;color:var(--fail)}
    @media (max-width:1100px){.layout{grid-template-columns:1fr}.side{position:static}}
    @media (max-width:860px){.toolbar{grid-template-columns:repeat(6,minmax(0,1fr))}.span-4,.span-3,.span-2,.span-12{grid-column:span 6}}
  </style>
</head>
<body>
  <div class="shell">
    <section class="hero panel">
      <p class="eyebrow">Scale250 Audit</p>
      <h1>Static local viewer for random concept and source-cell checks.</h1>
      <p class="sub">Open this file directly in a browser. It is manifest-driven, randomizable, and saves notes in your local browser storage on this machine.</p>
      <div class="chips" id="hero-chips"></div>
    </section>

    <section class="toolbar panel">
      <div class="control span-4">
        <label for="concept-search">Jump To Concept</label>
        <input id="concept-search" list="concept-list" placeholder="Type a concept and press Enter">
        <datalist id="concept-list"></datalist>
      </div>
      <div class="control span-3">
        <label for="stratum-filter">Stratum</label>
        <select id="stratum-filter"></select>
      </div>
      <div class="control span-3">
        <label for="source-filter">Source Cell</label>
        <select id="source-filter"></select>
      </div>
      <div class="control span-2">
        <label>View</label>
        <div class="mode-row">
          <button id="mode-concept" class="active">All 15</button>
          <button id="mode-source">Single Source</button>
        </div>
      </div>
      <div class="control span-12">
        <label>Sampling</label>
        <div class="btn-row">
          <button id="random-concept">Random Concept</button>
          <button id="random-stratum">Random In Stratum</button>
          <button id="random-source">Random Source Cell</button>
          <button id="next-unreviewed">Next Unreviewed</button>
          <button id="export-json">Export Notes JSON</button>
          <button id="export-csv">Export Notes CSV</button>
        </div>
      </div>
    </section>

    <div class="layout">
      <section class="main panel">
        <h2 id="title">Loading...</h2>
        <p class="subtitle" id="subtitle"></p>
        <div class="meta" id="meta"></div>
        <div id="grid-wrap"></div>
      </section>
      <aside class="side panel">
        <div class="stats">
          <div class="stat"><strong id="concept-reviewed">0</strong><span>Concepts reviewed</span></div>
          <div class="stat"><strong id="cell-reviewed">0</strong><span>Cells reviewed</span></div>
        </div>
        <div class="status-row">
          <button id="status-pass" class="pass">Pass</button>
          <button id="status-review" class="review">Review</button>
          <button id="status-fail" class="fail">Fail</button>
          <button id="status-clear">Clear</button>
        </div>
        <div class="control" style="margin-top:12px">
          <label for="notes">Notes For Current View</label>
          <textarea id="notes" placeholder="Write quick notes about image quality, semantic drift, duplicates, or anything suspicious."></textarea>
          <div class="hint" id="save-hint">Saved locally in your browser.</div>
        </div>
      </aside>
    </div>
  </div>
  <script id="audit-data" type="application/json">__DATA__</script>
  <script>
    const payload = JSON.parse(document.getElementById("audit-data").textContent);
    const concepts = payload.concepts;
    const conceptMap = new Map(concepts.map(c => [c.concept, c]));
    const sources = payload.sources;
    const strata = payload.strata;
    const storageKey = "scale250-audit-v1";
    const state = { concept: concepts[0].concept, mode: "concept", source: "imagenet", stratum: "all" };
    const store = loadStore();

    const el = {
      heroChips: document.getElementById("hero-chips"),
      conceptList: document.getElementById("concept-list"),
      conceptSearch: document.getElementById("concept-search"),
      stratum: document.getElementById("stratum-filter"),
      source: document.getElementById("source-filter"),
      title: document.getElementById("title"),
      subtitle: document.getElementById("subtitle"),
      meta: document.getElementById("meta"),
      grid: document.getElementById("grid-wrap"),
      notes: document.getElementById("notes"),
      hint: document.getElementById("save-hint"),
      conceptReviewed: document.getElementById("concept-reviewed"),
      cellReviewed: document.getElementById("cell-reviewed"),
      modeConcept: document.getElementById("mode-concept"),
      modeSource: document.getElementById("mode-source"),
      pass: document.getElementById("status-pass"),
      review: document.getElementById("status-review"),
      fail: document.getElementById("status-fail"),
    };

    init();

    function init() {
      el.conceptList.innerHTML = concepts.map(c => `<option value="${esc(c.concept)}"></option>`).join("");
      el.stratum.innerHTML = [`<option value="all">All strata</option>`, ...strata.map(s => `<option value="${esc(s)}">${esc(titleCase(s))}</option>`)].join("");
      el.source.innerHTML = [`<option value="all">Any source</option>`, ...sources.map(s => `<option value="${s}">${titleCase(s)}</option>`)].join("");
      bind();
      render();
    }

    function bind() {
      el.conceptSearch.addEventListener("keydown", (e) => {
        if (e.key !== "Enter") return;
        const concept = el.conceptSearch.value.trim();
        if (conceptMap.has(concept)) { state.concept = concept; render(); }
      });
      el.stratum.addEventListener("change", () => { state.stratum = el.stratum.value; render(); });
      el.source.addEventListener("change", () => { if (el.source.value !== "all") state.source = el.source.value; if (state.mode === "source") render(); });
      el.modeConcept.addEventListener("click", () => { state.mode = "concept"; render(); });
      el.modeSource.addEventListener("click", () => { state.mode = "source"; if (el.source.value === "all") el.source.value = state.source; render(); });
      document.getElementById("random-concept").addEventListener("click", () => chooseRandom(false));
      document.getElementById("random-stratum").addEventListener("click", () => chooseRandom(true));
      document.getElementById("random-source").addEventListener("click", chooseRandomSourceCell);
      document.getElementById("next-unreviewed").addEventListener("click", nextUnreviewed);
      document.getElementById("export-json").addEventListener("click", () => download("scale250_audit_notes.json", JSON.stringify(store, null, 2), "application/json"));
      document.getElementById("export-csv").addEventListener("click", exportCsv);
      el.pass.addEventListener("click", () => setStatus("pass"));
      el.review.addEventListener("click", () => setStatus("review"));
      el.fail.addEventListener("click", () => setStatus("fail"));
      document.getElementById("status-clear").addEventListener("click", clearEntry);
      el.notes.addEventListener("input", saveNotes);
    }

    function render() {
      const concept = conceptMap.get(state.concept);
      el.modeConcept.classList.toggle("active", state.mode === "concept");
      el.modeSource.classList.toggle("active", state.mode === "source");
      el.conceptSearch.value = concept.concept;
      el.stratum.value = state.stratum;
      el.source.value = state.mode === "source" ? state.source : "all";
      el.title.textContent = concept.concept;
      el.subtitle.textContent = state.mode === "concept" ? "Inspect all 15 images grouped by source." : `Inspect the 5-image ${titleCase(state.source)} cell.`;
      el.meta.innerHTML = [
        chip("Stratum", titleCase(concept.stratum)),
        chip("Selection", concept.selection_status.replaceAll("_", " ")),
        chip("Feasibility", concept.source_feasibility),
        chip("Balance", `${concept.source_mix_actual.imagenet}/${concept.source_mix_actual.openimages}/${concept.source_mix_actual.unsplash}`),
      ].join("");
      el.grid.innerHTML = renderSections(concept);
      renderAudit();
      renderHero();
    }

    function renderSections(concept) {
      const sectionSources = state.mode === "concept" ? sources : [state.source];
      return sectionSources.map(source => {
        const items = concept.images.filter(image => image.source === source);
        const meanClip = items.length ? (items.reduce((sum, x) => sum + x.clip_score, 0) / items.length).toFixed(3) : "-";
        return `<section class="section">
          <div class="section-head"><h3>${esc(titleCase(source))}</h3><span class="mini">${items.length} images, mean CLIP ${meanClip}</span></div>
          ${items.length ? `<div class="grid">${items.map(renderCard).join("")}</div>` : `<div class="empty">No images in this source cell.</div>`}
        </section>`;
      }).join("");
    }

    function renderCard(image) {
      return `<article class="card">
        <a href="${esc(image.src)}" target="_blank" rel="noreferrer"><img src="${esc(image.src)}" alt="${esc(image.filename)}" loading="lazy"></a>
        <div class="card-body">
          <span class="badge">${esc(image.source)}</span>
          <div class="filename">${esc(image.filename)}</div>
          <div class="small"><span>CLIP ${Number(image.clip_score).toFixed(4)}</span><span>${esc(image.storage_path)}</span></div>
        </div>
      </article>`;
    }

    function renderHero() {
      const reviewed = reviewCounts();
      const chips = [
        chip("Concepts", payload.totals.concepts),
        chip("Images", payload.totals.images),
        chip("Balance", "5 / 5 / 5"),
        chip("Reviewed concepts", reviewed.concepts),
        chip("Reviewed cells", reviewed.cells),
      ];
      el.heroChips.innerHTML = chips.join("");
      el.conceptReviewed.textContent = reviewed.concepts;
      el.cellReviewed.textContent = reviewed.cells;
    }

    function renderAudit() {
      const entry = store.entries[currentKey()] || {};
      el.pass.classList.toggle("active", entry.status === "pass");
      el.review.classList.toggle("active", entry.status === "review");
      el.fail.classList.toggle("active", entry.status === "fail");
      el.notes.value = entry.notes || "";
      el.hint.textContent = entry.updatedAt ? `Last saved ${new Date(entry.updatedAt).toLocaleString()}` : "Saved locally in your browser.";
    }

    function chooseRandom(restrictStratum) {
      const eligible = concepts.filter(c => !restrictStratum || state.stratum === "all" || c.stratum === state.stratum);
      if (!eligible.length) return;
      state.concept = sample(eligible).concept;
      state.mode = "concept";
      render();
    }

    function chooseRandomSourceCell() {
      const eligible = concepts.filter(c => state.stratum === "all" || c.stratum === state.stratum);
      if (!eligible.length) return;
      state.concept = sample(eligible).concept;
      state.source = el.source.value !== "all" ? el.source.value : sample(sources);
      state.mode = "source";
      render();
    }

    function nextUnreviewed() {
      const eligible = concepts.filter(c => (state.stratum === "all" || c.stratum === state.stratum) && !store.entries[`concept::${c.concept}`]);
      if (!eligible.length) { el.hint.textContent = "No unreviewed concepts remain for the current stratum filter."; return; }
      state.concept = sample(eligible).concept;
      state.mode = "concept";
      render();
    }

    function saveNotes() {
      const key = currentKey();
      const entry = store.entries[key] || {};
      entry.scope = state.mode;
      entry.concept = state.concept;
      entry.source = state.mode === "source" ? state.source : "";
      entry.notes = el.notes.value;
      entry.status = entry.status || "";
      entry.updatedAt = new Date().toISOString();
      store.entries[key] = entry;
      persist();
      renderAudit();
      renderHero();
    }

    function setStatus(status) {
      const key = currentKey();
      const entry = store.entries[key] || {};
      entry.scope = state.mode;
      entry.concept = state.concept;
      entry.source = state.mode === "source" ? state.source : "";
      entry.notes = el.notes.value;
      entry.status = status;
      entry.updatedAt = new Date().toISOString();
      store.entries[key] = entry;
      persist();
      renderAudit();
      renderHero();
    }

    function clearEntry() {
      delete store.entries[currentKey()];
      persist();
      renderAudit();
      renderHero();
    }

    function reviewCounts() {
      let conceptsReviewed = 0, cellsReviewed = 0;
      for (const key of Object.keys(store.entries)) {
        if (key.startsWith("concept::")) conceptsReviewed += 1;
        else if (key.startsWith("cell::")) cellsReviewed += 1;
      }
      return { concepts: conceptsReviewed, cells: cellsReviewed };
    }

    function currentKey() {
      return state.mode === "concept" ? `concept::${state.concept}` : `cell::${state.concept}::${state.source}`;
    }

    function loadStore() {
      try {
        const raw = localStorage.getItem(storageKey);
        const parsed = raw ? JSON.parse(raw) : { entries: {} };
        if (!parsed.entries || typeof parsed.entries !== "object") return { entries: {} };
        return parsed;
      } catch (_err) {
        return { entries: {} };
      }
    }

    function persist() { localStorage.setItem(storageKey, JSON.stringify(store)); }
    function sample(items) { return items[Math.floor(Math.random() * items.length)]; }
    function titleCase(value) { return value.replaceAll("_", " ").split(" ").map(part => part ? part[0].toUpperCase() + part.slice(1) : part).join(" "); }
    function chip(label, value) { return `<span class="chip"><strong>${esc(label)}</strong><span>${esc(value)}</span></span>`; }
    function esc(value) { return String(value).replaceAll("&","&amp;").replaceAll("<","&lt;").replaceAll(">","&gt;").replaceAll('"',"&quot;"); }
    function download(name, content, mime) { const blob = new Blob([content], {type: mime}); const url = URL.createObjectURL(blob); const a = document.createElement("a"); a.href = url; a.download = name; document.body.appendChild(a); a.click(); a.remove(); URL.revokeObjectURL(url); }
    function csvEscape(value) { const text = String(value ?? ""); return /[",\\n]/.test(text) ? `"${text.replaceAll('"','""')}"` : text; }
    function exportCsv() {
      const rows = [["key","scope","concept","source","status","notes","updatedAt"]];
      for (const [key, entry] of Object.entries(store.entries)) rows.push([key, entry.scope || "", entry.concept || "", entry.source || "", entry.status || "", entry.notes || "", entry.updatedAt || ""]);
      download("scale250_audit_notes.csv", rows.map(row => row.map(csvEscape).join(",")).join("\\n"), "text/csv");
    }
  </script>
</body>
</html>
"""


def build_payload(manifest_path: Path, output_path: Path) -> dict[str, Any]:
    manifest = json.loads(manifest_path.read_text())
    output_dir = output_path.parent
    concepts = []
    for concept_name, image_paths in manifest["concept_to_images"].items():
        metadata = manifest["concept_metadata"][concept_name]
        images = []
        for image_path in image_paths:
            image_file = Path(image_path)
            filename = image_file.name
            absolute_image_path = REPO_ROOT / image_file
            relative_src = os.path.relpath(absolute_image_path, output_dir).replace(os.sep, "/")
            images.append(
                {
                    "src": relative_src,
                    "storage_path": image_path,
                    "filename": filename,
                    "source": metadata["image_sources"][filename],
                    "clip_score": metadata["clip_scores"][filename],
                }
            )
        images.sort(key=lambda item: (SOURCE_ORDER.index(item["source"]), item["filename"]))
        concepts.append(
            {
                "concept": concept_name,
                "stratum": metadata["stratum"],
                "selection_status": metadata["selection_status"],
                "source_feasibility": metadata["source_feasibility"],
                "source_mix_actual": metadata["source_mix_actual"],
                "images": images,
            }
        )

    return {
        "generated_at": datetime.now(timezone.utc).isoformat(),
        "sources": list(SOURCE_ORDER),
        "strata": sorted({concept["stratum"] for concept in concepts}),
        "totals": {"concepts": len(concepts), "images": sum(len(concept["images"]) for concept in concepts)},
        "concepts": concepts,
    }


def main() -> None:
    parser = argparse.ArgumentParser(description="Build a static HTML viewer for auditing the 250-concept dataset.")
    parser.add_argument("--manifest-path", default=str(DEFAULT_MANIFEST_PATH))
    parser.add_argument("--output-path", default=str(DEFAULT_OUTPUT_PATH))
    args = parser.parse_args()

    manifest_path = Path(args.manifest_path).resolve()
    output_path = Path(args.output_path).resolve()
    output_path.parent.mkdir(parents=True, exist_ok=True)

    payload = build_payload(manifest_path, output_path)
    data_json = json.dumps(payload, separators=(",", ":"), ensure_ascii=True).replace("</", "<\\/")
    output_path.write_text(HTML_TEMPLATE.replace("__DATA__", data_json), encoding="utf-8")

    print(f"Manifest: {manifest_path}")
    print(f"Output: {output_path}")
    print(f"Concepts: {payload['totals']['concepts']}")
    print(f"Images: {payload['totals']['images']}")


if __name__ == "__main__":
    main()
