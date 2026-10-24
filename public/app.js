/* mcdakit — interface web. Tous les calculs sont faits par l'API Python (/api/*). */
"use strict";

// ============================================================================ utilitaires
const $ = (sel, root = document) => root.querySelector(sel);
const $$ = (sel, root = document) => [...root.querySelectorAll(sel)];
const el = (tag, attrs = {}, ...children) => {
  const e = document.createElement(tag);
  for (const [k, v] of Object.entries(attrs)) {
    if (k === "class") e.className = v;
    else if (k === "html") e.innerHTML = v;
    else if (k.startsWith("on")) e.addEventListener(k.slice(2), v);
    else if (v !== undefined && v !== null && v !== false) e.setAttribute(k, v === true ? "" : v);
  }
  for (const c of children.flat()) if (c !== null && c !== undefined) e.append(c.nodeType ? c : document.createTextNode(c));
  return e;
};
const esc = (s) => String(s ?? "").replace(/[&<>"]/g, (c) => ({ "&": "&amp;", "<": "&lt;", ">": "&gt;", '"': "&quot;" }[c]));
const num = (v) => (v === "" || v === null || v === undefined || Number.isNaN(Number(v)) ? null : Number(v));
const fmt = (v, d = 4) => {
  if (v === null || v === undefined) return "—";
  if (typeof v === "boolean") return v ? "oui" : "non";
  if (typeof v !== "number") return String(v);
  if (Number.isInteger(v)) return v.toLocaleString("fr-FR");
  return Math.abs(v) >= 1000 ? v.toLocaleString("fr-FR", { maximumFractionDigits: 0 })
    : v.toLocaleString("fr-FR", { maximumFractionDigits: d });
};
const pct = (v, d = 1) => (v === null || v === undefined ? "—" : (v * 100).toLocaleString("fr-FR", { maximumFractionDigits: d }) + " %");
const debounce = (fn, ms = 350) => { let t; return (...a) => { clearTimeout(t); t = setTimeout(() => fn(...a), ms); }; };
const store = {
  get(k) { try { return JSON.parse(localStorage.getItem(k)); } catch { return null; } },
  set(k, v) { try { localStorage.setItem(k, JSON.stringify(v)); } catch { /* stockage indisponible */ } },
};

function showStatus(msg, kind = "error", ms = 6000) {
  const s = $("#status");
  s.textContent = msg; s.className = "status" + (kind === "info" ? " info" : ""); s.hidden = false;
  clearTimeout(showStatus.t); showStatus.t = setTimeout(() => (s.hidden = true), ms);
}

async function api(path, body, { raw = false } = {}) {
  const opts = body === undefined ? {} : body instanceof FormData ? { method: "POST", body }
    : { method: "POST", headers: { "Content-Type": "application/json" }, body: JSON.stringify(body) };
  const r = await fetch("/api/" + path, opts);
  if (!r.ok) {
    let msg = r.statusText;
    try { const j = await r.json(); msg = typeof j.detail === "string" ? j.detail : JSON.stringify(j.detail); } catch { /* */ }
    throw new Error(msg);
  }
  return raw ? r : r.json();
}

function download(blob, name) {
  const a = el("a", { href: URL.createObjectURL(blob), download: name });
  document.body.append(a); a.click(); a.remove();
  setTimeout(() => URL.revokeObjectURL(a.href), 2000);
}

const SAATY = [9, 8, 7, 6, 5, 4, 3, 2, 1, -2, -3, -4, -5, -6, -7, -8, -9];   // gauche = 1er critère plus important
const parseNum = (x) => { if (typeof x === "number") return x; const s = String(x).replace(",", "."); if (s.includes("/")) { const [a, b] = s.split("/"); return Number(a) / Number(b); } return Number(s); };
const toSaaty = (a) => { const v = a >= 1 ? Math.round(a) : -Math.round(1 / a); return SAATY.reduce((b, o) => (Math.abs(o - v) < Math.abs(b - v) ? o : b), 1); };

// ============================================================================ état
let META = null;
let S = null;          // modèle éditable
let LAST = { validate: null, analyze: null, sens: null, delphi: null };

function blankModel() {
  return {
    name: "Nouveau projet", description: "",
    criteria: [
      { id: "C1", name: "Coût", direction: "min", unit: "€", aggregation: "sum", pref: { type: "usual" } },
      { id: "C2", name: "Qualité", direction: "max", unit: "/5", aggregation: "mean", pref: { type: "usual" } },
    ],
    attributes: [],
    alternatives: [
      { id: "A1", name: "Option 1", values: { C1: 100, C2: 3 } },
      { id: "A2", name: "Option 2", values: { C1: 80, C2: 2 } },
      { id: "A3", name: "Option 3", values: { C1: 120, C2: 5 } },
    ],
    weights: { method: "equal", direct: {}, rank: {}, ahp: {} },
    portfolio: { on: false, min: 2, max: 2 },
    constraints: [], overrides: [],
    ui: defaultUI(),
  };
}

function defaultUI() {
  return {
    methods: { wsm: { on: true, normalization: "minmax" }, wpm: { on: false, normalization: "max" },
      topsis: { on: true, normalization: "vector" }, promethee: { on: true } },
    primary: "topsis", sweep: null, mc: { n: 10000, spread: 0.2, uniform: false },
    profiles: [], equalProfile: true, scenarios: [], delphi: [{ alternative: "", column: "", r1: "", r2: "", r3: "" }], iqr: 1,
  };
}

/** dict API (format FORMAT.md) -> modèle éditable */
function fromProblem(d) {
  const m = blankModel();
  m.name = d.name || "Projet"; m.description = d.description || "";
  m.criteria = (d.criteria || []).map((c) => ({ id: c.id, name: c.name || c.id, direction: c.direction || "max", unit: c.unit || "",
    aggregation: c.aggregation || "mean", pref: { type: "usual", ...(c.preference || {}) } }));
  m.attributes = (d.attributes || []).map((a) => ({ id: a.id, name: a.name || a.id, unit: a.unit || "", aggregation: a.aggregation || "sum" }));
  m.alternatives = (d.alternatives || []).map((a) => ({ id: String(a.id), name: a.name || String(a.id), values: { ...(a.values || {}) } }));
  m.constraints = (d.constraints || []).map((c) => ({ column: c.column, op: c.op, value: c.value }));
  m.overrides = Object.entries(d.overrides || {}).flatMap(([a, ch]) => Object.entries(ch).map(([c, v]) => ({ alternative: a, column: c, value: v })));
  if (d.portfolio) {
    const s = d.portfolio.size ?? 2;
    const [lo, hi] = typeof s === "object" ? [s.min ?? 1, s.max ?? s.min ?? 1] : [s, s];
    m.portfolio = { on: true, min: lo, max: hi };
  }
  // Réglages suggérés par le fichier (ex. méthode utilisée dans l'article d'un cas publié)
  if (d.analysis && d.analysis.methods) {
    for (const k of Object.keys(m.ui.methods)) m.ui.methods[k].on = false;
    for (const [k, params] of Object.entries(d.analysis.methods)) m.ui.methods[k] = { ...(m.ui.methods[k] || {}), ...params, on: true };
    if (d.analysis.primary) m.ui.primary = d.analysis.primary;
  }
  let w = d.weights || { method: "equal" };
  if (!w.method) w = { method: "direct", values: w };
  const ids = m.criteria.map((c) => c.id);
  m.weights.method = w.method === "rank_sum" ? "roc" : w.method;
  if (w.method === "direct") m.weights.direct = { ...w.values };
  if (w.order) w.order.forEach((c, i) => (m.weights.rank[c] = i + 1));
  if (w.method === "ahp") {
    if (w.judgements) {
      for (const [k, v] of Object.entries(w.judgements)) { const [a, b] = k.split(">"); m.weights.ahp[`${a}|${b}`] = toSaaty(parseNum(v)); }
    } else if (w.matrix) {
      ids.forEach((a, i) => ids.forEach((b, j) => { if (j > i) m.weights.ahp[`${a}|${b}`] = toSaaty(parseNum(w.matrix[i][j])); }));
    }
  }
  return m;
}

const critIds = () => S.criteria.map((c) => String(c.id || "").trim()).filter(Boolean);
const attrIds = () => S.attributes.map((a) => String(a.id || "").trim()).filter(Boolean);

/** modèle éditable -> dict API */
function toProblem() {
  const ids = critIds();
  const cols = [...ids, ...attrIds()];
  const weights = (() => {
    const w = S.weights;
    if (w.method === "direct") return { method: "direct", values: Object.fromEntries(ids.map((c) => [c, num(w.direct[c]) ?? 1])) };
    if (w.method === "roc") return { method: "roc", order: [...ids].sort((a, b) => (num(w.rank[a]) ?? 99) - (num(w.rank[b]) ?? 99) || ids.indexOf(a) - ids.indexOf(b)) };
    if (w.method === "ahp") {
      const j = {};
      ids.forEach((a, i) => ids.slice(i + 1).forEach((b) => { const v = w.ahp[`${a}|${b}`] ?? 1; j[`${a}>${b}`] = v >= 1 ? v : `1/${-v}`; }));
      return { method: "ahp", judgements: j };
    }
    return { method: "equal" };
  })();
  const overrides = {};
  for (const o of S.overrides) {
    if (o.alternative && o.column && num(o.value) !== null) (overrides[o.alternative] ??= {})[o.column] = num(o.value);
  }
  const p = {
    name: S.name || "Projet", description: S.description || "",
    criteria: S.criteria.filter((c) => String(c.id || "").trim()).map((c) => {
      const pref = { type: c.pref?.type || "usual" };
      for (const k of ["q", "p", "s"]) if (num(c.pref?.[k]) !== null) pref[k] = num(c.pref[k]);
      return { id: String(c.id).trim(), name: c.name || c.id, direction: c.direction || "max", unit: c.unit || "", aggregation: c.aggregation || "mean", preference: pref };
    }),
    attributes: S.attributes.filter((a) => String(a.id || "").trim()).map((a) => ({ id: String(a.id).trim(), name: a.name || a.id, unit: a.unit || "", aggregation: a.aggregation || "sum" })),
    alternatives: S.alternatives.filter((a) => String(a.id || "").trim()).map((a) => ({
      id: String(a.id).trim(), name: a.name || a.id,
      values: Object.fromEntries(cols.filter((c) => num(a.values[c]) !== null).map((c) => [c, num(a.values[c])])),
    })),
    weights,
    constraints: S.constraints.filter((c) => c.column && c.op && num(c.value) !== null).map((c) => ({ column: c.column, op: c.op, value: num(c.value) })),
    overrides,
  };
  if (S.portfolio.on) p.portfolio = { size: { min: Number(S.portfolio.min), max: Math.max(Number(S.portfolio.min), Number(S.portfolio.max)) } };
  return p;
}

const methodsPayload = () => {
  const out = {};
  const ordered = [S.ui.primary, ...Object.keys(S.ui.methods).filter((k) => k !== S.ui.primary)];
  for (const k of ordered) {
    const m = S.ui.methods[k];
    if (!m?.on) continue;
    out[k] = m.normalization ? { normalization: m.normalization } : {};
  }
  return out;
};

const save = debounce(() => store.set("mcdakit.project", S), 400);
function changed({ structure = false } = {}) {
  save();
  if (structure) renderAlternatives();
  refreshValidation();
  invalidate();
}
function invalidate() { LAST.analyze = null; LAST.sens = null; const t = currentTab(); if (t === "results" || t === "sensitivity") refreshTab(t); }

// ============================================================================ tableau éditable générique
/**
 * cols : [{key, label, type: text|number|select, options, get(row), set(row, v), width}]
 * rows : tableau d'objets (modifié en place). onChange(structure) appelé à chaque modification.
 */
function editableTable(container, cols, rows, { onChange, newRow, structuralKeys = [] } = {}) {
  container.innerHTML = "";
  const tbody = el("tbody");
  const table = el("table", { class: "edit" },
    el("thead", {}, el("tr", {}, cols.map((c) => el("th", {}, c.label)), el("th"))), tbody);
  const draw = () => {
    tbody.innerHTML = "";
    rows.forEach((row, i) => {
      const tr = el("tr");
      for (const c of cols) {
        const val = c.get ? c.get(row) : row[c.key];
        let input;
        if (c.type === "select") {
          const opts = typeof c.options === "function" ? c.options() : c.options;
          input = el("select", {}, (c.allowEmpty ? [el("option", { value: "" }, "—")] : []),
            opts.map((o) => { const [v, l] = Array.isArray(o) ? o : [o, o]; return el("option", { value: v, selected: String(val) === String(v) }, l); }));
          if (val && !opts.some((o) => String(Array.isArray(o) ? o[0] : o) === String(val))) input.append(el("option", { value: val, selected: true }, val));
        } else {
          input = el("input", { type: c.type === "number" ? "number" : "text", step: "any", value: val ?? "", placeholder: c.placeholder || "" });
        }
        const structural = structuralKeys.includes(c.key);
        input.addEventListener(c.type === "select" || structural ? "change" : "input", () => {
          const v = c.type === "number" ? input.value : input.value;
          c.set ? c.set(row, v) : (row[c.key] = v);
          onChange?.(structural);
        });
        tr.append(el("td", {}, input));
      }
      tr.append(el("td", { class: "del" }, el("button", { class: "icon-btn", title: "Supprimer la ligne", "aria-label": "Supprimer",
        onclick: () => { rows.splice(i, 1); draw(); onChange?.(true); } }, "✕")));
      tbody.append(tr);
    });
  };
  draw();
  container.append(el("div", { class: "table-wrap" }, table),
    el("div", { class: "add-row" }, el("button", { class: "btn small", onclick: () => { rows.push(newRow()); draw(); onChange?.(true); } }, "+ Ajouter une ligne")));
}

/** tableau de lecture à partir du format API {index, columns, data} */
function dataTable(t, { labels = null, pctCols = [], digits = 4, highlight = null, indexLabel = "", mapCols = {} } = {}) {
  if (!t) return el("div");
  const head = el("tr", {}, el("th", {}, indexLabel), t.columns.map((c) => el("th", {}, c)));
  const body = t.index.map((idx, r) => {
    const cells = t.data[r].map((v, j) => {
      const col = t.columns[j];
      let text = mapCols[col] ? mapCols[col](v) : pctCols.includes(col) ? pct(v) : fmt(v, digits);
      return el("td", { class: typeof v === "number" ? "num" : "" }, text);
    });
    const name = labels && labels[idx] && labels[idx] !== idx ? `${idx} · ${labels[idx]}` : idx;
    return el("tr", { class: highlight && idx === highlight ? "best" : "" }, el("td", {}, name), cells);
  });
  return el("div", { class: "table-wrap" }, el("table", {}, el("thead", {}, head), el("tbody", {}, body)));
}

// ============================================================================ graphiques
function cssVar(n) { return getComputedStyle(document.documentElement).getPropertyValue(n).trim(); }
function plot(target, data, layout = {}) {
  if (!window.Plotly) return;
  const base = {
    paper_bgcolor: "rgba(0,0,0,0)", plot_bgcolor: "rgba(0,0,0,0)", font: { color: cssVar("--text"), family: "system-ui, sans-serif", size: 12 },
    margin: { t: 34, r: 16, b: 40, l: 60 }, colorway: ["#0f4c75", "#f5922f", "#2a9d8f", "#c44e8a", "#4fb3ec", "#8a6fdf", "#b8a200", "#d1495b"],
    xaxis: { gridcolor: cssVar("--border"), zerolinecolor: cssVar("--border") }, yaxis: { gridcolor: cssVar("--border"), zerolinecolor: cssVar("--border") },
  };
  const merged = { ...base, ...layout, xaxis: { ...base.xaxis, ...(layout.xaxis || {}) }, yaxis: { ...base.yaxis, ...(layout.yaxis || {}) } };
  Plotly.react(target, data, merged, { displayModeBar: false, responsive: true });
}
const shortLabel = (id, labels) => (S.portfolio.on ? id : labels?.[id] ?? id);

// ============================================================================ onglet Données
function renderData() {
  $("#p-name").value = S.name; $("#p-desc").value = S.description;
  const prefs = Object.keys(META.preference_functions);
  editableTable($("#criteria-table"), [
    { key: "id", label: "id", placeholder: "C1" },
    { key: "name", label: "nom" },
    { key: "direction", label: "sens", type: "select", options: [["max", "max"], ["min", "min"]] },
    { key: "unit", label: "unité" },
    { key: "aggregation", label: "agrégation", type: "select", options: META.aggregations },
    { key: "pref", label: "préférence", type: "select", options: prefs, get: (r) => r.pref?.type, set: (r, v) => ((r.pref ??= {}).type = v) },
    ...["q", "p", "s"].map((k) => ({ key: k, label: k, type: "number", get: (r) => r.pref?.[k], set: (r, v) => ((r.pref ??= {})[k] = v) })),
  ], S.criteria, { onChange: (s) => { changed({ structure: s }); if (s) renderWeightsInput(); },
    newRow: () => ({ id: `C${S.criteria.length + 1}`, name: "", direction: "max", unit: "", aggregation: "mean", pref: { type: "usual" } }), structuralKeys: ["id"] });

  editableTable($("#attributes-table"), [
    { key: "id", label: "id" }, { key: "name", label: "nom" }, { key: "unit", label: "unité" },
    { key: "aggregation", label: "agrégation", type: "select", options: META.aggregations },
  ], S.attributes, { onChange: (s) => changed({ structure: s }), newRow: () => ({ id: `X${S.attributes.length + 1}`, name: "", unit: "", aggregation: "sum" }), structuralKeys: ["id"] });

  renderAlternatives();
  $("#pf-on").checked = S.portfolio.on; $("#pf-min").value = S.portfolio.min; $("#pf-max").value = S.portfolio.max;
  setPortfolioDisabled();
}

function setPortfolioDisabled() { $("#pf-min").disabled = $("#pf-max").disabled = !S.portfolio.on; }

function renderAlternatives() {
  const cols = [...critIds(), ...attrIds()];
  editableTable($("#alternatives-table"), [
    { key: "id", label: "id" }, { key: "name", label: "nom" },
    ...cols.map((c) => ({ key: c, label: c, type: "number", get: (r) => r.values[c], set: (r, v) => (r.values[c] = v) })),
  ], S.alternatives, { onChange: (s) => changed({ structure: false }), newRow: () => ({ id: `A${S.alternatives.length + 1}`, name: "", values: {} }) });

  editableTable($("#constraints-table"), [
    { key: "column", label: "colonne", type: "select", options: () => [...critIds(), ...attrIds()], allowEmpty: true },
    { key: "op", label: "opérateur", type: "select", options: META.operators },
    { key: "value", label: "valeur", type: "number" },
  ], S.constraints, { onChange: () => changed(), newRow: () => ({ column: "", op: "<=", value: "" }) });

  editableTable($("#overrides-table"), [
    { key: "alternative", label: "alternative (ex. A+D)" },
    { key: "column", label: "colonne", type: "select", options: () => [...critIds(), ...attrIds()], allowEmpty: true },
    { key: "value", label: "valeur", type: "number" },
  ], S.overrides, { onChange: () => changed(), newRow: () => ({ alternative: "", column: "", value: "" }) });
}

const refreshValidation = debounce(async () => {
  const box = $("#data-summary");
  try {
    const v = await api("validate", { problem: toProblem() });
    LAST.validate = v;
    const total = v.candidates.index.length;
    const ok = v.matrix ? v.matrix.index.length : 0;
    box.className = "callout " + (ok ? "ok" : "bad");
    box.innerHTML = ok
      ? `Modèle valide : <b>${ok}</b> alternative(s) admissible(s) sur ${total}${S.portfolio.on ? " combinaisons" : ""}, ${critIds().length} critères.`
      : `Aucune alternative admissible : ${esc(v.matrix_error || "")}`;
    if (S.portfolio.on || S.constraints.length) {
      const feas = v.feasibility;
      const excluded = feas.index.filter((_, i) => !feas.data[i][feas.columns.length - 1]);
      if (excluded.length) box.innerHTML += `<br>Exclues par les contraintes : ${excluded.map(esc).join(", ")}.`;
    }
    renderWeightsOut(v);
    fillTargets();
  } catch (e) {
    LAST.validate = null;
    box.className = "callout bad"; box.textContent = "Modèle incomplet : " + e.message;
    $("#ahp-info").innerHTML = ""; $("#weights-chart").innerHTML = "";
  }
}, 300);

// ============================================================================ onglet Poids
const WHINT = {
  equal: "Tous les critères ont le même poids (démarche « intuitive » sans hiérarchie).",
  direct: "Saisissez des poids quelconques : ils sont normalisés pour sommer à 100 %.",
  roc: "Classez simplement les critères (1 = le plus important) : les poids sont déduits par la méthode ROC.",
  ahp: "Pour chaque paire, déplacez le curseur vers le critère le plus important. 1 = égale importance, 3 = modérée, 5 = forte, 7 = très forte, 9 = extrême.",
};

function renderWeightsInput() {
  $$("#wmode button").forEach((b) => b.classList.toggle("active", b.dataset.mode === S.weights.method));
  $("#wmode-hint").textContent = WHINT[S.weights.method];
  const box = $("#weights-input"); box.innerHTML = "";
  const ids = critIds();
  const name = (c) => S.criteria.find((x) => x.id === c)?.name || c;
  if (S.weights.method === "direct" || S.weights.method === "roc") {
    const key = S.weights.method === "direct" ? "direct" : "rank";
    const t = el("table", { class: "edit" }, el("thead", {}, el("tr", {}, el("th", {}, "critère"), el("th", {}, key === "direct" ? "poids" : "rang"))));
    const tb = el("tbody");
    ids.forEach((c, i) => {
      const val = S.weights[key][c] ?? (key === "direct" ? 1 : i + 1);
      const inp = el("input", { type: "number", step: "any", min: 0, value: val });
      inp.addEventListener("input", () => { S.weights[key][c] = inp.value; changed(); });
      tb.append(el("tr", {}, el("td", {}, `${c} — ${name(c)}`), el("td", {}, inp)));
    });
    t.append(tb); box.append(el("div", { class: "table-wrap", style: "max-width:520px" }, t));
  } else if (S.weights.method === "ahp") {
    const grid = el("div", { class: "ahp-grid" });
    ids.forEach((a, i) => ids.slice(i + 1).forEach((b) => {
      const k = `${a}|${b}`;
      const v = S.weights.ahp[k] ?? 1;
      const range = el("input", { type: "range", min: 0, max: SAATY.length - 1, step: 1, value: SAATY.indexOf(v), "aria-label": `${name(a)} contre ${name(b)}` });
      const la = el("b", {}, name(a)), lb = el("b", {}, name(b)), out = el("div", { class: "val" });
      const upd = () => {
        const s = SAATY[Number(range.value)];
        la.className = s > 1 ? "win" : ""; lb.className = s < -1 ? "win" : "";
        out.textContent = s === 1 ? "importance égale" : s > 1 ? `${name(a)} ${s}× plus important` : `${name(b)} ${-s}× plus important`;
        return s;
      };
      upd();
      range.addEventListener("input", () => { S.weights.ahp[k] = upd(); changed(); });
      grid.append(el("div", { class: "ahp-pair" }, el("div", { class: "labels" }, la, lb), range, out));
    }));
    box.append(grid);
  }
}

function renderWeightsOut(v) {
  const ids = Object.keys(v.weights);
  const names = ids.map((c) => S.criteria.find((x) => x.id === c)?.name || c);
  plot("weights-chart", [{ type: "bar", x: names, y: ids.map((c) => v.weights[c]), text: ids.map((c) => pct(v.weights[c])), textposition: "auto",
    marker: { color: cssVar("--accent") }, hovertemplate: "%{x} : %{y:.2%}<extra></extra>" }],
  { title: { text: "Poids normalisés", font: { size: 14 } }, yaxis: { tickformat: ".0%" }, height: 320 });
  const info = $("#ahp-info"); info.innerHTML = "";
  if (v.ahp) {
    const a = v.ahp;
    info.append(el("div", { class: "callout " + (a.consistent ? "ok" : "bad") },
      el("div", { class: "kpi" }, "Ratio de cohérence RC = ", el("b", {}, pct(a.cr, 2)), a.consistent ? " — cohérent (< 10 %)" : " — incohérent (≥ 10 %)"),
      el("div", { class: "hint" }, `λmax = ${fmt(a.lambda_max, 4)} · IC = ${fmt(a.ci, 4)} · IA = ${fmt(a.ri, 2)}`)));
    if (!a.consistent) info.append(el("p", {}, "Jugements à revoir en priorité :"), dataTable(a.worst, { indexLabel: "i", digits: 3 }));
    info.append(el("details", { class: "box" }, el("summary", {}, "Matrice de comparaisons"), dataTable(a.matrix, { digits: 3 })));
  }
}

// ============================================================================ onglet Delphi
function targets() {
  const base = S.alternatives.map((a) => a.id).filter(Boolean);
  const gen = S.portfolio.on && LAST.validate ? LAST.validate.candidates.index : [];
  return [...base, ...gen.filter((g) => !base.includes(g))];
}

function renderDelphi() {
  $("#delphi-iqr").value = S.ui.iqr;
  editableTable($("#delphi-table"), [
    { key: "alternative", label: "alternative / élément", type: "select", options: targets, allowEmpty: true },
    { key: "column", label: "critère", type: "select", options: critIds, allowEmpty: true },
    { key: "r1", label: "tour 1", placeholder: "2, 1, 3, 3, 2" },
    { key: "r2", label: "tour 2" }, { key: "r3", label: "tour 3" },
  ], S.ui.delphi, { onChange: () => { save(); runDelphi(); }, newRow: () => ({ alternative: "", column: "", r1: "", r2: "", r3: "" }) });
  runDelphi();
}

const runDelphi = debounce(async () => {
  const items = S.ui.delphi.map((r) => ({ alternative: r.alternative, column: r.column, rounds: [r.r1, r.r2, r.r3].filter((x) => String(x || "").trim()) }))
    .filter((r) => r.rounds.length);
  const out = $("#delphi-results"); out.innerHTML = ""; LAST.delphi = null;
  $("#delphi-apply").disabled = true;
  if (!items.length) return;
  try {
    const res = await api("delphi", { items, iqr_threshold: Number(S.ui.iqr) || 1 });
    LAST.delphi = res;
    const t = el("table", {}, el("thead", {}, el("tr", {}, ["cellule", "tour 1", "tour 2", "tour 3", "retenu", "statut"].map((h) => el("th", {}, h)))));
    const tb = el("tbody");
    res.forEach((r) => tb.append(el("tr", {},
      el("td", {}, `${r.alternative || "?"} × ${r.column || "?"}`),
      [0, 1, 2].map((k) => el("td", {}, r.rounds[k] ? `méd. ${fmt(r.rounds[k].median)} [${fmt(r.rounds[k].q1)}–${fmt(r.rounds[k].q3)}]` : "")),
      el("td", { class: "num" }, el("b", {}, fmt(r.value))), el("td", {}, r.status))));
    t.append(tb); out.append(el("div", { class: "table-wrap" }, t));
    $("#delphi-apply").disabled = !res.some((r) => r.alternative && r.column);
  } catch (e) { out.append(el("div", { class: "callout bad" }, e.message)); }
});

function applyDelphi() {
  const base = new Set(S.alternatives.map((a) => a.id));
  let n = 0;
  for (const r of LAST.delphi || []) {
    if (!r.alternative || !r.column) continue;
    if (base.has(r.alternative)) S.alternatives.find((a) => a.id === r.alternative).values[r.column] = r.value;
    else {
      S.overrides = S.overrides.filter((o) => !(o.alternative === r.alternative && o.column === r.column));
      S.overrides.push({ alternative: r.alternative, column: r.column, value: r.value });
    }
    n++;
  }
  renderAlternatives(); changed();
  showStatus(`${n} valeur(s) appliquée(s) au modèle (élément de base modifié ou surcharge de la combinaison).`, "info");
}

// ============================================================================ onglet Résultats
function renderMethodPicker() {
  const box = $("#method-picker"); box.innerHTML = "";
  for (const [k, label] of Object.entries(META.methods)) {
    const m = (S.ui.methods[k] ??= { on: false });
    const cb = el("input", { type: "checkbox", checked: m.on });
    const card = el("div", { class: "method-card" + (m.on ? " on" : "") }, el("label", {}, cb, label), el("p", {}, META.logic[k]));
    cb.addEventListener("change", () => { m.on = cb.checked; card.classList.toggle("on", m.on); fillPrimary(); save(); invalidate(); });
    if (META.default_normalization[k]) {
      m.normalization ??= META.default_normalization[k];
      const sel = el("select", { "aria-label": "Normalisation" }, Object.entries(META.normalizations).map(([v, l]) => el("option", { value: v, selected: v === m.normalization }, l)));
      sel.addEventListener("change", () => { m.normalization = sel.value; save(); invalidate(); });
      card.append(el("div", { class: "hint" }, "Normalisation"), sel);
    }
    box.append(card);
  }
  fillPrimary();
}

function fillPrimary() {
  const on = Object.keys(S.ui.methods).filter((k) => S.ui.methods[k].on);
  if (!on.includes(S.ui.primary)) S.ui.primary = on[0] || "topsis";
  const sel = $("#primary-method"); sel.innerHTML = "";
  on.forEach((k) => sel.append(el("option", { value: k, selected: k === S.ui.primary }, META.methods[k])));
}

async function runResults() {
  const out = $("#results-out");
  const methods = methodsPayload();
  if (!Object.keys(methods).length) { out.innerHTML = '<div class="callout warn">Sélectionnez au moins une méthode.</div>'; return; }
  out.innerHTML = '<p class="loading">Calcul…</p>';
  try {
    const r = LAST.analyze || (LAST.analyze = await api("analyze", { problem: toProblem(), methods }));
    out.innerHTML = "";
    const labels = r.labels;
    const ref = r.results[S.ui.primary] || Object.values(r.results)[0];
    const t = ref.table, scoreCol = 1;
    const second = t.index[1];
    out.append(el("div", { class: "callout ok", html:
      `<div class="kpi">Recommandation (${esc(ref.label)}) : <b>${esc(labels[ref.best])}</b> <span class="badge">${esc(ref.best)}</span> — ${esc(ref.score_label)} = ${fmt(ref.scores[ref.best])}</div>`
      + (second ? `Deuxième : ${esc(labels[second])} (écart ${fmt(t.data[1][t.columns.indexOf("écart au 1er")])})` : "") }));

    out.append(el("h2", {}, "Classements"));
    const grid = el("div", { class: "results-grid" });
    for (const [k, m] of Object.entries(r.results)) {
      const chart = el("div", { class: "chart" });
      const card = el("div", {}, chart, dataTable(m.table, { highlight: m.best, mapCols: { alternative: (v) => v, rang: (v) => String(v) } }));
      grid.append(card);
      const order = [...m.order].reverse();
      requestAnimationFrame(() => plot(chart, [{ type: "bar", orientation: "h", y: order.map((i) => shortLabel(i, labels)), x: order.map((i) => m.scores[i]),
        marker: { color: order.map((i) => (i === m.best ? cssVar("--orange") : "#9fc3dc")) },
        text: order.map((i) => fmt(m.scores[i])), textposition: "auto", hovertext: order.map((i) => labels[i]), hovertemplate: "%{hovertext}<br>%{x:.4f}<extra></extra>" }],
      { title: { text: m.label, font: { size: 14 } }, height: 90 + 34 * order.length, margin: { t: 34, r: 16, b: 30, l: S.portfolio.on ? 50 : 130 } }));
    }
    out.append(grid);

    if (r.comparison) {
      const c = r.comparison;
      out.append(el("h2", {}, "Comparaison des méthodes"));
      out.append(el("div", { class: "callout " + (c.same_winner ? "ok" : "warn"), html: c.same_winner
        ? `Toutes les méthodes désignent <b>${esc(labels[ref.best])}</b>.`
        : "Les méthodes divergent sur le 1er : " + Object.entries(c.winners).map(([k, v]) => `${esc(k)} → <b>${esc(labels[v])}</b>`).join(" · ") }));
      out.append(el("div", { class: "grid2" },
        el("div", {}, el("h3", {}, "Rangs par méthode"), dataTable(c.ranks, { mapCols: Object.fromEntries(c.ranks.columns.map((col) => [col, (v) => String(v)])) })),
        el("div", {}, el("h3", {}, "Corrélation de rang (τ de Kendall)"), dataTable(c.kendall_tau, { digits: 3 }),
          el("p", { class: "hint" }, "1 = classements identiques, −1 = inversés."))));
      if (c.divergent.length) out.append(el("p", { class: "hint" }, "Rangs divergents pour : " + c.divergent.map((i) => labels[i]).join(", ")
        + ". Causes typiques : compensation (additive vs surclassement), normalisation, seuils de préférence PROMETHEE, valeurs cardinales vs notes."));
    }

    const det = el("details", { class: "box" }, el("summary", {}, "Matrice de décision et détail des calculs"));
    if (LAST.validate) {
      det.append(el("h3", {}, "Candidates et admissibilité"), dataTable(LAST.validate.candidates));
      det.append(dataTable(LAST.validate.feasibility));
    }
    for (const m of Object.values(r.results)) {
      det.append(el("h3", {}, m.label));
      for (const [name, d] of Object.entries(m.details)) {
        if (Array.isArray(d)) det.append(el("p", {}, `${name} : ${d.length ? d.map((p) => p.join(" ? ")).join(", ") : "aucune"}`));
        else det.append(el("p", { class: "hint" }, name), dataTable(d));
      }
    }
    out.append(det);
  } catch (e) { out.innerHTML = `<div class="callout bad">${esc(e.message)}</div>`; }
}

// ============================================================================ onglet Sensibilité
function fillTargets() {
  const cols = [...critIds(), ...attrIds()];
  const sw = $("#sweep-crit"), cur = S.ui.sweep;
  sw.innerHTML = ""; critIds().forEach((c) => sw.append(el("option", { value: c, selected: c === cur }, `${c} — ${S.criteria.find((x) => x.id === c)?.name || c}`)));
  if (!critIds().includes(cur)) S.ui.sweep = critIds()[0];
  const va = $("#vt-alt"), vc = $("#vt-col"), a0 = va.value, c0 = vc.value;
  va.innerHTML = ""; targets().forEach((t) => va.append(el("option", { value: t, selected: t === a0 }, t)));
  vc.innerHTML = ""; cols.forEach((c) => vc.append(el("option", { value: c, selected: c === c0 }, c)));
}

function renderSensitivityInputs() {
  fillTargets();
  $("#mc-n").value = S.ui.mc.n; $("#mc-uniform").checked = S.ui.mc.uniform; $("#mc-spread").value = S.ui.mc.spread;
  $("#mc-spread").disabled = S.ui.mc.uniform; $("#mc-spread-out").textContent = pct(S.ui.mc.spread, 0);
  $("#equal-profile").checked = S.ui.equalProfile;
  const ids = critIds();
  editableTable($("#profiles-table"), [
    { key: "name", label: "profil" },
    ...ids.map((c) => ({ key: c, label: `× ${c}`, type: "number", get: (r) => r.mult?.[c] ?? 1, set: (r, v) => ((r.mult ??= {})[c] = v) })),
  ], S.ui.profiles, { onChange: () => { save(); sensChanged(); }, newRow: () => ({ name: `Profil ${S.ui.profiles.length + 1}`, mult: {} }) });
  editableTable($("#scenarios-table"), [
    { key: "name", label: "scénario" },
    { key: "alternative", label: "alternative / élément", type: "select", options: targets, allowEmpty: true },
    { key: "column", label: "colonne", type: "select", options: () => [...critIds(), ...attrIds()], allowEmpty: true },
    { key: "value", label: "nouvelle valeur", type: "number" },
  ], S.ui.scenarios, { onChange: () => { save(); sensChanged(); }, newRow: () => ({ name: `Scénario ${S.ui.scenarios.length + 1}`, alternative: "", column: "", value: "" }) });
}
const sensChanged = debounce(() => { LAST.sens = null; if (currentTab() === "sensitivity") runSensitivity(); }, 500);

async function runSensitivity() {
  const methods = methodsPayload();
  const primary = S.ui.primary;
  if (!methods[primary]) { $("#sens-intro").textContent = "Choisissez une méthode dans l'onglet Résultats."; return; }
  const profiles = {};
  for (const p of S.ui.profiles) if (p.name) profiles[p.name] = Object.fromEntries(Object.entries(p.mult || {}).filter(([, v]) => num(v) !== null).map(([k, v]) => [k, num(v)]));
  if (S.ui.equalProfile) profiles["Poids égaux"] = { __equal__: true };
  const scenarios = {};
  for (const s of S.ui.scenarios) if (s.name && s.alternative && s.column && num(s.value) !== null) (scenarios[s.name] ??= []).push([s.alternative, s.column, num(s.value)]);
  $("#sens-intro").innerHTML = `Méthode étudiée : <b>${esc(META.methods[primary])}</b> (méthode de référence de l'onglet Résultats). <span class="loading">Calcul…</span>`;
  try {
    const r = LAST.sens || (LAST.sens = await api("sensitivity", {
      problem: toProblem(), method: primary, params: methods[primary], methods, sweep_criterion: S.ui.sweep,
      mc_runs: Number(S.ui.mc.n) || 10000, mc_spread: S.ui.mc.uniform ? null : Number(S.ui.mc.spread), profiles, scenarios,
    }));
    const L = r.labels;
    $("#sens-intro").innerHTML = `Méthode étudiée : <b>${esc(META.methods[primary])}</b>. Gagnant de référence : <b>${esc(L[r.winner])}</b>. Où la décision bascule-t-elle ?`;
    const lbl = (v) => (v && v !== "—" ? `${v}${L[v] && L[v] !== v ? " · " + L[v] : ""}` : "—");
    const th = $("#thresholds"); th.innerHTML = "";
    th.append(dataTable(r.thresholds, { indexLabel: "critère", pctCols: ["poids actuel", "seuil haut", "seuil bas"],
      mapCols: { "variation relative min": (v) => pct(v, 0), "bascule vers (haut)": lbl, "bascule vers (bas)": lbl } }),
    el("p", { class: "hint" }, "Seuil = poids du critère à partir duquel le 1er change ; les autres poids s'ajustent proportionnellement."));

    if (r.sweep) {
      const sw = r.sweep;
      const traces = Object.entries(sw.scores).map(([a, ys]) => ({ type: "scatter", mode: "lines", name: shortLabel(a, L), x: sw.grid, y: ys,
        hovertemplate: `${esc(L[a])}<br>poids %{x:.0%} : %{y:.4f}<extra></extra>`, line: { width: a === r.winner ? 3.5 : 1.8 } }));
      plot("sweep-chart", traces, { height: 400, xaxis: { tickformat: ".0%", title: { text: `poids de ${sw.criterion}` } },
        shapes: [{ type: "line", x0: sw.current, x1: sw.current, yref: "paper", y0: 0, y1: 1, line: { dash: "dash", color: cssVar("--muted") } }],
        annotations: [{ x: sw.current, yref: "paper", y: 1.04, text: "poids actuel", showarrow: false, font: { size: 11 } }] });
    }
    if (r.monte_carlo) {
      const mc = r.monte_carlo, t = mc.table, fi = t.columns.indexOf("fréquence 1er");
      plot("mc-chart", [{ type: "bar", x: t.index.map((i) => shortLabel(i, L)), y: t.data.map((d) => d[fi]), text: t.data.map((d) => pct(d[fi])),
        textposition: "auto", marker: { color: cssVar("--accent") }, hovertext: t.index.map((i) => L[i]), hovertemplate: "%{hovertext}<br>%{y:.1%}<extra></extra>" }],
      { title: { text: `Fréquence de 1re place (${fmt(mc.n)} tirages)`, font: { size: 14 } }, yaxis: { tickformat: ".0%" }, height: 330 });
      const rd = mc.rank_distribution;
      plot("mc-heat", [{ type: "heatmap", z: rd.data, x: rd.columns.map((c) => `rang ${c}`), y: rd.index.map((i) => shortLabel(i, L)), colorscale: [[0, cssVar("--surface-2")], [1, cssVar("--accent")]],
        text: rd.data.map((row) => row.map((v) => (v ? pct(v, 0) : ""))), texttemplate: "%{text}", hovertemplate: "%{y} · %{x} : %{z:.1%}<extra></extra>", showscale: false }],
      { title: { text: "Distribution des rangs", font: { size: 14 } }, height: 330, yaxis: { autorange: "reversed" }, margin: { t: 34, r: 10, b: 40, l: S.portfolio.on ? 50 : 120 } });
    }
    $("#profiles-out").innerHTML = ""; if (r.profiles) $("#profiles-out").append(dataTable(r.profiles, { indexLabel: "profil", mapCols: { gagnant: lbl } }));
    $("#scenarios-out").innerHTML = ""; if (r.scenarios) $("#scenarios-out").append(dataTable(r.scenarios, { indexLabel: "scénario", mapCols: { gagnant: lbl } }));
    const d = r.diagnosis, cls = { robuste: "ok", "modérément robuste": "warn", fragile: "bad" }[d.level];
    $("#diagnosis").innerHTML = "";
    $("#diagnosis").append(el("div", { class: `callout ${cls}` }, el("div", { class: "kpi" }, "Recommandation ", el("b", {}, d.level), ` : ${L[d.winner]}`),
      el("ul", {}, d.reasons.map((x) => el("li", {}, x)))),
    el("p", { class: "hint" }, "Règles : fragile si 1er dans moins de 60 % des tirages, si un poids bascule pour moins de 10 % de variation relative ou si la majorité des scénarios basculent ; robuste si ≥ 90 % des tirages, aucune bascule de poids en deçà de 50 %, méthodes concordantes et aucun scénario de basculement ; modérément robuste sinon."));
  } catch (e) { $("#sens-intro").innerHTML = `<span class="callout bad">${esc(e.message)}</span>`; }
}

async function runValueThreshold() {
  const out = $("#vt-out"); out.innerHTML = '<p class="loading">Recherche…</p>';
  const methods = methodsPayload();
  try {
    const r = await api("value-threshold", { problem: toProblem(), method: S.ui.primary, params: methods[S.ui.primary] || {},
      alternative: $("#vt-alt").value, column: $("#vt-col").value, low: Number($("#vt-lo").value), high: Number($("#vt-hi").value) });
    const L = LAST.validate?.labels || {};
    out.innerHTML = "";
    out.append(r.threshold === null
      ? el("div", { class: "callout ok" }, `Aucun basculement entre ${$("#vt-lo").value} et ${$("#vt-hi").value} (valeur actuelle ${fmt(r.current)}).`)
      : el("div", { class: "callout warn" }, `Valeur actuelle ${fmt(r.current)} → le 1er devient ${L[r.new_winner] || r.new_winner} à partir de ≈ ${fmt(r.threshold, 3)}.`));
  } catch (e) { out.innerHTML = `<div class="callout bad">${esc(e.message)}</div>`; }
}

function prefillValueRange() {
  const a = $("#vt-alt").value, c = $("#vt-col").value;
  let cur = null;
  const base = S.alternatives.find((x) => x.id === a);
  if (base) cur = num(base.values[c]);
  else if (LAST.validate) { const t = LAST.validate.candidates, i = t.index.indexOf(a), j = t.columns.indexOf(c); if (i >= 0 && j >= 0) cur = t.data[i][j]; }
  if (cur === null) return;
  const span = Math.abs(cur) || 1;
  $("#vt-lo").value = +(cur - span).toPrecision(6); $("#vt-hi").value = +(cur + span).toPrecision(6);
}

// ============================================================================ onglet Rapport
async function makeReport(format) {
  const methods = methodsPayload();
  const profiles = {};
  for (const p of S.ui.profiles) if (p.name) profiles[p.name] = Object.fromEntries(Object.entries(p.mult || {}).filter(([, v]) => num(v) !== null).map(([k, v]) => [k, num(v)]));
  if (S.ui.equalProfile) profiles["Poids égaux"] = { __equal__: true };
  const scenarios = {};
  for (const s of S.ui.scenarios) if (s.name && s.alternative && s.column && num(s.value) !== null) (scenarios[s.name] ??= []).push([s.alternative, s.column, num(s.value)]);
  const win = format === "html" ? window.open("", "_blank") : null;
  if (win) win.document.write("<p style='font-family:system-ui;padding:2rem'>Génération du rapport…</p>");
  try {
    const r = await api("report", { problem: toProblem(), methods, primary: S.ui.primary, format, mc_runs: Math.min(Number(S.ui.mc.n) || 5000, 20000),
      mc_spread: S.ui.mc.uniform ? null : Number(S.ui.mc.spread), profiles, scenarios }, { raw: true });
    const blob = await r.blob();
    if (format === "html") {
      const url = URL.createObjectURL(blob);
      if (win) win.location.href = url; else download(blob, "rapport_mcda.html");
    } else download(blob, format === "md" ? "rapport_mcda.md" : "calculs_mcda.xlsx");
  } catch (e) { if (win) win.close(); showStatus(e.message); }
}

// ============================================================================ navigation & chargement
const currentTab = () => $(".tabs button.active")?.dataset.tab;
function refreshTab(t) {
  if (t === "weights") renderWeightsInput();
  if (t === "delphi") renderDelphi();
  if (t === "results") runResults();
  if (t === "sensitivity") { renderSensitivityInputs(); runSensitivity(); }
}
function selectTab(t) {
  $$(".tabs button").forEach((b) => b.classList.toggle("active", b.dataset.tab === t));
  $$(".panel").forEach((p) => p.classList.toggle("active", p.id === "tab-" + t));
  refreshTab(t);
  // Plotly mesure la largeur au rendu : on recalcule une fois le panneau visible.
  requestAnimationFrame(() => $$(`#tab-${t} .js-plotly-plot`).forEach((g) => window.Plotly && Plotly.Plots.resize(g)));
}

function load(model) {
  S = model; S.ui = { ...defaultUI(), ...(model.ui || {}) };
  LAST = { validate: null, analyze: null, sens: null, delphi: null };
  renderData(); renderMethodPicker(); renderWeightsInput(); save(); refreshValidation();
  refreshTab(currentTab());
}

async function init() {
  try { META = await api("meta"); } catch (e) { showStatus("API indisponible : " + e.message, "error", 60000); return; }
  const sel = $("#example-select");
  try { (await api("examples")).forEach((x) => sel.append(el("option", { value: x.id }, x.name))); } catch { /* */ }
  sel.addEventListener("change", async () => {
    if (!sel.value) return;
    try { load(fromProblem(await api("examples/" + sel.value))); showStatus("Exemple chargé.", "info", 2500); } catch (e) { showStatus(e.message); }
    sel.value = "";
  });
  $("#import-file").addEventListener("change", async (ev) => {
    const f = ev.target.files[0]; if (!f) return;
    const fd = new FormData(); fd.append("file", f);
    try { load(fromProblem(await api("import", fd))); showStatus(`« ${f.name} » importé.`, "info", 2500); } catch (e) { showStatus("Import impossible : " + e.message); }
    ev.target.value = "";
  });
  $("#new-project").addEventListener("click", () => { if (confirm("Remplacer le projet courant par un projet vierge ?")) load(blankModel()); });
  $$(".tabs button").forEach((b) => b.addEventListener("click", () => selectTab(b.dataset.tab)));
  $("#p-name").addEventListener("input", (e) => { S.name = e.target.value; save(); LAST.analyze = null; });
  $("#p-desc").addEventListener("input", (e) => { S.description = e.target.value; save(); });
  $("#pf-on").addEventListener("change", (e) => { S.portfolio.on = e.target.checked; setPortfolioDisabled(); changed(); });
  $("#pf-min").addEventListener("input", (e) => { S.portfolio.min = Number(e.target.value) || 1; changed(); });
  $("#pf-max").addEventListener("input", (e) => { S.portfolio.max = Number(e.target.value) || 1; changed(); });
  $$("#wmode button").forEach((b) => b.addEventListener("click", () => { S.weights.method = b.dataset.mode; renderWeightsInput(); changed(); }));
  $("#delphi-iqr").addEventListener("input", (e) => { S.ui.iqr = e.target.value; save(); runDelphi(); });
  $("#delphi-apply").addEventListener("click", applyDelphi);
  $("#primary-method").addEventListener("change", (e) => { S.ui.primary = e.target.value; save(); LAST.sens = null; runResults(); });
  $("#sweep-crit").addEventListener("change", (e) => { S.ui.sweep = e.target.value; save(); sensChanged(); });
  $("#mc-n").addEventListener("change", (e) => { S.ui.mc.n = Number(e.target.value); save(); sensChanged(); });
  $("#mc-uniform").addEventListener("change", (e) => { S.ui.mc.uniform = e.target.checked; $("#mc-spread").disabled = e.target.checked; save(); sensChanged(); });
  $("#mc-spread").addEventListener("input", (e) => { S.ui.mc.spread = Number(e.target.value); $("#mc-spread-out").textContent = pct(S.ui.mc.spread, 0); save(); sensChanged(); });
  $("#equal-profile").addEventListener("change", (e) => { S.ui.equalProfile = e.target.checked; save(); sensChanged(); });
  $("#vt-alt").addEventListener("change", prefillValueRange);
  $("#vt-col").addEventListener("change", prefillValueRange);
  $("#vt-run").addEventListener("click", runValueThreshold);
  $$("[data-report]").forEach((b) => b.addEventListener("click", () => makeReport(b.dataset.report)));
  $("#save-json").addEventListener("click", () => download(new Blob([JSON.stringify(toProblem(), null, 2)], { type: "application/json" }), "projet_mcda.json"));
  $("#save-xlsx").addEventListener("click", async () => {
    try { download(await (await api("export-xlsx", { problem: toProblem() }, { raw: true })).blob(), "projet_mcda.xlsx"); } catch (e) { showStatus(e.message); }
  });

  // Liens depuis la page d'accueil : /app?exemple=<id> ou /app?nouveau=1
  const q = new URLSearchParams(location.search);
  const saved = store.get("mcdakit.project");
  const hasSaved = saved && saved.criteria;
  const replace = () => !hasSaved || confirm("Un projet est déjà enregistré dans ce navigateur. Le remplacer ?\n(Annuler pour le reprendre.)");
  if (q.get("exemple") && replace()) {
    try { load(fromProblem(await api("examples/" + encodeURIComponent(q.get("exemple"))))); } catch { load(blankModel()); }
  } else if (q.has("nouveau") && replace()) {
    load(blankModel());
  } else if (hasSaved) load(saved);
  else {
    try { load(fromProblem(await api("examples/stratema_methode2"))); } catch { load(blankModel()); }
  }
  if (q.toString()) history.replaceState(null, "", location.pathname);
  prefillValueRange();
}

document.addEventListener("DOMContentLoaded", init);
