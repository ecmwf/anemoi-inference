// (C) Copyright 2026 Anemoi contributors. Licensed under the Apache Licence Version 2.0.
//
// Single-page viewer for the configuration reference. DATA is injected when the page is built.
// No network access, no dependencies: the page must work from file:// and on Read the Docs.

"use strict";

const STORAGE_KEY = "anemoi-inference-config-builder";
const KINDS = Object.fromEntries(DATA.kinds.map((k) => [k.id, k]));
const COMPONENTS = DATA.components;
const EXAMPLES = Object.fromEntries(DATA.examples.map((e) => [e.id, e]));
const TOP = COMPONENTS["config/run"];
const LIST_SLOTS = new Set(["pre_processors", "post_processors"]);

const $ = (sel, root = document) => root.querySelector(sel);
const esc = (s) =>
  String(s).replace(/[&<>"']/g, (c) => ({ "&": "&amp;", "<": "&lt;", ">": "&gt;", '"': "&quot;", "'": "&#39;" })[c]);
const norm = (name) => String(name).replace(/_/g, "-");

function storage(fn, fallback) {
  try {
    return fn(window.localStorage);
  } catch (e) {
    return fallback;
  }
}

function toast(msg) {
  const el = $("#toast");
  el.textContent = msg;
  el.classList.add("show");
  clearTimeout(toast.timer);
  toast.timer = setTimeout(() => el.classList.remove("show"), 1800);
}

function findComponent(kind, name) {
  const wanted = norm(name);
  for (const id of KINDS[kind] ? KINDS[kind].components : []) {
    const c = COMPONENTS[`${kind}/${id}`];
    if (norm(c.name) === wanted || c.aliases.some((a) => norm(a) === wanted)) return c;
  }
  return null;
}

// ---------------------------------------------------------------------------
// YAML emitter (plain data only)
// ---------------------------------------------------------------------------

const RESERVED = /^(true|false|yes|no|on|off|null|~|y|n)$/i;

function scalar(v) {
  if (v === null || v === undefined) return "null";
  if (typeof v === "boolean") return v ? "true" : "false";
  if (typeof v === "number") return Number.isFinite(v) ? String(v) : Number.isNaN(v) ? ".nan" : v > 0 ? ".inf" : "-.inf";
  const s = String(v);
  const needsQuote =
    s === "" ||
    RESERVED.test(s) ||
    /^[-+]?(\d[\d_]*\.?\d*([eE][-+]?\d+)?|\.\d+|\.(inf|nan))$/i.test(s) ||
    /^[\s\-?:,\[\]{}#&*!|>'"%@`]/.test(s) ||
    /(: |\s#|\s$|\n|\t)/.test(s) ||
    s.endsWith(":");
  return needsQuote ? JSON.stringify(s) : s;
}

function isPlainObject(v) {
  return v !== null && typeof v === "object" && !Array.isArray(v);
}

function toYaml(value, indent = 0) {
  const pad = " ".repeat(indent);
  if (Array.isArray(value)) {
    if (!value.length) return "[]";
    return value
      .map((item) => {
        if (isPlainObject(item) && Object.keys(item).length) {
          const body = toYaml(item, indent + 2);
          return `${pad}- ${body.slice(indent + 2)}`;
        }
        if (Array.isArray(item) && item.length) {
          return `${pad}-\n${toYaml(item, indent + 2)}`;
        }
        return `${pad}- ${isPlainObject(item) ? "{}" : Array.isArray(item) ? "[]" : scalar(item)}`;
      })
      .join("\n");
  }
  if (isPlainObject(value)) {
    const keys = Object.keys(value);
    if (!keys.length) return "{}";
    return keys
      .map((k) => {
        const v = value[k];
        const key = scalar(k);
        if ((isPlainObject(v) || Array.isArray(v)) && (Array.isArray(v) ? v.length : Object.keys(v).length)) {
          return `${pad}${key}:\n${toYaml(v, indent + 2)}`;
        }
        return `${pad}${key}: ${Array.isArray(v) ? "[]" : isPlainObject(v) ? "{}" : scalar(v)}`;
      })
      .join("\n");
  }
  return pad + scalar(value);
}

function highlightYaml(text) {
  return text
    .split("\n")
    .map((line) => {
      const m = line.match(/^(\s*(?:- )?)([^\s:#][^:#]*?)(:)(\s.*|$)/);
      let out;
      if (m) out = `${esc(m[1])}<span class="y-key">${esc(m[2])}</span>${m[3]}${esc(m[4])}`;
      else out = esc(line);
      return out.replace(/&lt;[^&]*?&gt;/g, (ph) => `<span class="y-ph">${ph}</span>`);
    })
    .join("\n");
}

// ---------------------------------------------------------------------------
// Component helpers
// ---------------------------------------------------------------------------

function placeholder(field) {
  if (field.example !== null && field.example !== undefined) return clone(field.example);
  if (field.hasDefault && field.default !== null) return clone(field.default);
  if (field.many) return [];
  return `<${field.name}>`;
}

function clone(v) {
  return v === undefined ? v : JSON.parse(JSON.stringify(v));
}

// Smallest configuration for a component: just its name, or its required fields.
function minimalConfig(comp) {
  const required = comp.fields.filter((f) => f.required);
  if (!required.length) return comp.name;
  if (required.length === 1 && comp.positional === required[0].name) {
    return { [comp.name]: placeholder(required[0]) };
  }
  return { [comp.name]: Object.fromEntries(required.map((f) => [f.name, placeholder(f)])) };
}

// Split a component config into [name, options], following Registry.from_config.
function splitComponent(value, comp) {
  if (typeof value === "string") return [value, {}];
  if (isPlainObject(value) && "_type" in value) {
    const { _type, ...rest } = value;
    return [_type, rest];
  }
  if (isPlainObject(value) && Object.keys(value).length === 1) {
    const [name, inner] = Object.entries(value)[0];
    if (isPlainObject(inner)) return [name, { ...inner }];
    if (inner === null) return [name, {}];
    const c = comp || null;
    if (c && c.positional) return [name, { [c.positional]: inner }];
    return [name, { _positional: inner }];
  }
  return [null, null];
}

function looksPerDataset(value, kind) {
  if (!isPlainObject(value) || "_type" in value) return false;
  const keys = Object.keys(value);
  return !(keys.length === 1 && findComponent(kind, keys[0]));
}

// ---------------------------------------------------------------------------
// Validation of the builder contents against the documented options
// ---------------------------------------------------------------------------

function checkSlot(value, field, path, issues) {
  if (value === null || value === undefined) return;
  if (field.perDataset && looksPerDataset(value, field.ref) && !(field.many && Array.isArray(value))) {
    for (const [ds, sub] of Object.entries(value)) checkSlot(sub, { ...field, perDataset: false }, `${path}.${ds}`, issues);
    return;
  }
  if (field.many) {
    if (!Array.isArray(value)) {
      issues.push(`${path}: expected a list`);
      return;
    }
    value.forEach((item, i) => checkSlot(item, { ...field, many: false, perDataset: false }, `${path}[${i}]`, issues));
    return;
  }
  const [name] = splitComponent(value);
  if (!name) {
    issues.push(`${path}: cannot read a component`);
    return;
  }
  const comp = findComponent(field.ref, name);
  if (!comp) {
    issues.push(`${path}: unknown ${KINDS[field.ref] ? KINDS[field.ref].title.toLowerCase() : field.ref} '${name}'`);
    return;
  }
  const [, opts] = splitComponent(value, comp);
  checkOptions(comp, opts, `${path}.${name}`, issues);
}

function checkOptions(comp, opts, path, issues) {
  const byName = Object.fromEntries(comp.fields.map((f) => [f.name, f]));
  for (const [key, val] of Object.entries(opts)) {
    const f = byName[key];
    if (key === "_positional") {
      issues.push(`${path}: '${comp.name}' has no short form`);
      continue;
    }
    if (!f) {
      if (!comp.extraFields) issues.push(`${path}: unknown option '${key}'`);
      continue;
    }
    if (f.choices && !f.choices.includes(val)) issues.push(`${path}.${key}: expected one of ${f.choices.join(", ")}`);
    if (typeof val === "string" && /^<.*>$/.test(val)) issues.push(`${path}.${key}: replace the placeholder`);
    if (f.ref) checkSlot(val, f, `${path}.${key}`, issues);
  }
  for (const f of comp.fields) {
    if (f.required && !(f.name in opts)) issues.push(`${path}: missing required option '${f.name}'`);
  }
}

function validate(config) {
  const issues = [];
  checkOptions(TOP, config, "config", issues);
  return issues.map((i) => i.replace(/^config\./, ""));
}

// ---------------------------------------------------------------------------
// Builder state
// ---------------------------------------------------------------------------

const Builder = {
  config: null,

  init() {
    const saved = storage((s) => s.getItem(STORAGE_KEY), null);
    try {
      this.config = saved ? JSON.parse(saved) : null;
    } catch (e) {
      this.config = null;
    }
    if (!isPlainObject(this.config)) this.reset(false);
    this.render();
  },

  reset(render = true) {
    const checkpoint = TOP.fields.find((f) => f.name === "checkpoint");
    this.config = { checkpoint: checkpoint ? placeholder(checkpoint) : "<checkpoint>" };
    if (render) this.changed("Builder reset");
  },

  changed(message) {
    storage((s) => s.setItem(STORAGE_KEY, JSON.stringify(this.config)));
    this.render();
    if (message) toast(message);
  },

  setTop(key, value) {
    // Keep keys in the order of the top-level reference so the file reads naturally.
    const order = TOP.fields.map((f) => f.name);
    const next = { ...this.config, [key]: value };
    const sorted = {};
    for (const k of order) if (k in next) sorted[k] = next[k];
    for (const k of Object.keys(next)) if (!(k in sorted)) sorted[k] = next[k];
    this.config = sorted;
  },

  addComponent(comp) {
    const slot = KINDS[comp.kind].slot;
    const value = minimalConfig(comp);
    if (comp.kind === "clusters") {
      const runner = this.config.runner;
      const [rname, ropts] = runner ? splitComponent(runner, findComponent("runners", splitComponent(runner)[0] || "")) : [null, {}];
      const opts = rname === "parallel" ? { ...ropts } : rname && rname !== "auto" ? { base_runner: rname } : {};
      opts.cluster = value;
      this.setTop("runner", { parallel: opts });
      return this.changed(`Set runner.parallel.cluster to ${comp.name}`);
    }
    if (comp.kind === "grib_templates") {
      const out = this.config.output;
      const [oname, oopts] = out ? splitComponent(out, findComponent("outputs", splitComponent(out)[0] || "")) : [null, null];
      if (oname !== "grib") return toast("Add a grib output first");
      const templates = Array.isArray(oopts.templates) ? oopts.templates : oopts.templates ? [oopts.templates] : [];
      oopts.templates = [...templates, value];
      this.setTop("output", { grib: oopts });
      return this.changed(`Added ${comp.name} to output.grib.templates`);
    }
    if (!slot) return toast("This component cannot be added directly");
    if (LIST_SLOTS.has(slot)) {
      const list = Array.isArray(this.config[slot]) ? this.config[slot] : [];
      this.setTop(slot, [...list, value]);
      return this.changed(`Appended ${comp.name} to ${slot}`);
    }
    this.setTop(slot, value);
    this.changed(`Set ${slot} to ${comp.name}`);
  },

  // Add one option to the component already placed in its slot.
  addField(comp, field) {
    if (comp.kind === "config") {
      this.setTop(field.name, placeholder(field));
      return this.changed(`Set ${field.name}`);
    }
    const slot = KINDS[comp.kind].slot;
    if (!slot || !(slot in this.config)) {
      this.addComponent(comp);
      if (!slot) return;
    }
    let holder = this.config[slot];
    if (looksPerDataset(holder, comp.kind) && !Array.isArray(holder)) {
      return toast(`${slot} is set per dataset: edit the downloaded file instead`);
    }
    let index = null;
    if (LIST_SLOTS.has(slot)) {
      index = holder.map((v) => norm(splitComponent(v)[0])).lastIndexOf(norm(comp.name));
      if (index < 0) {
        this.addComponent(comp);
        holder = this.config[slot];
        index = holder.length - 1;
      }
    }
    const current = index === null ? holder : holder[index];
    let [name, opts] = splitComponent(current, comp);
    if (!name || norm(name) !== norm(comp.name)) {
      this.addComponent(comp);
      return this.addField(comp, field);
    }
    opts = { ...opts, [field.name]: placeholder(field) };
    const value = { [name]: opts };
    if (index === null) this.setTop(slot, value);
    else {
      const list = [...holder];
      list[index] = value;
      this.setTop(slot, list);
    }
    this.changed(`Set ${slot}.${name}.${field.name}`);
  },

  addExample(ex) {
    for (const [key, value] of Object.entries(clone(ex.data))) {
      if (key === "checkpoint" && this.config.checkpoint && !/^<.*>$|^\/path\/to\//.test(String(this.config.checkpoint))) continue;
      if (LIST_SLOTS.has(key) && Array.isArray(value) && Array.isArray(this.config[key])) {
        this.setTop(key, [...this.config[key], ...value]);
      } else {
        this.setTop(key, value);
      }
    }
    this.changed(`Merged "${ex.title}"`);
  },

  remove(key) {
    const next = { ...this.config };
    delete next[key];
    this.config = next;
    this.changed(`Removed ${key}`);
  },

  // Edits made in the builder form. Paths are lists of keys and list indices from the root.
  setValue(path, value) {
    const parent = getAt(this.config, path.slice(0, -1));
    parent[path[path.length - 1]] = value;
    this.save();
    this.renderOutput();
  },

  removePath(path) {
    if (path.length === 1) return this.remove(path[0]);
    const parent = getAt(this.config, path.slice(0, -1));
    const key = path[path.length - 1];
    if (Array.isArray(parent)) parent.splice(key, 1);
    else delete parent[key];
    this.changed();
  },

  addItem(path) {
    const list = getAt(this.config, path);
    const last = list.length ? list[list.length - 1] : "";
    list.push(isPlainObject(last) || Array.isArray(last) ? "" : typeof last === "string" ? "" : last);
    this.changed();
  },

  save() {
    storage((s) => s.setItem(STORAGE_KEY, JSON.stringify(this.config)));
  },

  yaml() {
    return toYaml(this.config) + "\n";
  },

  render() {
    $("#builder-form").innerHTML = renderForm(this.config);
    const missing = TOP.fields.filter((f) => !(f.name in this.config));
    $("#builder-add-top").innerHTML =
      `<option value="">Add a top-level option…</option>` +
      missing.map((f) => `<option value="${esc(f.name)}">${esc(f.name)}</option>`).join("");
    this.renderOutput();
  },

  // Refresh everything except the form, so typing keeps focus.
  renderOutput() {
    $("#builder-count").textContent = Object.keys(this.config).length;
    $("#builder-yaml").innerHTML = highlightYaml(this.yaml());
    const issues = validate(this.config);
    $("#builder-issues").innerHTML = issues.length
      ? `<ul>${issues.map((i) => `<li>${esc(i)}</li>`).join("")}</ul>`
      : `<span class="ok">No problems found by the page. Run <code>anemoi-inference validate</code> for a full check.</span>`;
  },
};

// ---------------------------------------------------------------------------
// Builder form
// ---------------------------------------------------------------------------

function getAt(root, path) {
  return path.reduce((node, key) => node[key], root);
}

const isPlaceholder = (v) => typeof v === "string" && /^<.*>$/.test(v);

// Read what the user typed the way YAML would, but keep values such as "0079" as strings.
function parseScalar(text, previous) {
  const t = text.trim();
  if (t === "" && isPlaceholder(previous)) return previous;
  if (typeof previous === "string" && !isPlaceholder(previous) && !/^(true|false|null|~)$|^-?\d/.test(previous)) {
    if (!/^(true|false|null|~)$/i.test(t) && !/^-?(0|[1-9]\d*)(\.\d+)?([eE][-+]?\d+)?$/.test(t)) return text;
  }
  if (/^(true|false)$/i.test(t)) return t.toLowerCase() === "true";
  if (/^(null|~)$/i.test(t)) return null;
  if (/^-?(0|[1-9]\d*)(\.\d+)?([eE][-+]?\d+)?$/.test(t)) return Number(t);
  if (/^\.nan$/i.test(t)) return NaN;
  return text;
}

function scalarText(v) {
  if (v === null || v === undefined) return "null";
  if (typeof v === "number" && Number.isNaN(v)) return ".nan";
  return String(v);
}

const pathAttr = (path) => esc(JSON.stringify(path));

function removeButton(path, label) {
  return `<button class="icon-btn small" data-remove-path="${pathAttr(path)}" aria-label="Remove ${esc(label)}" title="Remove">&times;</button>`;
}

function scalarInput(value, path, label) {
  if (typeof value === "boolean") {
    return `<label class="check"><input type="checkbox" data-path="${pathAttr(path)}" ${value ? "checked" : ""}> ${value}</label>`;
  }
  const ph = isPlaceholder(value);
  return `<input type="text" class="${ph ? "ph" : ""}" data-path="${pathAttr(path)}" aria-label="${esc(label)}"
    value="${ph ? "" : esc(scalarText(value))}" placeholder="${ph ? esc(value) : ""}" spellcheck="false">`;
}

function renderEntry(label, value, path) {
  const isList = Array.isArray(value);
  if (isPlainObject(value) || isList) {
    const empty = isList ? !value.length : !Object.keys(value).length;
    const body = empty ? `<div class="fempty">${isList ? "empty list" : "empty mapping"}</div>` : renderChildren(value, path);
    const add = isList && value.every((v) => !isPlainObject(v) && !Array.isArray(v))
      ? `<button class="btn small" data-add-item="${pathAttr(path)}">+ item</button>`
      : "";
    return `<div class="fgroup"><div class="fhead"><span class="fkey">${esc(label)}</span>${add}${removeButton(path, label)}</div>
      <div class="fbody">${body}</div></div>`;
  }
  return `<div class="frow"><span class="fkey">${esc(label)}</span>${scalarInput(value, path, label)}${removeButton(path, label)}</div>`;
}

function renderChildren(value, path) {
  if (Array.isArray(value)) return value.map((v, i) => renderEntry("-", v, [...path, i])).join("");
  return Object.entries(value)
    .map(([k, v]) => renderEntry(k, v, [...path, k]))
    .join("");
}

function renderForm(config) {
  return Object.keys(config).length ? renderChildren(config, []) : `<p class="hint">Nothing yet.</p>`;
}

function download(text, filename) {
  const blob = new Blob([text], { type: "application/x-yaml" });
  const url = URL.createObjectURL(blob);
  const a = document.createElement("a");
  a.href = url;
  a.download = filename;
  document.body.appendChild(a);
  a.click();
  a.remove();
  setTimeout(() => URL.revokeObjectURL(url), 1000);
}

function copy(text) {
  const done = () => toast("Copied");
  if (navigator.clipboard && window.isSecureContext) {
    navigator.clipboard.writeText(text).then(done, () => fallbackCopy(text, done));
  } else fallbackCopy(text, done);
}

function fallbackCopy(text, done) {
  const ta = document.createElement("textarea");
  ta.value = text;
  ta.style.position = "fixed";
  ta.style.opacity = "0";
  document.body.appendChild(ta);
  ta.select();
  try {
    document.execCommand("copy");
    done();
  } catch (e) {
    toast("Copy failed");
  }
  ta.remove();
}

// ---------------------------------------------------------------------------
// Rendering
// ---------------------------------------------------------------------------

function kindLink(kind) {
  const k = KINDS[kind];
  return k ? `<a href="#/${kind}">${esc(k.title)}</a>` : esc(kind);
}

function compLink(id) {
  const c = COMPONENTS[id];
  if (!c) return esc(id);
  return `<a href="#/${c.kind}/${encodeURIComponent(c.name)}"><code>${esc(c.name)}</code></a>`;
}

function renderSidebar(filter = "") {
  const q = filter.trim().toLowerCase();
  const route = location.hash.slice(2);
  const parts = [];
  for (const kind of DATA.kinds) {
    const items = kind.components
      .map((name) => COMPONENTS[`${kind.id}/${name}`])
      .filter((c) => kind.id !== "config")
      .filter((c) => !q || matches(c, q));
    const showKind = !q || items.length || (kind.id === "config" && matches(TOP, q));
    if (!showKind) continue;
    const active = route === kind.id ? " active" : "";
    parts.push(
      `<div class="group"><a class="${active}" href="#/${kind.id}">${esc(kind.title)}</a><ul>` +
        items
          .map((c) => {
            const cls = route === `${c.kind}/${c.name}` ? "active" : "";
            return `<li><a class="${cls}" href="#/${c.kind}/${encodeURIComponent(c.name)}">${esc(c.name)}</a></li>`;
          })
          .join("") +
        `</ul></div>`
    );
  }
  const exActive = route.startsWith("examples") ? " active" : "";
  parts.push(`<div class="group"><a class="${exActive}" href="#/examples">Examples</a></div>`);
  $("#sidebar").innerHTML = parts.join("") || `<p class="empty">No matches.</p>`;
}

function matches(comp, q) {
  if (comp.name.toLowerCase().includes(q) || comp.title.toLowerCase().includes(q)) return true;
  if (comp.summary.toLowerCase().includes(q)) return true;
  return comp.fields.some((f) => f.name.toLowerCase().includes(q));
}

function renderHome() {
  const cards = DATA.kinds
    .map((k) => {
      const n = k.id === "config" ? `${TOP.fields.length} keys` : `${k.components.length} components`;
      return `<a class="card" href="#/${k.id}"><h3>${esc(k.title)}</h3><p>${k.summary}</p><div class="meta">${n}</div></a>`;
    })
    .join("");
  return `
    <h1>Configuration reference</h1>
    <div class="prose">${DATA.intro}</div>
    <div class="cards">${cards}
      <a class="card" href="#/examples"><h3>Examples</h3><p>Complete configurations you can copy or merge into the builder.</p>
      <div class="meta">${DATA.examples.length} examples</div></a>
    </div>`;
}

function renderKind(kindId) {
  const kind = KINDS[kindId];
  if (kindId === "config") return renderComponent(TOP, true);
  const rows = kind.components
    .map((name) => COMPONENTS[`${kindId}/${name}`])
    .map(
      (c) =>
        `<a href="#/${c.kind}/${encodeURIComponent(c.name)}"><span class="n">${esc(c.name)}${statusBadge(c)}</span><span class="s">${c.summary}</span></a>`
    )
    .join("");
  return `
    <div class="crumbs"><a href="#/">Reference</a></div>
    <h1>${esc(kind.title)}</h1>
    <p class="lead">${kind.summary}</p>
    <div class="prose">${kind.description}</div>
    <div class="comp-list">${rows}</div>`;
}

const STATUS_BADGES = {
  draft: ["auto-generated", "Generated automatically. It may contain errors or be simplistic."],
  "AI-reviewed": ["ai", "Written by an AI and not yet reviewed by a person. It may contain errors."],
};

const STATUS_NOTES = {
  draft: "This page was generated automatically. It may contain errors or be simplistic.",
  "AI-reviewed": "This page was written by an AI and has not yet been reviewed by a person. It may contain errors.",
};

function statusBadge(comp) {
  const badge = STATUS_BADGES[comp.status];
  return badge ? `<span class="badge ${badge[0]}" title="${esc(badge[1])}">${esc(badge[0] === "ai" ? "AI-written" : badge[0])}</span>` : "";
}

function fieldRow(comp, f) {
  const badges = [];
  if (f.required) badges.push('<span class="badge req">required</span>');
  if (f.perDataset) badges.push('<span class="badge" title="May be given per dataset for multi-dataset checkpoints">per dataset</span>');
  let type = esc(f.type || "");
  if (f.ref) {
    const what = f.many ? `list of ${kindLink(f.ref)}` : kindLink(f.ref);
    type = `${what}${type ? `<br>${type}` : ""}`;
  }
  let def = "";
  if (f.defaultText) def = `Default: ${esc(f.defaultText)}`;
  else if (f.hasDefault) def = `Default: <code>${esc(toYaml(f.default).trim())}</code>`;
  const choices = f.choices ? `<div class="default">One of: ${f.choices.map((c) => `<code>${esc(c)}</code>`).join(" ")}</div>` : "";
  const canAdd = comp.kind === "config" || KINDS[comp.kind].slot;
  const add = canAdd
    ? `<div class="add"><button class="btn small" data-add-field="${esc(f.name)}">Add to config</button></div>`
    : "";
  return `<tr id="field-${esc(f.name)}">
    <td class="name"><code>${esc(f.name)}</code>${badges.join("")}</td>
    <td><div class="desc">${f.description}</div>${def ? `<div class="default">${def}</div>` : ""}${choices}
      <div class="type">${type}</div>${add}</td>
  </tr>`;
}

function fieldTable(comp, fields) {
  return `<table class="fields"><thead><tr><th>Option</th><th>Description</th></tr></thead>
    <tbody>${fields.map((f) => fieldRow(comp, f)).join("")}</tbody></table>`;
}

function exampleCard(ex, current) {
  const tags = ex.components
    .filter((id) => id !== current)
    .map((id) => compLink(id))
    .join("");
  return `<div class="example" id="ex-${esc(ex.id)}">
    <div class="head"><h3>${esc(ex.title)}</h3>
      <div class="actions">
        <button class="btn small" data-copy-example="${esc(ex.id)}">Copy</button>
        <button class="btn small primary" data-add-example="${esc(ex.id)}">Add to config</button>
      </div></div>
    <div class="desc">${ex.description}</div>
    ${ex.generated ? '<p class="note">Generated automatically. It may contain errors or be simplistic.</p>' : ""}
    ${ex.html}
    ${tags ? `<div class="tags">Also uses: ${tags}</div>` : ""}
  </div>`;
}

// Place a value at a path such as "runner.parallel.cluster" or "post_processors[]".
function wrapAt(path, value) {
  return path
    .split(".")
    .reverse()
    .reduce((inner, part) => (part.endsWith("[]") ? { [part.slice(0, -2)]: [inner] } : { [part]: inner }), value);
}

function shortForms(comp) {
  if (comp.kind === "config") return "";
  const path = KINDS[comp.kind].path || KINDS[comp.kind].slot || "...";
  const forms = [];
  const minimal = minimalConfig(comp);
  forms.push(toYaml(wrapAt(path, minimal)));
  if (comp.positional && typeof minimal === "string") {
    const f = comp.fields.find((x) => x.name === comp.positional);
    if (f && !f.many) forms.push(toYaml(wrapAt(path, { [comp.name]: placeholder(f) })));
  }
  const where = "";
  return `<h2>Minimal configuration</h2>${where}${forms
    .map((y) => `<div class="snippet"><pre>${highlightYaml(y)}</pre></div>`)
    .join('<p class="muted" style="margin:6px 0">or, using the short form:</p>')}`;
}

function renderComponent(comp, isTop = false) {
  const own = comp.fields.filter((f) => !f.inheritedFrom);
  const groups = {};
  for (const f of comp.fields.filter((x) => x.inheritedFrom)) (groups[f.inheritedFrom] ||= []).push(f);
  const inherited = Object.entries(groups)
    .map(
      ([from, fields]) => `<details class="field-group" open><summary>Options shared by all ${
        KINDS[from.split("/")[0]] ? esc(KINDS[from.split("/")[0]].title.toLowerCase()) : esc(from)
      }</summary>${fieldTable(comp, fields)}</details>`
    )
    .join("");
  const ids = isTop ? DATA.examples.filter((e) => e.id.startsWith("config/")).map((e) => e.id) : comp.examples;
  const examples = ids.map((id) => EXAMPLES[id]).filter(Boolean);
  const canAdd = !isTop && (KINDS[comp.kind].slot || comp.kind === "clusters" || comp.kind === "grib_templates");
  const aliases = comp.aliases.length
    ? `<p class="muted">Deprecated names: ${comp.aliases.map((a) => `<code>${esc(a)}</code>`).join(" ")}</p>`
    : "";
  const extra = comp.extraFields ? `<div class="callout">${comp.extraFields}</div>` : "";
  const seeAlso = comp.seeAlso.length
    ? `<h2>See also</h2><ul>${comp.seeAlso
        .map((s) => (COMPONENTS[s] ? `<li>${compLink(s)}</li>` : `<li>${esc(s)}</li>`))
        .join("")}</ul>`
    : "";
  return `
    <div class="crumbs"><a href="#/">Reference</a>${isTop ? "" : ` › ${kindLink(comp.kind)}`}</div>
    <h1>${isTop ? "Top-level options" : esc(comp.name)}${statusBadge(comp)}</h1>
    <p class="lead">${comp.summary}</p>
    ${STATUS_NOTES[comp.status] ? `<p class="note">${esc(STATUS_NOTES[comp.status])}</p>` : ""}
    ${aliases}
    ${canAdd ? `<div class="actions"><button class="btn primary" data-add-component="${esc(comp.id)}">Add to config</button></div>` : ""}
    <div class="prose">${comp.description}</div>
    ${shortForms(comp)}
    <h2>Options</h2>
    ${own.length ? fieldTable(comp, own) : '<p class="muted">This component has no options of its own.</p>'}
    ${inherited}
    ${extra}
    ${comp.notes ? `<h2>Notes</h2><div class="prose">${comp.notes}</div>` : ""}
    <h2>Examples</h2>
    ${examples.length ? examples.map((e) => exampleCard(e, comp.id)).join("") : '<p class="muted">No examples yet.</p>'}
    ${seeAlso}`;
}

function renderExamples() {
  const byKind = {};
  for (const ex of DATA.examples) (byKind[ex.id.split("/")[0]] ||= []).push(ex);
  const sections = Object.entries(byKind)
    .map(([kind, list]) => `<h2>${KINDS[kind] ? esc(KINDS[kind].title) : esc(kind)}</h2>${list.map((e) => exampleCard(e, null)).join("")}`)
    .join("");
  return `<div class="crumbs"><a href="#/">Reference</a></div><h1>Examples</h1>
    <p class="lead">Every example is a complete configuration.</p>${sections}`;
}

function currentComponent() {
  const [kind, name] = location.hash.slice(2).split("/");
  if (kind === "config") return TOP;
  return name ? COMPONENTS[`${kind}/${decodeURIComponent(name)}`] : null;
}

function route() {
  const path = location.hash.replace(/^#\/?/, "");
  const [kind, name] = path.split("/");
  let html;
  if (!path) html = renderHome();
  else if (kind === "examples") html = renderExamples();
  else if (KINDS[kind] && !name) html = renderKind(kind);
  else if (KINDS[kind] && COMPONENTS[`${kind}/${decodeURIComponent(name)}`])
    html = renderComponent(COMPONENTS[`${kind}/${decodeURIComponent(name)}`]);
  else html = `<h1>Not found</h1><p><a href="#/">Back to the reference</a></p>`;
  $("#main").innerHTML = html;
  renderSidebar($("#search").value);
  document.body.classList.remove("nav-open");
  window.scrollTo(0, 0);
}

// ---------------------------------------------------------------------------
// Events
// ---------------------------------------------------------------------------

function setBuilderOpen(open) {
  document.body.classList.toggle("builder-open", open);
  $("#builder-toggle").setAttribute("aria-expanded", String(open));
  storage((s) => s.setItem(`${STORAGE_KEY}-open`, open ? "1" : "0"));
}

function initTheme() {
  const saved = storage((s) => s.getItem(`${STORAGE_KEY}-theme`), null);
  if (saved) document.documentElement.dataset.theme = saved;
  $("#theme-toggle").addEventListener("click", () => {
    const dark =
      document.documentElement.dataset.theme === "dark" ||
      (!document.documentElement.dataset.theme && window.matchMedia("(prefers-color-scheme: dark)").matches);
    const next = dark ? "light" : "dark";
    document.documentElement.dataset.theme = next;
    storage((s) => s.setItem(`${STORAGE_KEY}-theme`, next));
  });
}

document.addEventListener("click", (e) => {
  const t = e.target.closest("button");
  if (!t) return;
  if (t.dataset.addComponent) {
    Builder.addComponent(COMPONENTS[t.dataset.addComponent]);
    setBuilderOpen(true);
  } else if (t.dataset.addField) {
    const comp = currentComponent();
    const field = comp && comp.fields.find((f) => f.name === t.dataset.addField);
    if (field) {
      Builder.addField(comp, field);
      setBuilderOpen(true);
    }
  } else if (t.dataset.addExample) {
    Builder.addExample(EXAMPLES[t.dataset.addExample]);
    setBuilderOpen(true);
  } else if (t.dataset.copyExample) {
    copy(EXAMPLES[t.dataset.copyExample].text);
  } else if (t.dataset.removePath) {
    Builder.removePath(JSON.parse(t.dataset.removePath));
  } else if (t.dataset.addItem) {
    Builder.addItem(JSON.parse(t.dataset.addItem));
  } else if (t.dataset.tab) {
    for (const tab of document.querySelectorAll("[data-tab]")) {
      tab.classList.toggle("active", tab === t);
      tab.setAttribute("aria-selected", String(tab === t));
    }
    $("#builder-form").hidden = t.dataset.tab !== "form";
    $("#builder-yaml-wrap").hidden = t.dataset.tab !== "yaml";
  }
});

$("#builder-form").addEventListener("input", (e) => {
  const el = e.target;
  if (!el.dataset.path || el.type === "checkbox") return;
  const path = JSON.parse(el.dataset.path);
  const value = parseScalar(el.value, getAt(Builder.config, path));
  el.classList.toggle("ph", isPlaceholder(value));
  Builder.setValue(path, value);
});

$("#builder-form").addEventListener("change", (e) => {
  const el = e.target;
  if (el.type !== "checkbox" || !el.dataset.path) return;
  Builder.setValue(JSON.parse(el.dataset.path), el.checked);
  el.parentElement.lastChild.textContent = ` ${el.checked}`;
});

$("#builder-add-top").addEventListener("change", (e) => {
  const field = TOP.fields.find((f) => f.name === e.target.value);
  if (field) Builder.addField(TOP, field);
});

$("#builder-toggle").addEventListener("click", () => setBuilderOpen(!document.body.classList.contains("builder-open")));
$("#builder-close").addEventListener("click", () => setBuilderOpen(false));
$("#builder-download").addEventListener("click", () => download(Builder.yaml(), "config.yaml"));
$("#builder-copy").addEventListener("click", () => copy(Builder.yaml()));
$("#builder-clear").addEventListener("click", () => Builder.reset());
$("#nav-toggle").addEventListener("click", () => {
  const open = !document.body.classList.contains("nav-open");
  document.body.classList.toggle("nav-open", open);
  $("#nav-toggle").setAttribute("aria-expanded", String(open));
});
$("#search").addEventListener("input", (e) => renderSidebar(e.target.value));
window.addEventListener("hashchange", route);

initTheme();
Builder.init();
setBuilderOpen(storage((s) => s.getItem(`${STORAGE_KEY}-open`) === "1", false));
route();
