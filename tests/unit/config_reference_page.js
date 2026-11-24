// (C) Copyright 2026 Anemoi contributors.
//
// This software is licensed under the terms of the Apache Licence Version 2.0
// which can be obtained at http://www.apache.org/licenses/LICENSE-2.0.
//
// In applying this licence, ECMWF does not waive the privileges and immunities
// granted to it by virtue of its status as an intergovernmental organisation
// nor does it submit to any jurisdiction.

// Exercise the configuration reference page under Node with a minimal DOM stub.
// Usage: node config_reference_page.js <index.html>
// Prints JSON: {errors: [...], yaml: [{data, yaml}], parse: [...]} for test_config_reference.py to check.

"use strict";

const fs = require("fs");
const vm = require("vm");

const html = fs.readFileSync(process.argv[2], "utf8");
const script = html.match(/<script>([\s\S]*)<\/script>/)[1];

function makeElement(id) {
  const classes = new Set();
  return {
    id,
    innerHTML: "",
    textContent: "",
    value: "",
    hidden: false,
    dataset: {},
    style: {},
    listeners: {},
    classList: {
      add: (c) => classes.add(c),
      remove: (c) => classes.delete(c),
      contains: (c) => classes.has(c),
      toggle: (c, force) => ((force === undefined ? !classes.has(c) : force) ? classes.add(c) : classes.delete(c)),
    },
    setAttribute() {},
    addEventListener(type, fn) {
      this.listeners[type] = fn;
    },
    appendChild() {},
    remove() {},
    click() {},
    select() {},
  };
}

const elements = {};
const documentListeners = {};
const store = {};
const context = {
  document: {
    body: makeElement("body"),
    documentElement: makeElement("html"),
    querySelector: (sel) => (elements[sel] ||= makeElement(sel)),
    querySelectorAll: () => [],
    addEventListener: (type, fn) => (documentListeners[type] = fn),
    createElement: () => makeElement("created"),
    execCommand() {},
  },
  window: {
    localStorage: {
      getItem: (k) => (k in store ? store[k] : null),
      setItem: (k, v) => (store[k] = String(v)),
    },
    matchMedia: () => ({ matches: false }),
    addEventListener() {},
    scrollTo() {},
    isSecureContext: false,
  },
  location: { hash: "" },
  navigator: {},
  setTimeout: () => 0,
  clearTimeout: () => {},
  Blob: function () {},
  URL: { createObjectURL: () => "blob:", revokeObjectURL() {} },
  console,
};
vm.createContext(context);
vm.runInContext(script, context);

const checks = `
(() => {
  const errors = [];
  const yamls = [];
  const guard = (label, fn) => {
    try {
      const out = fn();
      if (typeof out === "string" && /undefined|\\[object Object\\]/.test(out)) errors.push(label + ": renders 'undefined' or '[object Object]'");
      return out;
    } catch (e) {
      errors.push(label + ": " + e.stack);
    }
  };

  // Every route renders.
  const routes = ["", "#/examples", ...DATA.kinds.map((k) => "#/" + k.id),
    ...Object.values(COMPONENTS).map((c) => "#/" + c.kind + "/" + encodeURIComponent(c.name))];
  for (const hash of routes) {
    location.hash = hash;
    guard("route " + hash, () => { route(); return $("#main").innerHTML; });
  }

  // The YAML emitter round-trips every example and some awkward values.
  const awkward = {
    s: ["", "true", "no", "null", "0079", "1e3", "-1", ".5", "*_50", "a: b", "a #b", "x:", "<path>", "- x", "{x}", "@x", " pad", "ok"],
    n: [0, 1.5, -3, true, false, null],
    nested: [{ a: [1, { b: [] }], c: {} }, [], [[1, 2], []]],
    "key with: colon": 1,
    "2t": "msl",
  };
  yamls.push({ data: awkward, yaml: toYaml(awkward) });
  for (const ex of DATA.examples) yamls.push({ data: ex.data, yaml: toYaml(ex.data) });

  // The builder accepts every component, option and example.
  Builder.reset(false);
  for (const c of Object.values(COMPONENTS)) {
    if (c === TOP) continue;
    guard("add " + c.id, () => Builder.addComponent(c));
    for (const f of c.fields) guard("add field " + c.id + "." + f.name, () => Builder.addField(c, f));
  }
  for (const f of TOP.fields) guard("add top " + f.name, () => Builder.addField(TOP, f));
  yamls.push({ data: Builder.config, yaml: Builder.yaml() });
  for (const ex of DATA.examples) {
    guard("example " + ex.id, () => Builder.addExample(ex));
    yamls.push({ data: Builder.config, yaml: Builder.yaml() });
  }
  guard("validate", () => validate(Builder.config).join("\\n"));
  guard("form", () => { Builder.render(); return $("#builder-form").innerHTML; });

  // Editing values through the form: required top-level keys, one optional key, and a component's short form.
  Builder.reset(false);
  const slotted = Object.values(COMPONENTS).find(
    (c) => c !== TOP && KINDS[c.kind].slot && !LIST_SLOTS.has(KINDS[c.kind].slot) && c.fields.some((f) => f.required)
  );
  if (slotted) Builder.addComponent(slotted);
  const optional = TOP.fields.find((f) => !f.required && !f.ref && !f.many && !f.nested);
  if (optional) Builder.addField(TOP, optional);
  Builder.render();
  const form = $("#builder-form");
  const input = (path, value) => form.listeners.input({ target: { dataset: { path: JSON.stringify(path) }, type: "text", value, classList: { toggle() {} } } });
  for (const f of TOP.fields.filter((x) => x.required && !isPlainObject(Builder.config[x.name]))) {
    guard("edit " + f.name, () => input([f.name], "/data/value"));
  }
  if (optional) guard("edit " + optional.name, () => input([optional.name], "72"));
  if (slotted && isPlainObject(Builder.config[KINDS[slotted.kind].slot])) {
    const slot = KINDS[slotted.kind].slot;
    guard("edit " + slot, () => input([slot, Object.keys(Builder.config[slot])[0]], "value"));
  }
  const edited = clone(Builder.config);
  if (optional) {
    guard("remove", () => Builder.removePath([optional.name]));
    if (optional.name in Builder.config) errors.push("removePath did not remove " + optional.name);
  }
  yamls.push({ data: Builder.config, yaml: Builder.yaml() });

  const parse = [
    ["72", 0], ["0079", "x"], ["true", "x"], ["null", 1], ["", "<path>"], ["abc", 1], ["1.5", "<x>"], ["-2", 0], ["1e3", 0],
  ].map(([text, previous]) => ({ text, previous, value: parseScalar(text, previous) }));

  return JSON.stringify({ errors, yaml: yamls, parse, edited });
})()
`;
process.stdout.write(vm.runInContext(checks, context));
