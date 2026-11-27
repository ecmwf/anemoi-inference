# Configuration reference

A static HTML page that documents every key of an anemoi-inference run configuration. It has a builder to
assemble a `config.yaml` and download it. The page is generated from the YAML files in this directory.

```
index.yaml                       page title, kinds, their registries, and hints for the generator
components/<kind>/<name>.yaml    one spec per component (fields, short form, notes)
components/<kind>/_<set>.yaml    fields shared by several components (via `inherits`)
components/config/run.yaml       top-level keys
examples/<kind>/*.yaml           hand-written, complete run configurations
site/                            page template, CSS and JS, identical in every anemoi package
refgen.py                        generator, checker and builder
```

Other anemoi packages use the same layout, formats and page. Neither `refgen.py` nor `site/` needs to be read to
document a component: edit the YAML files. Changes to `site/` must be copied to every package that has a
`docs/config_reference/`.

## Commands

Run from the package root with the development environment:

```bash
python docs/config_reference/refgen.py generate      # refresh draft specs from the code
python docs/config_reference/refgen.py check         # compare specs and examples with the code
python docs/config_reference/refgen.py build -o out  # write out/index.html
python docs/config_reference/refgen.py introspect outputs grib   # what the code says about one component
```

`tests/unit/test_docs.py` runs the same checks. It also fails if `generate` would change any file.

## Workflow

- **New or changed component:** run `generate`. Specs with `status: draft` are rewritten from the constructor
  signature and NumPy docstring, so better docstrings give better docs.
- **Curating a spec:** change `status` from `draft` first, then edit by hand. Use `AI-reviewed` for specs curated by
  an AI agent and `reviewed` once a person has checked them. Neither is overwritten. The checker still fails if they
  drift from the code, and it requires a description for every field. The page shows a badge for `draft` and
  `AI-reviewed` specs.
- **Examples:** add a complete configuration under `examples/<kind>/`. The first comment line is the title and the
  following comment lines are the description. Components without an example show only their minimal
  configuration.
- **Fields that take other components:** add them to `field_refs` in `index.yaml`, so the page links them and the
  checker validates their contents.
- **Where components go:** a kind's `slot` (top-level key) or `path` in `index.yaml` tells the builder where to put
  them. `path` segments alternate key and component name, and `[]` marks a list: `output.grib.templates[]`.

## Spec format

```yaml
name: grib                  # as written in YAML; underscores and hyphens are equivalent
status: draft | AI-reviewed | reviewed
summary: One sentence.
description: Markdown (paragraphs, lists, `code`, **bold**, [links](url), fenced code).
positional: path            # option set by the short form `grib: out.grib`
inherits: [outputs/_common]
fields:
  path:
    type: path              # free text shown to readers
    required: true
    default: ...            # or default_text: "same as `input`"
    description: Markdown.
    example: forecast.grib  # used by the builder
    choices: [a, b]
    ref: outputs            # value is a component of this kind
    many: true              # ... or a list of them
    per_dataset: true       # may be keyed by dataset name
hidden: [purpose]           # constructor parameters that are not user-facing
extra_fields: Markdown describing what unknown keys do, when the code accepts them.
aliases: [old-name]         # registry aliases
notes: Markdown.
see_also: [outputs/tee]
```

## Sphinx and Read the Docs

The page is built with the Sphinx docs and needs no extra steps on Read the Docs.

- `docs/conf.py` connects a `build-finished` handler. After an HTML build, it loads `refgen.py` and writes the page
  to `<html output>/config-reference/index.html`. Other builders, and builds that failed, are skipped.
- `docs/inference/configs/reference.rst` links to the page (`../../config-reference/index.html`) and is listed in the
  User Guide toctree in `docs/index.rst`.
- `config_reference` is in `exclude_patterns`, so Sphinx does not read this directory as documentation source.

The page is one self-contained file with relative hash links. It works from `file://`, from any static host and
under a Read the Docs version prefix.

To build locally, run the normal sphinx build command, then open
`docs/_build/html/config-reference/index.html`.

Read the Docs installs the package with the `docs` extra, so `refgen.py` can import the registries. Its only other
dependencies are `pyyaml` and `pygments`, which Sphinx already pulls in. If a component's module fails to import,
the page still builds, but the CI checks report the failure.

The user guide pages for top-level options, inputs, outputs, processors, GRIB output and parallel inference link to
the matching section or component with relative hash links such as `../../config-reference/index.html#/outputs/grib`.
