Every image in this folder is generated from real project data by
[`scripts/make_docs_assets.py`](../../scripts/make_docs_assets.py) — none of it
is hand-drawn or illustrative. Regenerate everything with:

```bash
pixi run -e ml python scripts/make_docs_assets.py
```

or a single asset with `--only {architecture,worked-example,regional-map,model-lineage}`.
See the script's own docstring for full usage, including targeting a different
example cell or region.
