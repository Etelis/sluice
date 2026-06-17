# Contributing to Sluice

Thanks for your interest! Sluice is a small, focused vLLM plugin; contributions
that keep it that way are very welcome.

## Development setup

```bash
git clone https://github.com/Etelis/sluice
cd sluice
pip install -e ".[dev]"      # into an environment that already has vLLM
```

Sluice has no runtime dependency on a *specific* vLLM version pinned in
`pyproject.toml`, but it does monkeypatch an internal factory and rely on the
modular `quant_method.apply(layer, …, topk_ids=…)` signature. Develop against a
vLLM build you can run end to end, and note the version in any PR.

## Before opening a PR

```bash
ruff check src examples       # lint (line length 88)
ruff format --check src examples
python -m py_compile src/sluice/*.py
```

If you have GPUs, the correctness gate is the bit-compare:

```bash
python examples/bitcompare_v2lite.py > baseline.txt
SLUICE_SLOTS=64 python examples/bitcompare_v2lite.py > sluice.txt
diff baseline.txt sluice.txt && echo "BIT-IDENTICAL"
```

Offloaded greedy decode must be **bit-identical** to the resident baseline. Test
both `SLUICE_SLOTS=64` (fits-all) and a small value (LRU/eviction path).

## Scope and style

- Match vLLM's style; keep code self-documenting and comments minimal.
- The offloader must stay backend-agnostic — it only manages `param.data` and
  `expert_map`. See [docs/ARCHITECTURE.md](docs/ARCHITECTURE.md).
- High-value directions: cross-layer expert prefetch, offloader-aware KV
  profiling, and upstreaming a generic `on_experts_selected` hook so the
  monkeypatch is no longer needed.

## Regenerating the icon

```bash
python assets/make_icon.py    # writes assets/sluice-icon.png + sluice-social.png
```

By contributing you agree your work is licensed under Apache-2.0, and you sign
off your commits (`git commit -s`, Developer Certificate of Origin).
