# Diagnostic scripts

These are the scratch tools used while developing this repository. They are kept
because they document how each conclusion was reached, but they are **not part
of the supported install** — `scripts/install.sh` does not use them.

The two supported diagnostics are in the parent directory:

- `diagnostics/diagnose.py` — check which of the three bugs affect your machine
- `diagnostics/benchmark_linear.py` — measure the GEMM anomaly directly

## What is here

| File | Purpose |
|---|---|
| `_ab_test.sh` | A/B the dequant cache on/off (produced the 296 s vs OOM result) |
| `_cost_split.sh` | Split per-layer cost into dequant / transpose / matmul |
| `_cost_model.sh` | Estimate per-forward cost from layer counts |
| `_cache_viable.sh` | Measure whether 13.25 GiB of cached weights fits |
| `_sim_cache.sh` | Simulate full-resident caching (265/265 tensors converted) |
| `_check_tag.sh` | Confirm weights arrive at `F.linear` tagged |
| `_instrument_live.sh` | Count fast vs slow `F.linear` calls in a live server |
| `_breakdown.sh` | Attention vs linear cost, token-count scaling |
| `_why_slow.sh` | Inspect which marker a live `ops.py` carries |
| `_reapply.sh` | Restore `ops.py` from backup and re-apply cleanly |
| `_install_repo_patch.sh` | Install the repo's patch exactly as `install.sh` does |
| `_restore.sh` | Roll a doubled-up VRAM patch back to a single application |
| `_validate.sh` | Syntax-check every script and run `install.sh --dry-run` |
| `_test_patches.sh` | Dry-run both source patches against a live install |

## The bug these caught

`_test_patches.sh` and `_install_repo_patch.sh` exist because of a real
regression: the shipped `ops.py` patch imported a module name
(`dspk_layout_marker`) that the repository does not contain, and
`patch_gguf_ops.py` treated the stale marker as "already patched, works as-is".

Result: the patch reported success, the weights were never tagged, and every
sampling step took **2185 seconds** instead of ~20. Silent no-ops are the worst
failure mode, so `patch_gguf_ops.py` now:

1. distinguishes the current marker from stale ones and **replaces** stale ones
2. runs a `_verify()` step that resolves the imported module path on disk and
   rolls back if it does not exist

Measure, do not assume — the patch "being applied" is not evidence that it works.
