"""Patch ComfyUI-GGUF so dequantized weights are tagged for the fast GEMM path.

WHAT IT CHANGES
===============

``ComfyUI-GGUF``'s ``GGMLOps.Linear.forward_ggml_cast_weights`` dequantizes the
weight on every call and hands it straight to ``F.linear``::

    def forward_ggml_cast_weights(self, input):
        weight, bias = self.cast_bias_weight(input)
        return torch.nn.functional.linear(input, weight, bias)

The weight arrives as ``[out_features, in_features]``, which is the transposed
layout that hipBLASLt handles ~400x slower on gfx1200 (see
``patches/gfx1200_layout_patch.py``).

This patch transposes it to ``[in_features, out_features]`` *once*, marks it,
and lets the patched ``F.linear`` use a plain ``matmul``.

MEMORY
======

The transposed buffer replaces the original (the local name is rebound and the
original becomes garbage), so steady-state memory is unchanged. This is
deliberate: keeping both copies would add ~13 GiB for a 7B DiT and trigger
``hipBLASLt OOM``.

Idempotent; writes a timestamped ``.bak`` beside the original.
"""

from __future__ import annotations

import argparse
import datetime
import os
import py_compile
import re
import shutil

MARKER = "GFX1200 layout fix"
# Marker used by the original hand-written patch; detected so that an install
# patched by an earlier version is not patched twice.
ALIASES = (MARKER, "DSPK gfx1200 layout fix")
OPS_REL = os.path.join("custom_nodes", "ComfyUI-GGUF", "ops.py")

OLD = """        def forward_ggml_cast_weights(self, input):
            weight, bias = self.cast_bias_weight(input)
            return torch.nn.functional.linear(input, weight, bias)"""

NEW = '''        def forward_ggml_cast_weights(self, input):
            weight, bias = self.cast_bias_weight(input)
            # --- gfx1200 layout fix ------------------------------------------
            # RDNA4 + ROCm 7.2 runs transposed-operand GEMMs ~400x slower, and
            # F.linear's [out, in] weight layout hits exactly that path. Store
            # the weight as [in, out] contiguous instead and let the patched
            # F.linear use torch.matmul.
            #
            # The transposed buffer replaces the original rather than being
            # added beside it, so memory is unchanged. Do NOT cache both.
            #
            # Optional extra: with GFX1200_CACHE_DEQUANT=1 the fp16 result is
            # kept across steps, removing the per-step dequantization cost
            # (~38 ms/layer, the dominant term). Costs ~13 GiB of VRAM.
            if (weight is not None and weight.dim() == 2
                    and weight.dtype in (torch.float16, torch.bfloat16)):
                try:
                    from gfx1200_layout_patch import mark_transposed as _mark
                    try:
                        from dequant_cache import enabled as _cache_on, \\
                            get_or_convert as _cached
                        if _cache_on():
                            weight = _cached(id(self), lambda: weight)
                        else:
                            weight = _mark(weight.t().contiguous())
                    except ImportError:
                        weight = _mark(weight.t().contiguous())
                except Exception:
                    pass
            # --- end gfx1200 layout fix --------------------------------------
            return torch.nn.functional.linear(input, weight, bias)'''


def find_ops(explicit: str | None) -> str:
    if explicit:
        return os.path.abspath(explicit)
    comfy = os.environ.get("COMFYUI_PATH", "/opt/ComfyUI")
    cand = os.path.join(comfy, OPS_REL)
    if os.path.isfile(cand):
        return os.path.abspath(cand)
    raise SystemExit(
        "ERROR: ComfyUI-GGUF/ops.py not found. "
        "Install it first, or pass --ops-path.")


def _already_patched(src: str) -> bool:
    """True only if the CURRENT marker is present.

    An older marker variant is NOT acceptable: the original hand-written patch
    imported a module named ``dspk_layout_marker`` which does not ship with this
    repository, so the import failed silently inside its own try/except and the
    weights were never tagged -- the patch appeared applied but did nothing.
    Detecting that case matters; treating it as "works as-is" is how a silent
    no-op ships.
    """
    return MARKER in src


def _strip_old_patch(src: str) -> str:
    """Remove any previously applied gfx1200 block so it can be re-applied."""
    for begin_tpl in ("            # --- DSPK gfx1200 layout fix",
                      "            # --- gfx1200 layout fix"):
        for end_tpl in ("            # --- end DSPK gfx1200 layout fix",
                        "            # --- end gfx1200 layout fix"):
            while True:
                i = src.find(begin_tpl)
                j = src.find(end_tpl)
                if i == -1 or j == -1:
                    break
                j = src.find("\n", j)
                if j == -1:
                    j = len(src)
                src = src[:i] + src[j + 1:]
    return src


def patch(ops_path: str, dry_run: bool = False) -> int:
    if not os.path.isfile(ops_path):
        print(f"ERROR: {ops_path} not found")
        return 1

    src = open(ops_path, encoding="utf-8").read()

    if _already_patched(src):
        print(f"already patched with the current marker: {ops_path}")
        return 0

    stale = [m for m in ALIASES if m in src]
    if stale:
        print(f"found stale patch ({stale[0]}); replacing it.")
        print("  reason: the old marker imported a module that is not shipped,")
        print("          so it silently did nothing.")
        src = _strip_old_patch(src)

    if OLD not in src:
        print("ERROR: could not find the expected forward_ggml_cast_weights block.")
        print("       The ComfyUI-GGUF version may differ from the one this")
        print("       patch was written against (see README for the tested commit).")
        return 1

    if dry_run:
        print(f"[dry-run] would patch {ops_path}")
        return 0

    stamp = datetime.datetime.now().strftime("%Y%m%d-%H%M%S")
    backup = f"{ops_path}.gfx1200-bak-{stamp}"
    shutil.copy2(ops_path, backup)
    open(ops_path, "w", encoding="utf-8").write(src.replace(OLD, NEW, 1))

    try:
        py_compile.compile(ops_path, doraise=True)
    except py_compile.PyCompileError as exc:
        shutil.copy2(backup, ops_path)
        print("ERROR: syntax check failed, rolled back.")
        print(exc)
        return 1

    print(f"patched : {ops_path}")
    print(f"backup  : {backup}")

    # Verify the tag actually resolves. A patch that imports a missing module is
    # worse than no patch, because it looks applied.
    if not _verify(ops_path):
        shutil.copy2(backup, ops_path)
        print("ERROR: post-patch verification failed; rolled back.")
        return 1
    return 0


def _verify(ops_path: str) -> bool:
    """Confirm the patched block references a module that actually exists."""
    src = open(ops_path, encoding="utf-8").read()
    m = re.search(r"from ([\w\.]+) import mark_transposed", src)
    if not m:
        print("VERIFY: could not find the mark_transposed import in the patch")
        return False
    module = m.group(1)
    node_dir = os.path.join(os.path.dirname(os.path.dirname(ops_path)),
                            "gfx1200_layout")
    candidate = os.path.join(node_dir, f"{module}.py")
    if not os.path.isfile(candidate):
        print(f"VERIFY: ops.py imports '{module}' but {candidate} does not exist")
        print("        the tag would never be applied (silent no-op)")
        return False
    print(f"VERIFY: ops.py imports '{module}' -> found {candidate}")
    return True


def unpatch(ops_path: str) -> int:
    src = open(ops_path, encoding="utf-8").read()
    if MARKER not in src:
        print("not patched; nothing to do")
        return 0

    start = src.find("            # --- gfx1200 layout fix")
    end = src.find("            # --- end gfx1200 layout fix")
    if start == -1 or end == -1:
        print("ERROR: malformed markers; restore from a .bak")
        return 1
    end = src.find("\n", src.find("----", end)) + 1

    stamp = datetime.datetime.now().strftime("%Y%m%d-%H%M%S")
    shutil.copy2(ops_path, f"{ops_path}.unpatch-bak-{stamp}")
    open(ops_path, "w", encoding="utf-8").write(src[:start] + src[end:])
    py_compile.compile(ops_path, doraise=True)
    print(f"unpatched: {ops_path}")
    return 0


def main(argv=None) -> int:
    ap = argparse.ArgumentParser(description=__doc__.split("\n")[0])
    ap.add_argument("--ops-path", help="path to ComfyUI-GGUF/ops.py")
    ap.add_argument("--dry-run", action="store_true")
    ap.add_argument("--unpatch", action="store_true")
    args = ap.parse_args(argv)

    p = find_ops(args.ops_path)
    print(f"ops.py: {p}")
    return unpatch(p) if args.unpatch else patch(p, args.dry_run)


if __name__ == "__main__":
    raise SystemExit(main())
