"""Patch ComfyUI's free-VRAM query for ROCm/WSL on gfx1200.

THE BUG
=======

Under WSL2 the AMD GPU is reached through ``/dev/dxg`` rather than the ``amdgpu``
kernel module, so ``amdsmi`` cannot initialise and::

    torch.cuda.mem_get_info()   ->   (4096, 16974905344)
                                     ^^^^ bogus: 4 KiB "free" on a 16 GiB card

``torch.cuda.get_device_properties(0).total_memory`` is correct (15.81 GiB), and
allocation works fine. But ComfyUI's ``get_free_memory()`` believes there is no
VRAM, so it drops into its lowest-VRAM mode and reports "0.00 MB usable", which
makes sampling crawl.

THE FIX
=======

ComfyUI derives free memory from ``mem_get_info()`` when no torch allocations
are pending. We substitute a sane value derived from the device total, minus
what torch already reserved.

``DSPK_USABLE_VRAM_MB`` (or ``GFX1200_USABLE_VRAM_MB``) overrides the ceiling.
A conservative default is used because an over-optimistic value makes ComfyUI
load everything and later die with ``hipBLASLt OOM``.

This module edits ``comfy/model_management.py`` in place, is idempotent, and
keeps a timestamped ``.bak`` next to the original.
"""

from __future__ import annotations

import argparse
import datetime
import os
import py_compile
import shutil
import sys

# Markers from the original hand-written patch, kept so that installs patched
# by an earlier version are detected instead of being patched a second time.
MARKER = "GFX1200_ROCM_FREE_MEM_PATCH"
ALIASES = (MARKER, "DSPK_ROCM_FREE_MEM_PATCH")
BEGIN = f"            # --- {MARKER}"
END = f"            # --- end {MARKER}"

TARGET_REL = os.path.join("comfy", "model_management.py")

BLOCK = f'''{BEGIN} ---------------------------------
            # See patches/patch_vram_reporting.py in the gfx1200 toolkit.
            #
            # Under WSL2 the AMD GPU is reached via /dev/dxg (no amdgpu module),
            # so amdsmi fails and torch.cuda.mem_get_info() reports a bogus
            # 4096-byte "free" on a 16 GiB card. ComfyUI would then offload
            # everything ("0.00 MB usable"). Reconstruct a plausible value.
            #
            # Note: over-reporting is dangerous -- ComfyUI will load the whole
            # model and later fail with "hipBLASLt OOM (192 MiB)". Keep the
            # ceiling conservative or set GFX1200_USABLE_VRAM_MB explicitly.
            import os as _os
            _dev_total = 0
            try:
                _dev_total = torch.cuda.get_device_properties(dev).total_memory
            except Exception:
                _dev_total = _mem_total_cuda

            if _dev_total > 0 and mem_free_cuda < (_dev_total * 0.01):
                import logging as _logging

                _cap = globals().get("_GFX1200_USABLE_VRAM_BYTES", None)
                if _cap is None:
                    _env = (_os.environ.get("GFX1200_USABLE_VRAM_MB")
                            or _os.environ.get("DSPK_USABLE_VRAM_MB"))
                    if _env:
                        try:
                            _cap = int(float(_env) * 1024 * 1024)
                        except Exception:
                            _cap = None
                    if _cap is None:
                        # Conservative default; raise only after verifying that
                        # a real generation completes without hipBLASLt OOM.
                        _cap = 4864 * 1024 * 1024
                    globals()["_GFX1200_USABLE_VRAM_BYTES"] = _cap
                    _logging.warning(
                        "[gfx1200] mem_get_info() free=%%d B is bogus; "
                        "using usable cap %%d MiB (device reports %%d MiB)",
                        mem_free_cuda, _cap // (1024 * 1024),
                        _dev_total // (1024 * 1024))

                mem_free_cuda = max(0, _cap - mem_reserved)
            {END} -----------------------------'''


def find_comfy_root(explicit: str | None) -> str:
    if explicit:
        return os.path.abspath(explicit)
    env = os.environ.get("COMFYUI_PATH")
    if env:
        return os.path.abspath(env)
    for cand in ("/opt/ComfyUI", os.path.expanduser("~/ComfyUI"),
                 os.path.join(os.getcwd(), "ComfyUI")):
        if os.path.isfile(os.path.join(cand, TARGET_REL)):
            return os.path.abspath(cand)
    raise SystemExit(
        "ERROR: could not locate ComfyUI. Pass --comfy-path or set COMFYUI_PATH.")


def patch(comfy_root: str, dry_run: bool = False) -> int:
    path = os.path.join(comfy_root, TARGET_REL)
    if not os.path.isfile(path):
        print(f"ERROR: {path} not found")
        return 1

    src = open(path, encoding="utf-8").read()

    present = [m for m in ALIASES if m in src]
    if present:
        print(f"already patched: {path} (marker: {present[0]})")
        if present[0] != MARKER:
            print("  note: this is an older marker variant; it works, but re-run")
            print("        with --unpatch then --patch to migrate it.")
        return 0

    anchor = "            mem_free_cuda, _mem_total_cuda = torch.cuda.mem_get_info(dev)"
    if anchor not in src:
        print("ERROR: anchor line not found; ComfyUI version may differ.")
        print("       Look for 'mem_get_info(dev)' in", path)
        return 1

    # Replace the anchor with itself + our block.
    new_src = src.replace(anchor, anchor + "\n\n" + BLOCK, 1)

    if dry_run:
        print(f"[dry-run] would patch {path} ({len(new_src) - len(src)} bytes added)")
        return 0

    stamp = datetime.datetime.now().strftime("%Y%m%d-%H%M%S")
    backup = f"{path}.gfx1200-bak-{stamp}"
    shutil.copy2(path, backup)
    open(path, "w", encoding="utf-8").write(new_src)

    try:
        py_compile.compile(path, doraise=True)
    except py_compile.PyCompileError as exc:
        shutil.copy2(backup, path)
        print("ERROR: syntax check failed, rolled back.")
        print(exc)
        return 1

    print(f"patched : {path}")
    print(f"backup  : {backup}")
    print("Set GFX1200_USABLE_VRAM_MB to override the reported ceiling.")
    return 0


def unpatch(comfy_root: str) -> int:
    path = os.path.join(comfy_root, TARGET_REL)
    src = open(path, encoding="utf-8").read()
    if MARKER not in src:
        print("not patched; nothing to do")
        return 0

    i = src.find(BEGIN)
    j = src.find(END)
    if i == -1 or j == -1:
        print("ERROR: malformed patch markers; restore from a .bak file")
        return 1
    j += len(END) + len(" -----------------------------")
    # also swallow the blank line we inserted before the block
    out = src[:i].rstrip("\n") + "\n" + src[j:].lstrip("\n")

    stamp = datetime.datetime.now().strftime("%Y%m%d-%H%M%S")
    shutil.copy2(path, f"{path}.unpatch-bak-{stamp}")
    open(path, "w", encoding="utf-8").write(out)
    py_compile.compile(path, doraise=True)
    print(f"unpatched: {path}")
    return 0


def main(argv=None) -> int:
    ap = argparse.ArgumentParser(description=__doc__.split("\n")[0])
    ap.add_argument("--comfy-path", help="ComfyUI root (default: auto-detect)")
    ap.add_argument("--dry-run", action="store_true")
    ap.add_argument("--unpatch", action="store_true")
    args = ap.parse_args(argv)

    root = find_comfy_root(args.comfy_path)
    print(f"ComfyUI root: {root}")
    return unpatch(root) if args.unpatch else patch(root, args.dry_run)


if __name__ == "__main__":
    raise SystemExit(main())
