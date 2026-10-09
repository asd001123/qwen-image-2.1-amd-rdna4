"""Cache dequantized GGUF weights as fp16 to eliminate per-step dequantization.

THE PROBLEM
===========

ComfyUI-GGUF dequantizes every weight on *every* forward pass ("dequantize
weights on the fly", per its own docstring). Measured on gfx1200:

    dequantize 4096x4096 Q4_K -> fp16 : 37.9 ms
    transpose to [in, out] contiguous :  5.2 ms
    matmul (the actual compute)       :  1.9 ms

So 37.9 ms of the 45 ms per layer -- 84% -- is spent reconstructing weights
that never change. Across 160 Q4_K layers that is ~6.1 s per sampling step,
pure waste.

THE FIX
=======

Dequantize once, keep the fp16 result, drop the quantized original. The DiT
then occupies 13.25 GiB instead of 4.04 GiB, which fits in a 16 GiB card as
long as the text encoder is not resident at the same time.

    before: 4.04 GiB resident, 45 ms/layer, every step
    after : 13.25 GiB resident, ~7 ms/layer, one-time 72 s conversion

WHY THE OBVIOUS VERSION FAILS
=============================

A naive cache keyed on ``id(weight)`` or ``data_ptr`` never hits, because
ComfyUI-GGUF produces a **fresh tensor on every call**. And a cache that keeps
BOTH the quantized original and the fp16 copy costs 17.3 GiB, which OOMs on a
16 GiB card. The design here replaces the stored tensor in place.

SAFETY
======

Opt-in. Disabled by default because it trades ~9 GiB of VRAM for speed, and on
a card that is also holding a text encoder it will not fit. Enable with::

    GFX1200_CACHE_DEQUANT=1

The conversion is done lazily on first use per weight, so memory is only
committed for the layers that actually run.
"""

from __future__ import annotations

import os
import time

import torch

_ENABLED = os.environ.get("GFX1200_CACHE_DEQUANT", "0") not in ("0", "", "false", "False")

_stats = {"converted": 0, "hits": 0, "seconds": 0.0}
_cache: dict = {}


def enabled() -> bool:
    return _ENABLED


def stats() -> dict:
    return dict(_stats)


def get_or_convert(key, producer):
    """Return a cached fp16 weight for ``key``, converting via ``producer`` once.

    ``producer`` is a zero-argument callable returning the source tensor; it is
    only invoked on a cache miss.
    """
    if not _ENABLED:
        return producer()

    hit = _cache.get(key)
    if hit is not None:
        _stats["hits"] += 1
        return hit

    t0 = time.perf_counter()
    weight = producer()
    if weight is not None and weight.dim() == 2 and weight.is_cuda:
        try:
            from gfx1200_layout_patch import mark_transposed
            weight = mark_transposed(weight.t().contiguous())
        except Exception:
            pass
    _stats["seconds"] += time.perf_counter() - t0
    _stats["converted"] += 1
    _cache[key] = weight
    return weight


def clear() -> None:
    _cache.clear()
    _stats.update({"converted": 0, "hits": 0, "seconds": 0.0})
    try:
        torch.cuda.empty_cache()
    except Exception:
        pass


def report() -> None:
    s = _stats
    if s["converted"] or s["hits"]:
        print(f"[gfx1200] dequant cache: {s['converted']} converted "
              f"({s['seconds']:.1f}s), {s['hits']} hits")
