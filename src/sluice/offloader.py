# SPDX-License-Identifier: Apache-2.0
"""Routing-aware MoE expert-weight streaming offloader (Sluice).

Keeps a FusedMoE layer's per-rank expert shard in host CPU memory and streams
only the experts the router selects into a small fixed-size GPU cache each
forward pass. The resident working set follows the router's top-k decisions
rather than being fixed at load time.

Load path (works for unquantized and quantized experts):
  1. ``wrap_modules`` (model construction, before checkpoint load): move each
     MoE layer's expert params to CPU so the loader fills host memory, and
     record the layer for later cache installation.
  2. The loader runs ``process_weights_after_loading`` inside vLLM's
     ``device_loading_context``, which stages each layer's experts back onto the
     GPU for processing (e.g. a CUDA-only FP4/FP8 marlin repack) and then
     restores them to CPU. Only one layer's experts are on the GPU at a time,
     so an expert shard far larger than GPU memory can still be processed.
  3. ``post_init`` (after load completes, i.e. after the context restore):
     install a small ``[num_slots, ...]`` GPU cache per layer as the live
     weights, hold the processed experts in host memory, and wrap the layer's
     ``quant_method.apply`` so the routing hook fires every forward.

Installing the cache in ``post_init`` (not inside ``process_weights``) is
required: ``device_loading_context`` records each param's pre-context device
(CPU, because ``wrap_modules`` parked it there) and restores it on exit, which
would move our GPU cache straight back to CPU.

The kernel reads ``layer.expert_map`` (global expert id -> resident row) to
index expert weights; this offloader rewrites that map per step to point the
selected experts at their GPU cache slots, so no MoE kernel changes are needed.
Requires a backend that applies ``expert_map`` (TRITON for unquantized, MARLIN
for NVFP4/FP8); FlashInfer-style monolithic backends ignore it.
"""

import itertools
from collections.abc import Generator
from dataclasses import dataclass, field

import torch
import torch.nn as nn

from vllm.logger import init_logger
from vllm.model_executor.offloader.base import BaseOffloader, should_pin_memory

logger = init_logger(__name__)

# Shared/global tensors (broadcast activation scales, bias/index tables), not
# per-expert weights. Mirrors the exclusions RoutedExperts applies for EPLB.
NON_EXPERT_WEIGHTS = frozenset(
    {
        "e_score_correction_bias",
        "w13_input_scale",
        "w2_input_scale",
        "hash_indices_table",
    }
)


@dataclass
class _ExpertLayerCache:
    """Per-layer streaming state for a single ``RoutedExperts`` module."""

    local_num_experts: int
    num_slots: int
    device: torch.device
    param_names: list[str] = field(default_factory=list)
    gpu_cache: dict[str, torch.Tensor] = field(default_factory=dict)
    cpu_store: dict[str, torch.Tensor] = field(default_factory=dict)
    local_of: list[int] = field(default_factory=list)
    expert_map_buf: torch.Tensor | None = None
    map_host: torch.Tensor | None = None
    slot_of: dict[int, int] = field(default_factory=dict)
    expert_in_slot: list[int | None] = field(default_factory=list)
    free_slots: list[int] = field(default_factory=list)
    lru: list[int] = field(default_factory=list)
    warned_overflow: bool = False


class ExpertStreamOffloader(BaseOffloader):
    """Stream routed-expert weights CPU->GPU per forward pass, by routing.

    Args:
        expert_cache_slots: GPU-resident expert slots per MoE layer, per rank.
            Must be >= the distinct local experts a single step can select.
    """

    def __init__(self, expert_cache_slots: int):
        assert expert_cache_slots > 0
        self.expert_cache_slots = expert_cache_slots
        self.pin_memory = should_pin_memory()
        self._caches: dict[int, _ExpertLayerCache] = {}
        self._layers: list[nn.Module] = []
        logger.info(
            "Sluice ExpertStreamOffloader enabled (%d cache slots per layer).",
            expert_cache_slots,
        )

    @staticmethod
    def _is_moe_layer(module: nn.Module) -> bool:
        return (
            hasattr(module, "local_num_experts")
            and hasattr(module, "global_num_experts")
            and hasattr(module, "quant_method")
        )

    @staticmethod
    def _per_expert_params(module: nn.Module, local_n: int) -> dict[str, nn.Parameter]:
        return {
            name: p
            for name, p in module.named_parameters(recurse=False)
            if name not in NON_EXPERT_WEIGHTS and p.dim() >= 1 and p.shape[0] == local_n
        }

    # -- model construction (pre-load) --------------------------------------

    def wrap_modules(
        self,
        modules_generator: Generator[nn.Module, None, None],
    ) -> list[nn.Module]:
        # Consume lazily; prepare each layer as it is built so the GPU never
        # holds more than one layer's experts at a time during load.
        modules = []
        for module in modules_generator:
            for sub in module.modules():
                if self._is_moe_layer(sub):
                    self._prepare_layer(sub)
            modules.append(module)
        return modules

    def _prepare_layer(self, module: nn.Module) -> None:
        """Move a layer's (still-empty) experts to CPU so the checkpoint loads
        into host memory, and record the layer for cache installation."""
        quant_method = module.quant_method
        if quant_method is None or getattr(quant_method, "is_monolithic", False):
            return
        if getattr(module, "rocm_aiter_fmoe_enabled", False):
            return
        local_n = int(module.local_num_experts)
        per_expert = self._per_expert_params(module, local_n)
        if not per_expert:
            return

        for p in per_expert.values():
            p.data = p.data.to("cpu")

        if all(id(module) != id(m) for m in self._layers):
            self._layers.append(module)

    # -- after load (post_init, after device_loading_context restore) -------

    def post_init(self) -> None:
        for module in self._layers:
            self._install_cache(module)
            self._wrap_apply(module)
            torch.accelerator.empty_cache()

    @staticmethod
    def _infer_device(module: nn.Module) -> torch.device:
        for t in itertools.chain(
            module.parameters(recurse=False), module.buffers(recurse=False)
        ):
            if t is not None and t.device.type != "cpu":
                return t.device
        from vllm.platforms import current_platform

        idx = torch.cuda.current_device() if torch.cuda.is_available() else 0
        return torch.device(current_platform.device_type, idx)

    def _install_cache(self, module: nn.Module) -> None:
        """Capture the layer's processed (host-resident) experts and install a
        small GPU slot cache as the live weights."""
        local_n = int(module.local_num_experts)
        global_n = int(module.global_num_experts)
        per_expert = self._per_expert_params(module, local_n)
        if not per_expert:
            return
        device = self._infer_device(module)
        num_slots = min(self.expert_cache_slots, local_n)

        cache = _ExpertLayerCache(
            local_num_experts=local_n,
            num_slots=num_slots,
            device=device,
            param_names=list(per_expert.keys()),
            expert_in_slot=[None] * num_slots,
            free_slots=list(range(num_slots)),
        )
        for name, p in per_expert.items():
            cpu = p.data.to("cpu")
            if self.pin_memory:
                cpu = cpu.pin_memory()
            cache.cpu_store[name] = cpu
            slot = torch.empty((num_slots, *p.shape[1:]), dtype=p.dtype, device=device)
            cache.gpu_cache[name] = slot
            p.data = slot

        orig_map = module.expert_map
        if orig_map is not None:
            cache.local_of = orig_map.to("cpu").tolist()
            map_dtype = orig_map.dtype
        else:
            cache.local_of = list(range(global_n))
            map_dtype = torch.int32
        cache.expert_map_buf = torch.full(
            (global_n,), -1, dtype=map_dtype, device=device
        )
        # layer.expert_map is a property returning the _expert_map buffer.
        module.register_buffer("_expert_map", cache.expert_map_buf, persistent=False)
        map_host = torch.full((global_n,), -1, dtype=map_dtype, device="cpu")
        if self.pin_memory:
            map_host = map_host.pin_memory()
        cache.map_host = map_host

        self._caches[id(module)] = cache
        logger.info(
            "Sluice: %d local experts -> %d GPU slots, %d params/expert.",
            local_n,
            num_slots,
            len(per_expert),
        )

    def _wrap_apply(self, module: nn.Module) -> None:
        """Wrap the layer's modular ``quant_method.apply`` so the routing-aware
        streaming hook fires before the fused-MoE kernel. This is what makes
        Sluice a plugin: no edit to vLLM's MoE runner source is required, since
        ``apply`` already receives ``layer`` and ``topk_ids``."""
        quant_method = module.quant_method
        if quant_method is None or getattr(quant_method, "_sluice_wrapped", False):
            return
        original_apply = quant_method.apply
        offloader = self

        def apply(*args, **kwargs):
            layer = kwargs.get("layer", args[0] if args else None)
            topk_ids = kwargs.get("topk_ids")
            if layer is not None and topk_ids is not None:
                offloader.on_experts_selected(layer, topk_ids)
            return original_apply(*args, **kwargs)

        quant_method.apply = apply
        quant_method._sluice_wrapped = True

    # -- forward path (fired from the wrapped quant_method.apply) ------------

    def on_experts_selected(
        self,
        module: nn.Module,
        topk_ids: torch.Tensor,
    ) -> None:
        cache = self._caches.get(id(module))
        if cache is None:
            return
        assert cache.map_host is not None and cache.expert_map_buf is not None

        selected = torch.unique(topk_ids).tolist()
        local_of = cache.local_of
        map_host = cache.map_host
        expert_map_buf = cache.expert_map_buf
        map_host.fill_(-1)
        protected: set[int] = set()
        fits_all = cache.num_slots >= cache.local_num_experts
        for g in selected:
            if g < 0 or g >= len(local_of):
                continue
            local = local_of[g]
            if local < 0:
                continue
            if fits_all:
                slot = local
                if local not in cache.slot_of:
                    cache.slot_of[local] = local
                    self._stream_in(cache, local, local)
            else:
                slot = cache.slot_of.get(local)
                if slot is None:
                    slot = self._acquire_slot(cache, local, protected)
                    if slot is None:
                        self._warn_overflow(cache)
                        continue
                    self._stream_in(cache, local, slot)
                else:
                    cache.lru.remove(slot)
                    cache.lru.append(slot)
            map_host[g] = slot
            protected.add(slot)

        expert_map_buf.copy_(map_host)

    # -- internals ----------------------------------------------------------

    def _acquire_slot(
        self, cache: _ExpertLayerCache, local: int, protected: set[int]
    ) -> int | None:
        if cache.free_slots:
            slot = cache.free_slots.pop()
        else:
            victim = next(
                (i for i, s in enumerate(cache.lru) if s not in protected), None
            )
            if victim is None:
                return None
            slot = cache.lru.pop(victim)
            old_local = cache.expert_in_slot[slot]
            if old_local is not None:
                cache.slot_of.pop(old_local, None)

        cache.expert_in_slot[slot] = local
        cache.slot_of[local] = slot
        cache.lru.append(slot)
        return slot

    def _stream_in(self, cache: _ExpertLayerCache, local: int, slot: int) -> None:
        for name, gpu in cache.gpu_cache.items():
            gpu[slot].copy_(cache.cpu_store[name][local], non_blocking=True)

    def _warn_overflow(self, cache: _ExpertLayerCache) -> None:
        if not cache.warned_overflow:
            cache.warned_overflow = True
            logger.warning(
                "Sluice: expert_cache_slots=%d is smaller than the experts "
                "selected in a single step; excess experts are skipped this "
                "step. Expected during the startup profiling run; increase "
                "SLUICE_SLOTS if it happens during real decode.",
                cache.num_slots,
            )
