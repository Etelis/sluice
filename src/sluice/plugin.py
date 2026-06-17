# SPDX-License-Identifier: Apache-2.0
"""Sluice vLLM plugin entry point.

Registered under the ``vllm.general_plugins`` entry-point group (see
``pyproject.toml``). vLLM calls :func:`register` in every engine/worker process
at startup, before the model is built. When ``SLUICE_SLOTS`` is set, we
monkeypatch vLLM's ``create_offloader`` factory so the active offloader becomes
the Sluice :class:`~sluice.offloader.ExpertStreamOffloader`. Everything else
flows through vLLM's existing ``BaseOffloader`` lifecycle (``wrap_modules`` ->
``post_init``) and the offloader's own ``quant_method.apply`` wrapping, so no
vLLM source files are modified.
"""

import os

from vllm.logger import init_logger

logger = init_logger(__name__)

SLOTS_ENV = "SLUICE_SLOTS"


def register() -> None:
    """vLLM ``general_plugins`` hook. No-op unless ``SLUICE_SLOTS`` is set."""
    raw = os.environ.get(SLOTS_ENV)
    if not raw:
        return
    try:
        slots = int(raw)
        assert slots > 0
    except (ValueError, AssertionError):
        logger.warning("Sluice: ignoring invalid %s=%r (want a positive int).",
                       SLOTS_ENV, raw)
        return

    import vllm.model_executor.offloader.base as base_mod

    from sluice.offloader import ExpertStreamOffloader

    def _create_offloader(offload_config):  # noqa: ANN001
        return ExpertStreamOffloader(expert_cache_slots=slots)

    # Patch the factory at its definition...
    base_mod.create_offloader = _create_offloader
    # ...and at every module that imported it by name before us.
    try:
        import vllm.v1.worker.gpu_model_runner as gmr

        if hasattr(gmr, "create_offloader"):
            gmr.create_offloader = _create_offloader
    except Exception:  # pragma: no cover - defensive across vLLM versions
        logger.debug("Sluice: gpu_model_runner not patchable; relying on base.")

    logger.info("Sluice: activated via %s=%d (expert offloading on).",
                SLOTS_ENV, slots)
