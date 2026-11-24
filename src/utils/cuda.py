"""CUDA resource ownership: shared allocator, bounded workspaces and explicit failures."""

import gc
from functools import lru_cache


class GPUExecutionError(RuntimeError):
    """A requested GPU operation failed; never interpreted as an infeasible candidate."""


@lru_cache(maxsize=1)
def initialize() -> dict:
    """Use the same non-pooling allocator for CuPy/cuDF/cuML on a small dedicated GPU."""
    try:
        import cupy as cp
        import cudf
        import cuml
        import rmm
        from rmm.allocators.cupy import rmm_cupy_allocator
        properties = cp.cuda.runtime.getDeviceProperties(cp.cuda.Device().id)
        rmm.mr.set_current_device_resource(rmm.mr.CudaMemoryResource())
        cp.cuda.set_allocator(rmm_cupy_allocator)
        cp.zeros(1).sum().item()
        free, total = cp.cuda.runtime.memGetInfo()
        name = properties["name"]
        return {"device": cp.cuda.Device().id, "name": name.decode() if isinstance(name, bytes) else str(name),
                "compute_capability": f"{properties['major']}.{properties['minor']}",
                "total_vram_bytes": total, "free_vram_at_start_bytes": free,
                "cuda_runtime": cp.cuda.runtime.runtimeGetVersion(),
                "allocator": "shared_rmm_cuda_without_retained_pool",
                "cupy": cp.__version__, "cudf": cudf.__version__, "cuml": cuml.__version__}
    except (ImportError, RuntimeError) as exc:
        raise GPUExecutionError(
            "GPU solicitada, mas CUDA/RAPIDS não está disponível. "
            "Use o ambiente .venv-gpu descrito em GPU.md. Não houve fallback para CPU."
        ) from exc


def release() -> None:
    """Release unused fold arrays/models before the next candidate claims GPU memory."""
    import cupy as cp
    gc.collect()
    cp.cuda.get_current_stream().synchronize()


def memory_failure(exc: Exception, stage: str) -> None:
    """Promote allocation failures to fatal errors rather than silently skipping candidates."""
    message = str(exc).lower()
    if any(token in message for token in ("out of memory", "outofmemory", "bad_alloc", "memoryallocation", "memory allocation")):
        raise GPUExecutionError(
            f"Memória GPU insuficiente em {stage}. Nenhum dado foi descartado e não houve fallback para CPU. "
            "Feche outros processos GPU ou use um dispositivo com mais VRAM."
        ) from exc
