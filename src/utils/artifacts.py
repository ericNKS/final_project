"""Gravação de imagens sem truncar arquivos existentes no Windows/WSL."""
import os
from pathlib import Path
import tempfile


def atomic_image_save(path, writer):
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    with tempfile.NamedTemporaryFile(dir=path.parent, suffix=path.suffix, delete=False) as temp:
        temporary = Path(temp.name)
    try:
        writer(temporary)
        os.replace(temporary, path)
    finally:
        temporary.unlink(missing_ok=True)


def savefig(path, **kwargs):
    import matplotlib.pyplot as plt
    atomic_image_save(path, lambda temporary: plt.savefig(temporary, **kwargs))
