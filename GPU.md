# Execução GPU

Use o ambiente `.venv-gpu` instalado por `make setup` a partir de `requirements.txt`, com driver NVIDIA compatível com CUDA 13. Execute `make train`, `make verify` e `make test-gpu`. Não há fallback automático para CPU.

Leitura, validação, coordenação temporal, banco e gráficos usam CPU; engenharia de históricos, transformações, balanceamento, classificadores, métricas e calibração usam CUDA. Contêineres exigem NVIDIA Container Toolkit/CDI além do driver. Consulte [README.md](README.md) e [docs/GPU_VALIDATION.md](docs/GPU_VALIDATION.md).
