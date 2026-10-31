# Treinamento na RTX 4060

No PowerShell, na pasta do projeto:

```powershell
.\run_gpu.ps1
```

O script usa Ubuntu-24.04 no WSL2 e o ambiente isolado
`/opt/final-project-gpu`. O `main.py` exige GPU por padrão; para executar
explicitamente o experimento na CPU, use `TRAINING_BACKEND=cpu`.

Para reproduzir o ambiente em Ubuntu com Python 3.12 e driver compatível
com CUDA 13, crie um venv e instale `requirements-gpu.lock.txt`
com `pip install -r requirements-gpu.lock.txt --extra-index-url https://pypi.nvidia.com`.
O arquivo `requirements.txt` original continua destinado ao ambiente anterior.

Para retreinar apenas 2M e regenerar os gráficos de 10k com as previsões já
salvas, execute `.\run_gpu.ps1 --reuse-10k-results`. As previsões finais de
2M ficam em `resultados/tabelas/predicoes_teste_2m.parquet`.
Depois disso, `python plot_model_diagnostics.py` recria os gráficos de
acertos/falsos positivos e trade-off dos três modelos finais sem treinar.

- XGBoost: CUDA com `tree_method="hist"`.
- Random Forest e Regressão Logística: estimadores nativos do cuML,
  com adaptação à interface de pipelines e calibração do scikit-learn.
- SMOTE/SMOTETomek: `cuml.accel` acelera a busca de vizinhos na GPU.
  Os logs do acelerador indicam onde cada chamada foi executada.
- Carregamento, organização de dados, partes da reamostragem, métricas e
  gráficos ainda usam CPU. Não é esperado uso constante de 100% da GPU.

Os modelos cuML usam implementações diferentes das versões scikit-learn;
as métricas devem ser recalculadas. A divisão treino/validação/teste e as
configurações dos experimentos foram preservadas. Os estimadores nativos
não fazem fallback de treinamento para CPU. Os artefatos com cuML devem
ser carregados no ambiente GPU compatível, com `cuml.accel` instalado antes
dos imports de scikit-learn.

Teste de integração das 24 configurações, incluindo calibração e serialização.
Execute no PowerShell, na pasta raiz do projeto; o WSL usa essa pasta como diretório de trabalho:

```powershell
wsl -d Ubuntu-24.04 -u root -- /opt/final-project-gpu/bin/python -m unittest discover -s tests -p test_gpu_runtime.py
```

Referências: [instalação RAPIDS](https://docs.rapids.ai/install/),
[aceleração e compatibilidade](https://docs.nvidia.com/cuml/latest/cuml-accel/compatibility/).
