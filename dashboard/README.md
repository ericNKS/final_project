# Dashboard Pix

Abra `resultados/dashboard/index.html` no navegador. O arquivo funciona offline, sem instalação de dependências e sem servidor.

## Visões

- Random Forest, XGBoost e regressão logística: indicadores do teste, Sankey das classes reais para as previstas e tabela das oito configurações de validação.
- Comparação: F1, tabela de métricas e Sankeys dos três modelos.
- Filtros dos diagramas: todas as transações, somente anomalias reais e somente erros. Os indicadores e as contagens abaixo dos diagramas sempre descrevem o teste completo.

As larguras são proporcionais às contagens. Fluxos de valor zero são omitidos. O filtro de erros normaliza cada diagrama pelo número de erros de seu modelo, por isso a comparação entre modelos deve considerar também os números, e não apenas as larguras.

## Atualização

Após gerar novos resultados, execute na raiz:

```powershell
.\.venv\Scripts\python.exe build_dashboard.py
```

O gerador lê os CSVs em `resultados/tabelas/`, os metadados em `resultados/metadados/` e este template. Os dados são incorporados ao HTML. Nenhum treinamento ou requisição externa é realizado. A análise textual da área comparativa descreve o experimento atual e deve ser revisada caso seus resultados mudem.

O arquivo editável é `dashboard/template.html`; `resultados/dashboard/index.html` é a saída gerada. Para preservar links para relatório e CSVs, mantenha a estrutura de `resultados/` ao compartilhar o projeto. O dashboard em si pode ser aberto como arquivo independente.

## Verificação

`node dashboard/check_dashboard.cjs` confere a execução do JavaScript, as quatro visões e os três filtros, incluindo a conservação dos fluxos e os totais selecionados. Esse teste não substitui inspeção visual no navegador.
