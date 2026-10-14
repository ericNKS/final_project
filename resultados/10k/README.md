# Gráficos do dataset Pix de 10 mil transações

Execute `py plot_pix_10k.py` na raiz do projeto. O script executa `src/notebooks/pix.py`, reutiliza seus atributos, hiperparâmetros e divisão estratificada (8.000 treino / 2.000 teste), e gera nove PNGs em `src/notebooks/graficos/10k/`.

O notebook compara XGBoost, Random Forest e Regressão Logística. Cada família usa Baseline, SMOTE, RandomUnderSampler e SMOTE + Tomek: 12 configurações. Regressão Logística usa StandardScaler antes da reamostragem. Os gráficos incluem comparação dos modelos, impacto da calibração, balanceamento, matrizes de confusão, ROC, precisão-recall, trade-off de limiar, distribuição de probabilidades e volume financeiro associado às classificações. As tabelas e os metadados ficam nesta pasta.

Para comparar limiares de forma pareada, todas as 12 configurações recebem calibração com cinco folds internos e avaliação com cinco folds externos, exclusivamente no treino. Isso inclui o Baseline. Os gráficos 1–3 mostram as métricas dos folds externos; as barras mostram média e desvio padrão. O XGBoost mantém os hiperparâmetros diferentes do notebook original, portanto sua comparação não isola o efeito do balanceamento.

As 12 configurações são ajustadas nas 8.000 transações e avaliadas nas mesmas 2.000 de teste. A melhor configuração de cada família é escolhida pelo F1 médio nos folds externos de treino e aparece nos gráficos 4–6 e 9. Os gráficos 7–8 usam a configuração com maior F1 médio de validação entre todas. O teste não escolhe o modelo nem o limiar; a curva de trade-off é apenas descritiva. As tabelas incluem todas as configurações, e o gerador reutiliza os resultados do notebook sem repetir os treinamentos.

O target representa 100 anomalias sintéticas (20 no teste). O gráfico financeiro mostra valores associados às classificações; não representa prejuízo real ou dinheiro protegido. O volume de anomalias é pequeno, limitando a estabilidade das métricas.

O CSV da versão 1 é reutilizado do cache local quando disponível. Se necessário, o acesso HTTPS usa os certificados confiáveis do sistema via `truststore`, mantendo a validação TLS. Os gráficos e resultados do dataset de 2 milhões são mantidos em seus caminhos originais.
