# Resultados: Random Forest, XGBoost e Regressão Logística

Base juniorbueno/pix-banking-transaction: 10000 registros; 8000 no treino e 2000 no teste. Rótulos: {'0': 9900, '1': 100}. Anomalia é o rótulo positivo e não constitui confirmação de fraude.

## Procedimento

Divisão estratificada 80/20, semente 42. Avaliação de 24 configurações (3 algoritmos × 4 estratégias × 2 conjuntos de parâmetros) nos mesmos 5 folds do treino. Encoder, padronização numérica da regressão logística e reamostragem ajustados somente no treino de cada fold. Seleção pelo maior F1 médio de validação, com desempate por AP e ROC-AUC. Parâmetros ajustados manualmente, reproduzindo os valores de main.py. O XGBoost padrão recebe semente 42. Os modelos de árvores usam até 4 processadores. Limiar padrão 0,5, sem otimização. Teste reservado usado somente após a seleção.

## Random Forest: validação no treino

| Modelo | Acurácia | Precisão | Recall | F1 | ROC-AUC | AP |
| --- | --- | --- | --- | --- | --- | --- |
| Random Forest | Sem reamostragem | Padrão | 0.9979 | 1.0000 | 0.7875 | 0.8779 | 0.9022 | 0.8034 |
| Random Forest | Sem reamostragem | Ajustada | 0.9976 | 1.0000 | 0.7625 | 0.8603 | 0.8836 | 0.8045 |
| Random Forest | SMOTE | Padrão | 0.9972 | 1.0000 | 0.7250 | 0.8377 | 0.8896 | 0.8033 |
| Random Forest | SMOTE | Ajustada | 0.9971 | 1.0000 | 0.7125 | 0.8279 | 0.8997 | 0.8053 |
| Random Forest | Under | Padrão | 0.9966 | 0.8638 | 0.8000 | 0.8253 | 0.8890 | 0.7495 |
| Random Forest | Under | Ajustada | 0.9963 | 0.8769 | 0.7375 | 0.7934 | 0.8781 | 0.7390 |
| Random Forest | SMOTETomek | Padrão | 0.9970 | 1.0000 | 0.7000 | 0.8194 | 0.9020 | 0.8046 |
| Random Forest | SMOTETomek | Ajustada | 0.9969 | 1.0000 | 0.6875 | 0.8115 | 0.9101 | 0.8048 |

## XGBoost: validação no treino

| Modelo | Acurácia | Precisão | Recall | F1 | ROC-AUC | AP |
| --- | --- | --- | --- | --- | --- | --- |
| XGBoost | Sem reamostragem | Padrão | 0.9926 | 0.8064 | 0.3500 | 0.4841 | 0.9117 | 0.6418 |
| XGBoost | Sem reamostragem | Ajustada | 0.9961 | 0.8599 | 0.7375 | 0.7863 | 0.9124 | 0.7637 |
| XGBoost | SMOTE | Padrão | 0.9975 | 0.9833 | 0.7625 | 0.8563 | 0.9189 | 0.7799 |
| XGBoost | SMOTE | Ajustada | 0.9979 | 0.9857 | 0.8000 | 0.8798 | 0.9100 | 0.8087 |
| XGBoost | Under | Padrão | 0.9889 | 0.4681 | 0.5625 | 0.5073 | 0.9081 | 0.5147 |
| XGBoost | Under | Ajustada | 0.9907 | 0.5348 | 0.6750 | 0.5922 | 0.9078 | 0.4945 |
| XGBoost | SMOTETomek | Padrão | 0.9972 | 0.9818 | 0.7375 | 0.8374 | 0.9212 | 0.7857 |
| XGBoost | SMOTETomek | Ajustada | 0.9975 | 0.9857 | 0.7625 | 0.8547 | 0.9067 | 0.8085 |

## Regressão Logística: validação no treino

| Modelo | Acurácia | Precisão | Recall | F1 | ROC-AUC | AP |
| --- | --- | --- | --- | --- | --- | --- |
| Regressão Logística | Sem reamostragem | Padrão | 0.9928 | 1.0000 | 0.2750 | 0.4289 | 0.8025 | 0.5095 |
| Regressão Logística | Sem reamostragem | Ajustada | 0.8266 | 0.0328 | 0.5750 | 0.0621 | 0.7827 | 0.3982 |
| Regressão Logística | SMOTE | Padrão | 0.8666 | 0.0416 | 0.5625 | 0.0774 | 0.7818 | 0.4471 |
| Regressão Logística | SMOTE | Ajustada | 0.8599 | 0.0406 | 0.5750 | 0.0757 | 0.7863 | 0.4485 |
| Regressão Logística | Under | Padrão | 0.9925 | 0.7456 | 0.3875 | 0.5035 | 0.8050 | 0.4520 |
| Regressão Logística | Under | Ajustada | 0.9919 | 0.8000 | 0.1875 | 0.3016 | 0.7957 | 0.3311 |
| Regressão Logística | SMOTETomek | Padrão | 0.8666 | 0.0416 | 0.5625 | 0.0774 | 0.7818 | 0.4471 |
| Regressão Logística | SMOTETomek | Ajustada | 0.8599 | 0.0406 | 0.5750 | 0.0757 | 0.7863 | 0.4485 |

## Configurações selecionadas

- Random Forest: Random Forest | Sem reamostragem | Padrão
- XGBoost: XGBoost | SMOTE | Ajustada
- Regressão Logística: Regressão Logística | Under | Padrão

## Comparação no teste reservado

| Modelo | Acurácia | Precisão | Recall | F1 | ROC-AUC | AP |
| --- | --- | --- | --- | --- | --- | --- |
| Random Forest | Sem reamostragem | Padrão | 0.9995 | 1.0000 | 0.9500 | 0.9744 | 0.9701 | 0.9505 |
| XGBoost | SMOTE | Ajustada | 0.9990 | 0.9500 | 0.9500 | 0.9500 | 0.9580 | 0.9481 |
| Regressão Logística | Under | Padrão | 0.9930 | 0.8000 | 0.4000 | 0.5333 | 0.9044 | 0.4774 |

- Random Forest: TN=1980, FP=0, FN=1, TP=19.
- XGBoost: TN=1979, FP=1, FN=1, TP=19.
- Regressão Logística: TN=1978, FP=2, FN=12, TP=8.

AP é average precision, separada de ROC-AUC; não é área trapezoidal da curva precisão–recall. Médias, desvios e tempos de validação estão no CSV. Métricas de teste vêm de uma única divisão e não são intervalos de confiança.

## Arquivos

- graficos/dados/: distribuição das classes.
- graficos/random_forest/: validação, matriz, curvas, importância e probabilidades.
- graficos/xgboost/ e logistic_regression/: validação, matriz, curvas e probabilidades.
- graficos/comparacao/: validação, métricas de teste, matrizes, curvas e erros.
- tabelas/metricas_validacao_cruzada.csv: médias, desvios e tempos das 24 configurações.
- tabelas/metricas_teste.csv: métricas e contagens das matrizes de confusão.
- tabelas/predicoes_teste.csv: probabilidades e previsões com índices originais.
- metadados/metadados.json: distribuição dos rótulos, divisão e configurações selecionadas.
- modelos/: random_forest.joblib, xgboost.joblib e logistic_regression.joblib: pipelines ajustados no treino.

## Limites de interpretação

Os resultados descrevem esta base e divisão, sem comprovar generalização para transações em produção. O uso de Status pressupõe disponibilidade no momento da previsão. Não foi auditada a origem ou independência temporal dos registros. SMOTE após one-hot pode gerar atributos categóricos fracionários. Importância soma categorias por atributo e usa redução de impureza; não representa causalidade e pode favorecer atributos com mais possibilidades de divisão.

Para reproduzir, execute na raiz: .\.venv\Scripts\python.exe evaluate_models.py

## Análise dos resultados desta execução

| Algoritmo | Configuração selecionada na validação | F1 médio CV | F1 no teste | Precisão | Recall | FP | FN |
| --- | --- | --- | --- | --- | --- | --- | --- |
| Random Forest | Padrão, sem reamostragem | 0,8779 | 0,9744 | 1,0000 | 0,9500 | 0 | 1 |
| XGBoost | Ajustada, com SMOTE | 0,8798 | 0,9500 | 0,9500 | 0,9500 | 1 | 1 |
| Regressão logística | Padrão, com Under | 0,5035 | 0,5333 | 0,8000 | 0,4000 | 2 | 12 |

Random Forest apresentou o maior F1 no teste reservado, com 19 das 20 anomalias detectadas e nenhum falso positivo entre 1.980 transações legítimas. XGBoost detectou as mesmas 19 anomalias, com um falso positivo. Na validação, o XGBoost selecionado teve F1 médio ligeiramente maior que o Random Forest selecionado; a ordenação do teste foi diferente.

A regressão logística detectou oito anomalias e deixou passar 12, além de gerar dois falsos positivos. A acurácia de 99,30% deve ser interpretada junto com o recall de 40%, devido ao forte desbalanceamento. Entre suas configurações, Under com parâmetros padrão teve o maior F1 médio; as configurações com SMOTE e SMOTETomek ficaram próximas de 0,08 de F1 na validação. Isso mostra que essas estratégias não melhoraram automaticamente o equilíbrio entre precisão e recall com os parâmetros e limiar usados.

A regressão logística selecionada teve ajuste final mais rápido nesta execução (cerca de 0,02 s), frente a cerca de 0,21 s do Random Forest e 0,48 s do XGBoost. Os tempos incluem o ajuste do pipeline e dependem da máquina e da carga de execução; não são medições de latência de previsão.

Esta comparação contém apenas 20 anomalias no teste: cada anomalia adicional detectada altera o recall em cinco pontos percentuais. Não foi realizado teste de significância entre algoritmos, e os resultados não estabelecem superioridade universal. As configurações foram selecionadas pela validação; o teste reservado foi usado para a comparação final.

## Verificação e navegação

Os três testes automatizados passaram. O verificador `verify_results.py` confirmou 24 configurações, seleção por validação, três pipelines carregáveis, métricas recalculadas a partir das previsões e integridade dos 20 gráficos. A execução do avaliador terminou sem avisos de convergência registrados no terminal.

Acesse o [índice dos resultados e gráficos](../README.md) para navegar pelas pastas. O gráfico de falsos positivos foi atualizado para incluir os três algoritmos. A distribuição de classes foi centralizada em `dados/`; uma cópia anterior foi preservada em `historico/`.


