# Avaliação temporal

Campeão escolhido exclusivamente pela AP da validação temporal: random_forest. Limiar escolhido por F1 no período de calibração. O teste é descritivo. Escores não são probabilidades calibradas. Os cenários têm alvos diferentes; valores sinalizados não demonstram dinheiro efetivamente protegido.

## Períodos e prevalência

```
  partition  rows  positives  prevalence                     start                       end
development  6983         47    0.006731 2025-06-20T18:55:56+00:00 2025-08-02T19:56:56+00:00
calibration   997         11    0.011033 2025-08-02T20:04:56+00:00 2025-08-08T11:29:56+00:00
       test  1996         18    0.009018 2025-08-08T11:43:57+00:00 2025-08-20T18:44:56+00:00
```

As prevalências podem mudar entre períodos. O limiar é fixado antes do teste; mudanças de distribuição não autorizam ajustá-lo usando o teste.

## Teste reservado

```
scenario                   run_id              family                              candidate  threshold  accuracy  precision   recall       f1  average_precision  roc_auc   tn  fp  fn  tp
     10k 20261123T223800_32d9e734 logistic_regression logistic_regression__original__default   0.430811  0.995491   1.000000 0.500000 0.666667           0.591841 0.956381 1978   0   9   9
     10k 20261123T223800_32d9e734       random_forest       random_forest__original__default   0.550000  0.998998   1.000000 0.888889 0.941176           0.944945 0.967728 1978   0   2  16
     10k 20261123T223800_32d9e734             xgboost        xgboost__class_weight__adjusted   0.887759  0.997495   0.842105 0.888889 0.864865           0.853594 0.972896 1975   3   2  16
```

## Limitações

- A fonte não informa quando os rótulos se tornaram conhecidos. O estudo assume disponibilidade no corte de treino; não comprova ausência de vazamento por atraso de rotulagem.
- O histórico offline assume disponibilidade imediata dos eventos anteriores. Na API, somente pagamentos executados antes do evento e confirmados até a análise entram no histórico.
- F1 é um objetivo experimental, sem custos financeiros de falsos positivos/negativos; o limiar não é uma política econômica ótima.
- Indicadores cadastrais e bancos precisam ser conhecidos antes da decisão. O alvo 10k é anomalia cadastral, não confirmação de fraude financeira.
- Cobertura temporal: 10000 linhas de origem; 24 datas inválidas excluídas, por classe: {'1': 24}. As métricas não avaliam essas exclusões nem demonstram detecção de anomalias de data.

## Justificativa de classificação

O balanceamento foi escolhido pela AP da validação progressiva, sem reamostrar validação ou teste. Pesos são calculados somente no treino de cada janela; configurações combinando pesos e síntese não são usadas. O limiar maximiza F1 em período posterior ao treino; seu desempenho nesse período é de ajuste, não de teste.

```
             family threshold_type  threshold  accuracy  precision   recall       f1  average_precision  roc_auc  tn  fp  fn  tp
logistic_regression        default   0.500000  0.994985        1.0 0.545455 0.705882           0.650864 0.855799 986   0   5   6
logistic_regression          tuned   0.430811  0.995988        1.0 0.636364 0.777778           0.650864 0.855799 986   0   4   7
      random_forest        default   0.500000  0.997994        1.0 0.818182 0.900000           0.820188 0.891112 986   0   2   9
      random_forest          tuned   0.550000  0.997994        1.0 0.818182 0.900000           0.820188 0.891112 986   0   2   9
            xgboost        default   0.500000  0.996991        0.9 0.818182 0.857143           0.706755 0.940162 985   1   2   9
            xgboost          tuned   0.887759  0.996991        0.9 0.818182 0.857143           0.706755 0.940162 985   1   2   9
```

Disponibilidade dos atributos: docs/METHODOLOGY.md. Consulte quality.json para exclusões e validation.json para contagens antes/depois da reamostragem e motivos de inviabilidade.
