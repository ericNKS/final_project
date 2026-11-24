# Avaliação temporal

Campeão escolhido exclusivamente pela AP da validação temporal: xgboost. Limiar escolhido por F1 no período de calibração. O teste é descritivo. Escores não são probabilidades calibradas. Os cenários têm alvos diferentes; valores sinalizados não demonstram dinheiro efetivamente protegido.

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
scenario                   run_id              family                                  candidate  threshold  accuracy  precision   recall       f1  average_precision  roc_auc   tn  fp  fn  tp
     10k 20261123T225344_36cb16d3 logistic_regression logistic_regression__class_weight__default   0.993508  0.995491   1.000000 0.500000 0.666667           0.592050 0.944107 1978   0   9   9
     10k 20261123T225344_36cb16d3       random_forest            random_forest__smotenc__default   0.540000  0.997495   0.882353 0.833333 0.857143           0.893457 0.961058 1976   2   3  15
     10k 20261123T225344_36cb16d3             xgboost            xgboost__class_weight__adjusted   0.925439  0.996994   0.875000 0.777778 0.823529           0.889686 0.978851 1976   2   4  14
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
logistic_regression        default   0.500000  0.830491   0.040698 0.636364 0.076503           0.643121 0.788309 821 165   4   7
logistic_regression          tuned   0.993508  0.995988   1.000000 0.636364 0.777778           0.643121 0.788309 986   0   4   7
      random_forest        default   0.500000  0.994985   0.750000 0.818182 0.782609           0.770273 0.898396 983   3   2   9
      random_forest          tuned   0.540000  0.995988   0.818182 0.818182 0.818182           0.770273 0.898396 984   2   2   9
            xgboost        default   0.500000  0.996991   0.900000 0.818182 0.857143           0.711894 0.933616 985   1   2   9
            xgboost          tuned   0.925439  0.996991   0.900000 0.818182 0.857143           0.711894 0.933616 985   1   2   9
```

Disponibilidade dos atributos: docs/METHODOLOGY.md. Consulte quality.json para exclusões e validation.json para contagens antes/depois da reamostragem e motivos de inviabilidade.
