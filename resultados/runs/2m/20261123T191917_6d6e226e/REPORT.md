# Avaliação temporal

Campeão escolhido exclusivamente pela AP da validação temporal: xgboost. Limiar escolhido por F1 no período de calibração. O teste é descritivo. Escores não são probabilidades calibradas. Os cenários têm alvos diferentes; valores sinalizados não demonstram dinheiro efetivamente protegido.

## Períodos e prevalência

```
  partition    rows  positives  prevalence                     start                       end
development 1396047       6883    0.004930 2024-01-01T03:00:00+00:00 2024-01-14T15:00:00+00:00
calibration  203777        701    0.003440 2024-01-14T16:00:00+00:00 2024-01-16T02:00:00+00:00
       test  400176       7792    0.019471 2024-01-16T03:00:00+00:00 2024-02-01T02:59:59+00:00
```

As prevalências podem mudar entre períodos. O limiar é fixado antes do teste; mudanças de distribuição não autorizam ajustá-lo usando o teste.

## Teste reservado

```
scenario                   run_id              family                                   candidate  threshold  accuracy  precision   recall       f1  average_precision  roc_auc     tn   fp   fn   tp
      2m 20261123T191917_6d6e226e logistic_regression logistic_regression__smotenc_tomek__default   0.950144  0.990599   0.769818 0.737808 0.753473           0.799322 0.993473 390665 1719 2043 5749
      2m 20261123T191917_6d6e226e       random_forest           random_forest__original__adjusted   0.325177  0.990482   0.787415 0.700205 0.741254           0.834584 0.991776 390911 1473 2336 5456
      2m 20261123T191917_6d6e226e             xgboost              xgboost__class_weight__default   0.983485  0.993413   0.931971 0.713809 0.808430           0.918749 0.996806 391978  406 2230 5562
```

## Limitações

- Rótulos disponíveis ao final do treino; atraso de confirmação não informado pela fonte.
- Histórico offline assume disponibilidade imediata de transações anteriores.
- Saldos anteriores devem estar disponíveis ao consumidor antes da execução.

## Justificativa de classificação

O balanceamento foi escolhido pela AP da validação progressiva, sem reamostrar validação ou teste. Pesos são calculados somente no treino de cada janela; configurações combinando pesos e síntese não são usadas. O limiar maximiza F1 em período posterior ao treino; seu desempenho nesse período é de ajuste, não de teste.

```
             family threshold_type  threshold  accuracy  precision   recall       f1  average_precision  roc_auc     tn   fp  fn  tp
logistic_regression        default   0.500000  0.969526   0.098645 0.965763 0.179006           0.348368 0.991529 196890 6186  24 677
logistic_regression          tuned   0.950144  0.994754   0.358243 0.663338 0.465233           0.348368 0.991529 202243  833 236 465
      random_forest        default   0.500000  0.997792   0.876877 0.416548 0.564797           0.693636 0.988117 203035   41 409 292
      random_forest          tuned   0.325177  0.997806   0.699686 0.634807 0.665669           0.693636 0.988117 202885  191 256 445
            xgboost        default   0.500000  0.977024   0.127735 0.974322 0.225860           0.737693 0.996701 198412 4664  18 683
            xgboost          tuned   0.983485  0.997973   0.726415 0.659058 0.691099           0.737693 0.996701 202902  174 239 462
```

Disponibilidade dos atributos: docs/METHODOLOGY.md. Consulte quality.json para exclusões e validation.json para contagens antes/depois da reamostragem e motivos de inviabilidade.
