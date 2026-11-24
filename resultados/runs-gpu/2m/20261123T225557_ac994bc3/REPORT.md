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
      2m 20261123T225557_ac994bc3 logistic_regression logistic_regression__smotenc_tomek__default   0.951277  0.990544   0.773770 0.726899 0.749603           0.802411 0.993574 390728 1656 2128 5664
      2m 20261123T225557_ac994bc3       random_forest           random_forest__original__adjusted   0.286255  0.992146   0.874859 0.696227 0.775388           0.868541 0.993420 391608  776 2367 5425
      2m 20261123T225557_ac994bc3             xgboost                 xgboost__original__adjusted   0.380241  0.992728   0.891814 0.713039 0.792469           0.885139 0.988193 391710  674 2236 5556
```

## Limitações

- A fonte não informa quando os rótulos se tornaram conhecidos. O estudo assume disponibilidade no corte de treino; não comprova ausência de vazamento por atraso de rotulagem.
- O histórico offline assume disponibilidade imediata dos eventos anteriores. Na API, somente pagamentos executados antes do evento e confirmados até a análise entram no histórico.
- F1 é um objetivo experimental, sem custos financeiros de falsos positivos/negativos; o limiar não é uma política econômica ótima.
- Aplicável somente a integrações com acesso legítimo aos saldos do pagador E do recebedor antes da decisão. O nome saldo_anterior não comprova disponibilidade. A API valida formato, mas não verifica a origem ou o instante da observação; sem esses saldos, este modelo não deve ser usado nem receber valores inventados.
- Cobertura temporal: 2000000 linhas de origem; 0 datas inválidas excluídas, por classe: {}. As métricas não avaliam essas exclusões nem demonstram detecção de anomalias de data.

## Justificativa de classificação

O balanceamento foi escolhido pela AP da validação progressiva, sem reamostrar validação ou teste. Pesos são calculados somente no treino de cada janela; configurações combinando pesos e síntese não são usadas. O limiar maximiza F1 em período posterior ao treino; seu desempenho nesse período é de ajuste, não de teste.

```
             family threshold_type  threshold  accuracy  precision   recall       f1  average_precision  roc_auc     tn   fp  fn  tp
logistic_regression        default   0.500000  0.969099   0.097873 0.971469 0.177830           0.345932 0.991726 196799 6277  20 681
logistic_regression          tuned   0.951277  0.994833   0.360759 0.650499 0.464122           0.345932 0.991726 202268  808 245 456
      random_forest        default   0.500000  0.997517   0.870722 0.326676 0.475104           0.626263 0.988658 203042   34 472 229
      random_forest          tuned   0.286255  0.996982   0.554569 0.623395 0.586971           0.626263 0.988658 202725  351 264 437
            xgboost        default   0.500000  0.997846   0.728223 0.596291 0.655686           0.688440 0.991136 202920  156 283 418
            xgboost          tuned   0.380241  0.997841   0.702326 0.646220 0.673105           0.688440 0.991136 202884  192 248 453
```

Disponibilidade dos atributos: docs/METHODOLOGY.md. Consulte quality.json para exclusões e validation.json para contagens antes/depois da reamostragem e motivos de inviabilidade.
