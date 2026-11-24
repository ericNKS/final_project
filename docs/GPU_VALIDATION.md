# Referência GPU reavaliada

Reavaliação em 2026-11-23 com `make evaluate`, preservando modelos, seleção e limiares. Relatórios atualizados por `make report`. Não houve novo treinamento nem escolha pelo teste.

| Cenário | Execução | Campeão / estratégia | Limiar | AP teste | F1 teste |
|---|---|---|---:|---:|---:|
| 10k | 20261123T225344_36cb16d3 | xgboost__class_weight__adjusted | 0.925439239 | 0.889686405 | 0.823529412 |
| 2m | 20261123T225557_ac994bc3 | xgboost__original__adjusted | 0.380241454 | 0.885138962 | 0.792468977 |

A referência 10k exclui 24 datas inválidas, todas positivas; o teste tem 1.996 registros, dos quais 18 positivos. O teste 2m tem 400.176 registros, dos quais 7.792 positivos. Os manifestos registram os cortes reais e as classes de cada partição.

O campeão 10k permanece XGBoost embora outro modelo tenha AP maior no teste: a escolha foi feita na validação. Não há estimativa de lucro ou prova de disponibilidade dos saldos de uma integração real. Consulte [METHODOLOGY.md](METHODOLOGY.md).

Verificações desta revisão: 22 testes temporais, de CLI e contrato HTTP aprovados; 65 testes CUDA aprovados (`RUN_GPU_TESTS=1`, incluindo candidatos, históricos, métricas e amostragem); `make verify` aprovou ambos os cenários. Os quatro avisos CUDA são ajustes automáticos de `n_bins` da Random Forest em conjuntos pequenos de teste. Os testes PostgreSQL não foram repetidos nesta revisão, que não alterou consultas, migrações ou ciclo de vida das transações.
