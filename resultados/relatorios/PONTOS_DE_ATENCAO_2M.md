# Comparação entre datasets de 10 mil e 2 milhões de registros e análise dos algoritmos no dataset de 2 milhões

Data: 13 de outubro de 2026. Escopo: comparação dos experimentos de 10 mil e 2 milhões de transações, seguida de análise detalhada de XGBoost, Random Forest e regressão logística na base de 2 milhões. “2M” representa registros, não parâmetros dos modelos.

## Parte I — Comparação dos algoritmos nos dois datasets

### Objetivo e organização da análise

O objetivo é comparar o comportamento dos três algoritmos em dois cenários de dados e, em seguida, aprofundar a discussão sobre seu funcionamento no dataset de 2 milhões de registros. Cada experimento treina e testa os modelos em partições da própria base. Essa comparação não corresponde a treinar em uma base e testar na outra.

Primeiro são apresentados os datasets, seus rótulos e os procedimentos de avaliação. Depois são comparados os resultados de cada algoritmo entre os cenários. A segunda parte discute o funcionamento, os erros e os pontos de atenção dos modelos no experimento de 2 milhões.

### Caracterização dos cenários

| Aspecto | Dataset de 10 mil | Dataset de 2 milhões |
|---|---|---|
| Treino / teste | 8.000 / 2.000 | 1.600.000 / 400.000 |
| Positivos no teste | 20 anomalias | 3.075 fraudes rotuladas |
| Proporção positiva no teste | 1,00% | 0,76875% |
| Natureza do problema | Anomalias sintéticas de cadastro e integridade, conforme descrição do projeto | Padrões financeiros em dados sintéticos derivados de PaySim |
| Informação disponível | Campos cadastrais e da transação | Valores, saldos, atributos temporais e derivados |
| Avaliação usada nesta comparação | Resultados e seleção registrados em `resultados/10k/` | Metadados do XGBoost e matrizes do teste de 400 mil |

As bases diferem em tamanho, atributos, prevalência e significado da classe positiva. Portanto, o estudo compara **dois cenários experimentais**; não isola o efeito de aumentar a quantidade de registros. Para investigar especificamente esse efeito, seria necessário treinar com amostras crescentes da mesma base, mantendo atributos, protocolo e teste fixos.

### Comparação dos resultados selecionados

Para 10 mil, foram usadas as configurações identificadas em `resultados/10k/metadados.json` como selecionadas por família, e suas métricas em `resultados/10k/metricas_teste.csv`. Isso evita escolher retrospectivamente a configuração com melhor resultado no teste. Esses artefatos representam um experimento diferente daquele preservado em `resultados/tabelas/`.

| Algoritmo | Base | Configuração | Precisão | Recall | F1 |
|---|---|---|---:|---:|---:|
| XGBoost | 10 mil | SMOTE | 95,00% | 95,00% | 95,00% |
| XGBoost | 2 milhões | Baseline | 81,06% | 79,61% | 80,33% |
| Random Forest | 10 mil | Baseline | 95,00% | 95,00% | 95,00% |
| Random Forest | 2 milhões | Baseline | 80,83% | 66,24% | 72,82% |
| Regressão logística | 10 mil | Under-sampling | 100,00% | 40,00% | 57,14% |
| Regressão logística | 2 milhões | SMOTE | 64,41% | 65,92% | 65,16% |

As métricas de 2 milhões foram recalculadas das matrizes, com as limitações documentadas na Parte II. As estratégias e configurações diferem entre os experimentos; a tabela compara os pipelines selecionados, não apenas o algoritmo isolado.

**XGBoost:** apresenta F1 de 95,00% no cenário de 10 mil e 80,33% no de 2 milhões, uma diferença de −14,67 pontos percentuais. A queda descreve a mudança de cenário, sem demonstrar que o volume maior prejudica o algoritmo. Na base maior, mantém o melhor equilíbrio observado entre precisão e recall.

**Random Forest:** empata com o XGBoost em precisão, recall e F1 no experimento selecionado de 10 mil. Na base de 2 milhões, apresenta F1 de 72,82%, diferença de −22,18 pontos percentuais. O recall de 66,24% evidencia a maior proporção de positivos não detectados nesse segundo cenário.

**Regressão logística:** no teste de 10 mil, identifica oito das 20 anomalias sem falsos positivos. A precisão de 100% acompanha recall de apenas 40%; não representa detecção completa. Na base maior, o recall sobe para 65,92%, enquanto a precisão cai para 64,41%, resultando em F1 de 65,16%. O aumento de 8,01 pontos percentuais de F1 não pode ser atribuído exclusivamente ao aumento da base, pois os atributos, rótulos e balanceamento também mudaram.

Há uma diferença importante de resolução estatística: no teste de 10 mil, cada positivo detectado altera o recall em cinco pontos percentuais; no teste de 2 milhões, em aproximadamente 0,0325 ponto percentual. O teste maior oferece mais casos positivos para análise, mas isso não elimina vieses de geração dos dados ou de validação. Não foi realizado teste de significância entre os cenários.

### Transição para o aprofundamento

Os resultados mostram que a posição relativa e o equilíbrio entre os tipos de erro variam conforme o cenário. Na base de 10 mil, XGBoost e Random Forest selecionados atingem o mesmo F1; na base de 2 milhões, o XGBoost apresenta vantagem. A partir dessa comparação, a análise se concentra no dataset de 2 milhões para examinar como cada algoritmo utiliza os atributos, quais erros produz e quais limitações condicionam a interpretação dos resultados.

## Parte II — Funcionamento e pontos de atenção no dataset de 2 milhões

## Parecer

O XGBoost apresenta o melhor equilíbrio entre detecção de fraudes e falsos positivos nos artefatos examinados. Entretanto, os resultados ainda não demonstram desempenho em operação real. Os principais pontos de atenção são: divergências na documentação das métricas, natureza sintética dos dados, ajuste do limiar final em registros usados no treinamento e ausência de validação temporal independente.

Esta análise conferiu o código, os metadados, o README e a imagem das matrizes de confusão. Não houve novo treinamento nem recálculo a partir das previsões individuais do experimento de 2M. Os CSVs em `resultados/tabelas/` pertencem ao experimento de 10 mil registros e não servem para validar estes resultados.

## 1. Resultados e consistência das evidências

O teste contém 3.075 fraudes e 396.925 transações legítimas. A tabela abaixo foi recalculada a partir das contagens de verdadeiros positivos (TP), falsos positivos (FP) e falsos negativos (FN) exibidas no gráfico `04_matrizes_confusao_teste_2m.png`. Os resultados do XGBoost também coincidem com o JSON do modelo salvo. Para os outros modelos, as contagens do gráfico são uma referência provisória até a conferência das previsões originais.

| Indicador | XGBoost — Baseline | Random Forest — Baseline | Regressão logística — SMOTE |
|---|---:|---:|---:|
| Acurácia | 99,70% | 99,62% | 99,46% |
| Precisão | 81,06% | 80,83% | 64,41% |
| Recall | 79,61% | 66,24% | 65,92% |
| F1 | 80,33% | 72,82% | 65,16% |
| Fraudes detectadas — TP | 2.448 | 2.037 | 2.027 |
| Fraudes não detectadas — FN | 627 | 1.038 | 1.048 |
| Alarmes incorretos — FP | 572 | 483 | 1.120 |
| Total de alertas — TP + FP | 3.020 | 2.520 | 3.147 |
| FP / transações legítimas | 0,144% | 0,122% | 0,282% |
| Limiar exibido | 0,6264 | 0,9697 | 0,9697 |

O README informa precisão/recall/F1 de 85,08%/61,20%/71,18% para Random Forest e 67,24%/58,44%/62,53% para regressão logística. Esses valores não correspondem às matrizes disponíveis. Portanto, não devem ser combinados em uma mesma apresentação como se fossem resultados reconciliados de uma única execução.

O README informa AP de 88,41%, 83,99% e 66,05%, respectivamente; o gráfico mostra valores arredondados compatíveis. Apenas o XGBoost tem esse valor confirmado no JSON consultado. No código, a métrica chamada “PR-AUC” é calculada por `average_precision_score`: deve ser apresentada como **Average Precision (AP)**, distinguindo-a da integração trapezoidal da curva precisão–recall.

## 2. Pontos de atenção por modelo

### XGBoost

O XGBoost combina árvores construídas sequencialmente: novas árvores buscam reduzir os erros do conjunto existente segundo a função de perda. Neste projeto, os cortes das árvores podem representar relações e interações entre valor, saldos e outros atributos. As configurações Baseline testadas usam pesos maiores para a classe positiva, e a classificação final aplica um limiar ajustado para F1. Essa capacidade de representar interações oferece uma explicação possível para o desempenho, mas atribuir o resultado a variáveis específicas exige análise de importância e testes de remoção de atributos.

É o candidato mais forte nesta comparação: detecta 79,61% das fraudes e possui F1 de 80,33%. Comparado ao Random Forest, identifica 411 fraudes adicionais, com 89 falsos positivos adicionais. A precisão dos dois modelos é muito próxima nas matrizes: 81,06% contra 80,83%.

O ponto crítico são as **627 fraudes não detectadas**, correspondentes a 20,39% dos casos positivos. O desempenho agregado não elimina a necessidade de investigar esses erros por valor, horário, tipo de transação e perfil de saldo. Também há 572 transações legítimas sinalizadas; aproximadamente 18,94% dos alertas são incorretos. Essa proporção é diferente da taxa de falsos positivos de 0,144%, cujo denominador são todas as transações legítimas.

### Random Forest

O Random Forest combina árvores treinadas com aleatoriedade nas amostras e na escolha de atributos, agregando seus escores. As configurações Baseline deste experimento usam `class_weight="balanced"` e limites de profundidade. O conjunto permite representar relações não lineares, mas o resultado depende da profundidade, dos pesos e do limiar. O recall menor não pode ser atribuído apenas à arquitetura da floresta; é necessário avaliar o comportamento ao variar o corte de decisão.

Gera menos falsos positivos e menos alertas totais, o que pode reduzir a demanda de revisão. Em contrapartida, deixa passar **1.038 fraudes, ou 33,76%**. A menor quantidade de falsos alarmes não implica precisão maior que a do XGBoost nos artefatos conferidos.

Antes de escolher o Random Forest pela carga operacional, é necessário comparar os modelos sob um mesmo orçamento de alertas ou uma mesma taxa de falsos positivos. Os limiares atuais representam pontos de operação diferentes e foram escolhidos para F1, não para uma restrição comum de capacidade.

### Regressão logística

A regressão logística calcula uma combinação linear dos atributos transformados e aplica a função logística para produzir um escore. O pipeline usa padronização dos atributos numéricos; a configuração selecionada para 2 milhões usa SMOTE. Sem a inclusão explícita de interações ou transformações adicionais, a fronteira é linear no espaço dos atributos processados. A interpretação dos coeficientes também deve considerar escala, correlação entre saldos e razões derivadas e o efeito da reamostragem.

Detecta praticamente a mesma quantidade de fraudes que o Random Forest, mas produz **637 falsos positivos adicionais**. Em relação ao XGBoost, detecta 421 fraudes a menos e gera 548 falsos alarmes a mais. Aproximadamente 35,59% de seus alertas são incorretos.

O resultado é compatível com a hipótese de que interações não lineares favoreçam as árvores, mas não prova essa explicação: preparação dos atributos, regularização, SMOTE e limiar também influenciam o desempenho. A regressão permanece útil como referência comparativa; qualquer vantagem de latência ou custo computacional deve ser medida no experimento de 2M, sem importar tempos do teste de 10 mil registros.

## 3. Limitações metodológicas prioritárias

**Natureza da base e origem do sinal.** A documentação do autor descreve uma base sintética derivada do simulador PaySim. Os saldos foram gerados com distribuições condicionadas ao rótulo de fraude. Assim, o desempenho pode refletir o reconhecimento das regras do gerador. A expansão por bootstrap com ruído também exige investigar similaridade entre exemplos de treino e teste. Esses fatos limitam a generalização para transações bancárias reais. [Fonte: documentação do dataset](https://huggingface.co/datasets/andremessina/pix-fraud-br).

**Ajuste do limiar com reutilização do treino — confirmado no código.** Em `main.py`, `XSubVal` é extraído de `XTrain`. O modelo final é ajustado em todo `XTrain` e, depois, seu limiar é escolhido usando `XSubVal`, com `cv="prefit"`. Portanto, essa etapa usa registros que o estimador já viu. Isso pode tornar a escolha do limiar otimista; não significa, por si só, que os 400 mil registros de teste foram usados no ajuste do estimador. A correção é reservar dados exclusivos para o limiar ou usar previsões fora da amostra.

**Seleção final usando o teste — confirmada no código.** O vencedor salvo é escolhido pelo maior AP em `final_test_results`. Consequentemente, o teste participa da escolha do algoritmo e deixa de ser uma avaliação inteiramente intocada do vencedor. Deve-se selecionar algoritmo, configuração e limiar na validação, deixando uma avaliação final separada.

**Históricos e separação temporal.** `prepare2mDataset` calcula contagem, média e mediana por pagador antes da divisão aleatória. Embora exclua o valor atual da média e da mediana, não ordena explicitamente os registros por data. “Anterior” representa a ordem das linhas, sem garantia de passado cronológico. É necessário auditar a ordenação e reconstruir atributos usando somente informações disponíveis no instante da transação. O split aleatório também não comprova desempenho em períodos futuros ou em clientes inéditos.

**Disponibilidade dos atributos.** O modelo usa saldos posteriores e informações do recebedor. Para bloquear uma operação antes de sua execução, é necessário verificar quais atributos estão disponíveis ou podem ser calculados naquele momento. A presença de saldos posteriores não comprova vazamento automaticamente, mas exige uma auditoria da origem e do momento de disponibilidade.

**Acurácia e desbalanceamento.** Prever todas as operações como legítimas já produziria 99,23% de acurácia nesse teste, com recall zero. Acurácia acima de 99% deve ser acompanhada de AP, recall, precisão e contagens de erros. ROC-AUC elevada também não substitui a avaliação do ponto de operação.

**Limiar não é calibração de probabilidade.** O procedimento otimiza um corte para F1; ele não demonstra que os escores correspondem a probabilidades reais de fraude. Um limiar de 0,9697 não significa confiabilidade de 96,97%. Pesos de classe e reamostragem reforçam a necessidade de avaliar a calibração em dados com prevalência representativa.

**SMOTE e incerteza.** O pipeline aplica SMOTE depois do processamento que inclui one-hot encoding; a interpolação pode gerar categorias fracionárias e combinações financeiras pouco plausíveis. Além disso, as 18 configurações usam uma subvalidação de 20 mil registros, com aproximadamente 154 fraudes. Os boxplots com configurações diferentes não são intervalos de confiança nem repetições independentes. Recomenda-se quantificar a variação entre períodos e a incerteza das diferenças entre modelos.

## 4. Interpretação do impacto financeiro

O JSON do XGBoost registra R$ 4.405.302.309,06 em transações rotuladas como fraude no teste. Destas, R$ 3.955.017.221,14 foram sinalizadas corretamente, equivalentes a 89,78% do volume; R$ 450.285.087,92 não foram detectados. O volume de transações legítimas sinalizadas foi de R$ 452.574.394,53.

O recall financeiro supera o recall por quantidade, indicando maior cobertura proporcional do valor do que do número de fraudes. Isso recomenda avaliar também as faixas de menor valor, para que bons resultados monetários não ocultem muitos casos não detectados.

Esses valores representam uma simulação de sinalização sobre a base. Não comprovam dinheiro efetivamente recuperado ou prejuízo evitado. O volume de falsos positivos também não equivale ao custo de revisão ou à perda financeira causada pelo bloqueio. Os valores financeiros dos outros modelos constam do README, mas não foram reconciliados aqui com previsões individuais; não devem fundamentar uma decisão definitiva antes dessa conferência.

## 5. Encaminhamento recomendado

1. Reconciliar README, gráficos e metadados a partir das previsões individuais dos três modelos; salvar métricas, configurações, partições e versão da base por execução.
2. Separar treino, seleção de configuração, ajuste de limiar e teste final, eliminando a reutilização identificada.
3. Auditar históricos, disponibilidade dos saldos e possível similaridade entre registros; executar validação temporal e, conforme o uso pretendido, por cliente.
4. Comparar modelos sob a mesma capacidade de revisão e avaliar erros por segmento, calibração e estabilidade dos resultados.
5. Manter o XGBoost como candidato principal para a próxima avaliação. A conclusão defensável é superioridade nos artefatos deste experimento, condicionada às limitações registradas.

## Fontes examinadas

- [Métricas do experimento de 10 mil](../10k/metricas_teste.csv), [configurações selecionadas](../10k/metadados.json) e [procedimento](../10k/README.md): comparação inicial entre cenários.

- [README principal](../../README.md): métricas publicadas e descrição do experimento.
- [Metadados do XGBoost](../modelos/melhor_modelo_pix.json): métricas e volumes financeiros persistidos.
- [Matrizes de confusão do teste de 400 mil](../../src/notebooks/graficos/04_matrizes_confusao_teste_2m.png): contagens usadas nos recálculos.
- [Pipeline principal](../../main.py): divisão, seleção, treinamento e ajuste de limiar.
- [Preparação dos dados](../../src/data/cleaning.py) e [pipeline de atributos](../../src/models/xgboost.py).
- [Metadados do experimento de 10 mil](../metadados/metadados.json): identificação dos resultados que não pertencem ao escopo de 2M.
- [Documentação original do dataset](https://huggingface.co/datasets/andremessina/pix-fraud-br): origem e geração sintética.
