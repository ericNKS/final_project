# Documentação da implementação do Random Forest

## 1. Objetivo

O Random Forest foi adicionado ao projeto de detecção de anomalias em transações Pix para permitir a comparação com o XGBoost já existente. A variável alvo é `Anomalia`: o projeto interpreta `0` como transação legítima e `1` como transação suspeita. Uma anomalia sinalizada pelo modelo não constitui, por si só, confirmação de fraude.

A implementação usa os mesmos dados de treinamento, atributos, estratégias de reamostragem e métricas do XGBoost. Isso permite comparar os algoritmos dentro do mesmo procedimento experimental.

## 2. Como o algoritmo funciona

O Random Forest combina várias árvores de decisão. Cada árvore aprende com uma amostra dos dados e considera subconjuntos de atributos durante a construção. Para classificação, o conjunto das árvores produz probabilidades para as classes e uma classe prevista.

Essa combinação permite representar relações não lineares entre os atributos da transação. No projeto, o algoritmo é fornecido pelo `RandomForestClassifier`, da biblioteca Scikit-learn. Não foi necessário implementar manualmente as árvores.

## 3. Arquivos envolvidos

| Arquivo | Papel na implementação |
| --- | --- |
| `src/models/random_forest.py` | Novo módulo responsável pela validação cruzada do Random Forest. |
| `main.py` | Integração dos experimentos, tabelas e gráficos comparativos. |
| `src/models/xgboost.py` | Módulo existente do qual são reutilizados o pipeline e as métricas. |
| `src/data/cleaning.py` | Preparação dos atributos, já existente e utilizada pelos dois algoritmos. |
| `requirements.txt` | Dependências do projeto, com duas versões corrigidas por incompatibilidade. |
| `.venv/` | Ambiente virtual criado com Python 3.13 para instalar e validar as dependências. |

O pré-processamento não foi recriado no módulo Random Forest: ele reutiliza a função `_getPipeline` e o dicionário `scoring` de `src/models/xgboost.py`. Isso evita manter duas definições diferentes para a mesma comparação. Como consequência, o novo módulo depende desse arquivo e da instalação do XGBoost.

## 4. Dados e preparação dos atributos

O fluxo existente carrega a base `juniorbueno/pix-banking-transaction`, do Kaggle, por meio de `kagglehub`. O arquivo utilizado é `comprovantes_pix_10000_anomalias.csv`.

Antes do treinamento, o projeto deriva atributos de data, documentos e bancos e remove campos como nomes, documentos completos, identificador da transação e chave Pix utilizada. O Random Forest recebe a mesma preparação já aplicada ao XGBoost.

Os atributos usados pelo pipeline são:

| Grupo | Atributos |
| --- | --- |
| Categóricos | `TipoChave`, `Status`, `Pagador_Banco`, `Recebedor_Banco`. |
| Numéricos | `Valor`, `Hora`, `Dia_Da_Semana`, `Dia_Do_Mes`, `Pagador_Is_CNPJ`, `Recebedor_Is_CNPJ`, `Pagador_CPF_Zerado`, `Mesmo_Banco`. |

As variáveis categóricas são transformadas com `OneHotEncoder(handle_unknown="ignore", sparse_output=False)`. Categorias desconhecidas na validação não provocam erro no encoder. As variáveis numéricas passam diretamente pelo `ColumnTransformer`, sem normalização adicional.

O processamento segue esta sequência:

```text
Dados preparados → Codificação das categorias → Reamostragem opcional → Random Forest
```

## 5. Novo módulo de avaliação

O arquivo `src/models/random_forest.py` disponibiliza a função:

```python
crossValidate(df, XTrain, yTrain, sampler=None, classifier=None)
```

- `XTrain`: atributos do conjunto de treinamento.
- `yTrain`: rótulos correspondentes.
- `sampler`: estratégia opcional de reamostragem.
- `classifier`: classificador opcional com parâmetros personalizados.
- `df`: argumento mantido para seguir a interface existente; não é utilizado dentro da função.

Quando nenhum classificador é informado, a função cria:

```python
RandomForestClassifier(random_state=42, n_jobs=-1)
```

`random_state=42` fixa a semente das operações aleatórias. `n_jobs=-1` permite utilizar os processadores disponíveis. Os demais parâmetros ficam nos padrões da biblioteca; na versão instalada, o modelo padrão possui 100 árvores.

A função monta o pipeline compartilhado e chama `cross_validate` com `cv=5` e as métricas existentes. O parâmetro `error_score="raise"` faz erros de treinamento serem apresentados diretamente, em vez de substituí-los silenciosamente por pontuações inválidas.

A função retorna os resultados de validação cruzada. Ela não devolve um modelo final treinado para uso em produção.

## 6. Estratégias de balanceamento

Foram adicionadas quatro estratégias ao `main.py`:

| Estratégia | Implementação | Finalidade |
| --- | --- | --- |
| Sem reamostragem | `sampler=None` | Estabelecer uma referência usando a distribuição original do treino. |
| SMOTE | `SMOTE(random_state=42)` | Criar exemplos sintéticos da classe minoritária. |
| Subamostragem | `RandomUnderSampler(sampling_strategy=0.1, random_state=42)` | Reduzir exemplos da classe majoritária. |
| SMOTETomek | `SMOTETomek(random_state=42)` | Combinar a geração de exemplos com a limpeza de pares Tomek. |

Na subamostragem, `sampling_strategy=0.1` representa a razão entre a quantidade da classe minoritária e da majoritária após a reamostragem. Não significa descartar 10% das transações. Essa configuração exige que a distribuição de entrada permita atingir a razão por redução da classe majoritária.

A reamostragem fica dentro do pipeline do `imbalanced-learn`. Em cada fold, ela é aplicada somente aos dados usados para treinar. A parcela de validação não é reamostrada. O encoder também é ajustado dentro de cada fold.

## 7. Configurações padrão e ajustada

Para cada estratégia, são executadas duas configurações, totalizando oito avaliações do Random Forest.

A configuração padrão usa `RandomForestClassifier(random_state=42, n_jobs=-1)`. A configuração ajustada é:

```python
RandomForestClassifier(
    n_estimators=200,
    max_depth=12,
    min_samples_leaf=2,
    class_weight="balanced" if forestSampler is None else None,
    random_state=42,
    n_jobs=-1,
)
```

| Parâmetro | Valor e intenção |
| --- | --- |
| `n_estimators` | 200 árvores para a configuração ajustada. |
| `max_depth` | Profundidade máxima de 12 para limitar a complexidade das árvores. |
| `min_samples_leaf` | Pelo menos 2 exemplos em cada folha para evitar folhas baseadas em um único registro. |
| `class_weight` | `"balanced"` apenas sem reamostragem, atribuindo maior peso à classe menos frequente; `None` nas demais estratégias. |
| `random_state` | 42 para fixar a semente. |
| `n_jobs` | -1 para permitir processamento paralelo. |

Esses valores foram definidos como configuração inicial de comparação. Não foi executada busca automática com GridSearchCV ou RandomizedSearchCV, e não há evidência de que sejam os melhores parâmetros. Como vários parâmetros mudam juntos, a comparação não isola o efeito individual de cada ajuste.

## 8. Divisão dos dados e métricas

O fluxo existente separa 80% dos dados para treinamento e 20% para teste, com estratificação pela coluna `Anomalia` e semente 42. O Random Forest usa o mesmo `XTrain` e `yTrain` do XGBoost.

A validação cruzada em cinco folds é realizada exclusivamente sobre o conjunto de treinamento. Na classificação binária, o Scikit-learn utiliza folds estratificados quando recebe `cv=5`; essa chamada não ativa embaralhamento adicional dos folds.

| Métrica | Interpretação |
| --- | --- |
| Acurácia | Proporção total de classificações corretas. |
| Precisão | Entre as transações classificadas como suspeitas, quantas pertencem à classe positiva. |
| Recall | Entre as transações realmente rotuladas como anômalas, quantas foram detectadas. |
| F1 | Média harmônica entre precisão e recall. |
| ROC-AUC | Capacidade de distinguir as classes a partir das pontuações do modelo em diferentes limiares. |

Precisão, recall e F1 utilizam `zero_division=0`. Isso evita erro de divisão quando determinada pontuação não pode ser calculada, por exemplo quando o modelo não prevê nenhuma transação positiva.

O projeto calcula a média dos cinco folds e a apresenta multiplicada por 100 nas tabelas e gráficos. Em dados desbalanceados, a acurácia deve ser interpretada junto com as métricas da classe positiva.

Apesar dos nomes retornados por `cross_validate`, como `test_f1`, essas pontuações pertencem aos folds de validação. Elas não são resultados do conjunto separado em `XTest` e `yTest`.

## 9. Integração com tabelas e gráficos

No `main.py`, a função do novo módulo é importada com o nome `crossValidateRandomForest`, evitando conflito com a função do XGBoost.

Foram criadas as listas `forestResults` e `adjustedForestResults`, que armazenam os resultados das configurações padrão e ajustada. A função existente `printTrainResult` apresenta duas tabelas do Random Forest.

Depois, os resultados são acrescentados às listas usadas pelos gráficos existentes. Cada gráfico passa a comparar oito combinações de algoritmo e balanceamento, com barras para as configurações padrão e ajustada.

Também foram substituídos nomes técnicos como `overXgbResults` por rótulos descritivos, como `XGBoost + SMOTE`, e a largura dos gráficos comparativos foi ampliada de 10 para 14 polegadas para acomodar os novos modelos.

As legendas existentes `Sem Tunagem` e `Com Tunagem` foram mantidas. Para o Random Forest, elas representam as configurações padrão e manualmente ajustada, sem busca automática de parâmetros.

## 10. Dependências e correções do ambiente

Foi criado o ambiente virtual `.venv` com o Python 3.13 disponível na máquina e instaladas todas as dependências de `requirements.txt`, incluindo as bibliotecas científicas e as ferramentas do Jupyter.

A instalação inicial terminou, mas os testes de importação identificaram duas incompatibilidades em execução:

| Combinação inicial | Problema encontrado | Correção aplicada |
| --- | --- | --- |
| Scikit-learn 1.9.0 e imbalanced-learn 0.14.0 | Importação de `_is_pandas_df`, indisponível nessa versão do Scikit-learn. | Scikit-learn alterado para 1.7.2. |
| Pandas 3.0.5 e Statsmodels 0.14.5 | Falha na importação de `statsmodels.api` relacionada à assinatura de `deprecate_kwarg`. | Pandas alterado para 2.3.3. |

As versões corrigidas foram instaladas no ambiente e registradas em `requirements.txt`. As demais versões declaradas foram mantidas.

## 11. Verificações realizadas

A validação incluiu:

1. Análise de sintaxe de `main.py` e `src/models/random_forest.py`.
2. Importação das dependências e dos módulos de dados, modelos e logging do projeto.
3. Execução de `pip check`, com o resultado final `No broken requirements found.`
4. Treinamento de Random Forest e XGBoost com as quatro estratégias de balanceamento, usando validação cruzada em cinco folds.
5. Verificação de que as cinco métricas retornavam cinco valores finitos em cada avaliação.

O teste de treinamento utilizou uma base sintética com 300 registros: 280 da classe 0 e 20 da classe 1. Foram usadas 10 árvores por modelo para uma verificação rápida da integração. Ao todo, oito combinações foram verificadas, com 40 ajustes nos folds.

Esses testes demonstram que as dependências, a codificação, a reamostragem, o treinamento e as métricas funcionam em conjunto. Eles não medem o desempenho real na base Pix nem validam a superioridade das configurações de 100 ou 200 árvores.

## 12. Como executar

Na raiz do projeto, execute no PowerShell:

```powershell
.\.venv\Scripts\python.exe main.py
```

Esse comando usa o ambiente no qual as dependências foram instaladas e testadas. O script executa os experimentos do XGBoost e do Random Forest e exibe os gráficos. O carregamento da base utiliza o Kagglehub e pode exigir acesso à rede quando o dataset não estiver disponível no cache.

Para recriar o ambiente em outra máquina com Python 3.13:

```powershell
py -3.13 -m venv .venv
.\.venv\Scripts\python.exe -m pip install -r requirements.txt
.\.venv\Scripts\python.exe -m pip check
```

Para abrir o JupyterLab no mesmo ambiente:

```powershell
.\.venv\Scripts\python.exe -m jupyterlab
```

## 13. Limitações e trabalho restante

A implementação inicial adicionou experimentos comparativos. Posteriormente, o script evaluate_models.py executou as 16 configurações na base Pix e avaliou os dois modelos selecionados no teste reservado. Gráficos, métricas e interpretação estão em resultados/relatorios/RELATORIO.md. O main.py mantém o fluxo original de validação cruzada; a avaliação final de teste é feita pelo novo script.

O script evaluate_models.py agora salva os pipelines selecionados e ajustados no conjunto de treino em resultados/modelos/random_forest.joblib e resultados/modelos/xgboost.joblib. Não foi ajustado o limiar de decisão, nem realizado treinamento em todos os 10.000 registros para produção. A escolha é baseada no F1 médio de validação, e os resultados de teste devem ser interpretados considerando o número reduzido de anomalias.

O SMOTE é aplicado após a codificação one-hot, como no pipeline existente. A interpolação pode produzir valores fracionários nas colunas categóricas codificadas; esse comportamento deve ser considerado ao interpretar os experimentos ou ao estudar uma alternativa específica para dados categóricos.

O módulo Random Forest reutiliza uma função privada (`_getPipeline`) localizada no módulo XGBoost. Se o projeto crescer, o pré-processamento e as métricas poderão ser movidos para um módulo compartilhado, preservando o procedimento usado na comparação.

