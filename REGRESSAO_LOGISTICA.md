# Documentação detalhada da regressão logística

## 1. Objetivo e estado da implementação

A regressão logística foi adicionada ao projeto para classificar transações Pix a partir do alvo `Anomalia` e permitir a comparação com XGBoost e Random Forest. A implementação utiliza `LogisticRegression`, do scikit-learn, com o mesmo conjunto de atributos e as mesmas estratégias de reamostragem dos demais algoritmos.

O projeto interpreta `0` como transação legítima e `1` como transação anômala. A previsão da classe positiva indica uma anomalia segundo os padrões aprendidos na base; ela não constitui confirmação de fraude.

O módulo está integrado ao `main.py` e ao `evaluate_models.py`. Na validação da implementação, os três testes automatizados passaram, incluindo as oito configurações em cinco folds com dados sintéticos. A avaliação completa na base Pix foi concluída: 24 configurações, cinco folds, 8.000 registros de treino e 2.000 de teste. A regressão logística selecionada usou Under e parâmetros padrão, com F1 de 0,5333, precisão de 0,80 e recall de 0,40 no teste. Os resultados dos três algoritmos estão em resultados/relatorios/RELATORIO.md e os gráficos organizados em resultados/README.md.

## 2. Fundamentos do algoritmo

### 2.1. Combinação linear e função logística

Apesar do nome, a regressão logística é utilizada aqui para classificação binária. Para cada transação, o modelo calcula uma combinação dos atributos transformados:

```text
z = b + w1*x1 + w2*x2 + ... + wp*xp
p(Anomalia = 1 | x) = 1 / (1 + exp(-z))
```

`b` é o intercepto, `wj` é o coeficiente aprendido para o atributo `xj`, e `p` é a probabilidade estimada da classe positiva. Os valores de `x` nessa expressão já passaram pela codificação das categorias e pela padronização dos atributos numéricos.

A função logística, também chamada de sigmoide, transforma a combinação linear em um valor entre zero e um. Valores positivos de `z` produzem probabilidades acima de 0,5; valores negativos produzem probabilidades abaixo de 0,5.

O projeto utiliza `predict_proba()` para obter probabilidades e `predict()` para obter as classes. O limiar de decisão permanece no padrão de 0,5, sem otimização. No caso binário com classes `[0, 1]`, uma pontuação exatamente nula, correspondente a probabilidade 0,5, é classificada como `0` pelo estimador.

### 2.2. Aprendizado e regularização

Durante o treinamento, o algoritmo ajusta os coeficientes para reduzir a perda logística, que penaliza previsões incompatíveis com os rótulos observados. Para uma observação com rótulo `y` e probabilidade `p`, a perda tem a forma:

```text
perda = -[y*log(p) + (1-y)*log(1-p)]
```

A configuração utilizada aplica a regularização L2 padrão da versão de scikit-learn declarada no projeto. Essa regularização penaliza coeficientes grandes. O parâmetro `C` controla inversamente sua intensidade: reduzir `C` aumenta a regularização.

A versão ajustada utiliza `C=0.1`, enquanto a versão padrão do experimento utiliza `C=1.0`. A configuração ajustada é uma hipótese experimental definida manualmente; seu nome não implica melhor desempenho.

### 2.3. Interpretação dos coeficientes

A relação linear é estabelecida no logaritmo das chances:

```text
log(p / (1-p)) = b + soma(wj*xj)
```

Mantidos os demais atributos constantes, um coeficiente positivo aumenta a pontuação da classe anômala, e um coeficiente negativo a reduz. Para atributos numéricos padronizados, a unidade corresponde a um desvio padrão aprendido no treino, quando o atributo tem variância positiva.

O encoder mantém todas as categorias, sem remover uma categoria de referência. Por isso, a interpretação categórica deve considerar diferenças entre coeficientes e a regularização, evitando tratar cada coeficiente isolado como uma comparação direta com uma categoria omitida. Os coeficientes expressam associações no modelo, não efeitos causais.

## 3. Organização dos arquivos

| Arquivo | Responsabilidade |
| --- | --- |
| `src/models/logistic_regression.py` | Criação do classificador, construção do pipeline e validação cruzada. |
| `src/models/xgboost.py` | Definição compartilhada dos atributos, pipeline básico e métricas. |
| `src/data/ingestion.py` | Carregamento da base e impressão das tabelas de validação. |
| `src/data/cleaning.py` | Extração de atributos e remoção de colunas. |
| `main.py` | Execução dos experimentos e exibição de gráficos interativos. |
| `evaluate_models.py` | Comparação dos candidatos, seleção por validação, teste reservado e gravação dos resultados. |
| `tests/test_logistic_regression.py` | Testes de integração com dados sintéticos. |
| `requirements.txt` | Versões declaradas das dependências. |

A função compartilhada `_getPipeline()` recebeu o argumento opcional `numeric_transformer`, cujo padrão é `"passthrough"`. A regressão logística fornece `StandardScaler()` nesse argumento. Os modelos de árvores continuam usando atributos numéricos sem padronização.

O novo módulo importa a função privada `_getPipeline` e `scoring` de `src/models/xgboost.py`. Assim, sua importação também depende da instalação de XGBoost. Uma futura extração desses componentes para um módulo comum pode reduzir esse acoplamento.

## 4. Origem e preparação dos dados

`loadData()` solicita ao Kagglehub o dataset identificado no código como `juniorbueno/pix-banking-transaction`. Em seguida, lê `comprovantes_pix_10000_anomalias.csv` usando ponto e vírgula como separador. O carregamento pode exigir rede quando a base não está disponível no cache.

A preparação segue esta ordem:

1. `extractDateFeatures()` converte `DataHora` e extrai hora, dia da semana e dia do mês. Datas inválidas recebem `-1` nos atributos derivados.
2. `extractPaymentTypeFeatures()` identifica a presença de `/` nos documentos como indicador de CNPJ e identifica o CPF do pagador igual a `000.000.000-00`.
3. `extractSameBankTransactionFeature()` compara o banco do pagador com o banco do recebedor.
4. `removeUnnecessaryColumns()` remove identificadores, nomes, documentos completos, chave Pix utilizada, data original, moeda e descrição.

Essas operações são transformações por registro; não calculam estatísticas agregadas de treino ou teste. A aprendizagem de categorias, médias e desvios ocorre posteriormente dentro do pipeline.

### 4.1. Atributos categóricos

| Atributo | Conteúdo utilizado |
| --- | --- |
| `TipoChave` | Tipo da chave Pix. |
| `Status` | Status registrado para a transação. |
| `Pagador_Banco` | Banco do pagador. |
| `Recebedor_Banco` | Banco do recebedor. |

### 4.2. Atributos numéricos

| Atributo | Conteúdo utilizado |
| --- | --- |
| `Valor` | Valor da transação. |
| `Hora` | Hora extraída da data, de 0 a 23, ou -1 para data inválida. |
| `Dia_Da_Semana` | Dia da semana, de 0 para segunda-feira a 6 para domingo, ou -1. |
| `Dia_Do_Mes` | Dia do mês, de 1 a 31, ou -1. |
| `Pagador_Is_CNPJ` | Indicador de presença de `/` no documento do pagador. |
| `Recebedor_Is_CNPJ` | Indicador de presença de `/` no documento do recebedor. |
| `Pagador_CPF_Zerado` | Indicador de CPF do pagador igual ao valor zerado esperado. |
| `Mesmo_Banco` | Indicador de igualdade entre os bancos. |

`Anomalia` é removida de `X` e utilizada como `y`. O `ColumnTransformer` seleciona explicitamente os atributos listados e descarta outras colunas. A quantidade final de atributos pode exceder 12, pois cada variável categórica gera várias colunas codificadas.

## 5. Pipeline de pré-processamento

```text
Dados preparados, sem a coluna Anomalia
                  |
                  v
          ColumnTransformer
          /               \
  Categóricos           Numéricos
  OneHotEncoder         StandardScaler
          \               /
           Matriz transformada
                  |
                  v
     Reamostragem opcional no treino
                  |
                  v
          LogisticRegression
```

### 5.1. Codificação categórica

O pipeline utiliza `OneHotEncoder(handle_unknown="ignore", sparse_output=False)`. Cada categoria conhecida no treino ganha uma coluna indicadora. Categorias desconhecidas durante a previsão são representadas por zeros no bloco da variável correspondente, sem erro por categoria nova.

A saída é uma matriz densa. Isso segue o padrão existente do projeto, mas pode consumir mais memória quando há muitas categorias distintas.

### 5.2. Padronização numérica

O `StandardScaler` aplica, para cada atributo numérico:

```text
x_padronizado = (x - media_do_treino) / desvio_do_treino
```

A transformação coloca os atributos em escalas comparáveis, favorecendo a otimização e tornando a penalização dos coeficientes menos dependente das unidades originais. Todos os oito atributos numéricos são padronizados, inclusive os indicadores binários. As colunas produzidas pelo one-hot não passam pelo scaler.

A padronização ocorre antes da reamostragem. Assim, suas estatísticas são estimadas sobre os registros originais do treino do fold. Essa ordem também altera as distâncias usadas por SMOTE em relação aos pipelines das árvores, que não padronizam os atributos numéricos.

### 5.3. Prevenção de vazamento entre treino e validação

O pipeline é do `imbalanced-learn`, permitindo incluir transformadores e reamostradores na mesma sequência. Em cada fold, encoder, scaler, reamostrador e classificador são ajustados usando apenas sua parcela de treinamento.

Na validação e na previsão, são aplicadas as transformações aprendidas e o classificador. A reamostragem não é aplicada: cada registro de entrada continua correspondendo a uma previsão. Ajustar o scaler ou executar SMOTE sobre toda a base antes de separar os folds alteraria esse procedimento e permitiria usar informação da validação no treino.

## 6. Interface do módulo

### 6.1. `getClassifier(adjusted=False, sampler=None)`

Retorna uma nova instância não treinada de `LogisticRegression`.

| Argumento | Efeito |
| --- | --- |
| `adjusted=False` | Usa `C=1.0` e `class_weight=None`. |
| `adjusted=True` | Usa `C=0.1`; aplica pesos balanceados somente se `sampler is None`. |
| `sampler` | Sua presença orienta a escolha de `class_weight`; esta função não executa reamostragem. |

As duas configurações usam `solver="lbfgs"`, `max_iter=2000` e `random_state=42`. O limite de iterações é uma escolha do experimento. A semente é mantida na configuração, mas não controla aleatoriedade do solver `lbfgs`; as sementes do split e dos reamostradores têm papel direto nas etapas correspondentes.

### 6.2. `getPipeline(sampler=None, classifier=None)`

Retorna um pipeline não treinado com as etapas `preprocessor`, `sampler` quando informado, e `classifier`. Se `classifier` for `None`, cria o classificador padrão com `getClassifier()`.

Para solicitar a configuração ajustada, é necessário criar explicitamente o classificador com `adjusted=True` e passá-lo ao pipeline. Informar apenas um sampler não ativa a configuração ajustada.

### 6.3. `crossValidate(df, XTrain, yTrain, sampler=None, classifier=None)`

Constrói o pipeline e chama `cross_validate()` com cinco folds e `error_score="raise"`. Falhas de treinamento são propagadas como exceções. Avisos de convergência continuam sendo avisos no fluxo normal; nos testes, são tratados como erros.

| Argumento | Papel |
| --- | --- |
| `df` | Mantido por compatibilidade com os demais módulos; não é utilizado internamente. |
| `XTrain` | DataFrame com os atributos preparados do conjunto de treinamento. |
| `yTrain` | Rótulos correspondentes aos registros de `XTrain`. |
| `sampler` | Reamostrador opcional executado dentro do pipeline. |
| `classifier` | Classificador opcional para substituir a configuração padrão. |

O retorno contém `fit_time`, `score_time`, `test_accuracy`, `test_precision`, `test_recall`, `test_f1` e `test_roc_auc`. Cada entrada contém cinco valores, um por fold. Nesse retorno, o prefixo `test_` identifica a parcela de validação do fold, não o conjunto de teste reservado de 20%.

A função retorna métricas e tempos; não retorna um modelo final treinado. Para esse objetivo, utilize `getPipeline().fit(...)` ou o avaliador completo.

## 7. Estratégias de balanceamento e parâmetros

O alvo desbalanceado pode favorecer um classificador que prevê quase sempre a classe majoritária. O projeto compara quatro estratégias para investigar esse efeito.

| Estratégia | Configuração | Comportamento no treino |
| --- | --- | --- |
| Sem reamostragem | `None` | Mantém a distribuição original. |
| SMOTE | `SMOTE(random_state=42)` | Gera exemplos sintéticos da classe minoritária por interpolação. |
| Under | `RandomUnderSampler(sampling_strategy=0.1, random_state=42)` | Reduz a classe majoritária até a razão minoritária/majoritária de 0,1. |
| SMOTETomek | `SMOTETomek(random_state=42)` | Combina SMOTE com remoção de pares Tomek. |

Na estratégia Under, `0.1` não significa remover 10% das transações. Por exemplo, com 20 registros minoritários, uma razão final de 0,1 corresponde a 200 majoritários. A distribuição de entrada precisa permitir atingir essa razão por subamostragem.

Com os parâmetros padrão utilizados, SMOTE precisa de pelo menos seis exemplos minoritários na parcela de treino de cada fold. Bases pequenas podem não satisfazer essa condição.

### 7.1. Oito configurações avaliadas

| Estratégia | C padrão | Peso padrão | C ajustado | Peso ajustado |
| --- | --- | --- | --- | --- |
| Sem reamostragem | 1,0 | None | 0,1 | balanced |
| SMOTE | 1,0 | None | 0,1 | None |
| Under | 1,0 | None | 0,1 | None |
| SMOTETomek | 1,0 | None | 0,1 | None |

`class_weight="balanced"` atribui pesos inversamente proporcionais à frequência das classes observadas no ajuste. Ele modifica a contribuição dos exemplos para o objetivo de treinamento, sem criar ou remover registros. Na configuração ajustada sem reamostragem, tanto `C` quanto os pesos mudam; portanto, a comparação não isola o efeito de apenas um parâmetro.

Não há `GridSearchCV`, busca aleatória ou otimização automática. As oito configurações correspondem a 40 ajustes nos cinco folds.

## 8. Validação cruzada, seleção e teste reservado

O fluxo completo do `evaluate_models.py` executa:

1. Carregamento e preparação dos dados.
2. Separação estratificada: 80% para treino e 20% para teste, com semente 42.
3. Construção de cinco folds estratificados sobre o treino com `shuffle=False`.
4. Avaliação das 24 configurações: três algoritmos, quatro estratégias e dois conjuntos de parâmetros.
5. Seleção de uma configuração por algoritmo pelo maior F1 médio de validação.
6. Desempate por maior AP e, depois, maior ROC-AUC médio.
7. Ajuste de cada configuração selecionada em todo o conjunto de treinamento.
8. Avaliação no teste reservado e gravação dos pipelines, tabelas e gráficos.

O avaliador reutiliza a mesma lista de índices dos folds em todas as configurações. São 120 ajustes de validação e três ajustes finais. O teste reservado não participa da seleção implementada.

O `crossValidate()` do módulo usa `cv=5`; para este classificador binário, o procedimento é estratificado e sem embaralhamento dos folds. No avaliador, a estratégia aparece de forma explícita em `StratifiedKFold`.

A função existente `getTrainTestSplit()` expõe `testSize` e `randomState`, mas atualmente usa internamente os valores fixos `0.2` e `42`. As chamadas do projeto usam esses mesmos valores. Alterar somente os argumentos dessa função não muda o split; o avaliador utiliza `train_test_split()` diretamente.

## 9. Métricas e interpretação

Considere a classe `1` como positiva:

| Contagem | Significado |
| --- | --- |
| TP | Transação anômala classificada como anômala. |
| TN | Transação legítima classificada como legítima. |
| FP | Transação legítima classificada como anômala. |
| FN | Transação anômala classificada como legítima. |

| Métrica | Definição ou interpretação |
| --- | --- |
| Acurácia | `(TP + TN) / total`; proporção geral de acertos. |
| Precisão | `TP / (TP + FP)`; proporção de anomalias reais entre os alertas. |
| Recall | `TP / (TP + FN)`; proporção de anomalias encontradas. |
| F1 | `2*TP / (2*TP + FP + FN)`; equilíbrio entre precisão e recall. |
| ROC-AUC | Resume a curva de recall contra taxa de falsos positivos ao variar o limiar. |
| AP | Average precision; resume precisão e recall com ponderação pelos incrementos de recall. |

Precisão, recall e F1 utilizam `zero_division=0`, evitando resultados indefinidos em casos sem denominador válido. ROC-AUC e AP usam pontuações ou probabilidades; não dependem apenas das classes produzidas pelo limiar de 0,5.

O módulo e o `main.py` usam as primeiras cinco métricas. O `evaluate_models.py` acrescenta AP. AP não é a área calculada por integração trapezoidal da curva precisão–recall.

Em bases desbalanceadas, a acurácia pode ser alta mesmo quando poucas anomalias são detectadas. Por exemplo, em uma base hipotética com 99% de transações legítimas, prever sempre `0` produz 99% de acurácia e recall zero para anomalias. Por isso, a análise deve considerar também F1, AP, precisão, recall e as contagens de FP e FN.

O `main.py` exibe as médias multiplicadas por 100. Os CSVs do avaliador armazenam métricas na escala de zero a um. O desvio padrão entre folds descreve variação entre essas partições; não é um intervalo de confiança do desempenho em produção.

## 10. Integração com os scripts

### 10.1. `main.py`

Executa os experimentos de XGBoost, Random Forest e regressão logística, imprime tabelas e exibe gráficos. Os resultados da regressão logística são adicionados às listas comparativas existentes.

As legendas `Sem Tunagem` e `Com Tunagem` são herdadas do fluxo original. Para a regressão logística, representam configurações padrão e ajustadas manualmente. O script faz validação no conjunto de treino; não executa a avaliação final no teste reservado nem salva o pipeline selecionado.

### 10.2. `evaluate_models.py`

Executa a comparação completa, seleciona candidatos e salva saídas em `resultados/`. Usa o backend gráfico `Agg`, gravando imagens sem abrir janelas. A validação dos folds é sequencial (`n_jobs=1`); os modelos de árvores são configurados para usar até quatro processadores.

A comparação mantém atributos e partições comuns, mas a regressão logística recebe padronização numérica. Essa diferença faz parte do pipeline avaliado e afeta também o espaço em que a reamostragem por vizinhança é calculada.

## 11. Arquivos produzidos pelo avaliador

Os arquivos abaixo foram produzidos na execução completa. O índice resultados/README.md inclui os 20 gráficos atuais e sua organização por algoritmo, dados e comparação.

| Caminho dentro de `resultados/` | Conteúdo |
| --- | --- |
| `modelos/logistic_regression.joblib` | Pipeline selecionado da regressão logística, ajustado no treino. |
| `graficos/logistic_regression/01_configuracoes_validacao.png` | Mapa de calor das oito configurações e seis métricas. |
| `graficos/logistic_regression/02_matriz_confusao.png` | Matriz de confusão no teste reservado. |
| `graficos/logistic_regression/03_curvas_roc_precisao_recall.png` | Curvas ROC e precisão–recall no teste. |
| `tabelas/metricas_validacao_cruzada.csv` | Médias, desvios e tempos das configurações avaliadas. |
| `tabelas/metricas_teste.csv` | Métricas de teste, configuração selecionada e TN, FP, FN e TP por algoritmo. |
| `tabelas/predicoes_teste.csv` | Índices originais, rótulos reais, probabilidades e previsões dos três algoritmos. |
| `metadados/metadados.json` | Distribuições das classes, tamanhos dos conjuntos, semente, folds e seleção. |
| `relatorios/RELATORIO.md` | Procedimento experimental, tabelas e interpretação dos resultados. |
| `graficos/comparacao/01_validacao_todas_configuracoes.png` | Comparação das 24 configurações na validação. |
| `graficos/comparacao/02_metricas_teste.png` | Comparação das métricas dos três modelos selecionados. |
| `graficos/comparacao/03_matrizes_confusao.png` | Matrizes de confusão no mesmo teste. |
| `graficos/comparacao/04_curvas_comparativas.png` | Curvas ROC e precisão–recall comparativas. |

As colunas específicas da regressão logística em `predicoes_teste.csv` são `prob_logistic_regression` e `pred_logistic_regression`.

O avaliador sobrescreve arquivos existentes com os mesmos nomes. O CSV de validação é salvo progressivamente; se a execução for interrompida, ele pode conter apenas parte das configurações e coexistir com resultados antigos nos demais arquivos. A mensagem `CONCLUÍDO` no terminal indica que o fluxo chegou ao fim.

## 12. Ambiente e execução

A implementação reutiliza dependências já declaradas: scikit-learn 1.7.2, imbalanced-learn 0.14.0, pandas 2.3.3, NumPy 2.5.2, XGBoost 3.4.1, Matplotlib 3.10.6, Joblib 1.5.2 e Kagglehub 0.4.0. Esses números descrevem `requirements.txt`; não representam uma recomendação de versões mais recentes.

Execute os comandos na raiz do projeto, em PowerShell, usando o ambiente virtual existente:

```powershell
# Conferir a consistência das dependências instaladas
.\.venv\Scripts\python.exe -m pip check

# Executar os testes com dados sintéticos
.\.venv\Scripts\python.exe -m unittest discover -s tests -v

# Exibir experimentos e gráficos interativos dos três algoritmos
.\.venv\Scripts\python.exe main.py

# Gerar métricas, gráficos, relatório e modelos selecionados
.\.venv\Scripts\python.exe evaluate_models.py
```

Para preparar um novo ambiente em uma cópia do projeto com Python 3.13 disponível:

```powershell
py -3.13 -m venv .venv
.\.venv\Scripts\python.exe -m pip install -r requirements.txt
.\.venv\Scripts\python.exe -m pip check
```

O avaliador executa todos os algoritmos. Não existe uma opção de linha de comando implementada para selecionar apenas a regressão logística.

## 13. Exemplo de uso independente do módulo

O exemplo abaixo carrega a base, prepara os atributos e avalia somente a configuração padrão da regressão logística. Pode ser salvo como script na raiz do projeto. Não realiza a seleção entre as oito configurações.

```python
from sklearn.model_selection import train_test_split
from src.data.ingestion import loadData
from src.data.cleaning import (
    extractDateFeatures,
    extractPaymentTypeFeatures,
    extractSameBankTransactionFeature,
    removeUnnecessaryColumns,
)
from src.models.logistic_regression import crossValidate, getPipeline

df = loadData()
for prepare in (
    extractDateFeatures,
    extractPaymentTypeFeatures,
    extractSameBankTransactionFeature,
    removeUnnecessaryColumns,
):
    df = prepare(df)

X = df.drop(columns="Anomalia")
y = df["Anomalia"]
X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.2, stratify=y, random_state=42
)

scores = crossValidate(df, X_train, y_train)
print("F1 médio de validação:", scores["test_f1"].mean())

pipeline = getPipeline()
pipeline.fit(X_train, y_train)
probabilidades = pipeline.predict_proba(X_test)[:, 1]
previsoes = pipeline.predict(X_test)
print("Primeiras probabilidades:", probabilidades[:5])
print("Primeiras classes:", previsoes[:5])
```

Para construir a configuração ajustada com SMOTE, substitua a construção do pipeline por:

```python
from imblearn.over_sampling import SMOTE
from src.models.logistic_regression import getClassifier, getPipeline

sampler = SMOTE(random_state=42)
classifier = getClassifier(adjusted=True, sampler=sampler)
pipeline = getPipeline(sampler=sampler, classifier=classifier)
```

### 13.1. Carregamento do modelo salvo

Após concluir o avaliador, o pipeline selecionado pode ser carregado com Joblib:

```python
import joblib

pipeline = joblib.load("resultados/modelos/logistic_regression.joblib")
# X_novo deve conter os 12 atributos preparados descritos neste documento.
probabilidades = pipeline.predict_proba(X_novo)[:, 1]
previsoes = pipeline.predict(X_novo)
```

`X_novo` deve ser fornecido pelo código consumidor. As funções de extração de atributos de `src/data/cleaning.py` não fazem parte do objeto salvo. O pipeline contém a codificação, a padronização, o reamostrador quando aplicável e o classificador; ele não recebe diretamente um CSV bruto sem os atributos derivados.

Não é necessário nem adequado ajustar novamente o scaler para prever novos registros. O pipeline reutiliza as estatísticas armazenadas no treino.

## 14. Testes realizados e alcance da validação

Os testes estão em `tests/test_logistic_regression.py` e usam uma base sintética de 300 registros: 280 da classe 0 e 20 da classe 1, com os mesmos nomes de atributos exigidos pelo pipeline.

| Teste | Verificação |
| --- | --- |
| `test_all_sampling_strategies_and_settings` | Executa quatro estratégias e duas configurações em cinco folds; exige cinco valores finitos por métrica e transforma avisos de convergência em erros. |
| `test_scaler_uses_training_data_and_accepts_unseen_categories` | Compara médias do scaler com os dados usados no ajuste, prevê categorias desconhecidas e verifica que a previsão não altera essas médias. |
| `test_tree_pipeline_keeps_unscaled_numeric_features` | Verifica que o pipeline compartilhado mantém `passthrough` como padrão numérico para as árvores. |

Na execução de validação da implementação, os três testes passaram. O primeiro realiza 40 ajustes nos folds; o segundo realiza um ajuste adicional. Também foi verificada a sintaxe dos módulos alterados e a importação do pipeline.

Esses testes demonstram funcionamento conjunto dos componentes nos cenários sintéticos verificados. Não comprovam qualidade preditiva na base Pix, ausência de vazamento em toda a origem dos dados, calibração das probabilidades nem execução completa dos relatórios e gráficos com dados reais.

## 15. Limitações e diagnóstico de problemas

### 15.1. Limites do modelo e do experimento

A regressão logística representa uma relação linear no logaritmo das chances. O pipeline atual não cria interações, termos polinomiais ou codificação cíclica para hora e dia. Padrões que dependam dessas relações podem não ser representados adequadamente.

SMOTE após one-hot pode gerar valores fracionários nas colunas categóricas. A interpolação também afeta indicadores binários padronizados. Esses registros sintéticos nem sempre correspondem a uma transação possível.

Reamostragem e pesos de classe alteram o ajuste em relação à prevalência original. As probabilidades produzidas não foram submetidas a uma etapa de calibração. Seu valor numérico deve ser avaliado antes de ser interpretado como frequência esperada de anomalias em uso real.

A divisão é aleatória e estratificada, sem separação temporal ou agrupamento por pagador e recebedor. A origem e a independência dos registros não foram auditadas nesta implementação. O uso de `Status` pressupõe que esse campo esteja disponível no momento em que a previsão seria realizada.

O pipeline não inclui imputação geral de valores ausentes. A preparação atual identifica CNPJ pelo formato textual e pressupõe documentos compatíveis; valores ausentes ou formatos diferentes podem exigir tratamento adicional. O valor sentinela `-1` das datas é tratado como número pelo modelo.

### 15.2. Problemas de execução

| Sintoma | Verificação sugerida |
| --- | --- |
| `ModuleNotFoundError` | Executar a partir da raiz e usar o Python da `.venv` com as dependências instaladas. |
| Coluna obrigatória ausente | Conferir os nomes das colunas brutas e executar todas as funções de preparação antes do pipeline. |
| Erro de vizinhos no SMOTE | Conferir a quantidade de exemplos minoritários em cada treino de fold; o padrão requer pelo menos seis. |
| Erro de razão no Under | Conferir se a razão 0,1 pode ser atingida removendo registros majoritários. |
| `ConvergenceWarning` | Conferir escalas, valores extremos e o número de iterações; o limite de 2000 não garante convergência para qualquer base. |
| Falha no carregamento do dataset | Conferir acesso ao Kaggle, cache local e presença do CSV esperado. |
| Ausência de `logistic_regression.joblib` | Confirmar que o avaliador chegou à etapa de ajuste final e gravação. Os testes não geram esse arquivo. |
| Resultados incompletos ou de execuções diferentes | Conferir a conclusão do avaliador e a presença das 24 configurações e dos três modelos nos CSVs. |

## 16. Possíveis extensões

Extensões futuras incluem busca de hiperparâmetros dentro do treino, ajuste de limiar por validação, calibração de probabilidades, tratamento explícito de ausentes, atributos cíclicos para tempo e avaliação por divisão temporal ou por grupos. Essas possibilidades não fazem parte da implementação atual.

A comparação final entre regressão logística, Random Forest e XGBoost deve se apoiar nos resultados do avaliador, considerando métricas, falsos positivos, falsos negativos e tempos. Na execução realizada, Random Forest obteve o maior F1 no teste reservado (0,9744), seguido de XGBoost (0,9500) e regressão logística (0,5333). Essa ordenação descreve apenas a base e a divisão avaliadas.


