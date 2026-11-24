# Protocolo temporal e condições de aplicação

A entrada oficial é `main.py`. `src/cli.py` centraliza as rotas e `src/workflow.py` executa o protocolo. Os cenários são problemas distintos: 10k prevê anomalia cadastral; 2m prevê fraude financeira. Não se transfere um modelo entre cenários. As listas ordenadas de colunas e a versão do contrato ficam em `src/data/temporal.py`; treino e API usam esse mesmo contrato.

## Particionamento e seleção

Após validar e excluir datas inválidas, ordenar por instante UTC e identificador estável. Datas sem fuso das fontes são interpretadas em Brasília. Os cortes aproximados são 70% desenvolvimento, 10% calibração e 20% teste. Eventos com o mesmo timestamp ficam juntos; por isso os tamanhos reais diferem dos percentuais nominais. O manifesto registra contagens, classes e limites reais, que prevalecem sobre percentuais arredondados.

Dentro do desenvolvimento, três janelas progressivas comparam configurações pela média da Average Precision (AP). A busca padrão usa os últimos 80 mil registros do desenvolvimento, preservando empates; `--tuning-rows 0` usa todo esse período. Escolhem-se configuração por família e campeão exclusivamente pela validação. O teste não participa da seleção, do pré-processamento, da reamostragem ou do ajuste do limiar. Um concorrente superar o campeão no teste não autoriza trocar o campeão.

Cada vencedor é ajustado no desenvolvimento completo. O limiar maximiza F1 no período reservado de calibração e é congelado antes do teste. O teste é avaliado uma vez para relatar generalização temporal; `evaluate` apenas reproduz a avaliação de artefatos congelados. O F1 de calibração mede ajuste, não validação independente.

## Balanceamento e limiar

Comparam-se original, pesos de classe, subamostragem, SMOTENC e SMOTENC com Tomek, em três famílias e duas configurações de parâmetros. A AP privilegia a ordenação dos positivos em dados desbalanceados, sem depender de limiar fixo. Não há técnica universalmente vencedora: a justificativa de cada escolha é a AP registrada em `validation.json`, com as contagens antes/depois da amostragem. Pesos e amostragem são calculados apenas no treino de cada janela; imputação, escalas e categorias também são ajustadas ali. Validação, calibração e teste mantêm suas distribuições naturais. Não se combinam pesos com síntese.

F1 equilibra precisão e revocação como objetivo experimental. Não foram fornecidos custos de investigação, bloqueio ou perda por fraude; o limiar não é financeiramente ótimo. Escores não são probabilidades calibradas. Alterar o objetivo exige nova calibração em dados apropriados e novo teste independente, nunca otimização no teste atual.

## Disponibilidade antes da decisão

| Cenário / atributos | Origem necessária e condição |
|---|---|
| Ambos: `amount`, tipo, horário | Solicitação anterior à execução; calendário derivado de `occurred_at` em Brasília |
| 10k: bancos, tipo de chave e indicadores documentais | Cadastro/roteamento conhecido antes da análise; não usar atualizações posteriores |
| 2m: `saldo_anterior_pagador` | Fonte autorizada observada antes da decisão |
| 2m: `saldo_anterior_recebedor` | Exige que a integração também tenha acesso autorizado ao saldo do recebedor antes da decisão; indisponível em muitas integrações do pagador |
| 2m: contagem, média e mediana do pagador | Somente eventos estritamente anteriores; exclui a própria transação e todos os eventos com horário igual |
| 2m: noite, dia útil e limite noturno | Derivações do horário e do valor, sem informação posterior |

**O modelo 2m só é aplicável se os dois saldos forem observáveis antes da decisão.** O nome de uma coluna não prova sua disponibilidade. A API exige os campos e valida seus valores, mas não autentica a origem nem comprova o instante em que foram observados. Não inventar saldos nem preencher zero para contornar falta de acesso. Uma integração sem esses atributos precisa de outro contrato, treinamento e avaliação; o modelo 10k tampouco é substituto para fraude financeira.

Rótulos, saldos posteriores, estado operacional e identificadores não são preditores. Na API, o banco calcula o histórico de pagamentos `completed` com `executed_at < occurred_at` e `confirmed_at <= analyzed_at`. Confirmações atrasadas não reescrevem fotografias de decisões passadas. A previsão não altera automaticamente o estado do pagamento e confirmação operacional não é rótulo de fraude.

## Limites dos dados de origem

As fontes não trazem timestamps de conhecimento dos rótulos. Assume-se que os rótulos de treino já eram conhecidos no corte; não é possível provar ausência de vazamento por atraso de rotulagem. O histórico offline também pressupõe disponibilidade imediata dos eventos anteriores. A API controla conhecimento de confirmações, mas isso não elimina as limitações da avaliação offline.

No 10k de referência, 24 datas inválidas são excluídas, todas da classe positiva: restam 9.976 linhas e 76 positivos dos 100 originais. As métricas temporais não cobrem essas 24 anomalias nem podem ser apresentadas como desempenho sobre todos os positivos originais. Consulte `quality.json`; não inventar ordem temporal para datas inválidas. Mudanças de prevalência entre períodos e a pequena quantidade de positivos de teste limitam a estabilidade das estimativas.

## Reprodução

Use `.venv-gpu/bin/python main.py` ou o Makefile. `train` produz uma execução nova, preserva anteriores e publica `active.json` somente ao concluir. `verify` confere artefatos e métricas; `report` gera tabelas e limitações. A API usa o campeão ativo, com validação de contrato, versões e hash. Não alterar manifestos ou modelos antigos para aparentar um novo treinamento. Atualizações de texto dos relatórios não mudam o experimento nem justificam retreino.
