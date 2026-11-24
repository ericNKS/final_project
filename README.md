# Classificação temporal de transações Pix

Experimentos separados para **10k: anomalias cadastrais** e **2M: fraude financeira**, com classificação antes da execução e API local com PostgreSQL 18.

O fluxo divide os dados cronologicamente, seleciona pela Average Precision (AP) na validação progressiva, ajusta o limiar por F1 em um período posterior e usa o teste somente para avaliação. Não há transferência automática de um modelo entre bases com alvos e atributos diferentes.

## Instalação e execução local

O projeto tem um único ponto de entrada, `main.py`, e um único arquivo de dependências, `requirements.txt`. O código continua organizado por responsabilidade dentro de `src/`.

Use Python 3.14, GPU NVIDIA compatível com CUDA 13, GNU Make e Docker Compose ou Podman Compose. O Makefile detecta Docker/Podman e usa diretamente `.venv-gpu/bin/python`, independentemente do ambiente ativo no terminal:

```bash
make setup
# Edite .env com as credenciais do seu banco.
make serve
```

`make setup` prepara `.venv-gpu`, instala as versões GPU, API e testes de `requirements.txt` e copia `.env.example` somente se `.env` ainda não existir. O ambiente CPU antigo não é sobrescrito; sua lista histórica está em [docs/CPU_ENVIRONMENT.md](docs/CPU_ENVIRONMENT.md).

`make serve` aguarda o PostgreSQL, aplica as migrações pendentes e inicia a API em <http://[::1]:8000/docs>. O `.env` é carregado pelo `main.py`, com suporte a Bash, Fish e outros shells sem executar `source`. Variáveis já exportadas têm prioridade.

| Comando | Comportamento |
|---|---|
| `make db` | Iniciar o PostgreSQL e aguardar ficar saudável |
| `make migrate` | Aplicar migrações pendentes |
| `make serve` | Iniciar banco, migrar e servir a API local |
| `make train SCENARIO=10k` | Treinar e avaliar 10k na GPU |
| `make train SCENARIO=2m` | Treinar e avaliar 2M na GPU |
| `make verify` | Verificar os resultados GPU dos dois cenários |
| `make evaluate SCENARIO=2m` | Reavaliar a execução congelada, sem retreinar |
| `make report` | Gerar gráficos e dashboards dos resultados existentes |
| `make test` | Executar a suíte comum |
| `make test-gpu` | Executar os testes CUDA reais |

O dashboard `index.html` inclui gráficos em alta resolução, legendas explicativas e tabelas com rolagem horizontal. As matrizes de confusão mostram contagens e percentuais por classe real; as curvas incluem referências e as barras exibem os valores. `make report` atualiza a apresentação usando somente os resultados salvos, sem retreinar os modelos.

`SCENARIO=all` é o padrão. `SOURCE=arquivo`, `RUN=pasta`, `RESULTS=pasta` e `ARGS="..."` passam opções aos comandos. Por exemplo:

```bash
make train SCENARIO=2m SOURCE=data/transacoes.parquet
make report SCENARIO=10k RUN=resultados/runs-gpu/10k/SEU_RUN_ID
make import-history SCENARIO=10k SOURCE=data/transacoes.csv CUTOFF=2025-02-01T00:00:00-03:00
```

Para executar diretamente, use o mesmo ponto de entrada:

```bash
.venv-gpu/bin/python main.py --help
.venv-gpu/bin/python main.py migrate
.venv-gpu/bin/python main.py serve
.venv-gpu/bin/python main.py train --scenario all --backend gpu
.venv-gpu/bin/python main.py verify --scenario all
.venv-gpu/bin/python main.py report --scenario all
```

Os antigos scripts de treinamento, gráficos e verificação da raiz foram removidos. Os scripts exploratórios foram arquivados como texto em [docs/archive](docs/archive/README.md), fora das rotas executáveis.

## Docker e Podman

O Dockerfile usa as mesmas dependências e o mesmo `main.py`. Os modelos, relatórios e cache ficam fora da imagem; o Compose monta `resultados`, `data` (somente leitura) e `.cache`. O `.env`, os ambientes virtuais e os dados não são enviados ao contexto de build.

Para executar CUDA dentro do contêiner, o host precisa do **NVIDIA Container Toolkit** além do driver. No Podman, os dispositivos NVIDIA precisam estar registrados via CDI. A reserva de GPU está declarada no Compose e não existe fallback para CPU. Veja a configuração oficial do [Container Toolkit](https://docs.nvidia.com/datacenter/cloud-native/container-toolkit/latest/install-guide.html) e do [CDI para Podman](https://docs.nvidia.com/datacenter/cloud-native/container-toolkit/latest/cdi-support.html).

```bash
make build
make up
make logs
```

`make up` inicia o banco, aplica migrações em um contêiner e inicia a API. Os endereços publicados são locais: <http://[::1]:8000/docs> e <http://127.0.0.1:8000/docs>. Dentro do contêiner, a API usa um socket IPv6 com suporte a conexões IPv4 mapeadas. O banco é acessado por `db`, sem alterar a URL local do seu `.env`. A aplicação roda como usuário sem privilégios; no Podman o Makefile configura `keep-id` para preservar a propriedade dos arquivos montados.

Para executar os demais comandos na imagem:

```bash
make container CMD="train --scenario 10k --backend gpu"
make container CMD="verify --scenario all"
make container CMD="report --scenario all"
make container CMD="import-history --scenario 10k --source data/transacoes.csv --cutoff 2025-02-01"
```

Antes de comandos de banco em `make container`, execute `make up` ou `make db` e `make container CMD="migrate"`. `make down` encerra os serviços sem apagar o volume PostgreSQL ou os arquivos de resultados. Use `COMPOSE="podman compose"` ou `COMPOSE="docker compose"` para escolher o mecanismo explicitamente.

## Protocolo e resultados

GPU é o padrão do treinamento; CPU exige `--backend cpu` explícito. CUDA é verificado antes da leitura dos dados. Históricos, transformações, SMOTENC/Tomek, classificadores, métricas e ajuste de limiar usam GPU; entrada, controle temporal, banco, persistência e gráficos usam CPU. Não há fallback silencioso ou descarte de linhas para contornar falta de VRAM.

Sem `--source`, os carregadores usam Kaggle/Hugging Face. CSV de 10k usa `;`, CSV de 2M usa vírgula, e ambos aceitam parquet. A busca usa até 80 mil registros do final do desenvolvimento, preservando empates de horário. `--tuning-rows 0` usa todo o desenvolvimento na busca; os vencedores sempre são ajustados no desenvolvimento completo.

Os resultados GPU ficam em `resultados/runs-gpu/<cenario>/<run_id>/`: manifesto com versões/hardware/hashes, validação das configurações, exclusões, três modelos finais, limiares, métricas e predições de teste, gráficos, `REPORT.md` e dashboard `index.html`. O `active.json` só é atualizado ao concluir a execução. A API carrega os campeões ativos; reinicie-a após treinar novos modelos.

Os resultados CPU continuam em `resultados/runs`. Sua reprodução exige o ambiente das versões indicadas nos manifestos; `--results` ou `MODEL_ROOT` permite selecionar esse diretório explicitamente. Os modelos `joblib` devem vir somente de diretórios locais confiáveis.

## API local

A documentação interativa está em <http://[::1]:8000/docs> e o contrato está em [docs/openapi.json](docs/openapi.json). Esta versão é destinada ao uso local do TCC.

O cenário 2m exige acesso autorizado aos saldos de ambas as partes antes da decisão. A API não comprova a origem desses saldos; sem esse acesso, o modelo não é aplicável. Consulte as condições e limitações em [METHODOLOGY.md](docs/METHODOLOGY.md).

### Analisar antes de executar

```bash
curl -g 'http://[::1]:8000/v1/transactions' \
  -H 'Content-Type: application/json' \
  -d '{"scenario":"2m","external_id":"pix-demo-001","payer_id":"pagador-opaco-001","occurred_at":"2026-11-22T12:00:00-03:00","amount":"150.00","attributes":{"tipo_transacao":"chave_pix","saldo_anterior_pagador":"1000.00","saldo_anterior_recebedor":"500.00"}}'
```

A resposta contém `id`, estado `pending`, escore, previsão, limiar, versão do modelo e fotografia dos atributos. `prediction=1` significa suspeita de fraude/anomalia e `prediction=0` significa não sinalizada. O `status` é independente: `pending`, `completed` e `cancelled` descrevem a execução do pagamento, não confirmam fraude. Toda análise nova começa em `pending`; somente a chamada de confirmação altera esse estado. A classificação não executa nem bloqueia um pagamento real. O histórico é calculado pelo banco; não envie agregações prontas. `payer_id` deve ser um identificador opaco, nunca nome ou documento pessoal.

Para 10k, use `scenario: "10k"` e atributos próprios:

```json
{"tipo_chave":"CPF","pagador_banco":"A","recebedor_banco":"B","pagador_is_cnpj":0,"recebedor_is_cnpj":1,"pagador_cpf_zerado":0}
```

Não são aceitos saldos posteriores, `Status`, rótulos reais ou atributos do outro cenário. Valores monetários são decimais, com até duas casas; o valor da transação pode ser zero para preservar o cenário de anomalias cadastrais. A data da requisição precisa incluir fuso.

### Confirmar o resultado operacional

Use o identificador retornado pela análise:

```bash
curl -g 'http://[::1]:8000/v1/transactions/ID_RETORNADO/confirmation' \
  -H 'Content-Type: application/json' \
  -d '{"status":"completed","executed_at":"2026-11-22T12:00:00-03:00"}'
```

Para cancelar: `{"status":"cancelled"}`. Só pagamentos concluídos alimentam o histórico. Repetir uma análise idêntica ou a mesma confirmação retorna o registro persistido; alterar conteúdo do mesmo identificador ou um estado final retorna `409`. Confirmações atrasadas não reescrevem decisões anteriores.

`POST /v1/transactions/batch` recebe `{"transactions":[...]}`, com no máximo 100 tentativas. O lote é atômico e as tentativas não entram no histórico antes da confirmação. Consultas:

- `GET /v1/transactions/{id}`: decisão persistida;
- `GET /v1/transactions?scenario=2m&limit=50&after=<cursor>`: página por cursor;
- `GET /v1/models`: versões, atributos, seleção e modelos indisponíveis;
- `GET /health`: banco, migração e campeões dos dois cenários. Retorna `503` se não estiver pronto.

## Importação opcional de histórico

O banco começa vazio. A importação é uma simulação explícita, separada por cenário e limitada por corte exclusivo, que não pode estar no futuro:

```bash
python main.py import-history --scenario 10k --source dados.csv --cutoff 2025-02-01T00:00:00-03:00
```

Somente linhas concluídas e anteriores ao corte são importadas. Datas inválidas são excluídas; alvos e documentos pessoais não são armazenados. Os pagadores recebem HMAC com `PAYER_HMAC_KEY`; use `src.api.history_import.pseudonym` com a mesma chave e cenário para referenciar esses pagadores em chamadas posteriores. A chave não deve mudar entre importações.

A confirmação é registrada no horário real da importação, sem fingir que a informação já era conhecida no passado. Lotes de 500 linhas permitem retomada idempotente; uma falha reverte o lote atual, mantendo os anteriores. Use o importador apenas para preparar a demonstração, não como avaliação temporal de produção.

## Verificação

```bash
make test
```

Os testes de PostgreSQL são executados apenas quando `TEST_DATABASE_URL` aponta para um banco de testes PostgreSQL 18. Eles criam um schema isolado, aplicam migrações e removem apenas esse schema ao terminar:

```bash
TEST_DATABASE_URL="sua_url_de_testes" make test
```

Para verificar o caminho GPU, execute: `make test-gpu` no ambiente `.venv-gpu`. Sem `RUN_GPU_TESTS=1`, a suíte comum não executa CUDA. Consulte [docs/METHODOLOGY.md](docs/METHODOLOGY.md) para disponibilidade dos atributos, balanceamento, limiares e limitações, e [docs/AUDIT.md](docs/AUDIT.md) para a auditoria do fluxo anterior.

Resultados anteriores em `resultados/10k`, `resultados/tabelas`, `resultados/modelos` e documentação histórica de algoritmos foram preservados. Eles não demonstram o protocolo temporal corrigido e não são carregados pela nova API.

Evidências CPU históricas: [docs/VALIDATION.md](docs/VALIDATION.md). Evidências CUDA atuais: [docs/GPU_VALIDATION.md](docs/GPU_VALIDATION.md).
