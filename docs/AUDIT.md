# Auditoria metodológica

A rota oficial substitui a exploração histórica por ordenação temporal, validação progressiva no desenvolvimento, calibração reservada e teste final. A seleção usa AP da validação e não métricas de teste. Pré-processamento e balanceamento são ajustados somente em treino. Históricos excluem o próprio evento e empates de horário.

Os scripts exploratórios antigos foram preservados como texto em [archive](archive/README.md), fora de `src/` e das rotas executáveis. Não representam evidência do protocolo corrigido.

Os contratos distintos de 10k/2m são compartilhados pelo treino e serviço. A disponibilidade real dos saldos e dos rótulos não é demonstrada pela fonte: veja as condições e limites em [METHODOLOGY.md](METHODOLOGY.md). Relatórios Markdown e HTML incluem essas ressalvas e a cobertura das exclusões por classe.

Os artefatos de referência foram preservados; correções de documentação e apresentação não alteram colunas, seleção, limiares ou modelos. A reavaliação deve usar os modelos congelados. A comparação de desempenho não permite selecionar novamente pelo teste.
