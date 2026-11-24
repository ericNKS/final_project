# Ambiente CPU histórico

Resultados em `resultados/runs` foram produzidos em ambiente distinto do GPU. As versões exatas estão no campo `versions` do manifesto de cada execução. Este documento não mantém uma segunda lista de dependências. Não carregar esses modelos em um ambiente incompatível nem editar versões no manifesto para contornar a verificação.

A instalação oficial atual usa `requirements.txt` e `.venv-gpu`. CPU exige `--backend cpu` explícito; o protocolo temporal é o mesmo.
