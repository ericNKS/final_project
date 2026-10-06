"""Gera dashboard offline a partir dos resultados, sem retreinar modelos."""
import csv
import json
from pathlib import Path

ROOT = Path(__file__).resolve().parent


def build():
    result = ROOT / 'resultados'
    def read_csv(name):
        with (result / 'tabelas' / name).open(encoding='utf-8-sig', newline='') as f:
            return list(csv.DictReader(f))
    metrics = read_csv('metricas_teste.csv')
    cv = read_csv('metricas_validacao_cruzada.csv')
    for row in metrics:
        for key in row.keys() - {'modelo', 'nome'}:
            row[key] = float(row[key])
        assert sum(row[k] for k in ('tn', 'fp', 'fn', 'tp')) == 2000
    for row in cv:
        for key in row.keys() - {'modelo', 'nome', 'estrategia', 'configuracao'}:
            row[key] = float(row[key])
    data = dict(models=metrics, cv=cv, metadata=json.loads(
        (result / 'metadados' / 'metadados.json').read_text(encoding='utf-8-sig')))
    template = (ROOT / 'dashboard' / 'template.html').read_text(encoding='utf-8')
    output = result / 'dashboard' / 'index.html'
    output.parent.mkdir(exist_ok=True)
    output.write_text(template.replace('__DATA__', json.dumps(data, ensure_ascii=False).replace('<', '\\u003c')), encoding='utf-8')
    print(output)


if __name__ == '__main__':
    build()
