"""Gera o grafico de falsos positivos a partir do teste ja executado."""
from pathlib import Path
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib.ticker import MaxNLocator
import pandas as pd

ROOT = Path(__file__).resolve().parent
metrics = pd.read_csv(ROOT / "resultados" / "tabelas" / "metricas_teste.csv")
predictions = pd.read_csv(ROOT / "resultados" / "tabelas" / "predicoes_teste.csv")
families = ["Random Forest", "XGBoost", "Regressão Logística"]
metrics = metrics.set_index("modelo").loc[families]
negative_count = int((predictions["anomalia_real"] == 0).sum())
for family, column in [
    ("Random Forest", "pred_random_forest"),
    ("XGBoost", "pred_xgboost"),
    ("Regressão Logística", "pred_logistic_regression"),
]:
    observed = int(((predictions["anomalia_real"] == 0) & (predictions[column] == 1)).sum())
    assert observed == int(metrics.loc[family, "fp"]), "Contagem inconsistente."
assert all(metrics["tn"] + metrics["fp"] == negative_count)

counts = metrics["fp"].astype(int).to_numpy()
rates = counts / negative_count * 100
fig, ax = plt.subplots(figsize=(9, 5.8))
fig.subplots_adjust(top=.77, bottom=.24, left=.12, right=.97)
bars = ax.bar(families, counts, color=["#1976d2", "#e76f51", "#2a9d8f"], width=.48, zorder=3)
for bar, count, rate in zip(bars, counts, rates):
    ax.annotate(f"{count} falso{'s' if count != 1 else ''} positivo{'s' if count != 1 else ''}\n"
                f"Taxa: {rate:.4f}%".replace(".", ","),
                (bar.get_x() + bar.get_width()/2, count),
                xytext=(0, 10), textcoords="offset points", ha="center", fontsize=12)
ax.set_ylim(0, max(1, int(counts.max())) * 1.45)
ax.yaxis.set_major_locator(MaxNLocator(integer=True))
ax.set_ylabel("Transações legítimas classificadas como anômalas")
ax.set_axisbelow(True)
ax.grid(axis="y", alpha=.25)
ax.spines[["top", "right"]].set_visible(False)
fig.suptitle("Falsos positivos no conjunto de teste", fontsize=18, y=.96)
fig.text(.5, .88,
         f"Mesmas {negative_count:,} transações legítimas avaliadas | Limiar 0,5".replace(",", "."),
         ha="center", fontsize=11, color="#444")
fig.text(.5, .10,
         "Configurações selecionadas pelo F1 médio de validação de cada algoritmo",
         ha="center", fontsize=10, color="#444")
fig.text(.5, .04, "Resultados do teste já realizado; não houve novo treinamento.",
         ha="center", fontsize=10, color="#444")
output = ROOT / "resultados" / "graficos" / "comparacao" / "05_falsos_positivos.png"
fig.savefig(output, dpi=180, facecolor="white")
plt.close(fig)
print(f"Grafico salvo em: {output}")
print(f"Falsos positivos: {dict(zip(families, counts.tolist()))}")

