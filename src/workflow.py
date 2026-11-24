"""Temporal experiment execution, immutable artifacts and descriptive reports."""

import hashlib
import importlib.metadata
import json
import os
from pathlib import Path
import tempfile
from datetime import datetime, timezone
from uuid import uuid4

import joblib
import numpy as np
import pandas as pd
from sklearn.metrics import (accuracy_score, average_precision_score, confusion_matrix,
                             f1_score, precision_recall_curve, precision_score,
                             recall_score, roc_auc_score)

from src.data.temporal import (CATEGORIES, CONTRACT_VERSION, NUMERICS, Scenario,
                               feature_frame, normalize, temporal_folds, temporal_split)
from src.models.experiments import (Candidate, FAMILIES, ThresholdModel, candidates,
                                    make_pipeline)

ROOT = Path(__file__).resolve().parents[1]
RESULTS = ROOT / "resultados" / "runs"
GPU_RESULTS = ROOT / "resultados" / "runs-gpu"


def methodology_limits(scenario: Scenario, quality: dict) -> list[str]:
    """State observability assumptions without claiming evidence absent from sources."""
    notes = [
        "A fonte não informa quando os rótulos se tornaram conhecidos. O estudo assume disponibilidade no corte de treino; não comprova ausência de vazamento por atraso de rotulagem.",
        "O histórico offline assume disponibilidade imediata dos eventos anteriores. Na API, somente pagamentos executados antes do evento e confirmados até a análise entram no histórico.",
        "F1 é um objetivo experimental, sem custos financeiros de falsos positivos/negativos; o limiar não é uma política econômica ótima.",
    ]
    if scenario == "2m":
        notes.append("Aplicável somente a integrações com acesso legítimo aos saldos do pagador E do recebedor antes da decisão. O nome saldo_anterior não comprova disponibilidade. A API valida formato, mas não verifica a origem ou o instante da observação; sem esses saldos, este modelo não deve ser usado nem receber valores inventados.")
    else:
        notes.append("Indicadores cadastrais e bancos precisam ser conhecidos antes da decisão. O alvo 10k é anomalia cadastral, não confirmação de fraude financeira.")
    excluded = quality.get("invalid_dates", 0)
    notes.append(f"Cobertura temporal: {quality.get('total', 'não informado')} linhas de origem; {excluded} datas inválidas excluídas, por classe: {quality.get('invalid_dates_by_class', {})}. As métricas não avaliam essas exclusões nem demonstram detecção de anomalias de data.")
    return notes


def write_json(path: Path, value: dict | list) -> None:
    """Atomically publish JSON; never expose a partly written active-model pointer."""
    path.parent.mkdir(parents=True, exist_ok=True)
    with tempfile.NamedTemporaryFile(mode="w", encoding="utf-8", dir=path.parent, delete=False) as file:
        temporary = Path(file.name)
        json.dump(value, file, ensure_ascii=False, indent=2, allow_nan=False)
    try:
        os.replace(temporary, path)
    finally:
        temporary.unlink(missing_ok=True)


def digest(path: Path) -> str:
    """Hash the serialized bytes to bind the trusted manifest to its artifact."""
    with path.open("rb") as stream:
        return hashlib.file_digest(stream, "sha256").hexdigest()


def load_source(scenario: Scenario, source: Path | None = None) -> pd.DataFrame:
    """Local CSV/parquet when supplied, otherwise the project's original sources."""
    if source is not None:
        return pd.read_parquet(source) if source.suffix == ".parquet" else pd.read_csv(source, sep=";" if scenario == "10k" else ",")
    from src.data.ingestion import loadFraud2mData, loadPix10kData
    return loadPix10kData() if scenario == "10k" else loadFraud2mData()


def metrics(y: pd.Series, probability: np.ndarray, threshold: float, backend: str = "cpu") -> dict:
    """Single-class AUC/AP is explicitly unavailable, never a fabricated number."""
    if backend == "gpu":
        from src.models.gpu_metrics import metrics as cuda_metrics
        return cuda_metrics(y, probability, threshold)
    prediction = probability >= threshold
    tn, fp, fn, tp = confusion_matrix(y, prediction, labels=[0, 1]).ravel()
    return {"accuracy": float(accuracy_score(y, prediction)),
            "precision": float(precision_score(y, prediction, zero_division=0)),
            "recall": float(recall_score(y, prediction, zero_division=0)),
            "f1": float(f1_score(y, prediction, zero_division=0)),
            "average_precision": float(average_precision_score(y, probability)) if y.nunique() == 2 else None,
            "roc_auc": float(roc_auc_score(y, probability)) if y.nunique() == 2 else None,
            "tn": int(tn), "fp": int(fp), "fn": int(fn), "tp": int(tp)}


def calibrate(y: pd.Series, probability: np.ndarray, backend: str = "cpu") -> float:
    """Exact F1 optimum over observed scores; tie-break by the higher threshold."""
    if backend == "gpu":
        from src.models.gpu_metrics import calibrate as cuda_calibrate
        return cuda_calibrate(y, probability)
    if y.nunique() != 2:
        raise ValueError("Calibração temporal exige as duas classes; não haverá fallback aleatório")
    precision, recall, thresholds = precision_recall_curve(y, probability)
    f1 = 2 * precision[:-1] * recall[:-1] / np.maximum(precision[:-1] + recall[:-1], 1e-15)
    return float(thresholds[np.flatnonzero(f1 == f1.max())[-1]])


def select(rows: list[dict], family: str | None = None) -> dict:
    """Only validation AP enters model selection, with a deterministic identifier tie-break."""
    eligible = [row for row in rows if row["status"] == "ok" and (family is None or row["family"] == family)]
    if not eligible:
        raise ValueError(f"Nenhuma configuração temporal viável: {family or 'todas'}")
    return sorted(eligible, key=lambda row: (-row["validation_ap"], row["candidate"]))[0]


def train(scenario: Scenario, source: Path | None = None, backend: str = "gpu",
          tuning_rows: int = 80000, results: Path | None = None) -> Path:
    """Select inside development, calibrate in the next period and freeze before testing."""
    if backend not in {"cpu", "gpu"} or tuning_rows < 0:
        raise ValueError("Backend inválido ou tamanho de amostra negativo")
    results = results if results is not None else (GPU_RESULTS if backend == "gpu" else RESULTS)
    hardware = None
    if backend == "gpu":
        from src.models.gpu import verify_backend
        hardware = verify_backend()
        print(f"CUDA confirmado: {hardware['name']} | {hardware['total_vram_bytes'] / 2**30:.1f} GiB", flush=True)
    run_id = datetime.now(timezone.utc).strftime("%Y%m%dT%H%M%S") + "_" + uuid4().hex[:8]
    output = results / scenario / run_id
    output.mkdir(parents=True, exist_ok=False)
    raw = load_source(scenario, source)
    frame, quality = normalize(raw, scenario)
    write_json(output / "quality.json", quality)
    try:
        X = feature_frame(frame, scenario, backend)
    except Exception as exc:
        if backend == "gpu":
            from src.utils.cuda import memory_failure
            memory_failure(exc, "engenharia de atributos temporais")
        raise
    y = frame.target.astype(int)
    split = temporal_split(frame)
    development = frame.loc[split.development]
    if tuning_rows and len(development) > tuning_rows:
        cutoff = development.occurred_at.iloc[-tuning_rows]
        tuning = development.loc[development.occurred_at >= cutoff]
    else:
        tuning = development
    folds = temporal_folds(tuning)
    validation_rows = []
    search = candidates()
    for candidate in search:
        print(f"[{scenario}] Validação temporal: {candidate.name}", flush=True)
        scores, fold_details = [], []
        try:
            for fit_index, val_index in folds:
                if y.loc[val_index].nunique() != 2:
                    raise ValueError("Janela de validação sem ambas as classes")
                pipeline = make_pipeline(candidate, scenario, y.loc[fit_index], backend)
                pipeline.fit(X.loc[fit_index], y.loc[fit_index])
                probability = pipeline.predict_proba(X.loc[val_index])[:, 1]
                if backend == "gpu":
                    from src.models.gpu_metrics import average_precision
                    scores.append(average_precision(y.loc[val_index], probability))
                else:
                    scores.append(float(average_precision_score(y.loc[val_index], probability)))
                sampler = pipeline.named_steps.get("sampler")
                before = {str(k): int(v) for k, v in y.loc[fit_index].value_counts().items()}
                # Resampling changes only the fit set; expose its intended counts without refitting.
                after = before.copy()
                if sampler is not None:
                    for label, count in sampler.sampling_strategy_.items():
                        after[str(label)] = int(count) if candidate.strategy == "under" else before[str(label)] + int(count)
                if "tomek" in pipeline.named_steps:
                    after = {"after_synthesis": after,
                             "after_tomek_total": int(len(pipeline.named_steps["tomek"].sample_indices_))}
                fold_details.append({"train": len(fit_index), "validation": len(val_index),
                                     "train_end": frame.loc[fit_index, "occurred_at"].max().isoformat(),
                                     "validation_start": frame.loc[val_index, "occurred_at"].min().isoformat(),
                                     "before_sampling": before, "after_sampling": after})
                del pipeline
                if backend == "gpu":
                    from src.utils.cuda import release
                    release()
            row = {"candidate": candidate.name, "family": candidate.family,
                   "strategy": candidate.strategy, "adjusted": candidate.adjusted,
                   "status": "ok", "validation_ap": float(np.mean(scores)), "fold_ap": scores,
                   "folds": fold_details}
        except ValueError as exc:
            if backend == "gpu":
                from src.utils.cuda import memory_failure
                memory_failure(exc, candidate.name)
            row = {"candidate": candidate.name, "family": candidate.family, "strategy": candidate.strategy,
                   "adjusted": candidate.adjusted, "status": "infeasible", "reason": str(exc)}
        validation_rows.append(row)
        write_json(output / "validation.json", validation_rows)
    champion = select(validation_rows)
    chosen = {family: select(validation_rows, family) for family in FAMILIES}
    versions = {package: importlib.metadata.version(package) for package in (
        "numpy", "pandas", "scikit-learn", "imbalanced-learn", "xgboost", "joblib")}
    if backend == "gpu":
        versions.update({package: importlib.metadata.version(package) for package in (
            "cupy-cuda13x", "cuml-cu13", "cudf-cu13", "rmm-cu13")})
    partitions = {}
    for name in ("development", "calibration", "test"):
        index = getattr(split, name)
        partitions[name] = {"rows": len(index), "start": frame.loc[index, "occurred_at"].min().isoformat(),
                            "end": frame.loc[index, "occurred_at"].max().isoformat(),
                            "classes": {str(k): int(v) for k, v in y.loc[index].value_counts().items()}}
    manifest = {"contract_version": CONTRACT_VERSION, "scenario": scenario, "run_id": run_id,
                "backend": backend, "hardware": hardware, "versions": versions, "features": CATEGORIES[scenario] + NUMERICS[scenario],
                "execution": {"feature_engineering": "cudf/cupy CUDA" if backend == "gpu" else "pandas CPU",
                              "metrics_and_threshold": "cupy CUDA" if backend == "gpu" else "sklearn CPU",
                              "input_validation_io_and_charts": "CPU"},
                "champion": champion["family"], "selection": "temporal_validation_average_precision",
                "threshold_objective": "calibration_f1", "partitions": partitions,
                "tuning_rows": len(tuning), "models": {}, "source_rows": len(raw),
                "source_sha256": digest(source) if source else None,
                "quality": quality,
                "assumptions": methodology_limits(scenario, quality)}
    calibration = pd.DataFrame({"target": y.loc[split.calibration]})
    for family, selected in chosen.items():
        print(f"[{scenario}] Ajuste final: {selected['candidate']} | {len(split.development):,} registros", flush=True)
        candidate = Candidate(family, selected["strategy"], selected["adjusted"])
        pipeline = make_pipeline(candidate, scenario, y.loc[split.development], backend)
        pipeline.fit(X.loc[split.development], y.loc[split.development])
        print(f"[{scenario}] Calibrando limiar de {family} no período reservado", flush=True)
        probabilities = pipeline.predict_proba(X.loc[split.calibration])[:, 1]
        threshold = calibrate(y.loc[split.calibration], probabilities, backend)
        calibration[f"{family}_score"] = probabilities
        artifact = output / f"{family}.joblib"
        joblib.dump(ThresholdModel(pipeline, threshold), artifact)
        manifest["models"][family] = {"artifact": artifact.name, "sha256": digest(artifact),
            "candidate": candidate.name, "strategy": candidate.strategy,
            "validation_ap": selected["validation_ap"], "threshold": threshold,
            "calibration_default": metrics(y.loc[split.calibration], probabilities, .5, backend),
            "calibration_tuned": metrics(y.loc[split.calibration], probabilities, threshold, backend),
            "execution": getattr(pipeline, "execution_", {"classifier": "sklearn/xgboost CPU"})}
        del pipeline
        if backend == "gpu":
            from src.utils.cuda import release
            release()
    # Frozen manifest exists before any test predictions are generated.
    calibration.to_parquet(output / "calibration.parquet", index=False)
    test = X.loc[split.test].copy()
    test["target"] = y.loc[split.test]
    test["occurred_at"] = frame.loc[split.test, "occurred_at"]
    test.to_parquet(output / "test.parquet", index=False)
    write_json(output / "manifest.json", manifest)
    evaluate(output)
    report(output)
    write_json(results / scenario / "active.json", {"run_id": run_id})
    return output


def load_manifest(run: Path) -> dict:
    """Reject legacy contracts before deserializing any model."""
    manifest = json.loads((run / "manifest.json").read_text(encoding="utf-8"))
    scenario = manifest["scenario"]
    if manifest["contract_version"] != CONTRACT_VERSION or manifest["features"] != CATEGORIES[scenario] + NUMERICS[scenario]:
        raise ValueError("Contrato do artefato incompatível; retreine pelo fluxo temporal")
    return manifest


def load_model(run: Path, manifest: dict, family: str) -> ThresholdModel:
    """Only locally trusted artifacts; manifests and model directories are not API inputs."""
    for package, version in manifest["versions"].items():
        if importlib.metadata.version(package) != version:
            raise ValueError(f"Versão incompatível de {package}; use o ambiente do treinamento")
    path = run / manifest["models"][family]["artifact"]
    if path.parent.resolve() != run.resolve() or digest(path) != manifest["models"][family]["sha256"]:
        raise ValueError("Integridade do modelo inválida")
    if manifest["backend"] == "gpu":
        from src.utils.cuda import initialize
        initialize()
    return joblib.load(path)


def evaluate(run: Path) -> pd.DataFrame:
    """Evaluate frozen family winners; never rank or update a model using test results."""
    manifest = load_manifest(run)
    test = pd.read_parquet(run / "test.parquet")
    y, X = test.target, test[manifest["features"]]
    predictions = test[["target", "amount", "occurred_at"]].copy()
    rows = []
    for family, config in manifest["models"].items():
        model = load_model(run, manifest, family)
        probability = model.predict_proba(X)[:, 1]
        rows.append({"scenario": manifest["scenario"], "run_id": manifest["run_id"],
                     "family": family, "candidate": config["candidate"], "threshold": model.threshold,
                     **metrics(y, probability, model.threshold, manifest["backend"])})
        predictions[f"{family}_score"] = probability
        predictions[f"{family}_prediction"] = (probability >= model.threshold).astype(int)
        del model
        if manifest["backend"] == "gpu":
            from src.utils.cuda import release
            release()
    result = pd.DataFrame(rows)
    result.to_csv(run / "metrics_test.csv", index=False)
    predictions.to_parquet(run / "predictions_test.parquet", index=False)
    return result


def report(run: Path) -> Path:
    """Generate charts and an offline dashboard from frozen results, without fitting."""
    import html
    from src.visualization.temporal_report import render_report_plots

    manifest = load_manifest(run)
    table = pd.read_csv(run / "metrics_test.csv")
    predictions = pd.read_parquet(run / "predictions_test.parquet")
    plots = run / "plots"
    plots.mkdir(exist_ok=True)
    validation = pd.DataFrame(json.loads((run / "validation.json").read_text()))
    calibration_rows = []
    for family, config in manifest["models"].items():
        for name in ("default", "tuned"):
            calibration_rows.append({"family": family, "threshold_type": name,
                "threshold": .5 if name == "default" else config["threshold"], **config[f"calibration_{name}"]})
    calibration_table = pd.DataFrame(calibration_rows)
    partition_table = pd.DataFrame([{"partition": name, "rows": partition["rows"],
        "positives": partition["classes"].get("1", 0),
        "prevalence": partition["classes"].get("1", 0) / partition["rows"],
        "start": partition["start"], "end": partition["end"]}
        for name, partition in manifest["partitions"].items()])
    render_report_plots(manifest, predictions, validation, calibration_table, plots)
    notes = ("Campeão escolhido exclusivamente pela AP da validação temporal: " + manifest["champion"] +
             ". Limiar escolhido por F1 no período de calibração. O teste é descritivo. "
             "Escores não são probabilidades calibradas. Os cenários têm alvos diferentes; "
             "valores sinalizados não demonstram dinheiro efetivamente protegido.")
    (run / "REPORT.md").write_text("# Avaliação temporal\n\n" + notes +
        "\n\n## Períodos e prevalência\n\n```\n" + partition_table.to_string(index=False) +
        "\n```\n\nAs prevalências podem mudar entre períodos. O limiar é fixado antes do teste; "
        "mudanças de distribuição não autorizam ajustá-lo usando o teste.\n\n## Teste reservado\n\n```\n" + table.to_string(index=False) +
        "\n```\n\n## Limitações\n\n" + "\n".join("- " + item for item in methodology_limits(manifest["scenario"], manifest.get("quality", {}))) +
        "\n\n## Justificativa de classificação\n\n" +
        "O balanceamento foi escolhido pela AP da validação progressiva, sem reamostrar validação ou teste. "
        "Pesos são calculados somente no treino de cada janela; configurações combinando pesos e síntese não são usadas. "
        "O limiar maximiza F1 em período posterior ao treino; seu desempenho nesse período é de ajuste, não de teste.\n\n```\n" +
        calibration_table.to_string(index=False) + "\n```\n\n" +
        "Disponibilidade dos atributos: docs/METHODOLOGY.md. Consulte quality.json para exclusões "
        "e validation.json para contagens antes/depois da reamostragem e motivos de inviabilidade.\n", encoding="utf-8")
    output = run / "index.html"
    output.write_text('<!doctype html><html lang="pt-BR"><meta charset="utf-8"><title>Avaliação temporal Pix</title>'
        '<meta name="viewport" content="width=device-width, initial-scale=1">'
        '<style>body{margin:0;background:#f1f5f9;color:#0f172a;font:16px/1.6 system-ui,sans-serif}'
        'main{max-width:1440px;margin:auto;padding:32px 24px}h1{font-size:2rem;line-height:1.2}'
        'h2{margin-top:40px}p{max-width:100ch;color:#475569}'
        'figure{margin:32px 0;padding:32px;background:white;border:1px solid #e2e8f0;border-radius:16px}'
        'img{display:block;width:100%;height:auto}figcaption{padding:24px 12px 8px;color:#475569}'
        '.table-scroll{overflow-x:auto;background:white;border-radius:12px;border:1px solid #e2e8f0}'
        'table{border-collapse:collapse;width:100%;font-size:14px}th,td{padding:12px;text-align:left;border:0;white-space:nowrap}'
        'th{background:#e2e8f0}tr:nth-child(even){background:#f8fafc}'
        '@media(max-width:600px){main{padding:20px 12px}figure{padding:8px}h1{font-size:1.5rem}}'
        '</style><main><h1>Avaliação temporal — ' + html.escape(manifest["scenario"]) + '</h1><p>' + html.escape(notes) + '</p>' +
        ('<nav aria-label="Gráficos"><a href="#curvas">Ver curvas de precisão-recall e ROC</a></nav>'
         if predictions.target.nunique() == 2 else '') +
        '<h2>Períodos e prevalência</h2>' + '<div class="table-scroll">' + partition_table.to_html(index=False, escape=True) + '</div>' +
        '<h2>Teste reservado</h2><div class="table-scroll">' + table.to_html(index=False, escape=True) + '</div><h2>Leitura dos resultados</h2>' + '<figure><img alt="AP média por modelo e estratégia de balanceamento na validação temporal" src="plots/balancing.png">'
        '<figcaption>AP é a precisão média ao longo dos níveis de recall. Valores maiores indicam melhor discriminação. Cada barra resume as configurações viáveis avaliadas na validação.</figcaption></figure>' +
        '<figure><img alt="F1 na calibração com limiar padrão e ajustado" src="plots/thresholds.png">'
        '<figcaption>F1 combina precisão e recall. Este período foi usado para ajustar o limiar; o ganho mostrado não comprova desempenho no teste.</figcaption></figure>' +
        '<figure><img alt="Matrizes de confusão com contagens e percentuais por classe real" src="plots/confusion.png">'
        '<figcaption>Azul: positivos identificados corretamente (1,1). Cada célula mostra a contagem e o percentual dentro da classe real.</figcaption></figure>' +
        ('<h2 id="curvas">Discriminação no teste: precisão-recall e ROC</h2>'
         '<figure><img alt="Curvas de precisão-recall e ROC no teste, com referências" src="plots/curves.png">'
         '<figcaption>Recall mede quantos positivos foram encontrados; precisão mede quantos sinalizados eram positivos. As linhas pontilhadas mostram referências. AUC resume a curva ROC; consulte também a curva de precisão-recall em bases desbalanceadas.</figcaption></figure>' if predictions.target.nunique() == 2 else '') +
        '<h2>Limitações e disponibilidade</h2><ul>' +
        ''.join('<li>' + html.escape(item) + '</li>' for item in
                methodology_limits(manifest['scenario'], manifest.get('quality', {}))) +
        '</ul></main></html>', encoding="utf-8")
    return output


def verify(run: Path) -> dict:
    """Read-only consistency proof for frozen decisions and temporal boundaries."""
    manifest = load_manifest(run)
    partitions = manifest["partitions"]
    if not (pd.Timestamp(partitions["development"]["end"]) < pd.Timestamp(partitions["calibration"]["start"]) and
            pd.Timestamp(partitions["calibration"]["end"]) < pd.Timestamp(partitions["test"]["start"])):
        raise ValueError("Sobreposição temporal nas partições")
    table = pd.read_csv(run / "metrics_test.csv").set_index("family")
    predictions = pd.read_parquet(run / "predictions_test.parquet")
    if len(predictions) != partitions["test"]["rows"]:
        raise ValueError("Contagem de teste incompatível")
    for family, config in manifest["models"].items():
        if digest(run / config["artifact"]) != config["sha256"]:
            raise ValueError("Integridade de artefato inválida")
        probability = predictions[f"{family}_score"].to_numpy()
        if not np.isfinite(probability).all() or not ((probability >= 0) & (probability <= 1)).all():
            raise ValueError("Escores inválidos")
        if not np.array_equal(probability >= config["threshold"], predictions[f"{family}_prediction"]):
            raise ValueError("Predições incompatíveis com limiar congelado")
        actual = metrics(predictions.target, probability, config["threshold"], manifest["backend"])
        for key, value in actual.items():
            stored = table.loc[family, key]
            if not (pd.isna(stored) if value is None else np.isclose(stored, value)):
                raise ValueError(f"Métrica divergente: {family}/{key}")
    return {"run_id": manifest["run_id"], "status": "verified", "rows": len(predictions)}
