"""Load only explicitly activated, locally trusted temporal model artifacts."""

import json
import math
from pathlib import Path

import pandas as pd

from src.workflow import GPU_RESULTS, load_manifest, load_model


class Registry:
    """Models are fixed for a process lifetime; restart after activating new versions."""

    def __init__(self, root: Path = GPU_RESULTS):
        self.models = {}
        self.manifests = {}
        self.errors = {}
        for scenario in ("10k", "2m"):
            try:
                pointer = json.loads((root / scenario / "active.json").read_text())
                run = root / scenario / pointer["run_id"]
                if run.parent.resolve() != (root / scenario).resolve():
                    raise ValueError("Versão ativa inválida")
                manifest = load_manifest(run)
                if manifest["scenario"] != scenario or manifest["run_id"] != pointer["run_id"]:
                    raise ValueError("Identidade do artefato incompatível")
                self.models[scenario] = load_model(run, manifest, manifest["champion"])
                self.manifests[scenario] = manifest
            except (OSError, ValueError, KeyError, ImportError) as exc:
                self.errors[scenario] = f"Modelo temporal indisponível: {type(exc).__name__}"

    def predict(self, scenario: str, features: dict) -> tuple[float, float, int, str, str]:
        """Never train or fall back to a legacy artifact during a request."""
        if scenario not in self.models:
            raise ValueError(self.errors.get(scenario, "Modelo indisponível"))
        manifest, model = self.manifests[scenario], self.models[scenario]
        frame = pd.DataFrame([features], columns=manifest["features"])
        score = float(model.predict_proba(frame)[0, 1])
        if not math.isfinite(score) or not 0 <= score <= 1 or not 0 <= model.threshold <= 1:
            raise ValueError("Modelo produziu escore ou limiar inválido")
        return score, float(model.threshold), int(score >= model.threshold), manifest["run_id"], manifest["champion"]

    def describe(self) -> dict:
        return {"available": {scenario: {"run_id": manifest["run_id"], "champion": manifest["champion"],
                    "features": manifest["features"], "selection": manifest["selection"],
                    "models": {family: {key: value for key, value in config.items() if key not in {"artifact", "sha256"}}
                               for family, config in manifest["models"].items()}}
                              for scenario, manifest in self.manifests.items()}, "unavailable": self.errors}
