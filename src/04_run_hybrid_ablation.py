"""Compara fusões uniforme e por PSO com o Stacking do benchmark."""

from __future__ import annotations

import argparse
import json
import time
from dataclasses import dataclass
from importlib import import_module
from pathlib import Path

import numpy as np
import pandas as pd
from sklearn.base import clone
from sklearn.metrics import balanced_accuracy_score, f1_score
from sklearn.model_selection import StratifiedKFold
from sklearn.preprocessing import LabelEncoder
from sklearn.utils import check_random_state
from threadpoolctl import threadpool_limits


# O prefixo numérico dos scripts requer importação por nome de módulo.
module_prefix = f"{__package__}." if __package__ else ""
benchmark = import_module(f"{module_prefix}02_run_benchmark")
fusion = import_module(f"{module_prefix}pso_adaptive_fusion")


@dataclass
class FoldResult:
    probabilities: np.ndarray
    fit_time: float
    weights: np.ndarray | None = None
    coefficients: np.ndarray | None = None
    oof_f1_macro: float | None = None
    base_fit_time: float | None = None
    oof_time: float | None = None
    pso_time: float | None = None


def evaluate_fold(X_train, y_train, X_test, random_state):
    """Ajusta as estratégias com o treino externo e prevê no teste externo."""
    models = benchmark.build_models(X_train, random_state=random_state)
    stacking = models[benchmark.MODEL_NAMES["stacking"]]
    pso = models[benchmark.MODEL_NAMES["pso"]]
    # Uma única configuração de paralelismo para a comparação de tempos.
    stacking.set_params(n_jobs=1, rf__classifier__n_jobs=1, knn__classifier__n_jobs=1)
    experts = [clone(estimator) for _, estimator in stacking.estimators]
    if np.bincount(y_train).min() < pso.cv:
        raise ValueError("Cada classe precisa estar presente em todos os folds internos.")
    inner_folds = list(StratifiedKFold(n_splits=pso.cv).split(X_train, y_train))
    if min(len(train) for train, _ in inner_folds) < experts[-1]["classifier"].n_neighbors:
        raise ValueError("Amostras insuficientes no treino interno para o k-NN.")

    start = time.perf_counter()
    oof = fusion._out_of_fold_probabilities(experts, X_train, y_train, inner_folds)
    oof_time = time.perf_counter() - start

    start = time.perf_counter()
    for expert in experts:
        expert.fit(X_train, y_train)
    base_time = time.perf_counter() - start

    learned = {}
    for name, adaptive in (("pso_fixed", False), ("pso_adaptive", True)):
        start = time.perf_counter()
        coefficients, oof_f1 = fusion._optimize_weights(
            oof, y_train, n_particles=pso.n_particles, n_iterations=pso.n_iterations,
            bound=pso.parameter_bound, rng=check_random_state(random_state), adaptive=adaptive,
        )
        learned[name] = (coefficients, oof_f1, time.perf_counter() - start)

    start = time.perf_counter()
    stacking.fit(X_train, y_train)
    stacking_time = time.perf_counter() - start

    probabilities = np.stack([expert.predict_proba(X_test) for expert in experts], axis=1)
    confidences = fusion._entropy_confidence(probabilities)
    uniform = np.zeros((2, len(experts)))
    results = {
        "uniform": FoldResult(
            probabilities=probabilities.mean(axis=1),
            weights=fusion._adaptive_weights(confidences, uniform),
            coefficients=uniform,
            oof_f1_macro=f1_score(y_train, oof.mean(axis=1).argmax(axis=1), average="macro"),
            fit_time=base_time, base_fit_time=base_time, oof_time=0.0, pso_time=0.0,
        )
    }
    for name, (coefficients, oof_f1, pso_time) in learned.items():
        results[name] = FoldResult(
            probabilities=fusion._fuse_probabilities(probabilities, confidences, coefficients),
            weights=fusion._adaptive_weights(confidences, coefficients),
            coefficients=coefficients, oof_f1_macro=oof_f1,
            fit_time=base_time + oof_time + pso_time,
            base_fit_time=base_time, oof_time=oof_time, pso_time=pso_time,
        )
    results["stacking"] = FoldResult(stacking.predict_proba(X_test), stacking_time)
    return results


def same_predictions_and_weights(first: FoldResult, second: FoldResult) -> bool:
    return np.array_equal(
        first.probabilities.argmax(axis=1), second.probabilities.argmax(axis=1)
    ) and all(
        np.allclose(left, right, rtol=0, atol=1e-12)
        for left, right in (
            (first.probabilities, second.probabilities),
            (first.coefficients, second.coefficients),
            (first.weights, second.weights),
        )
        if left is not None
    )


def run_ablation(data_dir, results_path, *, datasets=None, n_splits=benchmark.N_SPLITS,
                 random_state=benchmark.RANDOM_STATE, check_reproducibility=False):
    """Grava métricas por base, fold e estratégia em um CSV novo."""
    results_path = Path(results_path)
    if results_path.exists():
        raise FileExistsError(f"O CSV já existe: {results_path}. Escolha outro --results-path.")
    paths = {path.name[:-7]: path for path in sorted(Path(data_dir).glob("*.csv.gz"))}
    if not paths:
        raise FileNotFoundError(f"Nenhuma base preparada em {data_dir}.")
    if unknown := set(datasets or ()).difference(paths):
        raise ValueError(f"Bases desconhecidas: {', '.join(sorted(unknown))}.")

    cv = StratifiedKFold(n_splits=n_splits, shuffle=True, random_state=random_state)
    rows = []
    with threadpool_limits(limits=1):
        for name, path in paths.items():
            if datasets and name not in datasets:
                continue
            frame = pd.read_csv(path)
            X = frame.drop(columns="target")
            y = LabelEncoder().fit_transform(frame["target"])
            counts = np.bincount(y)
            if len(counts) < 2 or counts.min() < n_splits:
                raise ValueError(f"{name}: cada classe precisa estar presente em todos os folds externos.")
            for fold, (train, test) in enumerate(cv.split(X, y), start=1):
                args = (X.iloc[train], y[train], X.iloc[test], random_state)
                results = evaluate_fold(*args)
                repeated = evaluate_fold(*args) if check_reproducibility else None
                for strategy, result in results.items():
                    predictions = result.probabilities.argmax(axis=1)
                    f1 = f1_score(y[test], predictions, average="macro")
                    rows.append({
                        "dataset": name, "strategy": strategy, "fold": fold,
                        "n_splits": n_splits, "random_state": random_state,
                        "n_samples": len(y), "n_classes": len(counts),
                        "minority_count": counts.min(),
                        "test_minority_count": np.bincount(y[test], minlength=len(counts)).min(),
                        "f1_macro": f1,
                        "balanced_accuracy": balanced_accuracy_score(y[test], predictions),
                        "fit_time": result.fit_time,
                        "base_fit_time": result.base_fit_time,
                        "oof_time": result.oof_time,
                        "pso_time": result.pso_time,
                        "oof_f1_macro": result.oof_f1_macro,
                        "oof_minus_test_f1": (
                            None if result.oof_f1_macro is None else result.oof_f1_macro - f1
                        ),
                        "coefficients": (
                            None if result.coefficients is None else json.dumps(result.coefficients.tolist())
                        ),
                        "mean_weight_std": (
                            None if result.weights is None else result.weights.std(axis=0).mean()
                        ),
                        "reproducible": (
                            None if repeated is None
                            else same_predictions_and_weights(result, repeated[strategy])
                        ),
                    })

    metrics = pd.DataFrame(rows)
    results_path.parent.mkdir(parents=True, exist_ok=True)
    with results_path.open("x", encoding="utf-8", newline="") as output:
        metrics.to_csv(output, index=False, float_format="%.6f")
    return metrics


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--data-dir", type=Path, default=benchmark.PROCESSED_DIR)
    parser.add_argument("--datasets", nargs="+", help="Nomes das bases preparadas, sem .csv.gz.")
    parser.add_argument("--results-path", type=Path, required=True, help="CSV novo para os resultados.")
    parser.add_argument("--n-splits", type=int, default=benchmark.N_SPLITS)
    parser.add_argument("--random-state", type=int, default=benchmark.RANDOM_STATE)
    parser.add_argument("--check-reproducibility", action="store_true", help="Repete cada treino.")
    args = parser.parse_args()
    metrics = run_ablation(**vars(args))
    columns = ["f1_macro", "balanced_accuracy", "fit_time", "oof_minus_test_f1"]
    print(metrics.groupby(["dataset", "strategy"], sort=False)[columns].mean().round(4).to_string())
    if args.check_reproducibility:
        print(f"Resultados reproduzidos: {metrics['reproducible'].sum()}/{len(metrics)}")
    print(f"Métricas por fold em {args.results_path}")


if __name__ == "__main__":
    main()
