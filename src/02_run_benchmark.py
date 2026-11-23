"""Fase 2, 3 e 4: Benchmark com pré-processamento por fold e 5-Fold Stratified CV.

Lê as bases preparadas em ``data/processed/*.csv.gz`` e executa os 6 modelos sobre
``StratifiedKFold(n_splits=5, shuffle=True, random_state=42)`` e grava
``results/metrics_raw_refactored.csv`` com ``f1_macro``, ``balanced_accuracy`` e ``fit_time``.
"""

from __future__ import annotations

import argparse
import time
from pathlib import Path

import numpy as np
import pandas as pd
from sklearn.base import BaseEstimator, clone
from sklearn.compose import ColumnTransformer
from sklearn.ensemble import RandomForestClassifier, StackingClassifier
from sklearn.impute import SimpleImputer
from sklearn.metrics import balanced_accuracy_score, f1_score
from sklearn.model_selection import StratifiedKFold
from sklearn.neighbors import KNeighborsClassifier
from sklearn.neural_network import MLPClassifier
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import LabelEncoder, OneHotEncoder, StandardScaler
from sklearn.svm import SVC

if __package__:
    from .pso_adaptive_fusion import PSOAdaptiveFusionClassifier
else:
    from pso_adaptive_fusion import PSOAdaptiveFusionClassifier

RANDOM_STATE = 42
N_SPLITS = 5
MODEL_NAMES = {
    "svm": "SVM",
    "knn": "KNN",
    "rf": "Random Forest",
    "mlp": "MLP",
    "stacking": "Hibrido (Stacking)",
    "pso": "Hibrido (PSO Adaptativo)",
}

PROJECT_ROOT = Path(__file__).resolve().parent.parent
PROCESSED_DIR = PROJECT_ROOT / "data" / "processed"
RESULTS_DIR = PROJECT_ROOT / "results"
METRICS_PATH = RESULTS_DIR / "metrics_raw_refactored.csv"


def build_preprocessor(X: pd.DataFrame) -> ColumnTransformer:
    """Constrói o ColumnTransformer conforme a Fase 2 do protocolo experimental."""
    numeric_cols = X.select_dtypes(include=[np.number]).columns.tolist()
    categorical_cols = [col for col in X.columns if col not in numeric_cols]

    transformers: list[tuple[str, Pipeline, list[str]]] = []
    if numeric_cols:
        numeric_pipe = Pipeline(
            steps=[
                ("imputer", SimpleImputer(strategy="median")),
                ("scaler", StandardScaler()),
            ]
        )
        transformers.append(("num", numeric_pipe, numeric_cols))

    if categorical_cols:
        categorical_pipe = Pipeline(
            steps=[
                ("imputer", SimpleImputer(strategy="most_frequent")),
                ("encoder", OneHotEncoder(handle_unknown="ignore", sparse_output=False)),
            ]
        )
        transformers.append(("cat", categorical_pipe, categorical_cols))

    return ColumnTransformer(transformers=transformers, remainder="drop")


def build_models(X: pd.DataFrame, random_state: int = RANDOM_STATE) -> dict[str, BaseEstimator]:
    """Instancia os 6 modelos com pré-processamento ajustado em cada pipeline."""
    classifiers = {
        "SVM": SVC(kernel="rbf", probability=True, random_state=random_state),
        "KNN": KNeighborsClassifier(n_neighbors=5, n_jobs=-1),
        "Random Forest": RandomForestClassifier(
            n_estimators=100, random_state=random_state, n_jobs=-1
        ),
        "MLP": MLPClassifier(
            hidden_layer_sizes=(128, 64),
            early_stopping=True,
            max_iter=300,
            random_state=random_state,
        ),
    }
    models: dict[str, BaseEstimator] = {
        name: Pipeline(
            steps=[("preprocessor", build_preprocessor(X)), ("classifier", classifier)]
        )
        for name, classifier in classifiers.items()
    }
    # O Stacking clona e ajusta cada pipeline sobre o treino de cada fold interno.
    models[MODEL_NAMES["stacking"]] = StackingClassifier(
        estimators=[
            ("svm", clone(models["SVM"])),
            ("rf", clone(models["Random Forest"])),
            ("knn", clone(models["KNN"])),
        ],
        final_estimator=MLPClassifier(
            hidden_layer_sizes=(64,),
            early_stopping=True,
            max_iter=300,
            random_state=random_state,
        ),
        cv=3,
        stack_method="predict_proba",
        n_jobs=-1,
    )
    models[MODEL_NAMES["pso"]] = PSOAdaptiveFusionClassifier(
        preprocessor=build_preprocessor(X), random_state=random_state
    )

    return models


def load_completed_keys(metrics_path: Path) -> set[tuple[str, str, int]]:
    """Carrega execuções já salvas para permitir retomada segura com --resume."""
    if not metrics_path.exists():
        return set()
    existing = pd.read_csv(metrics_path)
    return {
        (str(row["dataset"]), str(row["model"]), int(row["fold"]))
        for _, row in existing.iterrows()
    }


def append_metric_row(metrics_path: Path, row: dict[str, object]) -> None:
    """Anexa uma linha de métrica ao CSV incrementalmente."""
    write_header = not metrics_path.exists() or metrics_path.stat().st_size == 0
    pd.DataFrame([row]).to_csv(metrics_path, mode="a", header=write_header, index=False)


def run_benchmark(
    dataset_filter: list[str] | None = None,
    n_splits: int = N_SPLITS,
    resume: bool = False,
    *,
    model_filter: list[str] | None = None,
    results_path: Path | None = None,
) -> None:
    """Executa a validação cruzada estratificada nas bases e modelos selecionados."""
    metrics_path = (Path(results_path) if results_path is not None else METRICS_PATH).resolve()
    cv = StratifiedKFold(n_splits=n_splits, shuffle=True, random_state=RANDOM_STATE)

    dataset_paths = {path.name[:-7]: path for path in sorted(PROCESSED_DIR.glob("*.csv.gz"))}
    if not dataset_paths:
        raise FileNotFoundError(
            f"Nenhum arquivo .csv.gz encontrado em {PROCESSED_DIR}. "
            "Execute primeiro: python src/01_prepare_data.py"
        )

    selected_datasets = (
        {name.removesuffix(".gz").removesuffix(".csv") for name in dataset_filter}
        if dataset_filter else set(dataset_paths)
    )
    if unknown_datasets := selected_datasets.difference(dataset_paths):
        raise ValueError(f"Bases desconhecidas: {', '.join(sorted(unknown_datasets))}.")
    selected_models = set(model_filter or MODEL_NAMES)
    if unknown_models := selected_models.difference(MODEL_NAMES):
        raise ValueError(f"Modelos desconhecidos: {', '.join(sorted(unknown_models))}.")
    model_names = [name for alias, name in MODEL_NAMES.items() if alias in selected_models]

    metrics_path.parent.mkdir(parents=True, exist_ok=True)
    if not resume and metrics_path.exists():
        raise FileExistsError(
            f"O CSV já existe: {metrics_path}. Use --resume ou outro --results-path."
        )
    completed = load_completed_keys(metrics_path) if resume else set()

    for dataset_name, dataset_path in dataset_paths.items():
        if dataset_name not in selected_datasets:
            continue
        df = pd.read_csv(dataset_path)
        X = df.drop(columns=["target"])
        y = LabelEncoder().fit_transform(df["target"])

        print(f"\n[02_run_benchmark] Base: {dataset_name} ({len(df)} amostras, {X.shape[1]} attrs)")

        for fold_idx, (train_idx, test_idx) in enumerate(cv.split(X, y), start=1):
            X_train, X_test = X.iloc[train_idx], X.iloc[test_idx]
            y_train, y_test = y[train_idx], y[test_idx]

            models = build_models(X_train, random_state=RANDOM_STATE)
            for model_name in model_names:
                if (dataset_name, model_name, fold_idx) in completed:
                    print(f"  [skip] fold {fold_idx}/{n_splits} | {model_name}")
                    continue

                model = models[model_name]
                t_start = time.perf_counter()
                model.fit(X_train, y_train)
                fit_time = time.perf_counter() - t_start

                y_pred = model.predict(X_test)
                f1 = f1_score(y_test, y_pred, average="macro")
                bal_acc = balanced_accuracy_score(y_test, y_pred)

                row = {
                    "dataset": dataset_name,
                    "model": model_name,
                    "fold": fold_idx,
                    "f1_macro": round(float(f1), 6),
                    "balanced_accuracy": round(float(bal_acc), 6),
                    "fit_time": round(float(fit_time), 4),
                }
                append_metric_row(metrics_path, row)
                print(
                    f"  fold {fold_idx}/{n_splits} | {model_name:18s} | "
                    f"F1={f1:.4f} | BalAcc={bal_acc:.4f} | t={fit_time:.2f}s"
                )

    print(f"\nMétricas brutas salvas em: {metrics_path}")


def parse_args() -> argparse.Namespace:
    """Interpreta argumentos opcionais de linha de comando."""
    parser = argparse.ArgumentParser(description="Executa benchmark 5-Fold Stratified CV.")
    parser.add_argument(
        "--datasets",
        nargs="+",
        default=None,
        help="Nomes das bases (ex: 07_ai4i_2020 08_electrical_grid) para filtrar execução.",
    )
    parser.add_argument(
        "--n-splits",
        type=int,
        default=N_SPLITS,
        help="Número de folds do StratifiedKFold (padrão: 5).",
    )
    parser.add_argument(
        "--models",
        nargs="+",
        choices=MODEL_NAMES,
        help="Modelos a executar (padrão: todos).",
    )
    parser.add_argument(
        "--results-path",
        type=Path,
        default=METRICS_PATH,
        help="CSV de saída (padrão: results/metrics_raw_refactored.csv).",
    )
    parser.add_argument(
        "--resume",
        action="store_true",
        help="Retoma o CSV e pula combinações de base, modelo e fold já gravadas.",
    )
    return parser.parse_args()


if __name__ == "__main__":
    args = parse_args()
    run_benchmark(
        dataset_filter=args.datasets,
        n_splits=args.n_splits,
        resume=args.resume,
        model_filter=args.models,
        results_path=args.results_path,
    )
