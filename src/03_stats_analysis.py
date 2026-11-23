"""Fase 4: Validação estatística (Friedman-Nemenyi), tabela de ranks e CD Diagram.

Lê ``results/metrics_raw_dry_bean.csv``, calcula médias, desvios padrão e ranks por base,
executa o teste não-paramétrico de Friedman seguido pelo pós-teste de Nemenyi
(Demšar, 2006) e grava os artefatos em ``results/stats_dry_bean/``.
"""

from __future__ import annotations

import math
from contextlib import ExitStack
from importlib import import_module
from itertools import product
from pathlib import Path
from typing import BinaryIO

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from scipy.stats import friedmanchisquare, studentized_range

try:
    import scikit_posthocs as sp
except ImportError:  # Fallback analítico caso scikit-posthocs ainda não esteja instalado.
    sp = None

ALPHA = 0.05
PRIMARY_METRIC = "f1_macro"

module_prefix = f"{__package__}." if __package__ else ""
preparation = import_module(f"{module_prefix}01_prepare_data")
benchmark = import_module(f"{module_prefix}02_run_benchmark")

METRICS_PATH = benchmark.METRICS_PATH
OUTPUT_DIR = benchmark.RESULTS_DIR / "stats_dry_bean"
KEY_COLUMNS = ["dataset", "model", "fold"]
METRIC_COLUMNS = [PRIMARY_METRIC, "balanced_accuracy", "fit_time"]


def validate_metrics(raw_df: pd.DataFrame) -> pd.DataFrame:
    """Confere a grade completa do protocolo antes de agregar os folds."""
    required = KEY_COLUMNS + METRIC_COLUMNS
    if "strategy" in raw_df or not set(required).issubset(raw_df):
        raise ValueError("CSV de benchmark inválido. Colunas esperadas: " + ", ".join(required))

    data = raw_df[required].copy()
    numeric = ["fold", *METRIC_COLUMNS]
    data[numeric] = data[numeric].apply(pd.to_numeric, errors="coerce")
    if (
        not np.isfinite(data[METRIC_COLUMNS].to_numpy(dtype=float, na_value=np.nan)).all()
        or not data[PRIMARY_METRIC].between(0, 1).all()
        or not data["balanced_accuracy"].between(0, 1).all()
        or not data["fit_time"].ge(0).all()
    ):
        raise ValueError("Métricas inválidas: use escores finitos em [0, 1] e tempos finitos não negativos.")

    folds = range(1, benchmark.N_SPLITS + 1)
    if not data["fold"].isin(folds).all() or data[["dataset", "model"]].isna().any().any():
        raise ValueError("Identificação de base, modelo ou fold inválida.")
    if data.duplicated(KEY_COLUMNS).any():
        raise ValueError("Há combinações duplicadas de base, modelo e fold.")

    datasets = {spec.name for spec in preparation.DATASET_SPECS}
    models = set(benchmark.MODEL_NAMES.values())
    expected = set(product(datasets, models, folds))
    observed = set(data[KEY_COLUMNS].itertuples(index=False, name=None))
    if observed != expected:
        raise ValueError(
            "Grade incompatível com o protocolo dry-bean: "
            f"{len(data)} de {len(expected)} registros esperados, "
            f"{len(expected - observed)} combinações faltantes e {len(observed - expected)} inesperadas."
        )
    return data


def compute_summary_tables(raw_df: pd.DataFrame) -> tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame]:
    """Agrega métricas por (base, modelo), calcula ranks e monta tabela formatada."""
    summary = (
        raw_df.groupby(["dataset", "model"], as_index=False)
        .agg(
            f1_macro_mean=("f1_macro", "mean"),
            f1_macro_std=("f1_macro", "std"),
            balanced_accuracy_mean=("balanced_accuracy", "mean"),
            balanced_accuracy_std=("balanced_accuracy", "std"),
            fit_time_mean=("fit_time", "mean"),
            fit_time_std=("fit_time", "std"),
        )
        .fillna(0.0)
    )

    f1_pivot = summary.pivot(index="dataset", columns="model", values="f1_macro_mean")
    ranks_pivot = f1_pivot.rank(axis=1, ascending=False, method="average")

    summary["f1_rank"] = summary.apply(
        lambda row: float(ranks_pivot.loc[row["dataset"], row["model"]]), axis=1
    )

    # Tabela formatada (Média ± DP) pronta para os slides.
    formatted = pd.DataFrame(index=f1_pivot.index)
    for model in f1_pivot.columns:
        sub = summary[summary["model"] == model].set_index("dataset")
        formatted[model] = [
            f"{sub.loc[ds, 'f1_macro_mean']:.4f} ± {sub.loc[ds, 'f1_macro_std']:.4f} "
            f"({ranks_pivot.loc[ds, model]:.1f})"
            for ds in f1_pivot.index
        ]

    avg_ranks = ranks_pivot.mean(axis=0)
    avg_time = summary.groupby("model")["fit_time_mean"].mean()

    formatted.loc["Rank Médio"] = [f"{avg_ranks[m]:.2f}" for m in f1_pivot.columns]
    formatted.loc["Tempo Médio de Treino (s)"] = [f"{avg_time[m]:.2f}s" for m in f1_pivot.columns]
    formatted = formatted.reset_index().rename(columns={"index": "dataset"})

    return summary, f1_pivot, formatted


def compute_nemenyi_and_cd(
    f1_pivot: pd.DataFrame, alpha: float = ALPHA
) -> tuple[pd.Series, pd.DataFrame, float]:
    """Calcula ranks médios, matriz de p-valores de Nemenyi e Diferença Crítica (CD)."""
    n_datasets, k_models = f1_pivot.shape
    ranks_pivot = f1_pivot.rank(axis=1, ascending=False, method="average")
    avg_ranks = ranks_pivot.mean(axis=0).sort_values(ascending=True)

    se = math.sqrt(k_models * (k_models + 1) / (6.0 * n_datasets))
    q_alpha = float(studentized_range.ppf(1.0 - alpha, k_models, np.inf) / math.sqrt(2.0))
    cd = q_alpha * se

    if sp is not None:
        pvalues = sp.posthoc_nemenyi_friedman(f1_pivot)
    else:
        models = list(f1_pivot.columns)
        mat = np.ones((k_models, k_models), dtype=float)
        for i, mi in enumerate(models):
            for j, mj in enumerate(models):
                if i < j:
                    q_stat = abs(avg_ranks[mi] - avg_ranks[mj]) / se * math.sqrt(2.0)
                    p_val = float(studentized_range.sf(q_stat, k_models, np.inf))
                    mat[i, j] = mat[j, i] = p_val
        pvalues = pd.DataFrame(mat, index=models, columns=models)

    return avg_ranks, pvalues, cd


def find_maximal_cliques(avg_ranks: pd.Series, pvalues: pd.DataFrame, alpha: float) -> list[tuple[int, int]]:
    """Identifica cliques máximos contíguos sem diferença estatisticamente significativa."""
    ordered = list(avg_ranks.index)
    n = len(ordered)
    cliques: list[tuple[int, int]] = []

    for i in range(n):
        j = i
        while j + 1 < n and all(
            float(pvalues.loc[ordered[a], ordered[b]]) >= alpha
            for a in range(i, j + 2)
            for b in range(a + 1, j + 2)
        ):
            j += 1
        if j > i:
            # Mantém apenas cliques maximais (não contidos em outro clique já registrado).
            if not any(c_start <= i and j <= c_end for c_start, c_end in cliques):
                cliques.append((i, j))
    return cliques


def plot_cd_diagram(
    avg_ranks: pd.Series,
    pvalues: pd.DataFrame,
    cd: float,
    chi2_stat: float,
    friedman_p: float,
    out_path: Path | BinaryIO,
    alpha: float = ALPHA,
) -> None:
    """Gera o Diagrama de Diferença Crítica de Demšar (2006) em alta resolução (300 DPI)."""
    ordered = list(avg_ranks.index)
    ranks = avg_ranks.values
    k = len(ordered)
    cliques = find_maximal_cliques(avg_ranks, pvalues, alpha)

    plt.rcParams.update({"font.family": "DejaVu Sans", "font.size": 10})
    fig, ax = plt.subplots(figsize=(10.5, 4.8), dpi=300)

    y_axis = 0.0
    low_rank, high_rank = 1.0, float(k)

    # Eixo principal invertido (melhor rank = 1 à direita, convenção de Demšar).
    ax.set_xlim(high_rank + 0.85, low_rank - 0.85)
    n_left = k // 2
    n_right = k - n_left
    max_rows = max(n_left, n_right, 2)
    y_min = -(0.65 + max_rows * 0.48 + len(cliques) * 0.22)
    ax.set_ylim(y_min, 1.35)
    ax.axis("off")

    # Linha horizontal do eixo de ranks e marcações.
    ax.plot([low_rank, high_rank], [y_axis, y_axis], color="#222222", linewidth=1.6)
    for tick in range(1, k + 1):
        ax.plot([tick, tick], [y_axis, y_axis + 0.12], color="#222222", linewidth=1.4)
        ax.text(tick, y_axis + 0.18, f"{tick}", ha="center", va="bottom", fontsize=10, fontweight="bold")
    for half_tick in np.arange(1.5, k, 1.0):
        ax.plot([half_tick, half_tick], [y_axis, y_axis + 0.06], color="#555555", linewidth=0.9)

    # Barra de escala da Diferença Crítica (CD) no topo.
    cd_y = 0.72
    cd_left = high_rank
    cd_right = high_rank - cd
    ax.plot([cd_left, cd_right], [cd_y, cd_y], color="#1f4e79", linewidth=2.2)
    ax.plot([cd_left, cd_left], [cd_y - 0.08, cd_y + 0.08], color="#1f4e79", linewidth=2.0)
    ax.plot([cd_right, cd_right], [cd_y - 0.08, cd_y + 0.08], color="#1f4e79", linewidth=2.0)
    ax.text(
        (cd_left + cd_right) / 2.0,
        cd_y + 0.12,
        f"CD = {cd:.3f} (α = {alpha:.2f})",
        ha="center",
        va="bottom",
        fontsize=10.5,
        fontweight="bold",
        color="#1f4e79",
    )

    # Rótulos dos modelos (melhores à direita, piores à esquerda).
    right_models = list(zip(ordered[:n_right], ranks[:n_right]))
    left_models = list(zip(ordered[n_right:][::-1], ranks[n_right:][::-1]))

    label_x_right = low_rank - 0.35
    label_x_left = high_rank + 0.35
    base_drop = -0.35 - len(cliques) * 0.20
    step_drop = 0.45

    for idx, (name, r_val) in enumerate(right_models):
        y_pos = base_drop - idx * step_drop
        ax.plot([r_val, r_val, label_x_right + 0.06], [y_axis, y_pos, y_pos], color="#333333", linewidth=1.1)
        ax.scatter([r_val], [y_axis], color="#c0392b" if idx == 0 else "#222222", s=26, zorder=4)
        ax.text(
            label_x_right,
            y_pos,
            f"{name} ({r_val:.2f})",
            ha="left",
            va="center",
            fontsize=10,
            fontweight="bold" if idx == 0 else "normal",
        )

    for idx, (name, r_val) in enumerate(left_models):
        y_pos = base_drop - idx * step_drop
        ax.plot([r_val, r_val, label_x_left - 0.06], [y_axis, y_pos, y_pos], color="#333333", linewidth=1.1)
        ax.scatter([r_val], [y_axis], color="#222222", s=26, zorder=4)
        ax.text(
            label_x_left,
            y_pos,
            f"{name} ({r_val:.2f})",
            ha="right",
            va="center",
            fontsize=10,
        )

    # Barras horizontais conectando grupos sem diferença estatisticamente significativa.
    for c_idx, (start_i, end_j) in enumerate(cliques):
        y_clique = -0.18 - c_idx * 0.18
        r_start = ranks[start_i] - 0.04
        r_end = ranks[end_j] + 0.04
        ax.plot([r_start, r_end], [y_clique, y_clique], color="#222222", linewidth=3.4, solid_capstyle="round")

    ax.set_title(
        f"Diagrama de Diferença Crítica (Friedman-Nemenyi | F1-Score Macro)\n"
        f"Friedman χ² = {chi2_stat:.4f}, p = {friedman_p:.4e}",
        fontsize=11.5,
        fontweight="bold",
        pad=10,
    )

    try:
        fig.savefig(out_path, format="png", dpi=300, bbox_inches="tight", facecolor="white")
    finally:
        plt.close(fig)


def main() -> None:
    """Executa toda a análise estatística e salva tabelas, relatório e gráfico."""
    raw_df = pd.read_csv(METRICS_PATH)
    if not isinstance(raw_df.index, pd.RangeIndex):
        raise ValueError("O CSV contém campos além das colunas declaradas.")
    summary_df, f1_pivot, formatted_df = compute_summary_tables(validate_metrics(raw_df))

    chi2_stat, friedman_p = friedmanchisquare(
        *[f1_pivot[col].values for col in f1_pivot.columns]
    )
    avg_ranks, pvalues, cd = compute_nemenyi_and_cd(f1_pivot, alpha=ALPHA)

    n_datasets, k_models = f1_pivot.shape
    decision = (
        "Rejeita-se H0 (há diferença estatisticamente significativa entre os modelos)."
        if friedman_p < ALPHA
        else "Não se rejeita H0 (sem evidência de diferença significativa entre os modelos)."
    )
    report_lines = [
        "=== Análise Estatística: Teste de Friedman + Pós-teste de Nemenyi ===",
        "Protocolo              : dry-bean",
        f"CSV de entrada         : {METRICS_PATH}",
        f"Métrica avaliada       : {PRIMARY_METRIC}",
        f"Bases (N)              : {n_datasets}",
        f"Modelos (k)            : {k_models}",
        f"Nível de significância : alpha = {ALPHA}",
        f"Estatística Friedman χ²: {chi2_stat:.6f}",
        f"p-valor (Friedman)     : {friedman_p:.6e}",
        f"Decisão (α={ALPHA})      : {decision}",
        f"Diferença Crítica (CD) : {cd:.4f}",
        "",
        "--- Ranks Médios (1 = melhor) ---",
        *[f"  {model:20s}: {rank:.4f}" for model, rank in avg_ranks.items()],
        "",
        "--- Matriz de p-valores (Nemenyi) ---",
        pvalues.round(6).to_string(),
    ]
    report_text = "\n".join(report_lines)
    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)
    with ExitStack() as rollback, ExitStack() as files:
        outputs = {}
        for filename in ("summary_table.csv", "summary_table_formatted.csv", "nemenyi_pvalues.csv",
                         "stats_summary.txt", "cd_diagram.png"):
            path = OUTPUT_DIR / filename
            try:
                outputs[filename] = files.enter_context(path.open("xb"))
            except FileExistsError:
                raise FileExistsError(f"O arquivo {path} já existe.") from None
            rollback.callback(path.unlink)

        summary_df.to_csv(outputs["summary_table.csv"], index=False)
        formatted_df.to_csv(outputs["summary_table_formatted.csv"], index=False)
        pvalues.to_csv(outputs["nemenyi_pvalues.csv"])
        outputs["stats_summary.txt"].write(report_text.encode("utf-8"))
        plot_cd_diagram(
            avg_ranks, pvalues, cd, float(chi2_stat), float(friedman_p),
            outputs["cd_diagram.png"], alpha=ALPHA,
        )
        files.close()
        rollback.pop_all()

    print(report_text)
    print(f"\nArtefatos salvos em: {OUTPUT_DIR}")


if __name__ == "__main__":
    try:
        main()
    except (ValueError, FileNotFoundError, FileExistsError) as error:
        raise SystemExit(str(error)) from None
