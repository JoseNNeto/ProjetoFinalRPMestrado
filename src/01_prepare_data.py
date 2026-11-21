"""Fase 1: Limpeza, padronização e subamostragem estratificada das 10 bases.

Lê os datasets originais (prioritariamente de ``data/raw/``, com fallback para
``dataset/``), aplica limpeza específica por base (remoção de IDs e atributos
com vazamento de dados), aplica subamostragem estratificada para 20.000
instâncias nas bases extensas e salva as versões padronizadas em
``data/processed/`` com a coluna alvo nomeada ``target``.
Também gera ``results/dataset_metadata.csv`` com estatísticas resumidas.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from pathlib import Path
from typing import Callable

import numpy as np
import pandas as pd
from sklearn.model_selection import train_test_split

RANDOM_STATE = 42
MAX_SAMPLES_DEFAULT = 20_000

PROJECT_ROOT = Path(__file__).resolve().parent.parent
RAW_SEARCH_DIRS = [PROJECT_ROOT / "data" / "raw", PROJECT_ROOT / "dataset"]
PROCESSED_DIR = PROJECT_ROOT / "data" / "processed"
RESULTS_DIR = PROJECT_ROOT / "results"


@dataclass(frozen=True)
class DatasetSpec:
    """Especificação de leitura e limpeza de uma base tabular."""

    name: str
    display_name: str
    domain: str
    filename: str
    target_col: str | None
    drop_cols: list[str] = field(default_factory=list)
    usecols: list[str] | None = None
    max_samples: int | None = MAX_SAMPLES_DEFAULT
    custom_clean: Callable[[pd.DataFrame], pd.DataFrame] | None = None


def _clean_flight_delay(df: pd.DataFrame) -> pd.DataFrame:
    """Filtra voos sem alvo definido e padroniza colunas pré-voo."""
    df = df.rename(columns=lambda col: col.strip())
    df = df[df["ArrDel15"].notna()].copy()
    df["ArrDel15"] = df["ArrDel15"].astype(int)
    return df


def _clean_steel_plates(df: pd.DataFrame) -> pd.DataFrame:
    """Converte as 7 colunas binárias exclusivas de defeito em alvo multiclasse."""
    fault_cols = [
        "Pastry",
        "Z_Scratch",
        "K_Scatch",
        "Stains",
        "Dirtiness",
        "Bumps",
        "Other_Faults",
    ]
    df = df.copy()
    df["target"] = df[fault_cols].idxmax(axis=1)
    return df.drop(columns=["id", *fault_cols])


def _clean_telco_churn(df: pd.DataFrame) -> pd.DataFrame:
    """Converte TotalCharges para numérico preservando NaNs para imputação no Pipeline."""
    df = df.copy()
    df["TotalCharges"] = pd.to_numeric(df["TotalCharges"], errors="coerce")
    return df


DATASET_SPECS: list[DatasetSpec] = [
    DatasetSpec(
        name="01_card_transdata",
        display_name="Card Transdata",
        domain="Finanças / Fraude",
        filename="card_transdata.csv.gz",
        target_col="fraud",
    ),
    DatasetSpec(
        name="02_smoking_drinking",
        display_name="Smoking & Drinking",
        domain="Saúde / Biometria",
        filename="smoking_driking_dataset_Ver01.csv.gz",
        target_col="DRK_YN",
    ),
    DatasetSpec(
        name="03_flight_delay",
        display_name="Flight Delay Prediction",
        domain="Aviação / Logística",
        filename="flight_data_2018_to_2022.csv.gz",
        target_col="ArrDel15",
        # Apenas atributos estritamente pré-voo (evita leakage pós-voo, data exata e IDs).
        usecols=[
            "Quarter",
            "Month",
            "DayofMonth",
            "DayOfWeek",
            "Operating_Airline ",
            "OriginState",
            "DestState",
            "CRSDepTime",
            "CRSArrTime",
            "CRSElapsedTime",
            "Distance",
            "DistanceGroup",
            "ArrDel15",
        ],
        custom_clean=_clean_flight_delay,
    ),
    DatasetSpec(
        name="04_emg_gesture",
        display_name="EMG Gesture Recognition",
        domain="Sinais Biomédicos",
        filename="EMG-data.csv.gz",
        target_col="class",
        drop_cols=["time", "label"],
    ),
    DatasetSpec(
        name="05_nasa_neo",
        display_name="NASA Nearest Earth Objects",
        domain="Astrofísica / Órbita",
        filename="nearest-earth-objects(1910-2024).csv.gz",
        target_col="is_hazardous",
        drop_cols=["neo_id", "name", "orbiting_body"],
    ),
    DatasetSpec(
        name="06_sdss_dr16",
        display_name="Sloan Digital Sky Survey DR16",
        domain="Astronomia",
        filename="sdss-IV-dr16-70k (1).csv.gz",
        target_col="class",
        drop_cols=["objid", "run", "rerun", "camcol", "field", "plate", "mjd", "fiberid"],
    ),
    DatasetSpec(
        name="07_ai4i_2020",
        display_name="AI4I 2020 Maintenance",
        domain="Indústria / Sensores",
        filename="ai4i2020.csv.gz",
        target_col="Machine failure",
        drop_cols=["UDI", "Product ID", "TWF", "HDF", "PWF", "OSF", "RNF"],
        max_samples=None,
    ),
    DatasetSpec(
        name="08_electrical_grid",
        display_name="Electrical Grid Stability",
        domain="Energia / Sistemas",
        filename="Data_for_UCI_named.csv.gz",
        target_col="stabf",
        drop_cols=["stab"],
        max_samples=None,
    ),
    DatasetSpec(
        name="09_steel_plates_faults",
        display_name="Steel Plates Faults",
        domain="Indústria / Metalurgia",
        filename="steel_plates_faults_original_dataset.csv.gz",
        target_col="target",
        max_samples=None,
        custom_clean=_clean_steel_plates,
    ),
    DatasetSpec(
        name="10_telco_churn",
        display_name="Telco Customer Churn",
        domain="Negócios / Telecom",
        filename="WA_Fn-UseC_-Telco-Customer-Churn.csv.gz",
        target_col="Churn",
        drop_cols=["customerID"],
        max_samples=None,
        custom_clean=_clean_telco_churn,
    ),
]


def resolve_raw_path(filename: str) -> Path:
    """Localiza o dataset bruto (.csv.gz ou .csv) nos diretórios de entrada."""
    stem_csv = filename[:-3] if filename.endswith(".gz") else filename
    candidates = [filename, stem_csv]
    for directory in RAW_SEARCH_DIRS:
        for candidate in candidates:
            path = directory / candidate
            if path.exists():
                return path
    searched = ", ".join(str(d) for d in RAW_SEARCH_DIRS)
    raise FileNotFoundError(f"Arquivo '{filename}' não encontrado em: {searched}")


def stratified_subsample(df: pd.DataFrame, max_samples: int, random_state: int) -> pd.DataFrame:
    """Reduz o dataframe de forma estratificada preservando a distribuição de ``target``."""
    if len(df) <= max_samples:
        return df.reset_index(drop=True)
    sampled_df, _ = train_test_split(
        df,
        train_size=max_samples,
        stratify=df["target"],
        random_state=random_state,
    )
    return sampled_df.reset_index(drop=True)


def prepare_single_dataset(spec: DatasetSpec) -> dict[str, object]:
    """Carrega, limpa, subamostra e salva uma base padronizada retornando seus metadados."""
    raw_path = resolve_raw_path(spec.filename)
    df = pd.read_csv(raw_path, usecols=spec.usecols)
    n_raw_rows = len(df)

    if spec.custom_clean is not None:
        df = spec.custom_clean(df)

    cols_to_drop = [col for col in spec.drop_cols if col in df.columns]
    if cols_to_drop:
        df = df.drop(columns=cols_to_drop)

    if spec.target_col is not None and spec.target_col != "target":
        df = df.rename(columns={spec.target_col: "target"})

    df = df[df["target"].notna()].copy()
    if spec.max_samples is not None:
        df = stratified_subsample(df, spec.max_samples, RANDOM_STATE)

    feature_cols = [col for col in df.columns if col != "target"]
    df = df[[*feature_cols, "target"]]

    out_path = PROCESSED_DIR / f"{spec.name}.csv.gz"
    df.to_csv(out_path, index=False, compression="gzip")

    X = df[feature_cols]
    n_num = len(X.select_dtypes(include=[np.number]).columns)
    n_cat = len(feature_cols) - n_num
    class_counts = df["target"].value_counts(normalize=True)

    return {
        "dataset": spec.name,
        "display_name": spec.display_name,
        "domain": spec.domain,
        "source_file": raw_path.name,
        "n_raw_rows": n_raw_rows,
        "n_used_rows": len(df),
        "n_features": len(feature_cols),
        "n_numeric_features": n_num,
        "n_categorical_features": n_cat,
        "n_classes": int(df["target"].nunique()),
        "majority_class_pct": round(float(class_counts.max() * 100), 2),
        "minority_class_pct": round(float(class_counts.min() * 100), 2),
        "imbalance_ratio": round(float(class_counts.max() / class_counts.min()), 2),
    }


def main() -> None:
    """Executa a preparação de todas as bases e grava a tabela de metadados."""
    PROCESSED_DIR.mkdir(parents=True, exist_ok=True)
    RESULTS_DIR.mkdir(parents=True, exist_ok=True)

    metadata_rows: list[dict[str, object]] = []
    for spec in DATASET_SPECS:
        print(f"[01_prepare_data] Processando {spec.name} ({spec.filename})...")
        meta = prepare_single_dataset(spec)
        metadata_rows.append(meta)
        print(
            f"  -> {meta['n_used_rows']} linhas | {meta['n_features']} atributos "
            f"| {meta['n_classes']} classes | IR={meta['imbalance_ratio']}"
        )

    metadata_df = pd.DataFrame(metadata_rows)
    meta_path = RESULTS_DIR / "dataset_metadata.csv"
    metadata_df.to_csv(meta_path, index=False)
    print(f"\nMetadados salvos em: {meta_path}")


if __name__ == "__main__":
    main()
