import io
import unittest
from contextlib import redirect_stdout
from importlib import import_module
from pathlib import Path
from tempfile import TemporaryDirectory
from unittest.mock import patch

import numpy as np
import pandas as pd
from scipy.stats import friedmanchisquare


stats = import_module("src.03_stats_analysis")


def benchmark_metrics():
    return pd.DataFrame([
        (spec.name, model, fold, 0.2 + j * 0.07 + fold / 100, 0.6, 1.0)
        for spec in stats.preparation.DATASET_SPECS
        for j, model in enumerate(stats.benchmark.MODEL_NAMES.values())
        for fold in range(1, 6)
    ], columns=stats.KEY_COLUMNS + stats.METRIC_COLUMNS)


class StatsAnalysisTests(unittest.TestCase):
    def test_invalid_grids_and_metrics_are_rejected(self):
        metrics = benchmark_metrics()
        cases = [
            ("Colunas esperadas", metrics.drop(columns="f1_macro")),
            ("CSV de benchmark", metrics.assign(strategy="uniform")),
            ("duplicadas", pd.concat([metrics, metrics.iloc[:1].assign(fold="1.0")])),
            ("1 combinações faltantes", metrics.iloc[1:]),
            ("inesperadas", metrics.replace({"dataset": {"04_dry_bean": "outra_base"}})),
            ("incompatível", metrics[metrics["model"] != "SVM"]),
            ("fold inválida", metrics.assign(fold=1.5)),
            ("inválidas", metrics.assign(f1_macro="texto")),
            ("inválidas", metrics.assign(balanced_accuracy=1.1)),
            ("inválidas", metrics.assign(fit_time=np.inf)),
        ]
        for message, invalid in cases:
            with self.subTest(message=message), self.assertRaisesRegex(ValueError, message):
                stats.validate_metrics(invalid)

    def test_analysis_paths_match_the_benchmark(self):
        self.assertEqual(stats.METRICS_PATH, stats.benchmark.METRICS_PATH)
        self.assertEqual(stats.OUTPUT_DIR, stats.benchmark.RESULTS_DIR / "stats_dry_bean")

    def test_generates_artifacts_from_fold_means_and_preserves_existing_files(self):
        metrics = benchmark_metrics()
        with TemporaryDirectory() as directory:
            root = Path(directory)
            source = root / "benchmark.csv"
            output = root / "analysis"
            metrics.to_csv(source, index=False)
            original = source.read_bytes()
            with patch.multiple(stats, METRICS_PATH=source, OUTPUT_DIR=output), redirect_stdout(io.StringIO()):
                stats.main()
            outputs = {path.name: path.read_bytes() for path in output.iterdir()}
            self.assertEqual(set(outputs), {
                "summary_table.csv", "summary_table_formatted.csv", "nemenyi_pvalues.csv",
                "stats_summary.txt", "cd_diagram.png",
            })
            means = metrics.groupby(stats.KEY_COLUMNS[:2])["f1_macro"].mean()
            summary = pd.read_csv(output / "summary_table.csv").set_index(stats.KEY_COLUMNS[:2])
            self.assertEqual(len(summary), 60)
            np.testing.assert_allclose(summary["f1_macro_mean"].reindex(means.index), means)
            statistic = friedmanchisquare(*means.unstack().to_numpy().T).statistic
            self.assertIn(f"{statistic:.6f}", outputs["stats_summary.txt"].decode())
            self.assertIn(str(source), outputs["stats_summary.txt"].decode())
            self.assertEqual(pd.read_csv(output / "nemenyi_pvalues.csv", index_col=0).shape, (6, 6))
            self.assertTrue(outputs["cd_diagram.png"].startswith(b"\x89PNG"))
            with patch.multiple(stats, METRICS_PATH=source, OUTPUT_DIR=output), self.assertRaises(FileExistsError):
                stats.main()
            self.assertEqual({path.name: path.read_bytes() for path in output.iterdir()}, outputs)
            self.assertEqual(source.read_bytes(), original)

            existing = root / "previous"
            existing.mkdir()
            plot = existing / "cd_diagram.png"
            plot.write_bytes(outputs["cd_diagram.png"])
            with patch.multiple(stats, METRICS_PATH=source, OUTPUT_DIR=existing), self.assertRaises(FileExistsError):
                stats.main()
            self.assertEqual(list(existing.iterdir()), [plot])
            self.assertEqual(plot.read_bytes(), outputs["cd_diagram.png"])

    def test_incomplete_csv_does_not_create_output_directory(self):
        with TemporaryDirectory() as directory:
            root = Path(directory)
            source, output = root / "incomplete.csv", root / "analysis"
            benchmark_metrics().iloc[1:].to_csv(source, index=False)
            with (
                patch.multiple(stats, METRICS_PATH=source, OUTPUT_DIR=output),
                self.assertRaisesRegex(ValueError, "faltantes"),
            ):
                stats.main()
            self.assertFalse(output.exists())
