import importlib.util
import io
from collections import Counter
from contextlib import redirect_stdout
from pathlib import Path
from tempfile import TemporaryDirectory
import unittest
from unittest.mock import patch

import numpy as np
import pandas as pd
from joblib import parallel_backend
from sklearn.compose import ColumnTransformer
from sklearn.model_selection import StratifiedKFold


benchmark_path = Path(__file__).resolve().parents[1] / "src" / "02_run_benchmark.py"
spec = importlib.util.spec_from_file_location("run_benchmark", benchmark_path)
benchmark = importlib.util.module_from_spec(spec)
spec.loader.exec_module(benchmark)


def small_dataset(n_classes, include_category=True):
    y = np.tile(np.arange(n_classes), 30)
    X = pd.DataFrame({"number": y * 10.0 + np.arange(len(y)) / len(y)})
    X.loc[::11, "number"] = np.nan
    if include_category:
        X["category"] = np.char.add("group_", y.astype(str))
        X.loc[::13, "category"] = np.nan
    return X, y


class BenchmarkTests(unittest.TestCase):
    def test_stacking_fits_transformers_on_each_internal_training_fold(self):
        X, y = small_dataset(n_classes=2)
        splits = list(StratifiedKFold(n_splits=3).split(X, y))
        for fold, (_, validation_indices) in enumerate(splits):
            X.loc[validation_indices, "category"] = f"fold_{fold}"
            X.loc[validation_indices, "number"] += fold * 100
        X.loc[::13, "category"] = np.nan

        hybrid = benchmark.build_models(X)["Hibrido (Stacking)"]
        expected_fits = Counter({tuple(X.index): 3})
        expected_fits.update({tuple(X.index[train]): 3 for train, _ in splits})
        observed_fits = Counter()
        fit_transform = ColumnTransformer.fit_transform

        def record_fit(preprocessor, data, *args, **kwargs):
            transformed = fit_transform(preprocessor, data, *args, **kwargs)
            observed_fits[tuple(data.index)] += 1

            numeric = preprocessor.named_transformers_["num"].named_steps
            median = data["number"].median()
            imputed = data["number"].fillna(median).to_numpy()
            np.testing.assert_allclose(numeric["imputer"].statistics_, [median])
            np.testing.assert_allclose(numeric["scaler"].mean_, [imputed.mean()])
            np.testing.assert_allclose(numeric["scaler"].var_, [imputed.var()])

            categorical = preprocessor.named_transformers_["cat"].named_steps
            mode = data["category"].mode().iloc[0]
            self.assertEqual(categorical["imputer"].statistics_[0], mode)
            self.assertEqual(
                set(categorical["encoder"].categories_[0]), set(data["category"].dropna())
            )
            return transformed

        hybrid.set_params(n_jobs=1)
        with patch.object(ColumnTransformer, "fit_transform", record_fit):
            hybrid.fit(X, y)

        self.assertEqual(observed_fits, expected_fits)

    def test_stacking_predicts_binary_and_multiclass_on_numeric_and_mixed_data(self):
        for n_classes in (2, 3):
            for include_category in (False, True):
                with self.subTest(n_classes=n_classes, include_category=include_category):
                    X, y = small_dataset(n_classes, include_category)
                    X_train, X_test = X.iloc[:-12], X.iloc[-12:].copy()
                    if include_category:
                        X_test["category"] = "unseen"
                    hybrid = benchmark.build_models(X_train)["Hibrido (Stacking)"]
                    with parallel_backend("threading"):
                        hybrid.fit(X_train, y[:-12])
                        predictions = hybrid.predict(X_test)
                        probabilities = hybrid.predict_proba(X_test)

                    self.assertEqual(predictions.shape, (len(X_test),))
                    self.assertTrue(np.isin(predictions, np.unique(y)).all())
                    self.assertEqual(probabilities.shape, (len(X_test), n_classes))
                    np.testing.assert_allclose(probabilities.sum(axis=1), 1)

    def test_benchmark_writes_metrics_for_all_models_and_five_folds(self):
        X, y = small_dataset(n_classes=2)
        with TemporaryDirectory() as directory:
            processed_dir = Path(directory) / "processed"
            processed_dir.mkdir()
            X.assign(target=y).to_csv(processed_dir / "small.csv.gz", index=False)
            results_dir = Path(directory) / "results"
            metrics_path = results_dir / "metrics_raw.csv"
            with (
                patch.multiple(
                    benchmark,
                    PROCESSED_DIR=processed_dir,
                    RESULTS_DIR=results_dir,
                    METRICS_PATH=metrics_path,
                ),
                parallel_backend("threading"),
                redirect_stdout(io.StringIO()),
            ):
                benchmark.run_benchmark()
            metrics = pd.read_csv(metrics_path)

        models = ("SVM", "KNN", "Random Forest", "MLP", "Hibrido (Stacking)")
        expected_runs = Counter(("small", model, fold) for model in models for fold in range(1, 6))
        self.assertEqual(
            Counter(metrics[["dataset", "model", "fold"]].itertuples(index=False, name=None)),
            expected_runs,
        )
        self.assertTrue(metrics["f1_macro"].between(0, 1).all())
        self.assertTrue(metrics["balanced_accuracy"].between(0, 1).all())
        self.assertTrue(metrics["fit_time"].ge(0).all())
