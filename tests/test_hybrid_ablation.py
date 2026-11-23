import unittest
from importlib import import_module
from pathlib import Path
from tempfile import TemporaryDirectory

import numpy as np
import pandas as pd
from threadpoolctl import threadpool_limits


ablation = import_module("src.04_run_hybrid_ablation")


class HybridAblationTests(unittest.TestCase):
    def test_folds_preserve_uniform_and_fixed_weights_and_match_production_pso(self):
        rng = np.random.RandomState(42)
        y = np.tile(np.arange(3), 20)
        X = pd.DataFrame({"numeric": y + rng.normal(size=len(y)), "category": y.astype(str)})
        X.loc[::7, "numeric"] = np.nan
        X.loc[::11, "category"] = np.nan
        X_train, X_test = X.iloc[:-12], X.iloc[-12:]
        y_train = y[:-12]
        with threadpool_limits(limits=1):
            results = ablation.evaluate_fold(X_train, y_train, X_test, 42)
            model = ablation.benchmark.build_models(X_train)[ablation.benchmark.MODEL_NAMES["pso"]]
            model.fit(X_train, y_train)
            expected = model.predict_proba(X_test)

        self.assertEqual(set(results), {"uniform", "pso_fixed", "pso_adaptive", "stacking"})
        uniform, fixed, adaptive = (results[name] for name in ("uniform", "pso_fixed", "pso_adaptive"))
        np.testing.assert_allclose(uniform.weights, 1 / 3)
        np.testing.assert_array_equal(fixed.coefficients[1], 0)
        np.testing.assert_allclose(fixed.weights, np.repeat(fixed.weights[:1], len(X_test), axis=0))
        np.testing.assert_allclose(adaptive.probabilities, expected, rtol=0, atol=1e-12)
        np.testing.assert_allclose(adaptive.coefficients, model.coefficients_, rtol=0, atol=1e-12)
        self.assertEqual(adaptive.oof_f1_macro, model.oof_f1_)
        for result in results.values():
            np.testing.assert_allclose(result.probabilities.sum(axis=1), 1)

    def test_two_fold_run_is_reproducible_and_preserves_existing_output(self):
        y = np.tile([0, 0, 1], 20)
        X = pd.DataFrame({"numeric": y * 2 + np.sin(np.arange(len(y))), "target": y})
        with TemporaryDirectory() as directory:
            data_dir = Path(directory)
            X.to_csv(data_dir / "small.csv.gz", index=False)
            output = data_dir / "new_results" / "ablation.csv"
            metrics = ablation.run_ablation(data_dir, output, n_splits=2, check_reproducibility=True)
            self.assertEqual(len(metrics), 8)
            self.assertFalse(metrics.duplicated(["dataset", "strategy", "fold"]).any())
            self.assertTrue(metrics["reproducible"].all())
            self.assertTrue(metrics["f1_macro"].between(0, 1).all())
            self.assertTrue(metrics["balanced_accuracy"].between(0, 1).all())
            pso_rows = metrics[metrics["strategy"].str.startswith("pso_")]
            np.testing.assert_allclose(
                pso_rows["fit_time"], pso_rows["base_fit_time"] + pso_rows["oof_time"] + pso_rows["pso_time"]
            )
            original = output.read_bytes()
            with self.assertRaises(FileExistsError):
                ablation.run_ablation(data_dir, output, n_splits=2)
            self.assertEqual(output.read_bytes(), original)
