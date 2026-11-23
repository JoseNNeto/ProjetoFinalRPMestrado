import unittest
from collections import Counter
from importlib import import_module
from unittest.mock import patch

import numpy as np
import pandas as pd
from sklearn.base import clone, is_classifier
from sklearn.compose import ColumnTransformer
from sklearn.datasets import make_classification
from sklearn.exceptions import NotFittedError
from sklearn.metrics import f1_score
from sklearn.model_selection import StratifiedKFold, cross_val_score

from src.pso_adaptive_fusion import (
    PSOAdaptiveFusionClassifier,
    _adaptive_weights,
    _entropy_confidence,
    _fuse_probabilities,
    _optimize_weights,
)


benchmark = import_module("src.02_run_benchmark")


class FusionMathTests(unittest.TestCase):
    def test_entropy_and_instance_weights(self):
        for n_classes in (2, 3):
            with self.subTest(n_classes=n_classes):
                uniform = np.full(n_classes, 1 / n_classes)
                certain = np.eye(n_classes)[0]
                probabilities = np.array([[certain, uniform, uniform], [uniform, certain, uniform]])
                confidence = _entropy_confidence(probabilities)
                np.testing.assert_allclose(confidence, [[1, 0, 0], [0, 1, 0]], atol=1e-15)

                coefficients = np.array([[0, 0, 0], [2, 2, 2]])
                weights = _adaptive_weights(confidence, coefficients)
                np.testing.assert_allclose(weights.sum(axis=1), 1)
                np.testing.assert_array_equal(weights.argmax(axis=1), [0, 1])
                expected = (np.exp(2) * certain + 2 * uniform) / (np.exp(2) + 2)
                fused = _fuse_probabilities(probabilities, confidence, coefficients)
                np.testing.assert_allclose(fused, [expected, expected])

                importance_only = np.array([[1, 0, -1], [0, 0, 0]])
                fixed_weights = _adaptive_weights(confidence, importance_only)
                np.testing.assert_allclose(fixed_weights[0], fixed_weights[1])
                self.assertEqual(fixed_weights[0].argmax(), 0)

    def test_pso_prioritizes_macro_f1_and_respects_bounds(self):
        probabilities = np.array([[[0.9, 0.1], [0.1, 0.9], [0.1, 0.9]],
                                  [[0.1, 0.9], [0.9, 0.1], [0.9, 0.1]]])
        y = np.array([0, 1])
        coefficients, score = _optimize_weights(
            probabilities, y, n_particles=12, n_iterations=6, bound=2,
            rng=np.random.RandomState(42),
        )

        baseline = f1_score(y, probabilities.mean(axis=1).argmax(axis=1), average="macro")
        fused = _fuse_probabilities(probabilities, _entropy_confidence(probabilities), coefficients)
        self.assertGreater(score, baseline)
        self.assertEqual(score, f1_score(y, fused.argmax(axis=1), average="macro"))
        self.assertEqual(score, 1.0)
        self.assertTrue((np.abs(coefficients[0]) <= 2).all())
        self.assertTrue(((0 <= coefficients[1]) & (coefficients[1] <= 2)).all())

    def test_pso_prefers_zero_coefficients_when_all_f1_scores_tie(self):
        probabilities = np.repeat([[[0.8, 0.2]], [[0.2, 0.8]]], 3, axis=1)
        coefficients, score = _optimize_weights(
            probabilities, np.array([0, 1]), n_particles=6, n_iterations=3, bound=4,
            rng=np.random.RandomState(42),
        )
        np.testing.assert_array_equal(coefficients, np.zeros((2, 3)))
        self.assertEqual(score, 1.0)


class PSOAdaptiveFusionClassifierTests(unittest.TestCase):
    def test_binary_predictions_and_sklearn_api(self):
        X, y = make_classification(n_samples=36, n_features=5, random_state=11)
        y = np.where(y == 0, "negativo", "positivo")
        model = PSOAdaptiveFusionClassifier(n_particles=5, n_iterations=3)
        self.assertTrue(is_classifier(model))
        self.assertEqual(clone(model).get_params(), model.get_params())
        with self.assertRaises(NotFittedError):
            model.predict(X)

        self.assertIs(model.fit(X, y), model)
        proba = model.predict_proba(X)
        np.testing.assert_array_equal(model.classes_, ["negativo", "positivo"])
        np.testing.assert_array_equal(model.predict(X), model.classes_[proba.argmax(axis=1)])
        self.assertEqual(proba.shape, (len(X), 2))
        np.testing.assert_allclose(proba.sum(axis=1), 1)
        scores = cross_val_score(model, X, y, cv=2, scoring="f1_macro")
        self.assertEqual(scores.shape, (2,))
        self.assertTrue(((0 <= scores) & (scores <= 1)).all())

    def test_multiclass_preprocessing_is_isolated_and_fit_is_reproducible(self):
        X, y = make_classification(
            n_samples=60, n_features=5, n_informative=4, n_redundant=0,
            n_classes=3, random_state=11,
        )
        X = pd.DataFrame(X, columns=[f"number_{i}" for i in range(X.shape[1])])
        X["category"] = np.where(X["number_0"] < 0, "left", "right")
        X.loc[::7, "number_0"] = np.nan
        X.loc[::9, "category"] = np.nan
        y = np.array([10, 20, 30])[y]
        train, test = next(StratifiedKFold(n_splits=5).split(X, y))
        X_train, X_test = X.iloc[train], X.iloc[test].copy()
        X_test["category"] = "unseen"

        model = PSOAdaptiveFusionClassifier(
            preprocessor=benchmark.build_preprocessor(X_train),
            n_particles=5, n_iterations=3, random_state=7,
        )
        original_fit_transform = ColumnTransformer.fit_transform
        observed_fits = Counter()

        def record_fit(preprocessor, data, *args, **kwargs):
            observed_fits[tuple(data.index)] += 1
            return original_fit_transform(preprocessor, data, *args, **kwargs)

        with patch.object(ColumnTransformer, "fit_transform", record_fit):
            model.fit(X_train, y[train])
            proba = model.predict_proba(X_test)

        expected_fits = Counter({tuple(X_train.index): 3})
        for inner_train, _ in StratifiedKFold(n_splits=3).split(X_train, y[train]):
            expected_fits[tuple(X_train.index[inner_train])] = 3
        self.assertEqual(observed_fits, expected_fits)
        self.assertFalse(hasattr(model.preprocessor, "transformers_"))
        np.testing.assert_array_equal(model.classes_, [10, 20, 30])
        self.assertEqual(proba.shape, (len(test), 3))
        self.assertTrue((proba >= 0).all())
        np.testing.assert_allclose(proba.sum(axis=1), 1)
        np.testing.assert_array_equal(model.predict(X_test), model.classes_[proba.argmax(axis=1)])

        repeated = clone(model).fit(X_train, y[train])
        np.testing.assert_array_equal(model.coefficients_, repeated.coefficients_)
        np.testing.assert_array_equal(proba, repeated.predict_proba(X_test))
        self.assertEqual(model.oof_f1_, repeated.oof_f1_)

    def test_fit_rejects_insufficient_class_counts_and_invalid_bounds(self):
        X = np.arange(20).reshape(10, 2)
        with self.assertRaisesRegex(ValueError, "pelo menos cv amostras"):
            PSOAdaptiveFusionClassifier().fit(X, [0] * 8 + [1] * 2)
        with self.assertRaisesRegex(ValueError, "parameter_bound"):
            PSOAdaptiveFusionClassifier(parameter_bound=np.inf).fit(X, [0] * 5 + [1] * 5)
