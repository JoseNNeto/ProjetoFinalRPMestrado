"""Fusão adaptativa de SVM, Random Forest e k-NN otimizada por PSO."""

from numbers import Integral, Real

import numpy as np
from sklearn.base import BaseEstimator, ClassifierMixin, clone
from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import StratifiedKFold, cross_val_predict
from sklearn.neighbors import KNeighborsClassifier
from sklearn.pipeline import Pipeline
from sklearn.svm import SVC
from sklearn.utils import check_random_state
from sklearn.utils.multiclass import check_classification_targets
from sklearn.utils.validation import (
    check_consistent_length,
    check_is_fitted,
    check_scalar,
    column_or_1d,
    validate_data,
)


def _entropy_confidence(probabilities):
    """Calcula 1 - entropia normalizada para cada amostra e especialista."""
    log_probabilities = np.zeros_like(probabilities)
    np.log(probabilities, out=log_probabilities, where=probabilities > 0)
    confidence = 1 + (probabilities * log_probabilities).sum(axis=-1) / np.log(
        probabilities.shape[-1]
    )
    return np.clip(confidence, 0, 1)


def _adaptive_weights(confidences, coefficients):
    importance, confidence_scale = coefficients
    logits = importance + confidence_scale * confidences
    logits -= logits.max(axis=1, keepdims=True)
    weights = np.exp(logits)
    return weights / weights.sum(axis=1, keepdims=True)


def _fuse_probabilities(probabilities, confidences, coefficients):
    weights = _adaptive_weights(confidences, coefficients)
    return np.einsum("ni,nic->nc", weights, probabilities)


def _optimize_weights(probabilities, y, *, n_particles, n_iterations, bound, rng):
    """Otimiza (F1 Macro, -norma quadrática média) em ordem lexicográfica."""
    n_experts = probabilities.shape[1]
    n_classes = probabilities.shape[2]
    confidences = _entropy_confidence(probabilities)
    true_counts = np.bincount(y, minlength=n_classes)

    def score(position):
        coefficients = position.reshape(2, n_experts)
        predicted = _fuse_probabilities(probabilities, confidences, coefficients).argmax(axis=1)
        predicted_counts = np.bincount(predicted, minlength=n_classes)
        true_positives = np.bincount(y[predicted == y], minlength=n_classes)
        macro_f1 = np.mean(2 * true_positives / (true_counts + predicted_counts))
        return macro_f1, -np.mean(position**2)

    lower = np.repeat([-bound, 0.0], n_experts)
    upper = np.full(2 * n_experts, bound)
    shape = (n_particles, 2 * n_experts)
    positions = rng.uniform(lower, upper, size=shape)
    positions[0] = 0  # Inclui a média uniforme entre as soluções candidatas.
    initial_speed = 0.1 * (upper - lower)
    velocities = rng.uniform(-initial_speed, initial_speed, size=shape)
    personal_best = positions.copy()
    personal_scores = [score(position) for position in positions]

    for _ in range(n_iterations):
        best = personal_best[max(range(n_particles), key=personal_scores.__getitem__)]
        velocities = (
            0.7 * velocities
            + 1.5 * rng.random_sample(shape) * (personal_best - positions)
            + 1.5 * rng.random_sample(shape) * (best - positions)
        )
        positions = np.clip(positions + velocities, lower, upper)
        for index, position in enumerate(positions):
            candidate_score = score(position)
            if candidate_score > personal_scores[index]:
                personal_best[index] = position
                personal_scores[index] = candidate_score

    winner = max(range(n_particles), key=personal_scores.__getitem__)
    return personal_best[winner].reshape(2, n_experts), personal_scores[winner][0]


class PSOAdaptiveFusionClassifier(ClassifierMixin, BaseEstimator):
    """Combina as probabilidades dos três especialistas com pesos por amostra.

    Passe um transformador não ajustado em ``preprocessor`` para dados que
    precisam de pré-processamento. Cada especialista recebe um clone próprio.
    ``cv`` define o número de folds internos estratificados, sem embaralhamento.
    O PSO inicializa ``n_particles`` e executa ``n_iterations`` atualizações.
    ``parameter_bound`` limita a importância a [-limite, limite] e a influência
    da confiança a [0, limite]. ``random_state`` controla o PSO, a SVM e a floresta.

    Após o ajuste, ``coefficients_`` tem duas linhas (importância e influência da
    confiança) na ordem SVM, Random Forest e k-NN. ``oof_f1_`` é o F1 Macro
    usado na otimização e deve ser interpretado como uma métrica de treinamento.
    """

    def __init__(
        self,
        *,
        preprocessor=None,
        cv=3,
        n_particles=24,
        n_iterations=30,
        parameter_bound=4.0,
        random_state=42,
    ):
        self.preprocessor = preprocessor
        self.cv = cv
        self.n_particles = n_particles
        self.n_iterations = n_iterations
        self.parameter_bound = parameter_bound
        self.random_state = random_state

    def fit(self, X, y):
        """Aprende a fusão com probabilidades out-of-fold do conjunto recebido."""
        check_scalar(self.cv, "cv", Integral, min_val=2)
        check_scalar(self.n_particles, "n_particles", Integral, min_val=2)
        check_scalar(self.n_iterations, "n_iterations", Integral, min_val=1)
        check_scalar(self.parameter_bound, "parameter_bound", Real)
        if not np.isfinite(self.parameter_bound) or self.parameter_bound <= 0:
            raise ValueError("parameter_bound deve ser finito e positivo.")
        rng = check_random_state(self.random_state)

        y = column_or_1d(y, warn=True)
        check_classification_targets(y)
        check_consistent_length(X, y)
        classes, encoded_y, counts = np.unique(y, return_inverse=True, return_counts=True)
        if classes.size < 2:
            raise ValueError("A classificação requer pelo menos duas classes.")
        if counts.min() < self.cv:
            raise ValueError("Cada classe deve ter pelo menos cv amostras para os folds internos.")

        classifiers = (
            SVC(kernel="rbf", probability=True, random_state=self.random_state),
            RandomForestClassifier(n_estimators=100, random_state=self.random_state),
            KNeighborsClassifier(n_neighbors=5),
        )
        estimators = [
            Pipeline([("preprocessor", clone(self.preprocessor)), ("classifier", classifier)])
            if self.preprocessor is not None else classifier
            for classifier in classifiers
        ]
        folds = list(StratifiedKFold(n_splits=self.cv).split(X, encoded_y))
        if min(len(train) for train, _ in folds) < classifiers[-1].n_neighbors:
            raise ValueError("Cada fold interno precisa de pelo menos 5 amostras de treino para o k-NN.")

        # O treino de cada clone inclui seu pré-processador, dentro do fold.
        oof_probabilities = np.stack(
            [
                cross_val_predict(estimator, X, encoded_y, cv=folds, method="predict_proba")
                for estimator in estimators
            ],
            axis=1,
        )
        coefficients, oof_f1 = _optimize_weights(
            oof_probabilities,
            encoded_y,
            n_particles=self.n_particles,
            n_iterations=self.n_iterations,
            bound=self.parameter_bound,
            rng=rng,
        )
        fitted_estimators = [estimator.fit(X, encoded_y) for estimator in estimators]

        # Preserva o DataFrame para que o ColumnTransformer possa selecionar por nome.
        validate_data(self, X, skip_check_array=True)
        self.classes_ = classes
        self.estimators_ = fitted_estimators
        self.coefficients_ = coefficients
        self.oof_f1_ = oof_f1
        return self

    def predict_proba(self, X):
        """Retorna probabilidades na ordem de ``classes_``."""
        check_is_fitted(self, "estimators_")
        validate_data(self, X, reset=False, skip_check_array=True)
        probabilities = np.stack([estimator.predict_proba(X) for estimator in self.estimators_], axis=1)
        return _fuse_probabilities(probabilities, _entropy_confidence(probabilities), self.coefficients_)

    def predict(self, X):
        """Retorna a classe com maior probabilidade após a fusão."""
        probabilities = self.predict_proba(X)
        return self.classes_[probabilities.argmax(axis=1)]
