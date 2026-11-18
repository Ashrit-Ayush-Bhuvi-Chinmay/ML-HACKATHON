"""Step 5: the model zoo.

Each model scores a (GPS point, candidate arc) pair with P(candidate is the reference arc).
The spec records family, configuration and training budget, so the comparison table is
generated from this file, not typed by hand.

`train_point_frac` < 1 trains on a random subset of training *points* (all candidates of a
chosen point are kept, so the per-point competition stays intact). It is used only where
fitting or prediction cost is otherwise impractical (k-NN, MLP) and is reported in the results.
"""
from __future__ import annotations

from dataclasses import dataclass
from typing import Callable

import lightgbm as lgb
import numpy as np
import xgboost as xgb
from sklearn.ensemble import RandomForestClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.naive_bayes import GaussianNB
from sklearn.neighbors import KNeighborsClassifier
from sklearn.neural_network import MLPClassifier
from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import StandardScaler

from .features import FEATURES_V1


@dataclass
class ModelSpec:
    name: str
    family: str
    params: dict
    build: Callable[[dict, int], object] | None  # None = special-cased baseline
    train_point_frac: float = 1.0
    epoch_history: bool = False
    features: list[str] | None = None  # None = the current full feature set


class DistanceGaussian:
    """Classic map-matching emission (Newson & Krumm 2009): weight exp(-d^2 / 2 sigma^2),
    normalised over the point's candidates; sigma = 1.4826 * median reference distance."""

    def fit(self, dist_pos: np.ndarray):
        self.sigma_ = float(1.4826 * np.median(dist_pos))
        return self

    def predict_proba_grouped(self, dist: np.ndarray, keys: np.ndarray) -> np.ndarray:
        w = np.exp(-0.5 * (dist / self.sigma_) ** 2)
        starts = np.flatnonzero(np.r_[True, keys[1:] != keys[:-1]])
        tot = np.repeat(np.add.reduceat(w, starts), np.diff(np.r_[starts, len(keys)]))
        return w / np.where(tot > 0, tot, 1.0)


def _logreg(p, seed):
    return make_pipeline(StandardScaler(), LogisticRegression(**p, random_state=seed))


def _gnb(p, seed):
    return GaussianNB(**p)


def _knn(p, seed):
    return make_pipeline(StandardScaler(), KNeighborsClassifier(**p))


def _rf(p, seed):
    return RandomForestClassifier(**p, n_jobs=-1, random_state=seed)


def _lgb(p, seed):
    return lgb.LGBMClassifier(**p, n_jobs=-1, random_state=seed, verbose=-1)


def _xgb(p, seed):
    # GPU (GTX 1660 Ti) via CUDA; identical model to the CPU "hist" method, just faster
    return xgb.XGBClassifier(**p, n_jobs=-1, random_state=seed, tree_method="hist", device="cuda",
                             eval_metric="logloss")


def _mlp(p, seed):
    return make_pipeline(StandardScaler(), MLPClassifier(**p, random_state=seed))


def _torch_pointwise(p, seed):
    from .torch_models import TorchScorer
    return TorchScorer(mode="pointwise", seed=seed, **p)


def _torch_listwise(p, seed):
    from .torch_models import TorchScorer
    return TorchScorer(mode="listwise", seed=seed, **p)


def _catboost(p, seed):
    from catboost import CatBoostClassifier
    return CatBoostClassifier(**p, random_seed=seed, verbose=0, task_type="GPU", devices="0")


_NN = {"hidden": (128, 64), "dropout": 0.1, "lr": 1e-3, "weight_decay": 1e-4,
       "batch_points": 4096, "max_epochs": 40, "patience": 5}


SPECS: dict[str, ModelSpec] = {
    "distance_gaussian": ModelSpec("distance_gaussian", "Classic baseline (distance only)", {"sigma": "1.4826*median"}, None),
    "gaussian_nb": ModelSpec("gaussian_nb", "Probabilistic (generative)", {"var_smoothing": 1e-9}, _gnb),
    "logreg": ModelSpec("logreg", "Linear", {"C": 1.0, "max_iter": 2000}, _logreg),
    "lightgbm": ModelSpec("lightgbm", "Boosting", {
        "n_estimators": 600, "learning_rate": 0.05, "num_leaves": 63, "min_child_samples": 50,
        "subsample": 0.8, "subsample_freq": 1, "colsample_bytree": 0.8}, _lgb),
    "random_forest": ModelSpec("random_forest", "Bagging ensemble", {
        "n_estimators": 200, "min_samples_leaf": 20, "max_features": "sqrt", "max_samples": 0.3}, _rf),
    "xgboost": ModelSpec("xgboost", "Boosting", {
        "n_estimators": 600, "learning_rate": 0.05, "max_depth": 8, "min_child_weight": 5,
        "subsample": 0.8, "colsample_bytree": 0.8}, _xgb),
    "knn": ModelSpec("knn", "Instance-based", {
        "n_neighbors": 50, "weights": "distance", "algorithm": "brute"}, _knn, train_point_frac=0.08),
    "mlp": ModelSpec("mlp", "Neural network", {
        "hidden_layer_sizes": (64, 32), "alpha": 1e-4, "batch_size": 2048, "learning_rate_init": 1e-3,
        "max_iter": 40, "early_stopping": True, "n_iter_no_change": 5}, _mlp,
        train_point_frac=0.5, epoch_history=True),
    "mlp_torch": ModelSpec("mlp_torch", "Neural network (pointwise MLP, GPU)", dict(_NN), _torch_pointwise,
                           epoch_history=True),
    "ranker_torch": ModelSpec("ranker_torch", "Neural network (listwise candidate ranker, GPU)", dict(_NN),
                              _torch_listwise, epoch_history=True),
    "catboost": ModelSpec("catboost", "Boosting (ordered boosting, GPU)", {
        "iterations": 1500, "learning_rate": 0.08, "depth": 8, "l2_leaf_reg": 3.0}, _catboost),
}
# Ablation: the same LightGBM on feature set v1 (no long-range heading).
SPECS["lightgbm_v1feats"] = ModelSpec(
    "lightgbm_v1feats", "Boosting (feature set v1: no long-range heading)",
    dict(SPECS["lightgbm"].params), _lgb, features=FEATURES_V1)
