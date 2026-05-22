from eval import Evaluator
import pandas as pd
import os

from tqdm import tqdm

import numpy as np
import torch
import random
import math

from tools import read_file, find_length_rank

from sklearn.decomposition import PCA as sklearn_PCA
from sklearn.utils.validation import check_array, check_is_fitted
from sklearn.preprocessing import StandardScaler
from scipy.stats import zscore
from scipy.spatial.distance import cdist


# =========================
# Sliding window utility
# =========================
class Window:
    def __init__(self, window):
        self.window = window

    def convert(self, X):

        X = np.asarray(X)

        if len(X.shape) == 1:
            X = X.reshape(-1, 1)

        n_samples, n_features = X.shape

        if n_samples < self.window:
            raise ValueError(
                "Window size larger than number of samples"
            )

        windows = []

        for i in range(n_samples - self.window + 1):
            windows.append(
                X[i:i + self.window].flatten()
            )

        return np.array(windows)


# =========================
# PCA Detector
# =========================
class PCA:

    def __init__(
        self,
        slidingWindow=100,
        sub=True,
        n_components=None,
        n_selected_components=None,
        contamination=0.1,
        copy=True,
        whiten=False,
        svd_solver='auto',
        tol=0.0,
        iterated_power='auto',
        random_state=0,
        weighted=True,
        standardization=True,
        zero_pruning=True,
        normalize=True
    ):

        self.slidingWindow = slidingWindow
        self.sub = sub
        self.n_components = n_components
        self.n_selected_components = n_selected_components
        self.contamination = contamination

        self.copy = copy
        self.whiten = whiten
        self.svd_solver = svd_solver
        self.tol = tol
        self.iterated_power = iterated_power
        self.random_state = random_state

        self.weighted = weighted
        self.standardization = standardization
        self.zero_pruning = zero_pruning
        self.normalize = normalize

    def fit(self, X, y=None):

        n_samples, n_features = X.shape

        # =========================
        # Sliding window conversion
        # =========================
        X = Window(
            window=self.slidingWindow
        ).convert(X)

        # =========================
        # Normalization
        # =========================
        if self.normalize:

            if n_features == 1:
                X = zscore(X, axis=0, ddof=0)
            else:
                X = zscore(X, axis=1, ddof=1)

            X = np.nan_to_num(X)

        X = check_array(X)

        # =========================
        # Standardization
        # =========================
        if self.standardization:

            self.scaler_ = StandardScaler()

            X = self.scaler_.fit_transform(X)

        # =========================
        # Remove zero columns
        # =========================
        if self.zero_pruning:

            non_zero_columns = np.any(X != 0, axis=0)

            X = X[:, non_zero_columns]

        # =========================
        # PCA
        # =========================
        self.detector_ = sklearn_PCA(
            n_components=self.n_components,
            copy=self.copy,
            whiten=self.whiten,
            svd_solver=self.svd_solver,
            tol=self.tol,
            iterated_power=self.iterated_power,
            random_state=self.random_state
        )

        self.detector_.fit(X)

        self.n_components_ = self.detector_.n_components_

        self.components_ = self.detector_.components_

        # =========================
        # Selected components
        # =========================
        if self.n_selected_components is None:

            self.n_selected_components_ = self.n_components_

        else:

            self.n_selected_components_ = min(
                self.n_selected_components,
                self.n_components_
            )

        # =========================
        # Component weights
        # =========================
        self.w_components_ = np.ones(
            [self.n_components_]
        )

        if self.weighted:

            self.w_components_ = (
                self.detector_.explained_variance_ratio_
            )

        # Avoid divide-by-zero
        self.w_components_ = np.clip(
            self.w_components_,
            1e-12,
            None
        )

        # =========================
        # Select smallest PCs
        # =========================
        self.selected_components_ = (
            self.components_[
                -self.n_selected_components_:,
                :
            ]
        )

        self.selected_w_components_ = (
            self.w_components_[
                -self.n_selected_components_:
            ]
        )

        # =========================
        # Decision scores
        # =========================
        self.decision_scores_ = np.sum(
            cdist(
                X,
                self.selected_components_
            ) / self.selected_w_components_,
            axis=1
        ).ravel()

        # =========================
        # Padding
        # =========================
        if self.decision_scores_.shape[0] < n_samples:

            left_pad = math.ceil(
                (self.slidingWindow - 1) / 2
            )

            right_pad = (
                self.slidingWindow - 1
            ) // 2

            self.decision_scores_ = np.array(
                [self.decision_scores_[0]] * left_pad
                + list(self.decision_scores_)
                + [self.decision_scores_[-1]] * right_pad
            )

        return self

    def decision_function(self, X):

        check_is_fitted(
            self.detector_,
            ['components_']
        )

        n_samples, n_features = X.shape

        X = Window(
            window=self.slidingWindow
        ).convert(X)

        if self.normalize:

            if n_features == 1:
                X = zscore(X, axis=0, ddof=0)
            else:
                X = zscore(X, axis=1, ddof=1)

            X = np.nan_to_num(X)

        X = check_array(X)

        if self.standardization:

            X = self.scaler_.transform(X)

        decision_scores_ = np.sum(
            cdist(
                X,
                self.selected_components_
            ) / self.selected_w_components_,
            axis=1
        ).ravel()

        # Padding
        if decision_scores_.shape[0] < n_samples:

            left_pad = math.ceil(
                (self.slidingWindow - 1) / 2
            )

            right_pad = (
                self.slidingWindow - 1
            ) // 2

            decision_scores_ = np.array(
                [decision_scores_[0]] * left_pad
                + list(decision_scores_)
                + [decision_scores_[-1]] * right_pad
            )

        return decision_scores_


# =========================
# Main
# =========================
def main():

    # Reproducibility
    seed = 2

    torch.manual_seed(seed)

    if torch.cuda.is_available():

        torch.cuda.manual_seed(seed)
        torch.cuda.manual_seed_all(seed)

    np.random.seed(seed)

    random.seed(seed)

    path = 'Datasets/TSB-AD-U/'

    file_list_path = (
        'Datasets/File_List/TSB-AD-U-Eva.csv'
    )

    file_list = pd.read_csv(
        file_list_path
    )['file_name'].values

    evaluator = Evaluator(metrics='restr')

    all_results = []

    results = []

    for filename in tqdm(
        file_list,
        desc="Running PCA"
    ):

        _, data, labels = read_file(
            path,
            filename
        )

        rank = find_length_rank(
            data[:, 0].reshape(-1, 1),
            rank=1
        )

        score = PCA(
            slidingWindow=rank,
            n_components=int(0.25*rank),
            n_selected_components=5
        ).fit(data).decision_scores_

        metrics = evaluator.metrics_fnc(
            score,
            labels,
            slidingWindow=rank
        )

        result = {
            'filename': filename
        }

        result.update(metrics)

        results.append(result)

    results_df = pd.DataFrame(results)

    all_results.append(results_df)

    combined_df = pd.concat(
        all_results,
        ignore_index=True
    )

    avg_metrics = (
        combined_df
        .groupby('filename')
        .mean(numeric_only=True)
        .mean()
        .round(3) * 100
    )

    os.makedirs(
        'results/Random',
        exist_ok=True
    )

    combined_df.to_csv(
        'results/Random/001.csv',
        index=False
    )

    print("\nAverage metrics:")
    print(avg_metrics)


if __name__ == '__main__':
    main()