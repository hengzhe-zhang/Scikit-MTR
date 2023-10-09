import matplotlib.pyplot as plt
import numpy as np
from sklearn.base import BaseEstimator, clone
from sklearn.datasets import make_regression
from sklearn.linear_model import LinearRegression
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score
from sklearn.model_selection import train_test_split


class MultiTargetRegressorStacking(BaseEstimator):
    def __init__(self, base_regressors, meta_regressors):
        self.base_regressors = base_regressors
        self.meta_regressors = meta_regressors

    def fit(self, X, Y):
        X = np.array(X)
        Y = np.array(Y)
        # Step 1: Train base regressors and collect predictions
        self.first_stage_models_ = [reg.fit(X, Y[:, i]) for i, reg in enumerate(self.base_regressors)]
        predictions = np.hstack([reg.predict(X).reshape(-1, 1) for reg in self.first_stage_models_])

        # Step 2: Train meta-models using predictions from base regressors as additional features
        X_augmented = np.hstack([X, predictions])
        self.second_stage_models_ = [reg.fit(X_augmented, Y[:, i]) for i, reg in enumerate(self.meta_regressors)]
        return self

    def predict(self, X):
        X = np.array(X)
        # Predict using base regressors
        predictions = np.hstack([reg.predict(X).reshape(-1, 1) for reg in self.first_stage_models_])
        X_augmented = np.hstack([X, predictions])

        # Predict using meta-models
        final_predictions = np.hstack([reg.predict(X_augmented).reshape(-1, 1) for reg in self.second_stage_models_])
        return final_predictions


if __name__ == '__main__':
    # Create synthetic data with 5 targets
    X, Y = make_regression(n_samples=1000, n_features=20, n_targets=5, noise=0.5)
    X_train, X_test, Y_train, Y_test = train_test_split(X, Y, test_size=0.2, random_state=0)

    # Initialize the model with base regressors and meta regressors
    base_regressors = [LinearRegression() for _ in range(Y.shape[1])]
    meta_regressors = [LinearRegression() for _ in range(Y.shape[1])]

    mtrs = MultiTargetRegressorStacking(base_regressors, meta_regressors)
    mtrs.fit(X_train, Y_train)
    predictions = mtrs.predict(X_test)

    # 1. Visualizing the predictions for each target
    n_targets = Y_test.shape[1]
    fig, axes = plt.subplots(n_targets, 1, figsize=(10, 5 * n_targets))
    for i in range(n_targets):
        axes[i].scatter(Y_test[:, i], predictions[:, i], alpha=0.5)
        axes[i].plot([Y_test.min(), Y_test.max()], [Y_test.min(), Y_test.max()], '--k', linewidth=2)
        axes[i].set_xlabel('True Target {}'.format(i + 1))
        axes[i].set_ylabel('Predicted Target {}'.format(i + 1))
        axes[i].set_title('Target {} Predictions vs True Values'.format(i + 1))
    plt.tight_layout()
    plt.show()

    # 2. Evaluating the model performance using metrics
    for i in range(n_targets):
        mae = mean_absolute_error(Y_test[:, i], predictions[:, i])
        mse = mean_squared_error(Y_test[:, i], predictions[:, i])
        r2 = r2_score(Y_test[:, i], predictions[:, i])
        print(f"Target {i + 1} - MAE: {mae:.2f}, MSE: {mse:.2f}, R^2: {r2:.2f}")
