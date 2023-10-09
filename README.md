# 🛠 scikit-mtr: Multi-Task Regression in Python

`scikit-mtr` provides a framework for multi-task regression using popular regression algorithms and introduces a
stacking method to combine different regressors for enhanced performance.

## 🌟 Features

- 📊 Support for multiple regressors: Decision Trees, Linear Regression, Random Forests, Extra Trees, MLP, and MOKP.
- 🏆 Stacking method to combine predictions from various regressors.

## 📦 Installation

```bash
pip install scikit-mtr
```

## ⚡ Quick Start

Get up and running with `scikit-mtr` in a flash:

```python
from sklearn.model_selection import train_test_split

from scikit_mtr.multi_output_tools import multi_output_regressor, load_data_by_sklearn

params = {
    'regressor': 'LR',
    'dataset': 41467,
    'random_seed': 0
}

# Load and split data
X, y = load_data_by_sklearn(params['dataset'])
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=params['random_seed'])

# Initialize regressor
model = multi_output_regressor(params['regressor'])
model.fit(X_train, y_train)

# 🎯 Make predictions
y_pred_test = model.predict(X_test)
```

## 📚 Using Stacking

`scikit-mtr` introduces a powerful stacking method where base regressors' predictions serve as features for
meta-regressors:

```python
from scikit_mtr.multi_output_stacking import MultiTargetRegressorStacking

base_regressors = [LinearRegression() for _ in range(y_train.shape[1])]
meta_regressors = [LinearRegression() for _ in range(y_train.shape[1])]

stacker = MultiTargetRegressorStacking(base_regressors, meta_regressors)
stacker.fit(X_train, y_train)
stacked_predictions = stacker.predict(X_test)
print(stacked_predictions)
```

## 🧪 Supported Regressors

Currently, the following regressors are in our arsenal:

- `DT`: 🌲 Decision Tree
- `LR`: 📈 Linear Regression
- `RF`: 🌳 Random Forest
- `ET`: 🍃 Extra Trees
- `MLP`: 🧠 Multi-layer Perceptron
- `MOKP`: 🔮 MOKP Regressor

💡 You can easily extend the `multi_output_regressor` function to add support for other regressors.