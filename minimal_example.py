from sklearn.linear_model import LinearRegression
from sklearn.model_selection import train_test_split

from scikit_mtr.multi_output_stacking import MultiTargetRegressorStacking
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
print(y_pred_test)

base_regressors = [LinearRegression() for _ in range(y_train.shape[1])]
meta_regressors = [LinearRegression() for _ in range(y_train.shape[1])]

stacker = MultiTargetRegressorStacking(base_regressors, meta_regressors)
stacker.fit(X_train, y_train)
stacked_predictions = stacker.predict(X_test)
print(stacked_predictions)
