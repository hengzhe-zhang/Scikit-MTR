import numpy as np
from sklearn.metrics import r2_score, mean_squared_error
from sklearn.model_selection import train_test_split

from scikit_mtr.multi_output_tools import multi_output_regressor, load_data_by_sklearn


def train_model(params):
    regressor = params['regressor']
    data_id = params['dataset']
    seed = params['random_seed']

    X, y = load_data_by_sklearn(data_id)
    print(X.shape, y.shape)
    X, y = np.array(X).astype(np.float32), np.array(y).astype(np.float32)
    x_train, x_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=seed)

    model = multi_output_regressor(regressor)
    model.fit(x_train, y_train)

    # Predictions for test and train sets
    y_pred_test = model.predict(x_test)
    y_pred_train = model.predict(x_train)

    # Calculate R2 for test and train sets
    individual_training_r2 = [r2_score(y_train[:, i], y_pred_train[:, i]) for i in range(y_train.shape[1])]
    training_r2 = np.mean(individual_training_r2)
    individual_test_r2 = [r2_score(y_test[:, i], y_pred_test[:, i]) for i in range(y_test.shape[1])]
    test_r2 = np.mean(individual_test_r2)

    # Calculate MSE for test and train sets
    training_mse = np.mean([mean_squared_error(y_train[:, i], y_pred_train[:, i]) for i in range(y_train.shape[1])])
    test_mse = np.mean([mean_squared_error(y_test[:, i], y_pred_test[:, i]) for i in range(y_test.shape[1])])

    result = {
        **params,
        'training_r2': training_r2,
        'test_r2': test_r2,
        'training_mse': training_mse,
        'test_mse': test_mse
    }
    return result


if __name__ == '__main__':
    for r in ['DT', 'LR', 'RF', 'MOKP', 'ET', 'MLP']:
        params = {
            'regressor': r,
            'dataset': 41467,
            'random_seed': 0,
        }

        print(train_model(params))
