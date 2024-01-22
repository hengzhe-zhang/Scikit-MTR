import openml
import pandas as pd
from pandas import CategoricalDtype
from sklearn.datasets import fetch_openml
from sklearn.ensemble import RandomForestRegressor, ExtraTreesRegressor
from sklearn.linear_model import LinearRegression
from sklearn.neural_network import MLPRegressor
from sklearn.preprocessing import LabelEncoder
from sklearn.tree import DecisionTreeRegressor
from sklearn.utils import Bunch

from openml_utils import remove_duplicate_features
from scikit_mtr.mkop.sk_mkop import MOKPRegressor

tag = "2019_multioutput_paper"


def load_datasets():
    tagged_datasets = openml.datasets.list_datasets(tag=tag, output_format="dict")
    datasets = []
    for dataset_id, dataset in tagged_datasets.items():
        # print('dataset_id:', dataset_id)
        datasets.append(dataset_id)
    return datasets


def load_data_by_openml(dataset_id):
    open_dataset = openml.datasets.get_dataset(
        dataset_id,
        download_data=True,
        download_qualities=True,
        download_features_meta_data=True,
    )
    x, y, categorical_indicator, attribute_names = open_dataset.get_data(
        dataset_format="dataframe"
    )
    return x, y


def load_data_by_sklearn(data_id):
    data: Bunch = fetch_openml(data_id=data_id, parser="auto")
    X, y = data.data, data.target
    x_encoded = input_encoding(X)
    x_encoded = remove_duplicate_features(x_encoded)
    y = target_encoding(y)
    return x_encoded, y


def input_encoding(df):
    categorical_columns = df.select_dtypes(include=["category"]).columns
    df_encoded = pd.get_dummies(df, columns=categorical_columns, drop_first=True)
    return df_encoded


def target_encoding(df):
    # Create a LabelEncoder object
    label_encoder = LabelEncoder()

    # Iterate through each column in the DataFrame
    for column in df.columns:
        # Check if the data type of the column is a string
        if isinstance(df[column].dtype, CategoricalDtype):
            # Use LabelEncoder to encode the values in this column
            with pd.option_context("mode.copy_on_write", True):
                df[column] = label_encoder.fit_transform(df[column])

    return df


def multi_output_regressor(regressor):
    if regressor == "DT":
        return DecisionTreeRegressor()
    elif regressor == "LR":
        return LinearRegression()
    elif regressor == "RF":
        return RandomForestRegressor()
    elif regressor == "ET":
        return ExtraTreesRegressor()
    elif regressor == "MLP":
        return MLPRegressor()
    elif regressor == "MOKP":
        return MOKPRegressor(ngen=50)
    else:
        raise ValueError("Invalid regressor type")


# load_datasets()
# load_data_by_openml()
if __name__ == "__main__":
    data_id = 41467
    print(load_data_by_sklearn(data_id))
    # print(load_data_by_openml(data_id))
