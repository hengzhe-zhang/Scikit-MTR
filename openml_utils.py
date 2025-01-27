import sys
from typing import List, Optional

import numpy as np
import pandas as pd
import sklearn
from category_encoders import BinaryEncoder
from category_encoders import TargetEncoder as SimpleTargetEncoder
from packaging import version
from sklearn.compose import ColumnTransformer
from sklearn.impute import SimpleImputer
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import (
    OneHotEncoder,
    LabelEncoder,
    OrdinalEncoder,
)

if sys.version_info >= (3, 9):
    from sklearn.preprocessing import TargetEncoder as TargetEncoderCV

from date_util import (
    replace_abbreviated_months_with_numbers,
    replace_abbreviated_days_with_numbers,
)


def remove_duplicate_features(x_encoded):
    # Drop duplicate columns
    x_encoded = x_encoded.loc[:, ~x_encoded.columns.duplicated(keep="first")]
    return x_encoded


def remove_columns_with_same_values(df, categorical_indicator, attribute_names):
    """
    Remove columns from a DataFrame where all values are the same.

    Parameters:
    df (pd.DataFrame): The DataFrame from which to remove columns.

    Returns:
    pd.DataFrame: A new DataFrame with columns removed.
    """
    column_selection = df.nunique() > 1
    return (
        df.loc[:, column_selection],
        [e for e, c in zip(categorical_indicator, list(column_selection)) if c],
        [e for e, c in zip(attribute_names, list(column_selection)) if c],
    )


def infer_categorical_features(X):
    categorical_features = []
    for feature in X.T:
        if len(np.unique(feature)) <= 5:
            categorical_features.append(True)
        else:
            categorical_features.append(False)
    return categorical_features


def mean_imputation_and_one_hot_encoding(
    x_train,
    x_test,
    y_train,
    y_test,
    categorical_indicator: Optional[List[bool]] = None,
    categorical_encoder="Onehot",
):
    # Check if x_train and x_test are DataFrames or NumPy arrays
    if isinstance(x_train, pd.DataFrame):
        for target_column in ["month", "day"]:
            if target_column in x_train.columns:
                replace_abbreviated_months_with_numbers(x_train, target_column)
                replace_abbreviated_days_with_numbers(x_train, target_column)
                replace_abbreviated_months_with_numbers(x_test, target_column)
                replace_abbreviated_days_with_numbers(x_test, target_column)

        if categorical_indicator is None:
            categorical_cols = x_train.select_dtypes(
                include=["category"]
            ).columns.tolist()
            numerical_cols = x_train.select_dtypes(
                exclude=["category"]
            ).columns.tolist()
        else:
            categorical_cols = x_train.columns[categorical_indicator].tolist()
            numerical_cols = x_train.columns[~pd.Series(categorical_indicator)].tolist()
    else:
        if categorical_indicator is None:
            raise ValueError("categorical_indicator must be provided for NumPy arrays.")
        categorical_cols = list(np.where(categorical_indicator)[0])
        numerical_cols = list(np.where(~np.array(categorical_indicator))[0])
    y_train, y_test = y_label_encoding(y_train, y_test)

    numerical_transformer = SimpleImputer(strategy="mean")

    # Creating a transformer for categorical features
    if categorical_encoder == "Ordinal":
        categorical_encoder_instance = OrdinalEncoder(
            handle_unknown="use_encoded_value", unknown_value=-1
        )
    elif categorical_encoder in ["TargetCV", "InferredTargetCV"]:
        categorical_encoder_instance = TargetEncoderCV(target_type="continuous")
    elif categorical_encoder == "Target":
        categorical_encoder_instance = SimpleTargetEncoder()
    elif categorical_encoder == "Binary":
        categorical_encoder_instance = BinaryEncoder()
    elif version.parse(sklearn.__version__) < version.parse("1.2"):
        categorical_encoder_instance = OneHotEncoder(
            handle_unknown="ignore", drop="if_binary", sparse=False
        )
    else:
        categorical_encoder_instance = OneHotEncoder(
            handle_unknown="ignore", drop="if_binary", sparse_output=False
        )
    categorical_transformer = Pipeline(
        steps=[
            ("imputer", SimpleImputer(strategy="most_frequent")),
            ("encoder", categorical_encoder_instance),
        ]
    )

    # Bundling transformers
    preprocessor = ColumnTransformer(
        transformers=[
            ("num", numerical_transformer, numerical_cols),
            ("cat", categorical_transformer, categorical_cols),
        ]
    )

    # Applying the transformations
    x_train_transformed = preprocessor.fit_transform(x_train, y=y_train)
    x_test_transformed = preprocessor.transform(x_test)

    if categorical_indicator is not None:
        num_numerical = len(numerical_cols)
        num_categorical = x_train_transformed.shape[1] - num_numerical
        categorical_indicator[:] = [False] * num_numerical + [True] * num_categorical

    return x_train_transformed, x_test_transformed, y_train, y_test


def y_label_encoding(y_train, y_test):
    y_train = np.array(y_train)
    y_test = np.array(y_test)
    # Label Encoding for y_train and y_test if they are not numeric (handling multi-target)
    if len(y_train.shape) == 1 or y_train.shape[1] == 1:  # Single target
        if not np.issubdtype(y_train.dtype, np.number):
            label_encoder = LabelEncoder()
            y_train = label_encoder.fit_transform(y_train)
            y_test = label_encoder.transform(y_test)
    else:  # Multi-target
        encoders = {}
        for col in range(y_train.shape[1]):
            if not np.issubdtype(y_train[:, col].dtype, np.number):
                label_encoder = LabelEncoder()
                y_train[:, col] = label_encoder.fit_transform(y_train[:, col])
                y_test[:, col] = label_encoder.transform(y_test[:, col])
                encoders[col] = label_encoder  # Store encoder for each column
    return y_train, y_test


def one_hot_encoding(x_merged: pd.DataFrame, categorical_indicator):
    # Identify categorical columns based on categorical_indicator
    categorical_columns = [
        x_merged.columns[i]
        for i, is_categorical in enumerate(categorical_indicator)
        if is_categorical
    ]
    # Perform one-hot encoding only on categorical columns
    x_encoded = pd.get_dummies(x_merged, columns=categorical_columns, drop_first=True)
    return x_encoded


def binary_encoding(x_merged: pd.DataFrame, categorical_indicator):
    binary_encoded_df = x_merged.copy()
    # Identify categorical columns based on categorical_indicator
    categorical_columns = [
        x_merged.columns[i]
        for i, is_categorical in enumerate(categorical_indicator)
        if is_categorical
    ]

    for col in categorical_columns:
        # Convert categorical values to integer codes
        codes = x_merged[col].astype("category").cat.codes
        # Convert integer codes to binary format
        max_val = codes.max()
        max_len = np.ceil(np.log2(max_val + 1)).astype(int)

        for i in range(max_len):
            shifted_values = np.right_shift(codes.values, i)
            binary_column = np.bitwise_and(shifted_values, 1)
            binary_encoded_df[f"{col}_bin_{i}"] = binary_column
        # Drop the original column
        binary_encoded_df.drop(col, axis=1, inplace=True)

    return binary_encoded_df


# Test function to verify the binary_encoding function
def test_binary_encoding():
    # Creating a sample DataFrame
    data = {
        "A": ["apple", "orange", "apple", "banana"],
        "B": [1, 2, 3, 4],
        "C": ["red", "orange", "green", "blue"],
    }
    df = pd.DataFrame(data)

    # Indicator for identifying categorical columns
    categorical_indicator = [True, False, True]

    # Applying binary encoding
    encoded_df = binary_encoding(df, categorical_indicator)

    # Checking the encoded DataFrame
    print("Binary Encoded DataFrame:")
    print(encoded_df)


if __name__ == "__main__":
    # Running the test
    test_binary_encoding()
