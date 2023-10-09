import numpy as np
import pandas as pd


def remove_duplicate_features(x_encoded):
    # Drop duplicate columns
    x_encoded = x_encoded.loc[:, ~x_encoded.columns.duplicated(keep='first')]
    return x_encoded


def one_hot_encoding(x_merged: pd.DataFrame, categorical_indicator):
    # Identify categorical columns based on categorical_indicator
    categorical_columns = [x_merged.columns[i] for i, is_categorical in enumerate(categorical_indicator)
                           if is_categorical]
    # Perform one-hot encoding only on categorical columns
    x_encoded = pd.get_dummies(x_merged, columns=categorical_columns, drop_first=True)
    return x_encoded


def binary_encoding(x_merged: pd.DataFrame, categorical_indicator):
    binary_encoded_df = x_merged.copy()
    # Identify categorical columns based on categorical_indicator
    categorical_columns = [x_merged.columns[i] for i, is_categorical in enumerate(categorical_indicator) if
                           is_categorical]

    for col in categorical_columns:
        # Convert categorical values to integer codes
        codes = x_merged[col].astype('category').cat.codes
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
        'A': ['apple', 'orange', 'apple', 'banana'],
        'B': [1, 2, 3, 4],
        'C': ['red', 'orange', 'green', 'blue']
    }
    df = pd.DataFrame(data)

    # Indicator for identifying categorical columns
    categorical_indicator = [True, False, True]

    # Applying binary encoding
    encoded_df = binary_encoding(df, categorical_indicator)

    # Checking the encoded DataFrame
    print("Binary Encoded DataFrame:")
    print(encoded_df)


if __name__ == '__main__':
    # Running the test
    test_binary_encoding()
