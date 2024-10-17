import unittest

import numpy as np
import pandas as pd

from openml_utils import mean_imputation_and_one_hot_encoding


# Unit tests
class TestMeanImputationAndOneHotEncoding(unittest.TestCase):
    def test_categorical_indicator_update_for_one_hot_with_cat_first(self):
        # Create a sample dataset with cat_feature as the first column
        x_train = pd.DataFrame(
            {
                "cat_feature": pd.Categorical(
                    np.random.choice(["A", "B", "C"], size=10)
                ),
                "num_feature1": np.random.rand(10),
                "num_feature2": np.random.rand(10),
            }
        )
        x_test = pd.DataFrame(
            {
                "cat_feature": pd.Categorical(
                    np.random.choice(["A", "B", "C"], size=5)
                ),
                "num_feature1": np.random.rand(5),
                "num_feature2": np.random.rand(5),
            }
        )
        y_train = np.random.randint(0, 2, 10)
        y_test = np.random.randint(0, 2, 5)

        categorical_indicator = [True, False, False]

        x_train_transformed, _, _, _ = mean_imputation_and_one_hot_encoding(
            x_train,
            x_test,
            y_train,
            y_test,
            categorical_indicator=categorical_indicator,
            categorical_encoder="Onehot",
        )

        # One-hot encoding expands cat_feature to three columns
        expected_categorical_indicator = [False, False, True, True, True]
        self.assertEqual(
            categorical_indicator,
            expected_categorical_indicator,
            "Categorical indicator was not updated correctly with numerical features first and categorical features appended.",
        )

        # Check that numerical columns are close to original values
        original_numerical = x_train[["num_feature1", "num_feature2"]].values
        transformed_numerical = x_train_transformed[
            :, [i for i, ind in enumerate(categorical_indicator) if not ind]
        ]
        self.assertEqual(
            transformed_numerical.shape[1],
            2,
            "Numerical features count mismatch in transformed output.",
        )
        np.testing.assert_allclose(
            transformed_numerical,
            original_numerical,
            rtol=1e-5,
            atol=1e-8,
            err_msg="Numerical features mismatch in values after transformation.",
        )

        # Check that all categorical (True) columns are binary (for one-hot encoding)
        cat_features_transformed = x_train_transformed[
            :, [i for i, ind in enumerate(categorical_indicator) if ind]
        ]
        self.assertTrue(
            np.all(np.isin(cat_features_transformed, [0, 1])),
            "Non-binary values found in one-hot encoded columns.",
        )

    def test_categorical_indicator_update_for_target_encoder_with_cat_first(self):
        # Create a sample dataset with cat_feature as the first column
        x_train = pd.DataFrame(
            {
                "cat_feature": pd.Categorical(
                    np.random.choice(["A", "B", "C"], size=10)
                ),
                "num_feature1": np.random.rand(10),
                "num_feature2": np.random.rand(10),
            }
        )
        x_test = pd.DataFrame(
            {
                "cat_feature": pd.Categorical(
                    np.random.choice(["A", "B", "C"], size=5)
                ),
                "num_feature1": np.random.rand(5),
                "num_feature2": np.random.rand(5),
            }
        )
        y_train = np.random.randint(0, 2, 10)
        y_test = np.random.randint(0, 2, 5)

        categorical_indicator = [True, False, False]

        x_train_transformed, _, _, _ = mean_imputation_and_one_hot_encoding(
            x_train,
            x_test,
            y_train,
            y_test,
            categorical_indicator=categorical_indicator,
            categorical_encoder="Target",
        )

        # Target encoding does not expand columns, so indicator remains the same
        expected_categorical_indicator = [False, False, True]
        self.assertEqual(
            categorical_indicator,
            expected_categorical_indicator,
            "Categorical indicator was not updated correctly with numerical features first and categorical features appended for target encoding.",
        )

        # Check that numerical columns are close to original values
        original_numerical = x_train[["num_feature1", "num_feature2"]].values
        transformed_numerical = x_train_transformed[
            :, [i for i, ind in enumerate(categorical_indicator) if not ind]
        ]
        self.assertEqual(
            transformed_numerical.shape[1],
            2,
            "Numerical features count mismatch in transformed output.",
        )
        np.testing.assert_allclose(
            transformed_numerical,
            original_numerical,
            rtol=1e-5,
            atol=1e-8,
            err_msg="Numerical features mismatch in values after transformation.",
        )


if __name__ == "__main__":
    unittest.main()
