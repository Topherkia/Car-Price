import os
import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler, OneHotEncoder
from sklearn.compose import ColumnTransformer
from sklearn.pipeline import Pipeline
from sklearn.linear_model import LinearRegression
from sklearn.metrics import root_mean_squared_error, r2_score

from csv_changer import CSVChanger


def preprocess_and_train_linear_regression(
    df: pd.DataFrame,
    target_column: str,
    test_size: float = 0.2,
    random_state: int | None = 42,
) -> tuple[Pipeline, dict[str, float]]:
    """
    Preprocesses features and trains a linear regression model on the given DataFrame,
    then evaluates it on a test split.

    Parameters
    ----------
    df : pd.DataFrame
        Input dataset containing features and the target variable.
    target_column : str
        Name of the column to use as the target variable.
    test_size : float, optional
        Proportion of the dataset to include in the test split (default is 0.2).
    random_state : int or None, optional
        Random seed used by train_test_split for reproducible splits (default is 42).

    Returns
    -------
    model_pipeline : sklearn.pipeline.Pipeline
        Fitted Scikit-Learn pipeline containing preprocessing and the linear regression model.
    metrics : dict
        Dictionary with evaluation metrics on the test set (e.g., {"R2": float, "RMSE": float}).
    """
    # 1. Feature cleaning
    df = df.copy()

    # Strip leading/trailing whitespaces in string columns
    str_cols = df.select_dtypes(include=["object"]).columns
    for col in str_cols:
        df[col] = df[col].str.strip()

    # Calculate 'car_age' based on current year and drop 'year'
    if "year" in df.columns:
        current_year = int(pd.Timestamp.now().year)
        df["car_age"] = current_year - df["year"]
        df = df.drop(columns=["year"])

    # Drop non-predictive/unnecessary columns if present
    cols_to_drop = [col for col in ["condition"] if col in df.columns]
    df = df.drop(columns=cols_to_drop)

    # Separate target and features
    X = df.drop(columns=[target_column])
    y = df[target_column]

    # 2. Define feature groups for preprocessing
    num_features = [col for col in ["mileage", "car_age"] if col in X.columns]
    cat_features = [
        col for col in ["brand", "title_status", "country", "color", "state", "model"]
        if col in X.columns
    ]

    cat_transformer = OneHotEncoder(
        handle_unknown="infrequent_if_exist",
        min_frequency=10,
        sparse_output=False,
    )

    preprocessor = ColumnTransformer(
        transformers=[
            ("num", StandardScaler(), num_features),
            ("cat", cat_transformer, cat_features),
        ]
    )

    # 3. Split data
    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=test_size, random_state=random_state
    )

    # 4. Build and train pipeline
    model_pipeline = Pipeline(
        steps=[
            ("preprocessor", preprocessor),
            ("regressor", LinearRegression()),
        ]
    )

    model_pipeline.fit(X_train, y_train)

    # 5. Evaluate
    y_pred = model_pipeline.predict(X_test)
    r2 = r2_score(y_test, y_pred)
    rmse = root_mean_squared_error(y_test, y_pred)

    metrics = {"R2": r2, "RMSE": rmse}

    return model_pipeline, metrics


if __name__ == "__main__":
    # 1. LOAD DATA
    project_root = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
    data_path = os.path.join(project_root, "data", "processed", "USA_cars_datasets_removed.csv")

    df = pd.read_csv(data_path)

    # Drop index, lot, and vin using CSVChanger
    df = CSVChanger.load_and_clean_columns(
        file_path="data/processed/USA_cars_datasets_removed.csv",
        columns_to_drop=["Unnamed: 0", "vin", "lot"],
    )

    # 2. TRAIN & EVALUATE
    model_pipeline, metrics = preprocess_and_train_linear_regression(
        df=df, target_column="price", test_size=0.2, random_state=42
    )

    print("=== Model Performance ===")
    print(f"R2 Score: {metrics['R2']:.4f}")
    print(f"RMSE: ${metrics['RMSE']:.2f}")