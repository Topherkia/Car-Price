import os
import pandas as pd
import matplotlib.pyplot as plt
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler, OneHotEncoder
from sklearn.compose import ColumnTransformer
from sklearn.pipeline import Pipeline
from sklearn.linear_model import LinearRegression
from sklearn.ensemble import RandomForestRegressor
from sklearn.tree import DecisionTreeRegressor
from sklearn.metrics import root_mean_squared_error, r2_score, mean_absolute_error, mean_absolute_percentage_error
from xgboost import XGBRegressor, sklearn
from csv_changer import CSVChanger


def preprocess_and_train_models(
    df: pd.DataFrame,
    target_column: str,
    test_size: float = 0.2,
    random_state: int | None = 42):
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


    # 4. Define Regressors
    regressors = {
        "Linear Regression": LinearRegression(),
        "Decision Tree Regressor": DecisionTreeRegressor(random_state=42),
        "Random Forest Regressor": RandomForestRegressor(n_estimators=100, random_state=42),
        "XGBoost Regressor": XGBRegressor(n_estimators=100, learning_rate=0.1, random_state=42)
    }

    results = {}

    # 5. Train and Compare Models
    # Set up a master figure and 3 subplots for plotting results (with dark theme)
    plt.style.use('dark_background')
    fig, axes = plt.subplots(1, 4, figsize=(15, 6))
    for i, (name, model) in enumerate(regressors.items()):
        pipeline = Pipeline(steps=[
            ('preprocessor', preprocessor),
            ('regressor', model)
        ])

        # Train
        pipeline.fit(X_train, y_train)

        # Predict & Evaluate
        y_pred = pipeline.predict(X_test)
        r2 = r2_score(y_test, y_pred)
        rmse = root_mean_squared_error(y_test, y_pred)
        mae = mean_absolute_error(y_test, y_pred)
        mape = mean_absolute_percentage_error(y_test, y_pred)

        results[name] = {"pipeline": pipeline, "R2": r2, "RMSE": rmse, "MAE": mae, "MAPE": mape}

        print(f"=== {name} Performance ===")
        print(f"R2 Score: {r2:.4f}")
        print(f"RMSE: ${rmse:.2f}")
        print(f"MAE: ${mae:.2f}")
        print(f"MAPE: {mape * 100:.2f}%\n")

        # Plot the evaluation using scatter plot for R-squared
        min_val = y_test.min()
        max_val = y_test.max()

        axes[i].scatter(y_test, y_pred, alpha=0.5, color='#A5FF1F')
        axes[i].plot([min_val, max_val], [min_val, max_val], color='red', linestyle='--')
        axes[i].set_title(f"{name} (R^2: {r2:.4f})")
        axes[i].set_xlabel('Actual Prices')
        axes[i].set_ylabel('Predicted Prices')
        axes[i].set_facecolor('#1F1F1F')

    plt.tight_layout()
    plt.show()

    return results


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

    model_results = preprocess_and_train_models(df=df, target_column="price")