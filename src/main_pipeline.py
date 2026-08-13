import os
import pandas as pd
import matplotlib.pyplot as plt
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler, OneHotEncoder
from sklearn.impute import SimpleImputer
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

    # Strip leading/trailing whitespace from string columns.
    str_cols = df.select_dtypes(include=["object"]).columns
    for col in str_cols:
        df[col] = df[col].astype("string").str.strip()

    # Make the target numeric and remove rows without a usable target.
    df[target_column] = pd.to_numeric(df[target_column], errors="coerce")
    df = df.dropna(subset=[target_column]).copy()

    # Remove impossible/non-positive target prices.
    df = df[df[target_column] > 0].copy()

    # Handle extreme price outliers for training/evaluation.
    #
    # The dataset contains an extreme price (around $12.3M) that is not
    # representative of normal vehicle prices. We use the IQR rule on the
    # training data only, preventing test-set information from leaking into
    # model fitting.
    y_all = df[target_column]
    q1 = y_all.quantile(0.25)
    q3 = y_all.quantile(0.75)
    iqr = q3 - q1
    lower_price = max(0.0, q1 - 1.5 * iqr)
    upper_price = q3 + 1.5 * iqr

    # Calculate 'car_age' based on the current year and drop 'year'.
    if "year" in df.columns:
        df["year"] = pd.to_numeric(df["year"], errors="coerce")
        current_year = int(pd.Timestamp.now().year)
        df["car_age"] = current_year - df["year"]
        df = df.drop(columns=["year"])

    # Drop non-predictive/unnecessary columns if present.
    cols_to_drop = [
        col for col in ["condition", "Unnamed: 0", "vin", "lot"]
        if col in df.columns
    ]
    if cols_to_drop:
        df = df.drop(columns=cols_to_drop)

    # Separate target and features.
    X = df.drop(columns=[target_column])
    y = df[target_column]

    # 2. Define feature groups for preprocessing.
    num_features = [
        col for col in ["mileage", "car_age"]
        if col in X.columns
    ]

    cat_features = [
        col for col in [
            "brand",
            "title_status",
            "country",
            "color",
            "state",
            "model",
        ]
        if col in X.columns
    ]

    # Numerical missing values -> median.
    # Categorical missing values -> most frequent value.
    #
    # Imputation is inside the sklearn pipeline so the values are learned
    # only from the training split, avoiding data leakage.
    num_transformer = Pipeline(steps=[
        ("imputer", SimpleImputer(strategy="median")),
        ("scaler", StandardScaler()),
    ])

    cat_transformer = Pipeline(steps=[
        ("imputer", SimpleImputer(strategy="most_frequent")),
        ("onehot", OneHotEncoder(
            handle_unknown="infrequent_if_exist",
            min_frequency=10,
            sparse_output=False,
        )),
    ])

    preprocessor = ColumnTransformer(
        transformers=[
            ("num", num_transformer, num_features),
            ("cat", cat_transformer, cat_features),
        ],
        remainder="drop",
    )

    # 3. Split data
    X_train, X_test, y_train, y_test = train_test_split(
        X,
        y,
        test_size=test_size,
        random_state=random_state,
    )

    # Remove extreme target-price outliers from TRAINING only.
    # The original test set is retained for an honest evaluation.
    train_mask = (
        (y_train >= lower_price)
        & (y_train <= upper_price)
    )

    X_train_clean = X_train.loc[train_mask].copy()
    y_train_clean = y_train.loc[train_mask].copy()

    removed_train = len(y_train) - len(y_train_clean)

    print("\n=== Dataset Preparation ===")
    print(f"Total usable rows: {len(df):,}")
    print(f"Training rows before outlier removal: {len(y_train):,}")
    print(f"Training outliers removed: {removed_train:,}")
    print(f"Training rows used: {len(y_train_clean):,}")
    print(f"Test rows: {len(y_test):,}")
    print(f"Price IQR bounds: ${lower_price:,.2f} - ${upper_price:,.2f}")

    # 4. Define regressors.
    regressors = {
        "Linear Regression": LinearRegression(),
        "Decision Tree Regressor": DecisionTreeRegressor(
            random_state=42
        ),
        "Random Forest Regressor": RandomForestRegressor(
            n_estimators=100,
            random_state=42,
            n_jobs=-1,
        ),
        "XGBoost Regressor": XGBRegressor(
            n_estimators=100,
            learning_rate=0.1,
            random_state=42,
            n_jobs=-1,
        ),
    }

    results = {}

    # 5. Train and compare models.
    plt.style.use("dark_background")
    fig, axes = plt.subplots(1, 4, figsize=(18, 6))

    for i, (name, model) in enumerate(regressors.items()):
        pipeline = Pipeline(steps=[
            ("preprocessor", preprocessor),
            ("regressor", model),
        ])

        # Train only on non-extreme target values.
        pipeline.fit(X_train_clean, y_train_clean)

        # Predict on the complete untouched test set.
        y_pred = pipeline.predict(X_test)

        # Full test-set metrics.
        r2_full = r2_score(y_test, y_pred)
        rmse_full = root_mean_squared_error(y_test, y_pred)
        mae_full = mean_absolute_error(y_test, y_pred)

        # Robust metrics excluding test-set target outliers.
        test_mask = (
            (y_test >= lower_price)
            & (y_test <= upper_price)
        )

        y_test_robust = y_test.loc[test_mask]
        y_pred_robust = y_pred[test_mask.to_numpy()]

        if len(y_test_robust) > 1:
            r2_robust = r2_score(y_test_robust, y_pred_robust)
            rmse_robust = root_mean_squared_error(
                y_test_robust,
                y_pred_robust,
            )
            mae_robust = mean_absolute_error(
                y_test_robust,
                y_pred_robust,
            )
        else:
            r2_robust = float("nan")
            rmse_robust = float("nan")
            mae_robust = float("nan")

        results[name] = {
            "pipeline": pipeline,
            "R2": r2_full,
            "RMSE": rmse_full,
            "MAE": mae_full,
            "R2_robust": r2_robust,
            "RMSE_robust": rmse_robust,
            "MAE_robust": mae_robust,
        }

        print(f"=== {name} Performance ===")
        print(f"Full test set:")
        print(f"  R²:   {r2_full:.4f}")
        print(f"  RMSE: ${rmse_full:,.2f}")
        print(f"  MAE:  ${mae_full:,.2f}")
        print(f"Robust test set (outliers excluded):")
        print(f"  R²:   {r2_robust:.4f}")
        print(f"  RMSE: ${rmse_robust:,.2f}")
        print(f"  MAE:  ${mae_robust:,.2f}\n")

        # Plot robust test performance so extreme prices do not flatten it.
        plot_mask = (
            (y_test >= lower_price)
            & (y_test <= upper_price)
        )

        actual_plot = y_test.loc[plot_mask].to_numpy()
        pred_plot = y_pred[plot_mask.to_numpy()]

        if len(actual_plot) > 0:
            min_val = min(actual_plot.min(), pred_plot.min())
            max_val = max(actual_plot.max(), pred_plot.max())

            axes[i].scatter(
                actual_plot,
                pred_plot,
                alpha=0.5,
            )
            axes[i].plot(
                [min_val, max_val],
                [min_val, max_val],
                linestyle="--",
            )

            axes[i].set_title(
                f"{name}\nRobust R²: {r2_robust:.4f}"
            )
            axes[i].set_xlabel("Actual Prices ($)")
            axes[i].set_ylabel("Predicted Prices ($)")
            axes[i].grid(True, linestyle=":", alpha=0.6)

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