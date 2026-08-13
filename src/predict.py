import os
import joblib
import pandas as pd
import matplotlib.pyplot as plt

# Dictionary mapping user choices to exact model filenames saved in Car-Price/models/
AVAILABLE_MODELS = {
    "1": ("Linear Regression", "linear_regression.joblib"),
    "2": ("Decision Tree Regressor", "decision_tree_regressor.joblib"),
    "3": ("Random Forest Regressor", "random_forest_regressor.joblib"),
    "4": ("XGBoost Regressor", "xgboost_regressor.joblib"),
}

# Ready-to-go sample attributes
DEFAULT_SAMPLE_CAR = {
    "year": 2018,
    "mileage": 45000.0,
    "brand": "ford",
    "model": "door",
    "title_status": "clean title",
    "country": "usa",
    "color": "black",
    "state": "texas",
}


def choose_model() -> tuple[str, str]:
    """Prompts the user to select one of the trained models."""
    print("=== Select a Model for Price Prediction ===")
    for key, (display_name, _) in AVAILABLE_MODELS.items():
        print(f"[{key}] {display_name}")

    while True:
        choice = input("\nEnter model number (1-4): ").strip()
        if choice in AVAILABLE_MODELS:
            return AVAILABLE_MODELS[choice]
        print("Invalid choice. Please enter a number between 1 and 4.")


def choose_prediction_mode() -> str:
    """Prompts the user to choose between dummy sample, manual input, and CSV evaluation."""
    print("\n=== Select Prediction Mode ===")
    print("[1] Use ready-to-go sample car attributes")
    print("[2] Insert car attributes manually")
    print("[3] Load CSV dataset file path and compare predictions with actual prices")

    while True:
        choice = input("\nEnter choice (1-3): ").strip()
        if choice in ["1", "2", "3"]:
            return choice
        print("Invalid choice. Please enter 1, 2, or 3.")


def preprocess_input(df: pd.DataFrame) -> pd.DataFrame:
    """Clean input data so it matches the features used during model training."""
    input_df = df.copy()

    # Remove columns that were explicitly excluded during training.
    cols_to_drop = [
        col for col in ["price", "Unnamed: 0", "vin", "lot", "condition"]
        if col in input_df.columns
    ]
    if cols_to_drop:
        input_df = input_df.drop(columns=cols_to_drop)

    # Convert year to car_age exactly as done by the training pipeline.
    if "year" in input_df.columns:
        input_df["year"] = pd.to_numeric(input_df["year"], errors="coerce")
        current_year = int(pd.Timestamp.now().year)
        input_df["car_age"] = current_year - input_df["year"]
        input_df = input_df.drop(columns=["year"])

    # Make sure numerical features are numeric.
    for col in ["mileage", "car_age"]:
        if col in input_df.columns:
            input_df[col] = pd.to_numeric(input_df[col], errors="coerce")

    # Strip whitespace from categorical features.
    str_cols = input_df.select_dtypes(include=["object"]).columns
    for col in str_cols:
        input_df[col] = input_df[col].astype(str).str.strip()

    # Keep exactly the features used by the training pipeline when available.
    expected_features = [
        "mileage",
        "car_age",
        "brand",
        "title_status",
        "country",
        "color",
        "state",
        "model",
    ]
    available_features = [col for col in expected_features if col in input_df.columns]

    return input_df[available_features]


def load_pipeline(model_filename: str):
    """Loads the trained model pipeline from the models directory."""
    script_dir = os.path.dirname(os.path.abspath(__file__))
    project_root = os.path.dirname(script_dir)
    model_path = os.path.join(project_root, "models", model_filename)

    if not os.path.exists(model_path):
        raise FileNotFoundError(
            f"Model file not found at '{model_path}'. Please run train_and_save.py first."
        )

    return joblib.load(model_path)


def get_manual_input() -> dict:
    """Prompts the user to input attributes for a single car manually."""
    print("\n--- Enter Car Attributes ---")
    car_attributes = {
        "year": int(input("Year: ").strip()),
        "mileage": float(input("Mileage: ").strip()),
        "brand": input("Brand: ").strip(),
        "model": input("Model: ").strip(),
        "title_status": input("Title Status: ").strip(),
        "country": input("Country: ").strip(),
        "color": input("Color: ").strip(),
        "state": input("State: ").strip(),
    }
    return car_attributes


def predict_single(car_attributes: dict, pipeline) -> float:
    """Predicts price for a single car attributes dictionary."""
    input_df = pd.DataFrame([car_attributes])
    processed_df = preprocess_input(input_df)
    predicted_price = pipeline.predict(processed_df)[0]
    return float(predicted_price)


def predict_and_plot_csv(csv_path: str, pipeline, model_name: str) -> None:
    """
    Load a CSV dataset, generate predictions, print evaluation metrics,
    and plot actual versus predicted prices.

    The CSV must contain a 'price' column.
    """
    if not os.path.exists(csv_path):
        raise FileNotFoundError(f"CSV file not found at path: {csv_path}")

    df = pd.read_csv(csv_path)

    # Find the actual price column.
    price_col = next(
        (col for col in ["price", "Price", "PRICE"] if col in df.columns),
        None,
    )

    if price_col is None:
        raise KeyError("CSV file must contain a 'price' column for comparison.")

    # Convert prices to numeric and remove invalid rows.
    df[price_col] = pd.to_numeric(df[price_col], errors="coerce")
    df = df.dropna(subset=[price_col]).copy()

    # Remove non-positive prices because they cannot represent a valid car price.
    df = df[df[price_col] > 0].copy()

    if df.empty:
        raise ValueError("No valid positive prices were found in the CSV.")

    # Keep the original actual prices.
    actual_prices = df[price_col].to_numpy(dtype=float)

    # Build the model input. preprocess_input removes the target and
    # the same non-predictive columns used during training.
    feature_df = df.drop(columns=[price_col])
    processed_df = preprocess_input(feature_df)

    # Remove rows with missing numerical values before prediction.
    valid_rows = ~processed_df.isna().any(axis=1)
    processed_df = processed_df.loc[valid_rows].copy()
    actual_prices = actual_prices[valid_rows.to_numpy()]

    if len(processed_df) == 0:
        raise ValueError("No valid rows remain after preprocessing.")

    # Generate predictions.
    predicted_prices = pipeline.predict(processed_df).astype(float)

    # Guard against NaN/inf model output.
    valid_predictions = (
        pd.Series(predicted_prices).replace([float("inf"), float("-inf")], pd.NA).notna()
    ).to_numpy()

    actual_prices = actual_prices[valid_predictions]
    predicted_prices = predicted_prices[valid_predictions]

    if len(predicted_prices) == 0:
        raise ValueError("The model produced no valid predictions.")

    # Print useful diagnostics so zero/invalid values are immediately visible.
    print(f"\nUsing model: {model_name}")
    print(f"Rows evaluated: {len(actual_prices):,}")
    print(
        f"Actual prices    -> min: ${actual_prices.min():,.2f}, "
        f"median: ${pd.Series(actual_prices).median():,.2f}, "
        f"mean: ${actual_prices.mean():,.2f}, "
        f"max: ${actual_prices.max():,.2f}"
    )
    print(
        f"Predicted prices -> min: ${predicted_prices.min():,.2f}, "
        f"median: ${pd.Series(predicted_prices).median():,.2f}, "
        f"mean: ${predicted_prices.mean():,.2f}, "
        f"max: ${predicted_prices.max():,.2f}"
    )

    # Evaluation metrics.
    from sklearn.metrics import (
        mean_absolute_error,
        mean_squared_error,
        r2_score,
    )

    mae = mean_absolute_error(actual_prices, predicted_prices)
    rmse = mean_squared_error(actual_prices, predicted_prices) ** 0.5
    r2 = r2_score(actual_prices, predicted_prices)

    print(f"MAE:  ${mae:,.2f}")
    print(f"RMSE: ${rmse:,.2f}")
    print(f"R²:   {r2:.4f}")

    # ------------------------------------------------------------------
    # Plot 1: Actual vs Predicted scatter plot.
    # This is much more informative than plotting two lines by row index.
    # ------------------------------------------------------------------
    plt.figure(figsize=(10, 7))
    plt.scatter(actual_prices, predicted_prices, alpha=0.45, s=20)

    min_price = min(actual_prices.min(), predicted_prices.min())
    max_price = max(actual_prices.max(), predicted_prices.max())

    plt.plot(
        [min_price, max_price],
        [min_price, max_price],
        linestyle="--",
        linewidth=2,
        label="Perfect Prediction",
    )

    plt.title(f"Actual vs. Predicted Car Prices - {model_name}")
    plt.xlabel("Actual Price ($)")
    plt.ylabel("Predicted Price ($)")
    plt.legend()
    plt.grid(True, linestyle=":", alpha=0.6)
    plt.tight_layout()
    plt.show()

    # ------------------------------------------------------------------
    # Plot 2: Sorted prices.
    # To prevent an extreme price outlier from flattening the graph,
    # display the 99th percentile and below.
    # ------------------------------------------------------------------
    upper_limit = float(pd.Series(actual_prices).quantile(0.99))
    plot_mask = actual_prices <= upper_limit

    sorted_indices = actual_prices[plot_mask].argsort()
    sorted_actual = actual_prices[plot_mask][sorted_indices]
    sorted_predicted = predicted_prices[plot_mask][sorted_indices]

    plt.figure(figsize=(12, 6))
    plt.plot(sorted_actual, label="Actual Price", linewidth=2)
    plt.plot(
        sorted_predicted,
        label=f"Predicted Price ({model_name})",
        linestyle="--",
        linewidth=2,
        alpha=0.85,
    )

    plt.title(
        f"Actual vs. Predicted Car Prices ({model_name}) - "
        "Sorted, excluding top 1% price outliers"
    )
    plt.xlabel("Car Samples (Sorted by Actual Price)")
    plt.ylabel("Price ($)")
    plt.legend()
    plt.grid(True, linestyle=":", alpha=0.6)
    plt.tight_layout()
    plt.show()


if __name__ == "__main__":
    # 1. Choose ML Model
    model_name, model_filename = choose_model()

    try:
        pipeline = load_pipeline(model_filename)
        # 2. Choose Prediction Mode
        mode_choice = choose_prediction_mode()

        if mode_choice == "1":
            # Ready-to-go Dummy Sample
            print("\n--- Using Ready-to-Go Sample Car Attributes ---")
            for k, v in DEFAULT_SAMPLE_CAR.items():
                print(f"  {k}: {v}")

            price = predict_single(DEFAULT_SAMPLE_CAR, pipeline)
            print(f"\nUsing model: {model_name}")
            print(f"Predicted Car Price: ${price:,.2f}")

        elif mode_choice == "2":
            # Manual Input Path
            sample_car = get_manual_input()
            price = predict_single(sample_car, pipeline)
            print(f"\nUsing model: {model_name}")
            print(f"Predicted Car Price: ${price:,.2f}")

        elif mode_choice == "3":
            # CSV File Batch Path & Matplotlib Visualization
            csv_path = input("\nEnter the path to the CSV dataset file: ").strip()
            predict_and_plot_csv(csv_path, pipeline, model_name)

    except (FileNotFoundError, KeyError, ValueError) as e:
        print(f"\nError: {e}")