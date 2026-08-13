import os
import joblib
import pandas as pd
from csv_changer import CSVChanger
from main_pipeline import preprocess_and_train_models


def train_and_save_all_models():
    # 1. Define true project root (one level up from src/)
    script_dir = os.path.dirname(os.path.abspath(__file__))
    project_root = os.path.dirname(script_dir)

    # Absolute path to data
    data_path = os.path.join(
        project_root, "data", "processed", "vehicles_with_index-2.csv"
    )
    # Target models directory at project root
    models_dir = os.path.join(project_root, "models")

    # Ensure output directory exists
    os.makedirs(models_dir, exist_ok=True)

    # 2. Load data directly using pandas to avoid path prepending inside CSVChanger
    df = pd.read_csv(data_path)

    # Drop non-predictive columns if they exist in the dataframe
    cols_to_drop = [col for col in ["Unnamed: 0", "vin", "lot"] if col in df.columns]
    if cols_to_drop:
        df = df.drop(columns=cols_to_drop)

    # 3. Train all regressors and retrieve results dictionary
    print("Training models...")
    model_results = preprocess_and_train_models(df=df, target_column="price")

    # 4. Save every model in the results dictionary
    print("\nSaving all trained models to disk:")
    for model_name, info in model_results.items():
        pipeline = info["pipeline"]

        # Convert model name to safe filename
        filename = f"{model_name.lower().replace(' ', '_')}-2.joblib"
        save_path = os.path.join(models_dir, filename)

        # Save pipeline
        joblib.dump(pipeline, save_path)
        print(f"  - Saved '{model_name}' -> {save_path}")


if __name__ == "__main__":
    train_and_save_all_models()