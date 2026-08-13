import os
from typing import Optional, List
import numpy as np
import pandas as pd

class CSVChanger2:
    STANDARD_VEHICLE_COLUMNS = [
        "Unnamed: 0", "price", "brand", "model", "year", "title_status",
        "mileage", "color", "vin", "lot", "state", "country", "condition"
    ]

    VEHICLE_COLUMN_RENAME_MAP = {
        "id": "lot", "manufacturer": "brand", "odometer": "mileage",
        "paint_color": "color", "VIN": "vin"
    }

    @staticmethod
    def load_and_clean_columns(file_path: str, columns_to_drop: Optional[List[str]] = None) -> pd.DataFrame:
        project_root = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
        df = pd.read_csv(os.path.join(project_root, file_path))
        if columns_to_drop:
            df = df.drop(columns=columns_to_drop, errors="ignore")
        return df

    @staticmethod
    def standardize_vehicles_dataset(df: pd.DataFrame) -> pd.DataFrame:
        """Convert vehicles.csv to the exact column schema of USA_cars_datasets_removed.csv."""
        result = df.copy()
        lower_to_actual = {str(c).lower(): c for c in result.columns}
        rename_map = {}
        for source, target in CSVChanger2.VEHICLE_COLUMN_RENAME_MAP.items():
            actual = lower_to_actual.get(source.lower())
            if actual is not None:
                rename_map[actual] = target
        result = result.rename(columns=rename_map)

        if "country" not in result.columns:
            result["country"] = "usa"
        else:
            result["country"] = result["country"].fillna("usa")

        for column in CSVChanger2.STANDARD_VEHICLE_COLUMNS:
            if column not in result.columns:
                result[column] = np.nan

        return result[CSVChanger2.STANDARD_VEHICLE_COLUMNS]

    @staticmethod
    def drop_invalid_rows(df: pd.DataFrame, column: str, invalid_values: Optional[List] = None) -> pd.DataFrame:
        if invalid_values is None:
            invalid_values = [0, np.nan]
        result = df.copy()
        if any(pd.isna(v) for v in invalid_values):
            result = result.dropna(subset=[column])
            invalid_values = [v for v in invalid_values if not pd.isna(v)]
        if invalid_values:
            result = result[~result[column].isin(invalid_values)]
        return result

    @staticmethod
    def replace_values(df: pd.DataFrame, column: str, invalid_values: Optional[List] = None,
                       strategy: str = "median", group_by: Optional[str] = None) -> pd.DataFrame:
        if invalid_values is None:
            invalid_values = [0, np.nan]
        result = df.copy()
        for value in invalid_values:
            if pd.isna(value):
                continue
            result[column] = result[column].replace(value, np.nan)
            if group_by and group_by in result.columns:
                if strategy == "mean":
                    result[column] = result.groupby(group_by)[column].transform(lambda x: x.fillna(x.mean()))
                elif strategy == "median":
                    result[column] = result.groupby(group_by)[column].transform(lambda x: x.fillna(x.median()))
        return result

    @staticmethod
    def save_data(df: pd.DataFrame, output_path: str, index: bool = False) -> None:
        project_root = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
        full_path = os.path.join(project_root, output_path)
        os.makedirs(os.path.dirname(full_path), exist_ok=True)
        df.to_csv(full_path, index=index)

    @staticmethod
    def find_rows(df: pd.DataFrame, column: str, value) -> pd.DataFrame:
        if callable(value):
            return df[df[column].apply(value)]
        return df[df[column] == value]

    @staticmethod
    def calculate_mean(df: pd.DataFrame, column: str, exclude_value: Optional[float] = None) -> float:
        if exclude_value is not None:
            return df[df[column] != exclude_value][column].mean()
        return df[column].mean()

    @staticmethod
    def calculate_median(df: pd.DataFrame, column: str, exclude_value: Optional[float] = None) -> float:
        if exclude_value is not None:
            return df[df[column] != exclude_value][column].median()
        return df[column].median()