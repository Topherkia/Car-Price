import pandas as pd
import numpy as np
import os

class CSVChanger:
    @staticmethod
    def load_and_clean_columns(file_path: str, columns_to_drop: list = None) -> pd.DataFrame:
        """
        Loads a CSV file into a pandas DataFrame and drops specified columns.
        """
        project_root = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
        df = pd.read_csv(os.path.join(project_root, file_path))
        
        if columns_to_drop:
            df = df.drop(columns=columns_to_drop, errors='ignore')
            
        return df

    @staticmethod
    def save_data(df: pd.DataFrame, output_path: str, index: bool = False) -> None:
        """
        Saves the cleaned DataFrame to a CSV file.
        """
        project_root = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
        df.to_csv(os.path.join(project_root, output_path), index=index)

    # ------------------ NEW METHODS ------------------ #

    @staticmethod
    def find_rows(df: pd.DataFrame, column: str, value) -> pd.DataFrame:
        """
        Finds and returns rows where a column matches a specific value or condition.
        
        Parameters:
        -----------
        df : pd.DataFrame
        column : str
            The column name to inspect.
        value : Any or callable
            The target value (e.g., 0) or a boolean condition/lambda (e.g., lambda x: x <= 0).
            
        Returns:
        --------
        pd.DataFrame
            Filtered subset of the DataFrame.
        """
        if callable(value):
            return df[df[column].apply(value)]
        return df[df[column] == value]
