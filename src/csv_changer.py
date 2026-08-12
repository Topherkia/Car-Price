import pandas as pd
import numpy as np
import os
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LinearRegression
from sklearn.metrics import mean_absolute_error, r2_score

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
    def drop_invalid_rows(df: pd.DataFrame, column: str, invalid_values: list = [0, np.nan]) -> pd.DataFrame:
        """
        Removes rows where a specific column contains invalid values (e.g., price = 0 or NaN).
        
        Parameters:
        -----------
        df : pd.DataFrame
        column : str
            Column to filter (e.g., 'price').
        invalid_values : list
            List of values considered invalid.
        """
        df_cleaned = df.copy()
        
        # Filter out NaNs if included in invalid_values
        if any(pd.isna(v) for v in invalid_values):
            df_cleaned = df_cleaned.dropna(subset=[column])
            invalid_values = [v for v in invalid_values if not pd.isna(v)]
            
        # Filter out remaining scalar invalid values (e.g., 0)
        if invalid_values:
            df_cleaned = df_cleaned[~df_cleaned[column].isin(invalid_values)]
            
        return df_cleaned
     
    @staticmethod
    def replace_values(df: pd.DataFrame, 
        column: str, 
        invalid_values: list = [0, np.nan], 
        strategy: str = 'median', 
        group_by: str = None
    ) -> pd.DataFrame:
        """
           Replace zero values with the median or mean.
        """
        result = df.copy()
        for val in invalid_values:
            if pd.isna(val):
                continue
            result[column] = result[column].replace(val, np.nan)
            if group_by and group_by in result.columns:
                if strategy == 'mean':
                    result[column] = result.groupby(group_by)[column].transform(lambda x: x.fillna(x.mean()))
                elif strategy == 'median':
                    result[column] = result.groupby(group_by)[column].transform(lambda x: x.fillna(x.median()))

        return result

    @staticmethod
    def save_data(df: pd.DataFrame, output_path: str, index: bool = False) -> None:
        """
        Saves the cleaned DataFrame to a CSV file.
        """
        project_root = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
        df.to_csv(os.path.join(project_root, output_path), index=index)

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

    @staticmethod
    def linear_regression(df: pd.DataFrame, target_column: str,  test_size: float = 0.2,
        random_state: int | None = 42):

        """
           Trains a linear regression model on the given DataFrame and evaluates it.

           Parameters
           ----------
           df : pd.DataFrame
           target_column : str
               Name of the column to use as the target variable.
           test_size : float, optional
               Proportion of the dataset to include in the test split (default is 0.2).
           random_state : int or None, optional
               Random seed used by train_test_split for reproducible splits (default is 42).

           Returns
           -------
           model : sklearn.linear_model.LinearRegression
               Fitted linear regression model.
           metrics : dict
               Dictionary with evaluation metrics on the test set
               (e.g., {"MAE": float, "R2": float}).
           """

        x = df.drop(columns=[target_column])
        y = df[target_column]

        x_train, x_test, y_train, y_test = train_test_split(
            x, y, test_size=test_size, random_state=random_state
        )
        model = LinearRegression()
        model.fit(x_train, y_train)
        y_pred = model.predict(x_test)

        mae = mean_absolute_error(y_test, y_pred)
        r2 = r2_score(y_test, y_pred)
        metrics = {"MAE": mae, "R2": r2}

        return model, metrics

