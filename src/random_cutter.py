import os
from datetime import datetime
from csv_changer import CSVChanger  # Importing the updated CSVChanger class


def create_partition(project_path: str, file_name: str, sample_size: int = 2500) -> None:
    """Loads a CSV dataset using CSVChanger, generates a random sample of rows,

    and saves the output to the project's 'data/processed' directory with a timestamp.
    """
    relative_file_path = os.path.join(project_path, file_name)

    # 1. Load data using CSVChanger
    df = CSVChanger.load_and_clean_columns(file_path=relative_file_path)

    # 2. Sample 2,500 random rows (handling cases where total rows < sample_size)
    actual_sample_size = min(len(df), sample_size)
    df_partitioned = df.sample(n=actual_sample_size, random_state=42)

    # 3. Prepare project-relative output directory and filename
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    base_name = os.path.splitext(file_name)[0]
    output_file_name = f"{base_name}_partition_{actual_sample_size}_{timestamp}.csv"

    # Resolves to: <project_root>/data/processed[cite: 1]
    output_dir = os.path.join(
        os.path.dirname(os.path.dirname(os.path.abspath(__file__))), 
        "data", 
        "processed"
    )
    os.makedirs(output_dir, exist_ok=True)

    output_path = os.path.join(output_dir, output_file_name)

    # 4. Save data using CSVChanger
    CSVChanger.save_data(df=df_partitioned, output_path=output_path, index=False)
    print(f"Partition successfully saved to: {output_path}")


if __name__ == "__main__":
    path_input = input("Enter the relative project path for the CSV file: ")
    file_input = input("Enter the CSV file name: ")
    create_partition(path_input, file_input)