import os
from datetime import datetime
from csv_changer import CSVChanger  # Importing the updated CSVChanger class[cite: 2]


def process_car_prices(project_path: str, file_name: str) -> None:
    """Loads a car dataset using CSVChanger, handles zero prices based on user choice,

    and saves the output to '~/data/processed' with a timestamp.
    """
    relative_file_path = os.path.join(project_path, file_name)

    # 1. Load data using CSVChanger[cite: 2]
    df = CSVChanger.load_and_clean_columns(file_path=relative_file_path)

    # Prompt user for handling method
    print("Choose how to handle cars with price = 0:")
    print("1: Remove rows")
    print("2: Replace with mean price")
    print("3: Replace with median price")

    choice = input("Enter choice (1, 2, or 3): ").strip()

    # Process price column according to selection
    if choice == "1":
        df = df[df["price"] != 0]
        strategy = "removed"
    elif choice == "2":
        # Calculate mean using CSVChanger method
        mean_price = CSVChanger.calculate_mean(
            df, column="price", exclude_value=0
        )
        df["price"] = df["price"].replace(0, mean_price)
        strategy = "mean_replaced"
    elif choice == "3":
        # Calculate median using CSVChanger method
        median_price = CSVChanger.calculate_median(
            df, column="price", exclude_value=0
        )
        df["price"] = df["price"].replace(0, median_price)
        strategy = "median_replaced"
    else:
        raise ValueError(
            "Invalid choice selected. Please run again with 1, 2, or 3."
        )

    # 2. Prepare output directory and filename
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    base_name = os.path.splitext(file_name)[0]
    output_file_name = f"{base_name}_{strategy}.csv"

    output_dir = os.path.join(os.path.dirname(os.path.dirname(os.path.abspath(__file__))), "data", "processed")
    os.makedirs(output_dir, exist_ok=True)

    output_path = os.path.join(output_dir, output_file_name)

    # 3. Save data using CSVChanger[cite: 2]
    CSVChanger.save_data(df=df, output_path=output_path, index=False)
    print(f"File successfully saved to: {output_path}")


if __name__ == "__main__":
    path_input = input("Enter the csv file path in the project: ")
    file_input = input("Enter the CSV file name: ")
    process_car_prices(path_input, file_input)