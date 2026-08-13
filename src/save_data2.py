import os
from csv_changer2 import CSVChanger2

def process_car_prices(project_path: str, file_name: str) -> None:
    """Load vehicles.csv, standardize it, handle zero prices, and save the result."""
    relative_file_path = os.path.join(project_path, file_name)
    df = CSVChanger2.load_and_clean_columns(file_path=relative_file_path)
    df = CSVChanger2.standardize_vehicles_dataset(df)

    print("Choose how to handle cars with price = 0:")
    print("1: Remove rows")
    print("2: Replace with mean price")
    print("3: Replace with median price")
    choice = input("Enter choice (1, 2, or 3): ").strip()

    if choice == "1":
        df = df[df["price"] != 0].copy()
        strategy = "removed"
    elif choice == "2":
        mean_price = CSVChanger2.calculate_mean(df, "price", exclude_value=0)
        df["price"] = df["price"].replace(0, mean_price)
        strategy = "mean_replaced"
    elif choice == "3":
        median_price = CSVChanger2.calculate_median(df, "price", exclude_value=0)
        df["price"] = df["price"].replace(0, median_price)
        strategy = "median_replaced"
    else:
        raise ValueError("Invalid choice selected. Please run again with 1, 2, or 3.")

    base_name = os.path.splitext(file_name)[0]
    output_dir = os.path.join(os.path.dirname(os.path.dirname(os.path.abspath(__file__))), "data", "processed")
    os.makedirs(output_dir, exist_ok=True)
    output_path = os.path.join(output_dir, f"{base_name}_{strategy}.csv")
    CSVChanger2.save_data(df, output_path, index=False)
    print(f"File successfully saved to: {output_path}")

if __name__ == "__main__":
    path_input = input("Enter the csv file path in the project: ").strip()
    file_input = input("Enter the CSV file name: ").strip()
    process_car_prices(path_input, file_input)