import pandas as pd
import matplotlib.pyplot as plt

# Define file name
file_name = 'USA_cars_datasets_processed_removed_20260812_133529.csv'

# Define path
file_path = '../data/processed/' + file_name

# Convert the CSV file into a Pandas DataFrame
df = pd.read_csv(file_path)

# Sort DataFrame by price so all cars are ordered continuously
df_sorted = df.sort_values('price').reset_index(drop=True)

# Calculate the min, max and median values for prices
min_price = df['price'].min()
median_price = df['price'].median()
max_price = df['price'].max()

# set the figure size
plt.figure(figsize=(12, 6))

# Setup the data for Matplotlib
categories =  ['Minimum', 'Median', 'Maximum']
values = [min_price, median_price, max_price]

# Create bar chart for all individual cars
plt.bar(df_sorted.index, df_sorted['price'], color='#A0C4DF', width=1.0, label='Car Prices')

# Add horizontal reference lines for Min, Median, and Max
plt.axhline(
    min_price, 
    color='#4C72B0', 
    linestyle='--', 
    linewidth=2, 
    label=f'Min Price (${min_price:,.0f})'
)
plt.axhline(
    median_price, 
    color='#55A868', 
    linestyle='--', 
    linewidth=2, 
    label=f'Median Price (${median_price:,.0f})'
)
plt.axhline(
    max_price, 
    color='#C44E52', 
    linestyle='--', 
    linewidth=2, 
    label=f'Max Price (${max_price:,.0f})'
)

# Labels and styling
plt.title("USA's Car Price Distribution (All Vehicles)", fontsize=14, pad=15)
plt.xlabel('Vehicles (Sorted by Price)')
plt.ylabel('Price ($)')
plt.legend(loc='upper left')
plt.grid(axis='y', linestyle=':', alpha=0.6)

# Render the plot
plt.tight_layout()
plt.show()
