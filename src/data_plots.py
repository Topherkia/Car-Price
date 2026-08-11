import pandas as pd
import matplotlib.pyplot as plt

# Define file name
file_name = 'USA_cars_datasets.csv'

# Define path
file_path = '../data/raw/' + file_name

# Convert the CSV file into a Pandas DataFrame
df = pd.read_csv(file_path)

# Calculate the min, max and median values for prices
min_price = df['price'].min()
median_price = df['price'].median()
max_price = df['price'].max()

# Setup the data for Matplotlib
categories =  ['Minimum', 'Median', 'Maximum']
values = [min_price, median_price, max_price]

# Create the bar chart
plt.bar(categories, values, color=['#4C72B0', '#55A868', '#C44E52'])

# Create labels
plt.title('USA\'s car price distribution (Min, Max, Median)')
plt.ylabel('Price ($)')
plt.xlabel('Categories')

# Add numeric value to the chart
for index, value in enumerate(values):
    plt.text(index, value, str(value), ha='center', va='bottom')

# Render the plot
plt.tight_layout()
plt.show()