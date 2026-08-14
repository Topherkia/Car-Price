# Car-Price

## Overview
### This is a machine learning mini project that predicts the price of a car based on various features such as make, model, year, mileage, and more. The project utilizes regression algorithms to provide accurate price predictions.

## Features
### Predicts the price of a car based on various features
### Utilizes regression algorithms for accurate price predictions
### Cleaned and standardized dataset
### Preprocessing pipeline for handling missing values and outliers
### Training and evaluation of multiple regression models
### Model selection based on performance metrics
### Ready-to-go sample attributes for testing and visualization
### CSV file handling and visualization using Matplotlib

## Installation
### To run this project, you need to install the following libraries:

```bash
#pip install these libraries in case you don't have them:
pip install scikit-learn
pip install numpy
pip install pandas
pip install matplotlib
pip install xgboost
pip install joblib
```
## Usage
### To use the project, you can follow these steps:

### * Choose ML Model
### * Choose Prediction Mode
### * Run the project
### 1. Choose ML Model
#### You can choose from the following models:

#### Linear Regression
#### Decision Tree Regressor
#### Random Forest Regressor
#### XGBoost Regressor
###
### To choose a model, you can run the choose_model() function in src/predict.py .
###
### 2. Choose Prediction Mode
#### You can choose from the following modes:

#### Ready-to-go Dummy Sample
#### Manual Input Path
#### CSV File Batch Path & Matplotlib Visualization
#### To choose a mode, you can run the choose_prediction_mode() function in 
#### src/predict.py .
#### 

### 3. Run the project
#### To run the project, you can run the predict_single() function in 
#### src/predict.py
#### for single predictions or the predict_and_plot_csv() function in 
#### src/predict.py
#### for batch predictions.

## Data
### The project uses the USA - 2025 - Car Price dataset which is located in the 
```bash data/raw```
### directory. The dataset contains information about various car features and their prices.

## Contributing
### Contributions are welcome! If you find any issues or have any suggestions, please open an issue or submit a pull request.

## Acknowledgments
### Matplotlib for data visualization
### XGBoost for machine learning algorithms
### Ayberk URAL's USA - 2025 - Car Price dataset in kaggle for providing the data used in this project.
