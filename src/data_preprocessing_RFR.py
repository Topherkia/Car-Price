import os
import pandas as pd
import matplotlib.pyplot as plt
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler, OneHotEncoder
from sklearn.compose import ColumnTransformer
from sklearn.pipeline import Pipeline
from sklearn.linear_model import LinearRegression
from sklearn.ensemble import RandomForestRegressor
from sklearn.tree import DecisionTreeRegressor
from sklearn.metrics import root_mean_squared_error, r2_score

# Step 1: Load Data
project_root = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
df = pd.read_csv(os.path.join(project_root, "data", "processed", "USA_cars_datasets_removed.csv"))

# Step 2: Clean Features
current_year = int(pd.Timestamp.now().year)
df['car_age'] = current_year - df['year']
df = df.drop(columns=['Unnamed: 0', 'vin', 'lot', 'year', 'condition'])

# Step 3: Define Features and Target
X = df.drop(columns=['price'])
y = df['price']

num_features = ['mileage', 'car_age']
cat_features = ['brand', 'title_status', 'country', 'color', 'state', 'model']

# Step 4: Setup Preprocessing Pipeline
preprocessor = ColumnTransformer(
    transformers=[
        ('num', StandardScaler(), num_features),
        ('cat', OneHotEncoder(handle_unknown='infrequent_if_exist', min_frequency=10, sparse_output=False), cat_features)
    ]
)

# Step 5: Split Data
X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.2, random_state=42
)

# Step 6: Define Regressors
regressors = {
    "Linear Regression": LinearRegression(),
    "Random Forest Regressor": RandomForestRegressor(n_estimators=100, random_state=42),
    "Decision Tree Regressor": DecisionTreeRegressor(random_state=42)
}

# Step 7: Train and Compare Models
# Set up a master figure and 3 subplots for plotting results (with dark theme)
plt.style.use('dark_background')
fig, axes = plt.subplots(1, 3, figsize=(15, 6))

for i, (name, model) in enumerate(regressors.items()):
    pipeline = Pipeline(steps=[
        ('preprocessor', preprocessor),
        ('regressor', model)
    ])
    
    # Train
    pipeline.fit(X_train, y_train)
    
    # Predict & Evaluate
    y_pred = pipeline.predict(X_test)
    r2 = r2_score(y_test, y_pred)
    rmse = root_mean_squared_error(y_test, y_pred)

    print(f"=== {name} Performance ===")
    print(f"R2 Score: {r2:.4f}")
    print(f"RMSE: ${rmse:.2f}\n")

    # Plot the evaluation using scatter plot for R-squared
    min_val = y_test.min()
    max_val = y_test.max()

    axes[i].scatter(y_test, y_pred, alpha=0.5, color='#A5FF1F')
    axes[i].plot([min_val, max_val], [min_val, max_val], color='red', linestyle='--')
    axes[i].set_title(f"{name} (R^2: {r2:.4f})")
    axes[i].set_xlabel('Actual Prices')
    axes[i].set_ylabel('Predicted Prices')
    axes[i].set_facecolor('#1F1F1F')

plt.tight_layout()
plt.show()