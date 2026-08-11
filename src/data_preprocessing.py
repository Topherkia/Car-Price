import pandas as pd
import numpy as np
import os
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler, OneHotEncoder
from sklearn.compose import ColumnTransformer
from sklearn.pipeline import Pipeline
from sklearn.ensemble import RandomForestRegressor
from sklearn.metrics import root_mean_squared_error, r2_score

# 1. LOAD DATA
project_root = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
df = pd.read_csv(os.path.join(project_root, "data", "raw", "USA_cars_datasets.csv"))

# 2. FEATURE CLEANING & DROP IDENTIFIERS
# Drop index, lot, and vin as they don't carry predictive value
df = df.drop(columns=['Unnamed: 0', 'vin', 'lot'], errors='ignore')

# Strip leading/trailing whitespaces in string columns
str_cols = df.select_dtypes(include=['object']).columns
for col in str_cols:
    df[col] = df[col].str.strip()

# Calculate 'car_age' based on current year
current_year = int(pd.Timestamp.now().year)
df['car_age'] = current_year - df['year']
df = df.drop(columns=['year'])

# 3. DEFINE FEATURE GROUPS FOR ENCODING
X = df.drop(columns=['price', 'condition']) # Drop 'condition' (auction time remaining)
y = df['price']

# Numerical features to scale
num_features = ['mileage', 'car_age']

# Categorical features for One-Hot Encoding (Low-to-Medium Cardinality)
cat_features = ['brand', 'title_status', 'country', 'color', 'state']

# High-cardinality categorical features (e.g., 'model' has 127 unique values)
# OneHotEncoder with handle_unknown='infrequent_if_exist' or min_frequency automatically groups rare models
cat_transformer = OneHotEncoder(
    handle_unknown='infrequent_if_exist', 
    min_frequency=10, 
    sparse_output=False
)

# 4. BUILD PREPROCESSING PIPELINE
preprocessor = ColumnTransformer(
    transformers=[
        ('num', StandardScaler(), num_features),
        ('cat', cat_transformer, cat_features + ['model'])
    ]
)

# 5. TRAIN / TEST SPLIT
X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.2, random_state=42
)

# 6. COMBINE PREPROCESSING & MODEL INTO A PIPELINE
model_pipeline = Pipeline(steps=[
    ('preprocessor', preprocessor),
    ('regressor', RandomForestRegressor(n_estimators=100, random_state=42))
])

# 7. TRAIN AND EVALUATE
model_pipeline.fit(X_train, y_train)
y_pred = model_pipeline.predict(X_test)

print("=== Model Performance ===")
print(f"R2 Score: {r2_score(y_test, y_pred):.4f}")
print(f"RMSE: ${root_mean_squared_error(y_test, y_pred):.2f}")
