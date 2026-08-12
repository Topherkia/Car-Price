import os
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler, OneHotEncoder
from sklearn.compose import ColumnTransformer
from sklearn.pipeline import Pipeline
from sklearn.metrics import root_mean_squared_error, r2_score
from xgboost import XGBRegressor

from csv_changer import CSVChanger

# 1. LOAD DATA
project_root = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
df = pd.read_csv(os.path.join(project_root, "data", "processed", "USA_cars_datasets_removed.csv"))

# 2. FEATURE CLEANING & DROP IDENTIFIERS
# Drop index, lot, and vin as they don't carry predictive value
unwanted_cols = ['Unnamed: 0', 'vin', 'lot']
df = CSVChanger.load_and_clean_columns(
    file_path="data/processed/USA_cars_datasets_removed.csv", 
    columns_to_drop=['Unnamed: 0', 'vin', 'lot']
)

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

# Categorical features for One-Hot Encoding
cat_features = ['brand', 'title_status', 'country', 'color', 'state']

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

# 6. COMBINE PREPROCESSING & XGBOOST MODEL INTO A PIPELINE
model_pipeline = Pipeline(steps=[
    ('preprocessor', preprocessor),
    ('regressor', XGBRegressor(n_estimators=100, learning_rate=0.1, random_state=42))
])

# 7. TRAIN AND EVALUATE
model_pipeline.fit(X_train, y_train)
y_pred = model_pipeline.predict(X_test)

print("=== XGBoost Regressor Performance ===")
print(f"R2 Score: {r2_score(y_test, y_pred):.4f}")
print(f"RMSE: ${root_mean_squared_error(y_test, y_pred):.2f}")