import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
import os

from sklearn.model_selection import train_test_split
from sklearn.preprocessing import OneHotEncoder, StandardScaler
from sklearn.pipeline import Pipeline
from sklearn.compose import ColumnTransformer
from sklearn.linear_model import LinearRegression
from sklearn.ensemble import RandomForestRegressor
from sklearn.metrics import r2_score, mean_absolute_error

# ---------------------------------------------------------
# 📁 Setup
# ---------------------------------------------------------
SEED = 42
np.random.seed(SEED)

output_path = os.path.join(os.getcwd(), "plots")
model_path = os.path.join(os.getcwd(), "models")
os.makedirs(output_path, exist_ok=True)
os.makedirs(model_path, exist_ok=True)

# ---------------------------------------------------------
# 📥 Load Data
# ---------------------------------------------------------
cars = pd.read_csv(r"PASTE_YOUR_FILE_PATH")

print("Shape:", cars.shape)
print(cars.info())

# ---------------------------------------------------------
# 🧹 Cleaning
# ---------------------------------------------------------
cars.drop(['MSRP', 'Invoice'], axis=1, errors='ignore', inplace=True)
cars.drop_duplicates(inplace=True)
cars.dropna(inplace=True)

# ---------------------------------------------------------
# 🎯 Target & Features
# ---------------------------------------------------------
TARGET = 'MPG_City'

X = cars.drop(TARGET, axis=1)
y = cars[TARGET]

# ---------------------------------------------------------
# 🧠 Feature Types
# ---------------------------------------------------------
num_cols = X.select_dtypes(include=['int64', 'float64']).columns
cat_cols = X.select_dtypes(include=['object']).columns

# ---------------------------------------------------------
# 🔄 Preprocessing Pipeline
# ---------------------------------------------------------
num_pipeline = Pipeline([
    ('scaler', StandardScaler())
])

cat_pipeline = Pipeline([
    ('onehot', OneHotEncoder(handle_unknown='ignore'))
])

preprocessor = ColumnTransformer([
    ('num', num_pipeline, num_cols),
    ('cat', cat_pipeline, cat_cols)
])

# ---------------------------------------------------------
# 🤖 Models
# ---------------------------------------------------------
models = {
    "LinearRegression": LinearRegression(),
    "RandomForest": RandomForestRegressor(n_estimators=100, random_state=SEED)
}

# ---------------------------------------------------------
# ✂️ Train/Test Split
# ---------------------------------------------------------
X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.2, random_state=SEED
)

# ---------------------------------------------------------
# 🏋️ Training + Evaluation
# ---------------------------------------------------------
results = {}

for name, model in models.items():
    pipe = Pipeline([
        ('preprocessor', preprocessor),
        ('model', model)
    ])

    pipe.fit(X_train, y_train)
    preds = pipe.predict(X_test)

    r2 = r2_score(y_test, preds)
    mae = mean_absolute_error(y_test, preds)

    results[name] = {"R2": r2, "MAE": mae}

    print(f"\n{name}")
    print("R2:", r2)
    print("MAE:", mae)

# ---------------------------------------------------------
# 📊 Visualization
# ---------------------------------------------------------

# 1. Distribution
sns.histplot(cars[TARGET], kde=True)
plt.savefig(os.path.join(output_path, "target_distribution.png"))
plt.clf()

# 2. Correlation Heatmap
plt.figure(figsize=(10, 7))
sns.heatmap(cars.corr(numeric_only=True), annot=True)
plt.savefig(os.path.join(output_path, "heatmap.png"))
plt.clf()

# 3. Regression Example
sns.regplot(x='Horsepower', y=TARGET, data=cars)
plt.savefig(os.path.join(output_path, "horsepower_vs_mpg.png"))
plt.clf()

# ---------------------------------------------------------
# 🌳 Feature Importance (Random Forest)
# ---------------------------------------------------------
rf_pipe = Pipeline([
    ('preprocessor', preprocessor),
    ('model', RandomForestRegressor(n_estimators=100, random_state=SEED))
])

rf_pipe.fit(X_train, y_train)

model = rf_pipe.named_steps['model']

importances = model.feature_importances_

# Get feature names after encoding
ohe = rf_pipe.named_steps['preprocessor'].named_transformers_['cat']['onehot']
encoded_features = list(ohe.get_feature_names_out(cat_cols))

all_features = list(num_cols) + encoded_features

feat_imp = pd.Series(importances, index=all_features).sort_values(ascending=False)

feat_imp.head(10).plot(kind='barh')
plt.savefig(os.path.join(output_path, "feature_importance.png"))
plt.clf()

# ---------------------------------------------------------
# 💾 Save Model
# ---------------------------------------------------------
import joblib
joblib.dump(rf_pipe, os.path.join(model_path, "car_model.pkl"))

print("\n✅ Pipeline complete")
print("📁 Plots saved in:", output_path)
print("📦 Model saved in:", model_path)
