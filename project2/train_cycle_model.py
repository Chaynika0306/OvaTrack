import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LinearRegression
from sklearn.impute import SimpleImputer
from sklearn.preprocessing import LabelEncoder
from sklearn.metrics import r2_score, mean_absolute_error
import joblib

# -----------------------------
# 1️⃣ Load and clean data
# -----------------------------
df = pd.read_csv("Menstural_cyclelength.csv")

# Normalize column names
df.columns = df.columns.str.strip().str.lower()

# Drop invalid ages
df = df[df['age'] >= 9]

# Drop rows where target is missing
df = df.dropna(subset=['cycle_length'])

# -----------------------------
# 2️⃣ Encode categorical data
# -----------------------------
if 'conception_cycle' in df.columns:
    le = LabelEncoder()
    df['conception_cycle'] = le.fit_transform(df['conception_cycle'].astype(str))

# -----------------------------
# 3️⃣ Select features and target
# -----------------------------
feature_cols = ['age', 'cycle_number', 'conception_cycle']
X = df[feature_cols]
y = df['cycle_length']

# -----------------------------
# 4️⃣ Handle missing values
# -----------------------------
imputer = SimpleImputer(strategy='mean')
X_imputed = imputer.fit_transform(X)

# -----------------------------
# 5️⃣ Train/test split
# -----------------------------
X_train, X_test, y_train, y_test = train_test_split(X_imputed, y, test_size=0.2, random_state=42)

# -----------------------------
# 6️⃣ Train Linear Regression model
# -----------------------------
model = LinearRegression()
model.fit(X_train, y_train)

# -----------------------------
# 7️⃣ Evaluate model
# -----------------------------
y_pred = model.predict(X_test)
r2 = r2_score(y_test, y_pred)
mae = mean_absolute_error(y_test, y_pred)

print("\n📊 Model Evaluation Results:")
print(f"R² Score: {r2:.3f}")
print(f"Mean Absolute Error: {mae:.2f} days")

# -----------------------------
# 8️⃣ Save model and imputer
# -----------------------------
joblib.dump(model, "cycle_model.pkl")
joblib.dump(imputer, "cycle_imputer.pkl")

print("\n✅ Model trained and saved successfully as 'cycle_model.pkl' and 'cycle_imputer.pkl'.")
