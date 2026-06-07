import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from sklearn.ensemble import RandomForestRegressor
from sklearn.model_selection import train_test_split
from sklearn.metrics import mean_absolute_error, r2_score

df = pd.read_csv("Video_Games.csv")

# Drop rows with missing Year or Publisher
df = df.dropna(subset=["Year", "Publisher"])

# Features
features = ["Platform", "Genre", "Year", "NA_Sales", "EU_Sales", "JP_Sales", "Other_Sales"]
target = "Global_Sales"

df = df[features + [target]]

# Encode categoricals
df = pd.get_dummies(df, columns=["Platform", "Genre"])

X = df.drop(columns=[target])
y = df[target]

X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

model = RandomForestRegressor(n_estimators=100, random_state=42)
model.fit(X_train, y_train)

preds = model.predict(X_test)

mae = mean_absolute_error(y_test, preds)
r2 = r2_score(y_test, preds)

print(f"MAE: {mae:.4f}")
print(f"R2 Score: {r2:.4f}")

# Feature importance plot
importances = pd.Series(model.feature_importances_, index=X.columns)
importances.nlargest(10).plot(kind="barh")
plt.title("Top 10 Feature Importances")
plt.tight_layout()
plt.savefig("feature_importance.png")
print("Plot saved.")