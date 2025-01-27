import numpy as np
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestClassifier
from sklearn.datasets import load_iris
import joblib

# Load dataset
iris = load_iris()
X = iris.data
y = iris.target

# Train a model
model = RandomForestClassifier(n_estimators=100, random_state=42)
model.fit(X, y)

# Save the model
joblib.dump(model, 'iris_model.pkl')
print("Model saved as 'iris_model.pkl'")

