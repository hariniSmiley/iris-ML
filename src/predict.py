# predict.py
import joblib
import numpy as np

# Load model and label encoder
model = joblib.load('iris_model.pkl')
le = joblib.load('label_encoder.pkl')

# Example input: [sepal_length, sepal_width, petal_length, petal_width]
input_data = np.array([[5.1, 3.5, 1.4, 0.2]])

# Predict
prediction = model.predict(input_data)
predicted_species = le.inverse_transform(prediction)

print(f"Predicted Iris species: {predicted_species[0]}")
