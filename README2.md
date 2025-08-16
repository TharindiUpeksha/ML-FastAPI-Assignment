# Iris Classification API

## Problem
Predict iris flower species (Setosa, Versicolor, Virginica) based on 4 features:
- Sepal length
- Sepal width
- Petal length
- Petal width

## Model
- Algorithm: RandomForestClassifier
- Dataset: Built-in Iris dataset from scikit-learn
- Accuracy: ~97% on test set

## API Endpoints
- `GET /` → Health check
- `POST /predict` → Make prediction
- `GET /model-info` → Get model details

## Example Request
```json
POST /predict
{
  "sepal_length": 5.1,
  "sepal_width": 3.5,
  "petal_length": 1.4,
  "petal_width": 0.2
}
