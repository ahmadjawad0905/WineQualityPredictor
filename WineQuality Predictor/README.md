# 🍷 Wine Quality Predictor

This project predicts the quality of wine based on its physicochemical properties using Machine Learning.

## Dataset
Wine Quality Dataset containing features like acidity, sugar, pH, alcohol, etc.

## Model
- Random Forest Regressor
- Train-Test Split: 80/20
- Feature Scaling: StandardScaler

## Evaluation
- Mean Squared Error (MSE)
- R² Score

## How to Run
1. Install dependencies  
   pip install -r requirements.txt
2. Run the script  
   python winequality.py

## Output
The model predicts a numerical wine quality score and classifies it as:
- Below Average
- Average
- Good Quality
