import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.ensemble import RandomForestRegressor
from sklearn.metrics import mean_squared_error, r2_score
import joblib

# Load data
data = pd.read_csv("WineQuality.csv")

data = data.drop(columns=['Id'])   # or whatever extra column is

# Split features and target
X = data.drop('quality', axis=1)
y = data['quality']

# Train-test split (80% train, 20% test)
X_train, X_test, y_train, y_test = train_test_split(
    X,
    y,
    test_size=0.2,
    random_state=42
)

# Scaling (optional for Random Forest, but OK)
scaler = StandardScaler()
X_train = scaler.fit_transform(X_train)
X_test = scaler.transform(X_test)

# Model
model = RandomForestRegressor(n_estimators=200, random_state=42)
model.fit(X_train, y_train)

# Prediction
y_pred = model.predict(X_test)

# Evaluation
print("MSE:", mean_squared_error(y_test, y_pred))
print("R² Score:", r2_score(y_test, y_pred))

def predictor():
    fixed_acidity = float(input("Enter Fixed Acidity: "))
    volatile_acidity = float(input("Enter Volatile Acidity: "))
    citric_acid = float(input("Enter Citric Acid: "))
    residual_sugar = float(input("Enter Residual Sugar: "))
    chlorides = float(input("Enter Chlorides: "))
    free_sulfur = float(input("Enter Free Sulfur Dioxide: "))
    total_sulfur = float(input("Enter Total Sulfur Dioxide: "))
    density = float(input("Enter Density: "))
    pH = float(input("Enter pH: "))
    sulphates = float(input("Enter Sulphates: "))
    alcohol = float(input("Enter Alcohol: "))

    # Arrange input in SAME order as training data
    user_input = [[
        fixed_acidity,
        volatile_acidity,
        citric_acid,
        residual_sugar,
        chlorides,
        free_sulfur,
        total_sulfur,
        density,
        pH,
        sulphates,
        alcohol
    ]]

    # Scale input
    user_input_scaled = scaler.transform(user_input)

    # Prediction
    prediction = model.predict(user_input_scaled)

    predicted_quality = prediction[0]
    
    print(f"\n🍷 Predicted Wine Quality: {predicted_quality:.2f}")
    if (predicted_quality>=0 and predicted_quality<=4):
        print("Below Average Quality Wine !")
    elif(predicted_quality>=5 and predicted_quality<=7):
        print("Average Quality Wine !")
    elif(predicted_quality>=8):
        print("Good Quality Wine !") 

predictor()
        




