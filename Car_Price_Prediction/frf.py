import joblib
model = joblib.load("car_price_predictor_optimized.pkl")
print(model.feature_names_in_)  # Vérifiez les colonnes utilisées
