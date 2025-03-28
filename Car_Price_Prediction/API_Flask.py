from flask import Flask, request, jsonify, render_template
import pandas as pd
import joblib

app = Flask(__name__)

# Charger le modèle entraîné
model = joblib.load("car_price_predictor_optimized.pkl")

# Charger les colonnes utilisées à l'entraînement
model_columns = joblib.load("model_columns.pkl")  # Sauvegarde lors de l'entraînement

@app.route("/")
def home():
    return render_template("index.html")

@app.route("/predict", methods=["POST"])
def predict():
    try:
        # Récupérer les données du formulaire
        car_age = int(request.form["car_age"])
        mileage = int(request.form["mileage"])
        brand = request.form["brand"]

        # Créer un dataframe avec les colonnes exactes du modèle
        data = {col: 0 for col in model_columns}  # Initialiser toutes les colonnes à 0
        data["car_age"] = car_age
        data["mileage"] = mileage

        # Vérifier si la marque existe dans les colonnes du modèle
        brand_column = f"brand_{brand}"
        if brand_column in data:
            data[brand_column] = 1  # Activer la marque sélectionnée

        df = pd.DataFrame([data])

        # Prédire le prix
        prediction = model.predict(df)[0]

        return render_template("index.html", prediction=round(float(prediction), 2))

    except Exception as e:
        return render_template("index.html", error=str(e))

if __name__ == "__main__":
    app.run(debug=True)
