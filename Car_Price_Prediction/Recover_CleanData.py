# Importation des librairies
import pandas as pd
from sklearn.model_selection import train_test_split, RandomizedSearchCV
from sklearn.preprocessing import OneHotEncoder
from sklearn.ensemble import RandomForestRegressor
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score
import joblib
from pymongo import MongoClient

# 📌 Connexion à MongoDB et récupération des données
client = MongoClient("mongodb://localhost:27017/")
db = client["car_db"]
collection = db["cars"]
data = list(collection.find({}, {"_id": 0}))  # Exclure l'ID MongoDB
df = pd.DataFrame(data)

# 📌 Nettoyage des données
# Suppression des valeurs aberrantes
df = df[(df["price"] > 500) & (df["price"] < 200000)]

# Création d'une nouvelle colonne "âge du véhicule"
df["car_age"] = 2025 - df["model_year"]
df = df.drop(columns=["model_year"], errors="ignore")  # Supprime "year" après création de car_age

# 📌 Séparer les colonnes numériques et catégorielles
cat_cols = df.select_dtypes(include=["object"]).columns
num_cols = df.select_dtypes(include=["int64", "float64"]).columns

# 📌 One-Hot Encoding des variables catégorielles
df = pd.get_dummies(df, columns=cat_cols, drop_first=True)

# 📌 Définition des features et de la target
X = df.drop(columns=["price"])  # On prédit le prix
y = df["price"]

# 📌 Séparer les données en Train/Test
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

print(f"✅ Données prêtes avec {X_train.shape[1]} features !")

# 📌 Définition des hyperparamètres pour l'optimisation
param_grid = {
    "n_estimators": [50, 100, 200, 300],  # Nombre d'arbres
    "max_depth": [None, 10, 20, 30],  # Profondeur des arbres
    "min_samples_split": [2, 5, 10],  # Min. d’échantillons pour diviser un nœud
}

# 📌 Initialisation du modèle RandomForest
rf = RandomForestRegressor(random_state=42)

# 📌 Optimisation avec RandomizedSearchCV
grid_search = RandomizedSearchCV(
    rf, param_grid, n_iter=10, cv=3, scoring="r2", n_jobs=-1, random_state=42
)
grid_search.fit(X_train, y_train)

# 📌 Meilleur modèle trouvé
best_model = grid_search.best_estimator_
print("✅ Meilleurs paramètres trouvés :", grid_search.best_params_)

# 📌 Prédictions
y_pred = best_model.predict(X_test)

# 📌 Évaluation du modèle optimisé
mae = mean_absolute_error(y_test, y_pred)
mse = mean_squared_error(y_test, y_pred)
r2 = r2_score(y_test, y_pred)

print(f"✅ Résultats après optimisation :")
print(f"MAE (Erreur absolue moyenne) : {mae:.2f}")
print(f"MSE (Erreur quadratique moyenne) : {mse:.2f}")
print(f"R² Score : {r2:.4f}")

# 📌 Sauvegarde du modèle optimisé
joblib.dump(df.columns.tolist(), "model_columns.pkl")  # df = ton dataframe d'entraînement
joblib.dump(best_model, "car_price_predictor_optimized.pkl")
print("✅ Modèle optimisé sauvegardé sous 'car_price_predictor_optimized.pkl' !")
