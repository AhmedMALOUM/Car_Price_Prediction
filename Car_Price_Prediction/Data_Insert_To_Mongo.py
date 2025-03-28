import pandas as pd
from pymongo import MongoClient

# Connexion à MongoDB
client = MongoClient("mongodb://localhost:27017/")
db = client["car_db"]
collection = db["cars"]

# Chargement du dataset
df = pd.read_csv("used_cars.csv")

# Nettoyage des données
df["milage"] = df["milage"].str.replace(" mi.", "", regex=True).str.replace(",", "").astype(float)  # Conversion en float
df["price"] = df["price"].str.replace("$", "").str.replace(",", "").astype(float)  # Conversion en float
df["model_year"] = df["model_year"].astype(int)  # Conversion en entier

# Remplacement des NaN par "Unknown"
df = df.fillna("Unknown")

# Insertion dans MongoDB
collection.insert_many(df.to_dict(orient="records"))

# Vérification après l'insertion
count = collection.count_documents({})
print(f"Import terminé ! {count} documents insérés.")