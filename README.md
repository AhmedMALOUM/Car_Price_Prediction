#  Description du Projet
Ce projet vise à prédire le prix des voitures d'occasion en fonction de plusieurs caractéristiques comme :

- L'âge du véhicule (en années)
- Le kilométrage (distance parcourue)
- La marque

L'application repose sur un modèle de machine learning **Random Forest Regressor**, entraîné sur un dataset de voitures d'occasion. Ce modèle a été optimisé et sauvegardé sous forme d'un fichier `.pkl` pour être réutilisé dans une **API Flask**.

L'objectif est de fournir une estimation rapide et précise du prix d'une voiture en fonction des informations saisies par l'utilisateur.

---

##  Structure du Répertoire
```
Car_Price_Prediction/
│── templates/                 # Dossier contenant les fichiers HTML pour l'interface utilisateur
│── API_Flask.py               # Script principal qui exécute l'API Flask
│── car_price_predictor_optimized.pkl  # Modèle de machine learning sauvegardé (Random Forest Regressor)
│── Data_Insert_To_Mongo.py     # Script d'insertion des données dans MongoDB
│── docker-compose.yml          # Fichier de configuration pour Docker
│── model_columns.pkl           # Fichier contenant les noms des colonnes utilisées à l'entraînement
│── Recover_CleanData.py        # Script de nettoyage et récupération des données
│── used_cars.csv               # Dataset utilisé pour entraîner le modèle
```

---

##  Détails du Modèle de Machine Learning

- **Modèle utilisé** : `RandomForestRegressor`
- **Librairie** : `scikit-learn`

### Entraînement :
- **Dataset utilisé** : `used_cars.csv`
- **Prétraitement** : One-Hot Encoding pour les variables catégorielles (ex : marque)
- **Normalisation des données** pour éviter les écarts trop importants

### Optimisation :
- Recherche d’hyperparamètres (`n_estimators`, `max_depth`, etc.)
- Sélection des features les plus importantes
- Format de sauvegarde : `joblib` / `.pkl`

---

##  Installation et Exécution

### 1️ Prérequis
- Python 3.x
- Flask
- Pandas
- Scikit-learn
- Joblib
- Docker
- MongoDB *(optionnel)*

### 2️ Installation
```bash
git clone https://github.com/ton-repo/Car_Price_Prediction.git
cd Car_Price_Prediction
pip install -r requirements.txt  # (Créer un fichier si besoin)
```

### 3️ Exécution de l'API Flask
```bash
python API_Flask.py
```
L'API sera accessible sur `http://127.0.0.1:5000/`.

### 4️ Exécution avec Docker
```bash
docker-compose up
```

---

## 🛠 Problèmes Connus et Solutions

 **Erreur de mismatch de colonnes**
- Vérifier que les features utilisées pour la prédiction correspondent à celles du modèle (`model_columns.pkl`).
- Utiliser `model.feature_names_in_` pour voir les noms des colonnes attendues.

 **Erreur "Feature names unseen at fit time"**
- Charger dynamiquement `model_columns.pkl` et s’assurer que le DataFrame de prédiction a les bonnes colonnes.

---

## 📈 Performances du Modèle
- **Score R²** : (À compléter après évaluation)
- **Erreur moyenne (MAE)** : (À ajouter)
- **Précision globale** : Bonne sur les données de test, mais améliorable avec plus de données.

---

##  Licence
Ce projet est sous la licence **MIT**.

🔹 **Auteur** : Ahmed MALOUM  
🔹 **Auteur** : Salah MOHAND KACI  
🔹 **© 2025 - Tous droits réservés.**
