# ==========================================================
# Application Flask pour utiliser le pipeline meta-learning
# ==========================================================

from flask import Flask, render_template, request
import numpy as np
import joblib

# ---------------------------------------------
# Charger les modèles sauvegardés
# ---------------------------------------------
scaler = joblib.load("../models/scaler.joblib")
dt_model = joblib.load("../models/dt_model.joblib")
rf_model = joblib.load("../models/rf_model.joblib")
svm_model = joblib.load("../models/svm_model.joblib")
nb_model = joblib.load("../models/nb_model.joblib")
knn_meta = joblib.load("../models/meta_knn_model.joblib")

# ---------------------------------------------
# Fonction utilitaire pour extraire méta-features
# ---------------------------------------------
def extract_meta_for_single(model, x):
    proba = model.predict_proba(x)[0]   # [P(0), P(1)]
    confiance_max = np.max(proba)
    margin = abs(proba[1] - proba[0])
    return confiance_max, margin

# ---------------------------------------------
# Pipeline final : sélection du meilleur modèle
# ---------------------------------------------
def meta_predict(raw_features):
    # Normalisation
    x_scaled = scaler.transform(raw_features)

    # Extraction des métafeatures
    dt_conf, dt_margin = extract_meta_for_single(dt_model, x_scaled)
    rf_conf, rf_margin = extract_meta_for_single(rf_model, x_scaled)
    svm_conf, svm_margin = extract_meta_for_single(svm_model, x_scaled)
    nb_conf, nb_margin = extract_meta_for_single(nb_model, x_scaled)

    meta = np.array([[dt_conf, dt_margin,
                      rf_conf, rf_margin,
                      svm_conf, svm_margin,
                      nb_conf, nb_margin]])

    # Choix du meilleur modèle
    best_model = knn_meta.predict(meta)[0]

    # Prédiction finale
    if best_model == "DT":
        pred = dt_model.predict(x_scaled)[0]
    elif best_model == "RF":
        pred = rf_model.predict(x_scaled)[0]
    elif best_model == "SVM":
        pred = svm_model.predict(x_scaled)[0]
    elif best_model == "NB":
        pred = nb_model.predict(x_scaled)[0]

    return best_model, int(pred)

# ---------------------------------------------
# Flask setup
# ---------------------------------------------
app = Flask(__name__)

@app.route("/")
def home():
    return render_template("form.html")

@app.route("/predict", methods=["POST"])
def predict():
    # Récupération des 6 features depuis le formulaire
    features = [
        float(request.form["Time_Index"]),
        float(request.form["Temperature"]),
        float(request.form["Humidity"]),
        float(request.form["Light"]),
        float(request.form["CO2"]),
        float(request.form["HumidityRatio"])
    ]

    raw = np.array(features).reshape(1, -1)

    # Pipeline meta-learning
    best_model, prediction = meta_predict(raw)

    return render_template("result.html",
                           model=best_model,
                           prediction=prediction)

if __name__ == "__main__":
    app.run(debug=True)