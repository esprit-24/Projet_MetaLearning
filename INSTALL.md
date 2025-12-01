# 📦 INSTALLATION — Guide pour exécuter le projet Meta-Learning (Occupancy)

Ce guide permet à n’importe quel utilisateur d’installer le projet et de lancer l’application Flask.

---

# ✅ 1. Cloner le projet

```bash
git clone https://github.com/TON_REPO/projet-datamining.git
cd projet-datamining
```

> Remplacer `TON_REPO` par votre URL Git réelle.

---

# 🧰 2. Créer un environnement Conda

```bash
conda create -n meta_learning python=3.10 -y
conda activate meta_learning
```

---

# 📦 3. Installer les dépendances

```bash
pip install numpy pandas scikit-learn flask matplotlib seaborn joblib jupyter
```

---

# 🗂 4. Vérifier la structure du projet

```
project/
├── data/
├── models/
├── flask_app/
└── notebook/
```

⚠️ Le dossier **models/** doit contenir :

```
scaler.joblib
dt_model.joblib
rf_model.joblib
svm_model.joblib
nb_model.joblib
meta_knn_model.joblib
```

---

# 🚀 5. Lancer Flask

```bash
cd flask_app
python app.py
```

Puis ouvrir :  
➡️ http://127.0.0.1:5000

---

# 🧪 6. Utilisation

Saisir les champs :  
Time_Index, Temperature, Humidity, Light, CO2, HumidityRatio  
→ Puis cliquer **Prédire**.

---

# ✔ Fin du guide
