# 🧠 Projet Data Mining — Meta-Learning pour la Prédiction d’Occupation

Ce projet a été réalisé dans le cadre du module **Data Mining (M2 GDIL)**.  
L’objectif principal est de concevoir un système intelligent capable de prédire si une salle est **occupée (1)** ou **non occupée (0)** à partir de données de capteurs IoT.

---

## 📁 Structure du Projet

```
project/
│
├── data/
│   ├── train.csv
│   ├── test.csv
│
├── models/
│   ├── scaler.joblib
│   ├── dt_model.joblib
│   ├── rf_model.joblib
│   ├── svm_model.joblib
│   ├── nb_model.joblib
│   └── meta_knn_model.joblib
│
├── notebook/
│   └── projet_datamining.ipynb
│
├── flask_app/
│   ├── app.py
│   └── templates/
│       ├── form.html
│       └── result.html
│
└── README.md
```

---

## 🎯 Objectifs

- Former 4 modèles de base (DT, RF, SVM, NB)  
- Extraire 8 métafeatures  
- Entraîner un méta-modèle KNN (k=5)  
- Construire un pipeline de sélection dynamique du meilleur modèle  
- Évaluer le pipeline final sur test.csv  
- Développer une application Flask pour tester les prédictions

---

## 🧩 Modèles de Base

Chaque modèle a été évalué avec :

- Accuracy  
- Precision  
- Recall  
- F1-score  
- Matrice de confusion  

---

## 🔬 Extraction des Métafeatures

Pour chaque modèle :

- **Confiance** : max(P(0), P(1))  
- **Margin** : abs(P(1) - P(0))  

Total : **8 métafeatures**.

---

## 🧠 Meta-Modèle (KNN)

Le méta-modèle apprend :

```
métafeatures → meilleur modèle (DT, RF, SVM, NB)
```

En cas d’égalité → modèle avec meilleure confiance.  
En cas d’erreur globale → fallback au modèle le plus confiant.

---

## 🔗 Pipeline Final

1. Normalisation  
2. Extraction des 8 métafeatures  
3. Sélection du meilleur modèle via KNN  
4. Prédiction finale via le modèle sélectionné  

---

## 📊 Évaluation Finale sur test.csv

Métriques évaluées :

- Accuracy  
- Precision  
- Recall  
- F1-score  
- Matrice de confusion finale  

---

## 🌐 Application Flask

Permet de saisir les features et d’obtenir :

- Le modèle sélectionné  
- La prédiction finale  

Démarrage :

```bash
cd flask_app
python app.py
```

Naviguer vers :

```
http://127.0.0.1:5000
```

---

## 💾 Sauvegarde

Les modèles sont sauvegardés en `.joblib` dans `/models`.

---

## 🎥 Vidéo de Démonstration (3 min)

La vidéo montre :

- Le notebook  
- Les modèles  
- Le pipeline  
- Flask  
- Une prédiction réelle  

---

## 📝 Auteur

Projet réalisé dans le cadre du Master 2 GDIL.
