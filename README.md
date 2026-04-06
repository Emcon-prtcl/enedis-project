# Enedis Consumption Analysis

## Description

Ce projet a été réalisé dans le cadre du cours de data science.  
L’objectif est d’analyser des données de consommation électrique (Enedis) afin de :

- regrouper les clients selon leur comportement,
- prédire leur type (résidence principale ou secondaire),
- prévoir la consommation future,
- générer des courbes réalistes.

## Données

Les données utilisées proviennent d’Open Data Enedis (RES2 – 6-9 kVA).

Chaque client possède :
- des mesures de puissance toutes les 30 minutes,
- sur une période d’environ un an.

Les valeurs sont converties en énergie (kWh).


## Étapes du projet

### 1. Préparation des données

- nettoyage des données,
- conversion des dates,
- sélection d’un échantillon de clients pour simplifier l’analyse.



### 2. Exploration

- analyse des courbes moyennes,
- comparaison semaine / week-end,
- calcul de l’énergie journalière.



### 3. Feature engineering

Création de variables comme :

- moyenne de consommation,
- écart-type,
- consommation jour/nuit,
- variabilité journalière.


### 4. Clustering

Utilisation de K-Means (k=2) pour regrouper les clients.

Résultat :
- un groupe avec forte consommation (probablement RP),
- un groupe avec faible consommation (probablement RS).



### 5. Classification

Entraînement de modèles supervisés :

- Logistic Regression → accuracy ≈ 0.95  
- Random Forest → accuracy ≈ 0.80  

Les résultats montrent que les groupes sont bien séparés.



### 6. Forecasting

Modèle simple de régression linéaire basé sur :

- consommation des jours précédents,
- moyenne glissante.

Erreur moyenne (MAE) ≈ 3000 kWh.



### 7. Génération de courbes

Génération de profils de consommation en utilisant :

- la moyenne des clusters,
- un bruit aléatoire.

Les courbes générées sont cohérentes avec les profils observés.



## Dashboard

Un dashboard Streamlit a été développé pour visualiser les résultats :

### bash
streamlit run app.py
