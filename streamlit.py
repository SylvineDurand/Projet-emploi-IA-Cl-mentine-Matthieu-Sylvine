#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Wed Feb  1 09:40:36 2023
@author: sylvine
"""

# base pour fichier stlit

import pandas as pd
# from datetime import timedelta
# import numpy as np

# # pour preprocessing
# from sklearn.impute import SimpleImputer
# from sklearn.compose import ColumnTransformer
from sklearn.feature_extraction.text import CountVectorizer
# from sklearn.preprocessing import OneHotEncoder

# # Pipeline and model
# from sklearn.pipeline import Pipeline
# from sklearn.model_selection import train_test_split
# from sklearn.linear_model import LinearRegression, Ridge, Lasso, ElasticNet
# from sklearn.ensemble import RandomForestRegressor 

# # Score of models
# from sklearn.metrics import r2_score, mean_squared_error

import streamlit as st

from fonctions import prepa_modele, test_modele, prediction_avec_input
# il va falloir changer le nom du script en un truc sans espaces
# peut être d'autres libraires à importer, à voir au fur et à mesure


#------------------------------------------------------------------------------
# Importer le df clean
df_model = pd.read_csv("df_clean.csv").dropna()
st.title("Notre appli du feu de dieu 🔥")

# 
#------------------------------------------------------------------------------
# Analyses exploratoires: les super graphes de Clémentine


#------------------------------------------------------------------------------
# Prédiction de salaire selon input de l'utilisateur
df_model = pd.read_csv("df_clean.csv").dropna()
#recup le df et cree le debut de la pipeline
prepa_modele()

#fonction integrant la pipeline complete avec options de target, modele, seed et est
test_modele()

# Partie où on récupère les inputs de l'utilisateur via menu déroulant, ou autre
# A MODIFIER POUR ST
st.header("Prédiction de salaire")
st.subheader("Veuillez sélectionner les paramètres")
# A MODIFIER POUR INCLURE UN CHOIX VIDE
#selectbox # A MODIFIER POUR INCLURE UN CHOIX VIDE
Input_intitule = st.selectbox(
    'Quel intitulé de poste recherchez-vous ?',
    (df_model["Intitule"].unique()))

Input_lieu = st.selectbox(
    'Dans quelle ville souhaitez-vous rechercher un poste ?',
    (df_model["Lieu"].unique()))

Input_contrat = st.selectbox(
    'Quel type de contrat recherchez-vous ?',
    (df_model["Type_poste"].unique()))

Input_societe = st.selectbox(
    'Dans quelle société souhaitez-vous travailler ?',
    (df_model["Société"].unique()))

#multiselect sur les compétences
vectorizer = CountVectorizer(tokenizer=lambda x: x.split(',')) 
vectorizer.fit_transform(df_model["Competences"])
X = vectorizer.get_feature_names_out()

Input_competences  = st.multiselect(
    'Quelles compétences possédez-vous?',
    X.tolist(),
    [' agile',' bases de données'])
Input_competences = ','.join(Input_competences)

Input = [Input_intitule, Input_competences, Input_lieu, Input_contrat, Input_societe]

# lacement de la prediction
predicted_min_sal, predicted_max_sal = prediction_avec_input(input = Input)   
# On recupère deux variables qu'on pourra intégrer dans le code st pour les afficher à l'utilisateur

st.subheader("Prédiction")
st.text("Votre salaire sera compris entre")
st.text(predicted_min_sal)
st.text("et")
st.text(predicted_max_sal )
st.text("€")

# st.text("Votre salaire sera compris entre", predicted_min_sal ," et ", predicted_max_sal,"€")

# affichage du résultat sous forme user friendly
# si c'était sous forme de graphique ça serait cool! Ca sera du bonus