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
import pandas_profiling
from streamlit_pandas_profiling import st_profile_report
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
df_complet = pd.read_csv("df_clean.csv")
df_complet_sans_dup = pd.read_csv("df_clean.csv").drop_duplicates()
df_model = pd.read_csv("df_clean.csv").dropna()
df_model_sans_dup = pd.read_csv("df_clean.csv").dropna().drop_duplicates()
st.title("Notre appli du feu de dieu 🔥")

#------------------------------------------------------------------------------
# Analyses exploratoires: les super graphes de Clémentine

st.header('Choisissez sur quel jeu de données vous voulez voir les analyses')

st.write ('Veux-tu toutes les données ou seulement les données non vides des salaires')

données = st.radio(
    "Sélectionnez les données dont vous voulez les analyses",
    ('Données complètes', 'Données complètes sans duplicat', 'Données non vides','Données non vides sans duplicats'))


st.subheader('`Streamlit profile report`')

if données == 'Données complètes' :
    pr = df_complet.profile_report()
    st_profile_report(pr)

if données == 'Données non vides' : 
    pr = df_model.profile_report()
    st_profile_report(pr)

if données == 'Données complètes sans duplicat' :
    pr = df_complet_sans_dup.profile_report()
    st_profile_report(pr)

if données == 'Données non vides sans duplicats' : 
    pr = df_model_sans_dup.profile_report()
    st_profile_report(pr)


df_intitule_salaire = pd.read_csv("df_poste2.csv")


data_intitule_min = pd.DataFrame({
    'index': df_intitule_salaire["Intitule"],
    'Salaire_minimum': df_intitule_salaire['Salaire_minimum'],
}).set_index('index')

st.bar_chart(data_intitule_min)


data_intitule_max = pd.DataFrame({
    'index': df_intitule_salaire["Intitule"],
    'Salaire_maximum': df_intitule_salaire['Salaire_maximum'],
}).set_index('index')

st.bar_chart(data_intitule_max)

df_cresults = pd.read_csv("df_cresults.csv")

data_df_cresults_min = pd.DataFrame({
    'index': df_cresults["Competence"],
    'Salaire_minimum': df_intitule_salaire['Salaire_minimum'],
}).set_index('index')

st.bar_chart(data_df_cresults_min)

data_df_cresults_max = pd.DataFrame({
    'index': df_cresults["Competence"],
    'Salaire_maximun': df_intitule_salaire['Salaire_maximum'],
}).set_index('index')

st.bar_chart(data_df_cresults_max)

df_salaire_moyen_par_competences = pd.read_csv("df_salaire_moyen.csv")

data_salaire_moyen_par_competences = pd.DataFrame({
    'index': df_salaire_moyen_par_competences["Competence"],
    'Salaire_moyen': df_salaire_moyen_par_competences['Salaire moyen'],
}).set_index('index')

st.bar_chart(data_salaire_moyen_par_competences)

df_contrat = pd.read_csv("df_contrat.csv")

data_contrat = pd.DataFrame({
    'index': df_contrat["Type_poste"],
    'Occurences': df_contrat['number of occurences'],
}).set_index('index')

st.bar_chart(data_contrat)


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