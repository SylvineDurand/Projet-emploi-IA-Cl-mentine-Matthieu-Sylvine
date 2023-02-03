#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Wed Feb  1 11:17:18 2023

@author: sylvine
"""


import pandas as pd
from datetime import timedelta
import numpy as np

# pour preprocessing
from sklearn.impute import SimpleImputer
from sklearn.compose import ColumnTransformer
from sklearn.feature_extraction.text import CountVectorizer
from sklearn.preprocessing import OneHotEncoder

# Pipeline and model
from sklearn.pipeline import Pipeline
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LinearRegression, Ridge, Lasso, ElasticNet
from sklearn.ensemble import RandomForestRegressor 

# Score of models
from sklearn.metrics import r2_score, mean_squared_error


####################################################
def prepa_modele():
    # 1. Drop les salaires NaN
    # probleme pour drop na sur le df_clean, on travaille en réimportant le csv
    global df_model
    df_model = pd.read_csv("df_clean.csv").dropna()
    
    # 2. Définition des x 
    global X
    X = df_model.drop(columns=['Date_de_publication','Salaire_minimum','Salaire_maximum'])
    
    # 3. Selection des variables categoriques sur lesquelles appliquer OneHot
    column_cat = X.select_dtypes(include=['object']).columns.drop(['Competences'])
    
    # 4. Creation des pipelines pour chaque type de variable
    transfo_cat = Pipeline(steps=[
        ('', SimpleImputer(strategy='most_frequent')),
        ('onehot', OneHotEncoder(handle_unknown='ignore', sparse = False))
    ])
         
    transfo_text = Pipeline(steps=[
        ('bow', CountVectorizer(tokenizer=lambda x: x.split(', ')) )
    ])
    
    # 5. Class ColumnTransformer: appliquer chacune des pipelines sur les bonnes colonnes en nommant chaque étape
    global preparation
    preparation = ColumnTransformer(
        transformers=[
            ('data_cat', transfo_cat , column_cat),
            #('data_artist', transfo_text , 'artist_name'),
            ('data_text', transfo_text , 'Competences')
        ])

# modele = LinearRegression()
def test_modele(target = "Minimum", seed = 42, modele = LinearRegression(), est = r2_score):
    if target == "Minimum":
        y = df_model['Salaire_minimum']
    elif target == "Maximum":
        y = df_model['Salaire_maximum']
    
    # Creation de la pipeline complète intégrant le modèle
    pipe_model = Pipeline(steps=[('preparation', preparation),
                            ('model',modele)])

    # Train test split
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=seed)

    # fit le model 
    pipe_model.fit(X_train, y_train)

    # predictions pour le model pré entrainé
    y_pred = pipe_model.predict(X_test)
         
    # Evaluer le modele
    print(f"Target = Salaire {target}, modèle = {modele}, seed = {seed}")
    print(f"Score du modèle sur le training: {pipe_model.score(X_train, y_train)}") 
    if est == mean_squared_error:
        print(f"Estimateur {est}: {est(y_test, y_pred,squared = False)}")
    else:
        print(f"Estimateur {est}: {est(y_test, y_pred)}")
    
    return pipe_model

def prediction_avec_input(input = ['','', '', '', ''], modele = LinearRegression(), seed = 42, est = r2_score):
    df_input = pd.DataFrame(np.array([input]),
                       columns=['Intitule', 'Competences', 'Lieu', 'Type_poste', 'Société'])
    modele_min = test_modele("Minimum", modele = modele, seed = seed, est = est)
    minimum_predit = modele_min.predict(df_input)[0]
    print("---------------------")

    modele_max = test_modele("Maximum", modele = modele, seed = seed, est = est)
    maximum_predit = modele_max.predict(df_input)[0]
    print("---------------------")
    
    print(f"Pour les caractéristiques suivantes : {input}")
    print(f"Le salaire sera compris entre {round(minimum_predit,2)} € et {round(maximum_predit, 2)} €") 
    
    predicted_min_sal = round(minimum_predit,2)
    predicted_max_sal = round(maximum_predit,2)
    return predicted_min_sal, predicted_max_sal
    