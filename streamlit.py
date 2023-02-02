#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Wed Feb  1 09:40:36 2023
@author: sylvine
"""

import pandas as pd
import pandas_profiling
from streamlit_pandas_profiling import st_profile_report
from sklearn.feature_extraction.text import CountVectorizer
import streamlit as st
import matplotlib.pyplot as plt
from fonctions import prepa_modele, test_modele, prediction_avec_input


#------------------------------------------------------------------------------
# Importer le df clean
df_complet = pd.read_csv("df_clean.csv")
df_complet_sans_dup = pd.read_csv("df_clean.csv").drop_duplicates()
df_model = pd.read_csv("df_clean.csv").dropna()
df_model_sans_dup = pd.read_csv("df_clean.csv").dropna().drop_duplicates()

#------------------------------------------------------------------------------
st.title("Marché de l'emploi data en Ile-de-France")
st.title("Une appli du feu de dieu 🔥")

tab1, tab2, tab3 = st.tabs(["Description des données", "Quelques graphes d'intérêt", "Prédiction de salaire"])

# Description des données avec st profile report
with tab1: 
    st.header('Choisissez quel jeu de données vous souhaitez découvrir')
        
    données = st.radio(
        "Voulez-vous toutes les données ou seulement les données pour lesquelles les salaires sont disponibles?",
        ('Données complètes', 'Données complètes sans duplicat', 'Données non vides','Données non vides sans duplicats'))
    
    
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


# Analyses exploratoires: les super graphes de Clémentine
with tab2:  
    # Salaire minimum par intitulé de poste
    st.subheader("Salaire par intitulé de poste")
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
    
    
    # Salaire minimum et maximum par compétence
    st.subheader("Salaire par compétence")
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
    
    # Salaire moyen par compétence
    df_salaire_moyen_par_competences = pd.read_csv("df_salaire_moyen.csv")
    
    data_salaire_moyen_par_competences = pd.DataFrame({
        'index': df_salaire_moyen_par_competences["Competence"],
        'Salaire_moyen': df_salaire_moyen_par_competences['Salaire moyen'],
    }).set_index('index')
    
    st.bar_chart(data_salaire_moyen_par_competences)
    
    # barplot type de contrat
    st.subheader("Répartition des types de contrats")
    df_contrat = pd.read_csv("df_contrat.csv")
    
    data_contrat = pd.DataFrame({
        'index': df_contrat["Type_poste"],
        'Occurences': df_contrat['number of occurences'],
    }).set_index('index')
    
    st.bar_chart(data_contrat)


#------------------------------------------------------------------------------
# Prédiction de salaire selon input de l'utilisateur
with tab3:
    df_model = pd.read_csv("df_clean.csv").dropna()
    #recup le df et cree le debut de la pipeline
    prepa_modele()
    
    #fonction integrant la pipeline complete avec options de target, modele, seed et est
    test_modele()
    
    # On récupère les inputs de l'utilisateur
    st.header("Veuillez sélectionner les paramètres")
    
    Input_intitule = st.selectbox(
        'Quel intitulé de poste recherchez-vous ?',
        ([""] + df_model["Intitule"].unique().tolist()))
    
    Input_lieu = st.selectbox(
        'Dans quelle ville souhaitez-vous rechercher un poste ?',
        ([""] + df_model["Lieu"].unique().tolist()))
    
    Input_contrat = st.selectbox(
        'Quel type de contrat recherchez-vous ?',
        ([""] + df_model["Type_poste"].unique().tolist()))
    
    Input_societe = st.selectbox(
        'Dans quelle société souhaitez-vous travailler ?',
        ([""] + df_model["Société"].unique().tolist()))
    
    
    vectorizer = CountVectorizer(tokenizer=lambda x: x.split(', ')) 
    vectorizer.fit_transform(df_model["Competences"])
    liste_competences = vectorizer.get_feature_names_out()
    
    Input_competences = st.multiselect(
        'Quelles compétences possédez-vous?',
        [""] + liste_competences.tolist(),
        [""])
    Input_competences = ','.join(Input_competences)
    
    Input = [Input_intitule, Input_competences, Input_lieu, Input_contrat, Input_societe]
    
    # lancement de la prediction
    st.subheader("Le salaire prédit est de:")
    
    #prediction pour salaire moyen = aucune donnée en entrée
    moyenne_minimum, moyenne_maximum = prediction_avec_input()  
    
    # On recupère salaires prédits
    predicted_min_sal, predicted_max_sal = prediction_avec_input(input = Input)   
    
    #calcul de la diff avec moyen
    diff_min = round(predicted_min_sal - moyenne_minimum, 2)
    diff_max = round(predicted_max_sal - moyenne_maximum, 2)
    
    #transfo en string pour affichage
    predicted_min_sal_str = str(predicted_min_sal)+" €"
    predicted_max_sal_str = str(predicted_max_sal)+" €"
    diff_min = str(diff_min)+ " € par rapport au salaire prédit moyen"
    diff_max = str(diff_max)+ " € par rapport au salaire prédit moyen"
    
    # affichage
    col1, col2 = st.columns(2)
    col1.metric("Salaire minimum", predicted_min_sal_str, diff_min)
    col2.metric("Salaire maximum", predicted_max_sal_str, diff_max)
    
    st.subheader("Positionnement des salaires prédits par rapport aux salaires observés")
    #histogrammes avec barre verticale marquant le salaire prédit
    fig, (ax1, ax2) = plt.subplots(1, 2)
    
    ax1.hist(df_model["Salaire_minimum"],  bins=20, color ="tomato" ) 
    ax1.axvline(predicted_min_sal, color='firebrick', linestyle='dashed', linewidth=1)
    ax1.title.set_text('Salaire minimum')
    ax1.tick_params(labelrotation=90)
    ax1.axis(ymin=0, ymax=14.5, xmin=29000, xmax=101000)
    
    ax2.hist(df_model["Salaire_maximum"], bins=20, color ="tomato") 
    ax2.axvline(predicted_max_sal, color='firebrick', linestyle='dashed', linewidth=1)
    ax2.title.set_text('Salaire maximum')
    ax2.tick_params(labelrotation=90)
    ax2.axis(ymin=0, ymax=14.5, xmin=29000, xmax=101000)
    
    st.pyplot(fig)
    
    if st.button("Etes-vous content de l'appli?"):
        st.balloons()
        








