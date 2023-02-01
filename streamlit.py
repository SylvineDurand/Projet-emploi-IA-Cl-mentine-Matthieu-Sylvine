#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Wed Feb  1 09:40:36 2023

@author: sylvine
"""

# base pour fichier stlit

import pandas as pd
from Projet_Emploi_IA.py import prepa_modele, test_modele, prediction_avec_input
# il va falloir changer le nom du script en un truc sans espaces
# peut être d'autres libraires à importer, à voir au fur et à mesure


#------------------------------------------------------------------------------
# Importer le df clean

#------------------------------------------------------------------------------
# Analyses exploratoires: les super graphes de Clémentine


#------------------------------------------------------------------------------
# Prédiction de salaire selon input de l'utilisateur

#recup le df et cree le debut de la pipeline
prepa_modele()

#fonction integrant la pipeline complete avec options de target, modele, seed et est
test_modele()

# Partie où on récupère les inputs de l'utilisateur via menu déroulant, ou autre
# A MODIFIER POUR ST
Input_intitule = 'data analyst'
Input_competences = 'support, si' #une string avec compétences séparées par virgules
Input_lieu = 'PARIS'
Input_contrat = 'cdi'
Input_societe = 'selescope'

Input = [Input_intitule, Input_competences, Input_lieu, Input_contrat, Input_societe]

# lacement de la prediction
predicted_min_sal, predicted_max_sal = prediction_avec_input(input = Input)   
# On recupère deux variables qu'on pourra intégrer dans le code st pour les afficher à l'utilisateur


# affichage du résultat sous forme user friendly
# si c'était sous forme de graphique ça serait cool! Ca sera du bonus