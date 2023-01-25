# Projet emploi IA
# Clémentine, Matthieu, Sylvine




# import librairies
import pandas as pd
import numpy as np

# Ouverture dataset
df = pd.read_json("https://raw.githubusercontent.com/SylvineDurand/Projet-emploi-IA-Cl-mentine-Matthieu-Sylvine/main/data.json")


# I. NETTOYAGE DONNEES
# Exploration du type de données
type(df.iloc[[0],[0]]) # est un df 
type(df.iloc[[0],[0]].values[0]) # np.ndarray
type(df.iloc[[0],[0]].values[0][0]) # list
type(df.iloc[[0],[0]].values[0][0][0]) # str

# 1. Création de la colonne Intitule
# Liste d'éléments à retirer notamment au milieu des strings
to_remove = ["[","]","'","(",")", 
             "H/F","F/H","h/f","f/h"]

# Liste d'éléments à retirer au début des strings
debut_to_remove = ["2019-moa-data-30912",
                   "2018-788",
                   "2016-433"]

contract = ["stage",
            "stagiaire",
            "cdi",
            "apprenti",
            "alternant",
            "alternance"]               

## Première fonction enlève les caractères sans utilité
def f1(x):
    x = x[0]
    x = x.lower()
    for i in to_remove:
        x = x.replace(i,"")    
    for i in debut_to_remove:
        x = x.replace(i,"")      
    x = x.strip(" \n-")
    return x

# créer colonne intermediaire
df["Int1"] = df["Intitulé du poste"].apply(f1)  

# 2e fonction récup l'info alternance pour utilisation ultérieure
def f2(x):
    y = ""
    for i in contract:
        if x.startswith(i):
            y =  y + i
    return y   

# créer colonne intermediaire, sera utilisée plus tard
df["Int2"] = df["Int1"].apply(f2)  

# 3e fonction enlève info stage etc, enleve caractères restant au début, récup première partie
def f3(x):
    for i in contract:
        x = x.replace(i,"") 
    x = x.strip(" -–:e")
    x = x.split(" - ")[0]
    x = x.split(" – ")[0]
    x = x.split(" / ")[0]
    return x

# Créer colonne finale    
df["Intitule"] = df["Int1"].apply(f3)  


# 2. Creation colonne Type_poste
def type_contrat(x):
    y = "raté"
    if x[0] == '':
        y = x[1][-1].split(" - ")[0],
    elif x[0] == 'stage':
        y = "Stage"
    elif x[0] == 'stagiaire':
        y = "Stage"
    elif x[0] == 'alternant':
        y = "Alternance"
    elif x[0] == 'alternance':
        y = "Alternance"
    elif x[0] == 'apprenti':
        y = "Apprentissage"
    elif x[0] == 'cdi':
        y = "cdi"
    return y

df['Type_poste'] = df[["Int2","Type de poste"]].apply(lambda x : type_contrat(x), axis = 1)
type(df.Type_poste[0]) # est un tuple, il faut le convertir

# Conversion des tuples en str
def convertTuple(tup):
    st = ''.join(map(str, tup))
    return st
 
df['Type_poste'] = df['Type_poste'].apply(lambda x : convertTuple(x))
type(df.Type_poste)  # series

# remove les \n
df['Type_poste'] = df['Type_poste'].apply(lambda x : x.strip("\n",).lower())


# Nettoyage modification des intitulés de poste au cas par cas
    # a voir plus tard si j'ai la foi

# retirer des lignes non pertinentes?
    # - 185 Assistant comptable???


