# Projet emploi IA
# Clémentine, Matthieu, Sylvine




# import librairies
import pandas as pd

# Ouverture dataset
df = pd.read_json("https://raw.githubusercontent.com/SylvineDurand/Projet-emploi-IA-Cl-mentine-Matthieu-Sylvine/main/data.json")


# NETTOYAGE DONNEES
type(df.iloc[[0],[0]]) # est un df 
type(df.iloc[[0],[0]].values[0]) # np.ndarray
type(df.iloc[[0],[0]].values[0][0]) # list
type(df.iloc[[0],[0]].values[0][0][0]) # str

# Création de la colonne Intitulé
# pour test sur 1 seul élément
test = df.iloc[[0],[0]].values[0][0][0]
type(test)
test

# Liste de caractères �  enlever
to_remove = ["[","]","'","(",")", 
             "H/F","F/H","h/f","f/h"
             ]

# pour test sur 1 seul élément
test = df.iloc[[0],[0]].values[0][0][0]
type(test)
test  

for i in to_remove:
    test = test.replace(i,"")
# on split sur le tiret
test = test.split(" - ")[0]
# on strip espace et antislash n
test.strip("").strip("\n")    

# Fonction �  appliquer sur la colonne "intitulé de poste" pour nettoyer
def func_intitule(x):
    x = x[0]
    x = x.split(" - ")[0]
    x = x.split(" – ")[0]
    x = x.split(" / ")[0]
    for i in to_remove:
        x = x.replace(i,"")    
    x = x.strip("").strip("\n").strip("-")
    return x
    
func_intitule(test)
    
df["Intitule"] = df["Intitulé du poste"].apply(func_intitule)    
    
# reste �  faire: 
    # - index 13 = alternant en CDI?
    # - 34 alternance (contrat pro ok)
    # - 49 stage chargé projet en CDI
    # - 55 commence par chiffre
    # - 139 2019 moa data etc*
    # - 185 Assistant comptable???


    

