# Projet emploi IA
# Clémentine, Matthieu, Sylvine




# import librairies
import pandas as pd
from datetime import timedelta
from sklearn.feature_extraction.text import CountVectorizer


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

#fonction pour la date et l'apply avec 
def date(df):
    df = df.strip("").strip("\n")
    df = df.split(" ")
    if df[-1] == "hier" or df[-1] == "heures":
        df = pd.to_datetime("2023/01/14")
    else:
        df = df[-2:]
        if df[-1] == "mois":
            x= int(df[-2])
            temps=x*31
            df = pd.to_datetime("2023/01/15")-timedelta(days=temps)
        else:
            temps=int(df[-2])
            df = pd.to_datetime("2023/01/15")-timedelta(days=temps)
    
        
    return df

df["Date de publication"] = df["Date de publication"].apply(date)

#fonction pour mise en miniscule pour les noms de sociétés
def nom(df):
    df = df[2].lower()
    
    return df

df["Type de poste"] = df["Type de poste"].apply(nom)



#fonction de salaire qui garde le salaire 
def salaire(df):
    if len(df) == 2:
        df = df[1]
        df = df.split("/ an")[0]
        df = df.split("/an")[0]
        df = df.replace("€","").replace(".","").replace(",00","")
          
    else:
        df = "NaN"
    
    return df

df["Salaire"] = df["lieu"].apply(salaire)


#fonctions qui font le salaire max et min en integer
def salaire_min(df):
    df = df.split("-")[0]
    if df != "NaN":
        df = int(df)
    return df
def salaire_max(df):
    df = df.split("-")[-1]
    if df != "NaN":
        df = int(df)
    return df

df["Salaire_minimun"] = df["Salaire"].apply(salaire_min)
df["Salaire_maximun"] = df["Salaire"].apply(salaire_max)
df=df.drop(["Salaire"],axis=1)


#join les éléments dans chaque liste de compétences en les changeant en string 
df['competences'] = [', '.join(map(str, l)) for l in df['competences']]


#retire les retours à la ligne des compétences 
def retour_a_la_ligne(value):
    return ''.join(value.splitlines())

df["competences"] = df["competences"].apply(retour_a_la_ligne)


#count-vectorize pour les compétences qui tranforme le nombre de mots en 1 utiliser dans un array
vectorizer = CountVectorizer()
X = vectorizer.fit_transform(df["competences"])
vectorizer.get_feature_names_out()

#compte le nombre de valeurs d'entreprise dans la colonne type de poste
pd.Series(df["Type de poste"]).value_counts()


print(X)



df.to_csv("test.csv")