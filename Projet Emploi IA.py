# -- coding: utf-8 --
# Projet emploi IA
# Clémentine, Matthieu, Sylvine




# import librairies
import pandas as pd
import numpy as np
from datetime import timedelta
from sklearn.feature_extraction.text import CountVectorizer
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestRegressor
from sklearn.metrics import r2_score

# Ouverture dataset
df = pd.read_json("https://raw.githubusercontent.com/SylvineDurand/Projet-emploi-IA-Cl-mentine-Matthieu-Sylvine/main/data.json")


# I. NETTOYAGE DONNEES
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


# 3. Creation colonne date
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

# 4. Creation colonne Société
#fonction pour mise en miniscule pour les noms de sociétés
def nom(df):
    df = df[2].lower()
    
    return df

df["Type de poste"] = df["Type de poste"].apply(nom)

data = df['lieu']
Lieu=[]
for i in data:
    list_lieu=[]
    for c in i[0] :
      list_lieu.append(c.replace('\n', ''))
    Lieu.append(' '.join(list_lieu))

data = Lieu
Lieu2 = []
for i in data :
     Lieu2.append(i.replace(" ", "")) 

data = Lieu2
LIEU = []
for i in data:
   LIEU.append(i.upper())

data = LIEU
LIEU2 = []
for i in data :
     LIEU2.append(i.replace("PARIS,", "").replace("Î", "I")
                  .replace(",FR", "").replace("É", "E")
                  .replace("(78)FACEGAREDESTQUENTINENYVELINE", "").replace("(92)-ILEDEFRANCE", "")
                  .replace("9ÈME(75)", "").replace("PARAYVIEILLEPOSTE", "PARAY-VIEILLE-POSTE")
                  .replace("09(75)", "").replace("20RUEHECTORMALOT(75-HEC)", "")
                  .replace("ILEDEFRANCE", "ILE-DE-FRANCE").replace(",ILE-DE-FRANCE","")
                  .replace("RUEILMALMAISON", "RUEIL-MALMAISON").replace("LADEFENSE", "LA-DEFENSE"))
df['lieu'] = LIEU2

# 5. Creation colonnes salaire
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

df["Salaire_minimum"] = df["Salaire"].apply(salaire_min)
df["Salaire_maximum"] = df["Salaire"].apply(salaire_max)
df=df.drop(["Salaire"],axis=1)

# 6. Creation colonne compétences
#join les éléments dans chaque liste de compétences en les changeant en string 
df['competences'] = [', '.join(map(str, l)) for l in df['competences']]


#retire les retours à la ligne des compétences 
def retour_a_la_ligne(value):
    return ''.join(value.splitlines())

df["competences"] = df["competences"].apply(retour_a_la_ligne)

# II. Préprocessing des données
# 1. Count vectorize compétences
#count-vectorize pour les compétences qui tranforme le nombre de mots en 1 utiliser dans un array
vectorizer = CountVectorizer()
X = vectorizer.fit_transform(df["competences"])
vectorizer.get_feature_names_out()


# III. Analyse exploratoire
# 1. Entreprises qui embauchent le plus
#compte le nombre de valeurs d'entreprise dans la colonne type de poste
pd.Series(df["Type de poste"]).value_counts()




df_clean = pd.DataFrame(list(zip(df["Date de publication"],df["Intitule"], df["competences"],df['lieu'],df["Salaire_minimum"],df["Salaire_maximum"],df['Type_poste'],df["Type de poste"])),columns =['Date_de_publication', 'Intitule',"Competences","Lieu","Salaire_minimum","Salaire_maximum","Type_poste","Société"])





df_clean.to_csv("df_clean.csv")