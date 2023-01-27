# -- coding: utf-8 --
# Projet emploi IA
# Clémentine, Matthieu, Sylvine




# Import librairies
# pour nettoyage data
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
from sklearn.linear_model import LinearRegression
from sklearn.ensemble import RandomForestRegressor # a garder? Matthieu qu'en penses tu?

# Score of models
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
df['Lieu'] = LIEU2

# 5. Creation colonnes salaire
#fonction de salaire qui garde le salaire 
def salaire(df):
    if len(df) == 2:
        df = df[1]
        df = df.split("/ an")[0]
        df = df.split("/an")[0]
        df = df.replace("€","").replace(".","").replace(",00","").replace("\n","")

    else:
        df = np.nan
    return df

df["Salaire"] = df["lieu"].apply(salaire)


#fonctions qui font le salaire max et min en integer
def salaire_min(df):
    if not df :
        df = df
    else:
        df = str(df)
        df = df.split("-")[0]
    return df
def salaire_max(df):
    if not df :
        df = df
    else:
        df = str(df)
        df = df.split("-")[-1]
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




df_clean = pd.DataFrame(list(zip(df["Date de publication"],df["Intitule"], df["competences"],df['Lieu'],df["Salaire_minimum"],df["Salaire_maximum"],df['Type_poste'],df["Type de poste"])),columns =['Date_de_publication', 'Intitule',"Competences","Lieu","Salaire_minimum","Salaire_maximum","Type_poste","Société"])


df_clean.to_csv("df_clean.csv")

############################################################
# Creation de la pipeline


# exploration du df_clean juste pour Sylvine qui le découvre =)
type(df_clean)
df_clean.shape

df_clean.columns
df_clean.Salaire_minimum.describe() 
# Les salaires sont des objets!!!! il faudrait les convertir en autre chose, a voir plus tard ?


# 1. Drop les salaires NaN
'''
df_test = df_clean.dropna()
df_test.shape #230 ne marche pas, à régler !!!
df_test['Salaire_minimum']

df_test.dropna().describe()
df_test.dropna().Salaire_minimum.describe()
# ne marche paaaaaas, je pense que ça vient du fait que les colonnes salaire sont des objets

'''

#gros probleme pour drop na sur le df_clean, je travaille en réimportant le csv qui, lui , est propre!
df_model = pd.read_csv("df_clean.csv").dropna()
df_model.describe()
df_model.Salaire_minimum.describe() # salaire est bien un float, ouf


# 2. Définition des y et x 
y = df_model['Salaire_minimum']
#y = df_model['Salaire_maximum']
X = df_model.drop(columns=['Date_de_publication','Unnamed: 0','Salaire_minimum','Salaire_maximum'])
X.head()


# 3. Selection des variables categoriques sur lesquelles appliquer OneHot
column_cat = X.select_dtypes(include=['object']).columns.drop(['Competences'])


# 4. Creation des pipelines pour chaque type de variable
transfo_cat = Pipeline(steps=[
    ('', SimpleImputer(strategy='most_frequent')),
    ('onehot', OneHotEncoder(handle_unknown='ignore', sparse = False))
])
     
transfo_text = Pipeline(steps=[
    ('bow', CountVectorizer())
])
      
 
# 5. Class ColumnTransformer: appliquer chacune des pipelines sur les bonnes colonnes en nommant chaque étape
preparation = ColumnTransformer(
    transformers=[
        ('data_cat', transfo_cat , column_cat),
        #('data_artist', transfo_text , 'artist_name'),
        ('data_track', transfo_text , 'Competences')
    ])


# 6. Creation du modèle
# modele choisi = reg lineaire car target est quantitative
model = LinearRegression()


# Creation de la pipeline complète intégrant le modèle
pipe_model = Pipeline(steps=[('preparation', preparation),
                        ('model',model)])
pipe_model

# display le diagramme de la pipeline dans spyder
from sklearn import set_config
set_config(display='diagram') 
pipe_model # j'arrive pas à le display dans spyder

# Train test split
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

X_train # 34 lignes pour 5 colonnes
X_train.columns

# fit le model 
pipe_model.fit(X_train, y_train)

# predictions pour le model pré entrainé
y_pred = pipe_model.predict(X_test)
     
# Evaluer le modele
print("MAE:", r2_score(y_test, y_pred))

# OK j'ai un MAE ridicule mais au moins ça tourne ! 
# Il va falloir affiner et choisir un meileur modèle


# Observation de y pour savoir quel modèle appliquer
import matplotlib.pyplot as plt
plt.hist(y) # quelle distribution merdique


#########################################
# creation d'une fonction pour automatiser les tests de modèle, 
# avec possibilité de changer de seed, model, y (min ou max)

#|--------------------------------------------------------------------
#TOUT CA A LANCER AVANT LA FONCTION ---------------
# 2. Définition des x 
X = df_model.drop(columns=['Date_de_publication','Unnamed: 0','Salaire_minimum','Salaire_maximum'])
# 3. Selection des variables categoriques sur lesquelles appliquer OneHot
column_cat = X.select_dtypes(include=['object']).columns.drop(['Competences'])

# 4. Creation des pipelines pour chaque type de variable
transfo_cat = Pipeline(steps=[
    ('', SimpleImputer(strategy='most_frequent')),
    ('onehot', OneHotEncoder(handle_unknown='ignore', sparse = False))
])
     
transfo_text = Pipeline(steps=[
    ('bow', CountVectorizer())
])
        
# 5. Class ColumnTransformer: appliquer chacune des pipelines sur les bonnes colonnes en nommant chaque étape
preparation = ColumnTransformer(
    transformers=[
        ('data_cat', transfo_cat , column_cat),
        #('data_artist', transfo_text , 'artist_name'),
        ('data_track', transfo_text , 'Competences')
    ])
#--------------------------------------------------------------

def test_modele(target = "Minimum", seed = 42, modele = LinearRegression(), est = r2_score):
    if target == "Minimum":
        y = df_model['Salaire_minimum']
    elif target == "Maximum":
        y = df_model['Salaire_maximum']
    # else:
    #     print("la target n'est pas le salaire min ou max")
    #     exit()

    
    # Creation de la pipeline complète intégrant le modèle
    pipe_model = Pipeline(steps=[('preparation', preparation),
                            ('model',modele)])
    print(pipe_model)

    # Train test split
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=seed)

    # fit le model 
    pipe_model.fit(X_train, y_train)

    # predictions pour le model pré entrainé
    y_pred = pipe_model.predict(X_test)
         
    # Evaluer le modele
    print(f"target = {target}, modèle = {modele}, estimateur {est}: {est(y_test, y_pred)}")

# Appliquer la fonction pour différents types de modèles
# reg lineaire classique
test_modele()
test_modele("Maximum")


#reg regularisée
# Ridge

#Lasso

#ElasticNet



test_modele(modele = RandomForestRegressor())


from sklearn.linear_model import LinearRegression
from sklearn.ensemble import RandomForestRegressor 
