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
from sklearn.linear_model import LinearRegression, Ridge, Lasso, ElasticNet
from sklearn.ensemble import RandomForestRegressor 

# Score of models
from sklearn.metrics import r2_score, mean_squared_error

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
def Intitule_1(x):
    x = x[0]
    x = x.lower()
    for i in to_remove:
        x = x.replace(i,"")    
    for i in debut_to_remove:
        x = x.replace(i,"")      
    x = x.strip(" \n-")
    return x

# créer colonne intermediaire
df["Int1"] = df["Intitulé du poste"].apply(Intitule_1)  

# 2e fonction récup l'info alternance pour utilisation ultérieure
def Intitule_2(x):
    y = ""
    for i in contract:
        if x.startswith(i):
            y =  y + i
    return y   

# créer colonne intermediaire, sera utilisée plus tard
df["Int2"] = df["Int1"].apply(Intitule_2)  

# 3e fonction enlève info stage etc, enleve caractères restant au début, récup première partie
def Intitule_3(x):
    for i in contract:
        x = x.replace(i,"") 
    x = x.strip(" -–:e")
    x = x.split(" - ")[0]
    x = x.split(" – ")[0]
    x = x.split(" / ")[0]
    return x

# Créer colonne finale    
df["Intitule"] = df["Int1"].apply(Intitule_3)  


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
# fonction de salaire qui garde le salaire 
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
df['competences'] = df['competences'].apply([lambda row: row.replace("é","e")])

############################################################
# II. Préprocessing des données
# 1. Count vectorize compétences
#count-vectorize pour les compétences qui tranforme le nombre de mots en 1 utiliser dans un array
vectorizer = CountVectorizer(tokenizer=lambda x: x.split(',')) 
X = vectorizer.fit_transform(df["competences"])
vectorizer.get_feature_names_out()


############################################################
# III. Analyse exploratoire
# 1. Entreprises qui embauchent le plus
#compte le nombre de valeurs d'entreprise dans la colonne type de poste
pd.Series(df["Type de poste"]).value_counts()


df_clean = pd.DataFrame(list(zip(df["Date de publication"],df["Intitule"], df["competences"],df['Lieu'],df["Salaire_minimum"],df["Salaire_maximum"],df['Type_poste'],df["Type de poste"])),columns =['Date_de_publication', 'Intitule',"Competences","Lieu","Salaire_minimum","Salaire_maximum","Type_poste","Société"])
df_clean.to_csv("df_clean.csv" , index=None)

# III. Analyse exploratoire
# 1. Les compétences les plus recherchées
#tableau des occurences de chaque competence à partir de la vectorisation
count_array = X.toarray()
df_competences = pd.DataFrame(data=count_array, columns = vectorizer.get_feature_names())
print(df_competences)
df_competences.shape
#Selection de la somme des occurences pour chaque competence & sort des plus demandées
df_competences2=df_competences.append(df_competences.sum(), ignore_index=True)
print(df_competences2)
df_competences3=df_competences2.loc[[230]].T
df_competences3.reset_index(inplace=True)
df_competences3.columns =['competences','number of occurences']
df_competences3.sort_values(by='number of occurences', ascending=False, inplace=True)
#Selection des competence qui aparaissent plus de 20 fois
df_competencesfinal = df_competences3.loc[df_competences3['number of occurences'] > 20]
df_competencesfinal.shape
#creation du plot pour les competences demandées plus de 10 fois
import matplotlib.pyplot as plt
x = df_competencesfinal['competences']
y = df_competencesfinal['number of occurences']
xlabel = df_competencesfinal['competences']

plt.bar(x, y, width = 0.6)
plt.xticks(xlabel, rotation=90)
plt.xlabel('Competence')
plt.ylabel('Number of occurence')
plt.title('Competences recherchées')
plt.show()

# 2. Les entreprises qui recrutent le plus
#nettoyage de la colonne entreprise
df_societe=pd.read_csv('df_clean.csv') 
data = df_societe['Société']
societe=[]
for i in data:
       societe.append(i.replace(' ', '_').replace(' ', '').replace("é", "e")
                      .replace('-','_').replace('&','_').upper())
data = societe
SOCIETE = []
for i in data:
   SOCIETE.append(i.upper())

df_societe2 = pd.DataFrame(SOCIETE, columns = ['societe'])

#recurrence des sociéte dans le dataset
df_societe_count = df_societe2.value_counts(ascending=False)
df_societefinal = df_societe_count.to_frame().reset_index()
df_societefinal.columns = ['societe', 'number of occurence']


#creation du nombre d'occurence
df_societefinal2 = df_societefinal.loc[df_societefinal['number of occurence'] > 5]
df_societefinal2.shape

#creation du plot pour les competences demandées plus de 5 fois

x_societe = df_societefinal2['societe']
y_societe = df_societefinal2['number of occurence']
xlabel = df_societefinal2['societe'] 

plt.bar(x_societe, y_societe, width = 0.6)
plt.xticks(xlabel, rotation=90)
plt.xlabel('societe')
plt.ylabel('Number of occurence')
plt.title('Sociétés recrutant le plus')
plt.show()

  
# 3. Les postes les mieux payés
df_poste=pd.read_csv('df_clean.csv') 
df_poste.columns
df_poste2 = df_poste.dropna().groupby(['Intitule']).median()
df_poste2.sort_values('Salaire_minimum', ascending=False, inplace=True)
df_poste2

df_poste2.to_csv("df_poste2.csv")



# 4. Les compétences les mieux payés
df_competence=pd.read_csv('df_clean.csv') 
df_competence.columns
df_competence2 = df_competence.dropna()
Test = df_competence2["Competences"].str.split(", ", n = -1, expand = True)
df_competence2bis = pd.concat([Test, df_competence2.reindex(Test.index)], axis=1)
df_competence2bis.rename(columns={0: "a", 1: "b", 2: "c", 3: "d", 4: "e"}, inplace=True)
df_competence2bis.columns
df_competence2bis2 = df_competence2bis.drop(['Date_de_publication', 'Intitule', 'Competences', 'Lieu', 'Type_poste', 'Société'], axis=1)
df_c0 = df_competence2bis2.drop(['b', 'c', 'd', 'e'], axis=1)
df_c1 = df_competence2bis2.drop(['a', 'c', 'd', 'e'], axis=1).rename(columns = {"b": "a"})
df_c2 = df_competence2bis2.drop(['a', 'b', 'd', 'e'], axis=1).rename(columns = {"c": "a"})
df_c3 = df_competence2bis2.drop(['a', 'b', 'c', 'e'], axis=1).rename(columns = {"d": "a"})
df_c4 = df_competence2bis2.drop(['a', 'b', 'c', 'd'], axis=1).rename(columns = {"e": "a"})
frames = [df_c0, df_c1, df_c2, df_c3, df_c4]
df_cfinal = pd.concat(frames)
df_cfinal.shape
df_cfinal.drop_duplicates(inplace=True)
df_cfinal.shape
df_cfinal.dropna(inplace=True)

df_cresults = df_cfinal.groupby(['a']).median().reset_index()
df_cresults.sort_values('Salaire_minimum', ascending=False, inplace=True)
df_cresults.rename(columns = {"a": "Competence"}, inplace=True)

df_cresults.to_csv("df_cresults.csv")

# 5. Le salaire moyen par compétence
dfcresults_mean = df_cresults.mean(axis=1)
dfcresults_mean2=dfcresults_mean.to_frame(name = 'Salaire moyen') 
df_salairemoyen = pd.concat([df_cresults, dfcresults_mean2.reindex(df_cresults.index)], axis=1)
df_salairemoyen.to_csv("df_salaire_moyen.csv")

# 6. Les types de contrat
df_contrat=pd.read_csv('df_clean.csv')

vectorizer = CountVectorizer(tokenizer=lambda x: x.split('/'))
X = vectorizer.fit_transform(df_contrat["Type_poste"])
vectorizer.get_feature_names_out()
print(X)
print(vectorizer.get_feature_names_out())

count_array = X.toarray()
df_contrat2 = pd.DataFrame(data=count_array, columns = vectorizer.get_feature_names())
print(df_contrat2)
df_contrat2.shape
df_contrat3 = df_contrat2.append(df_contrat2.sum(), ignore_index=True)
df_contrat4=df_contrat3.loc[[230]].T
df_contrat4.reset_index(inplace=True)
df_contrat4.columns =['Type_poste','number of occurences']
df_contrat4.sort_values(by='number of occurences', ascending=False, inplace=True)

x = df_contrat4['Type_poste']
y = df_contrat4['number of occurences']
xlabel = df_contrat4['Type_poste']

plt.bar(x, y, width = 0.6)
plt.xticks(xlabel, rotation=90)
plt.xlabel('Type_poste')
plt.ylabel('Number of occurence')
plt.title('differents type de postes')
plt.show()

df_contrat4.to_csv("df_contrat.csv")


############################################################
# IV. Creation de la pipeline et application des modèles

#Creation fonction pour préparer le modèle 
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
        ('bow', CountVectorizer(tokenizer=lambda x: x.split(',')) )
    ])
    
    # 5. Class ColumnTransformer: appliquer chacune des pipelines sur les bonnes colonnes en nommant chaque étape
    global preparation
    preparation = ColumnTransformer(
        transformers=[
            ('data_cat', transfo_cat , column_cat),
            #('data_artist', transfo_text , 'artist_name'),
            ('data_text', transfo_text , 'Competences')
        ])

prepa_modele()

# Creation d'une fonction pour automatiser les tests de modèle
# avec possibilité de changer de seed, model, y (min ou max)
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
    print(f"Estimateur {est}: {est(y_test, y_pred)}")
    
    return pipe_model

# Appliquer la fonction pour différents types de modèles
# reg lineaire classique
test_modele()
test_modele("Maximum")
test_modele(est = mean_squared_error)
test_modele("Maximum", est = mean_squared_error)

# reg regularisée
# Ridge
test_modele(modele = Ridge())
test_modele(modele = Ridge(), est = mean_squared_error)
test_modele("Maximum", modele = Ridge())
test_modele("Maximum", modele = Ridge(), est = mean_squared_error)

#Lasso
test_modele(modele = Lasso(max_iter=10000))
test_modele(modele = Lasso(max_iter=10000),est = mean_squared_error)
test_modele("Maximum",modele = Lasso(max_iter=10000))
test_modele("Maximum",modele = Lasso(max_iter=10000),est = mean_squared_error)

#ElasticNet
test_modele(modele = ElasticNet()) 
test_modele(modele = ElasticNet(),est = mean_squared_error) 
test_modele("Maximum", modele = ElasticNet()) 
test_modele("Maximum", modele = ElasticNet(),est = mean_squared_error) 

# Random forest
test_modele(modele = RandomForestRegressor())
test_modele(modele = RandomForestRegressor(),est = mean_squared_error)
test_modele("Maximum", modele = RandomForestRegressor())
test_modele("Maximum", modele = RandomForestRegressor(),est = mean_squared_error)
# Resultats dependent du hasard avec ce type de modèle, pas robuste du tout quand peu de données


############################################################
# V. Prediction salaire min et max avec des inputs de l'utilisateur

# 1. Recuperation d'un input depuis l'utilisateur pour chaque feature du modele
Input_intitule = 'data analyst'
Input_competences = 'support, si' #une string avec compétences séparées par virgules
Input_lieu = 'PARIS'
Input_contrat = 'cdi'
Input_societe = 'selescope'

# 2. Concaténation en une liste
Input = [Input_intitule, Input_competences, Input_lieu, Input_contrat, Input_societe]

# Fonction de prédiction
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
    
# 3. Lancement de la fonction de prediction    
prediction_avec_input() # si pas d'input de l'utilisateur
prediction_avec_input(input = Input)   

predicted_min_sal, predicted_max_sal = prediction_avec_input(input = Input)   



# notes de fin:
# si on poursuit la démarche sur du clustering, il suffit de modifier la pipeline pour que 
# le modele s'entraine juste sur des features
# difficulté sera surtout de comprendre où sont les variables dedans. 

exit()

#############################################
# 6. Bonus nuage de mots pour illustration
# CHANTIER EN COURS !!!

from wordcloud import WordCloud
import matplotlib.pyplot as plt
import numpy as np
from PIL import Image

vectorizer = CountVectorizer(tokenizer=lambda x: x.split(','))
X = vectorizer.fit_transform(df["competences"])
vectorizer.get_feature_names_out() 


corpus = " ".join([ligne for ligne in vectorizer.get_feature_names_out()])



wordcloud = WordCloud(background_color = 'white', max_words = 50).generate(X)
plt.imshow(wordcloud)
plt.axis("off")
plt.show();


# bonne source pour corriger ça? a creuser
# https://github.com/rachelrakov/Intro_to_Machine_Learning/blob/master/sections/word_cloud.md

words = np.array(vectorizer.get_feature_names())

# vectorizer_mat = vectorizer.toarray()
# docs = vectorizer_mat[(vectorizer_mat>10).any(axis=1)]


# cv = CountVectorizer(min_df=0, charset_error="ignore",                                               
#                          stop_words="english", max_features=200)
counts = vectorizer.fit_transform(df["competences"]).toarray().ravel()                                                  
words = np.array(vectorizer.get_feature_names()) 
# normalize                                                                                                                                             
counts = counts / float(counts.max())

