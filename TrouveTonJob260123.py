#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Thu Jan 26 15:47:53 2023
"""

# -- coding: utf-8 --
# Projet emploi IA
# Clémentine, Matthieu, Sylvine




# import librairies
import pandas as pd
import numpy as np
from datetime import timedelta
from sklearn.feature_extraction.text import CountVectorizer


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

#Creation de la colonne lieu
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

pd.Series(df["Type de poste"]).value_counts()

df_clean = pd.DataFrame(list(zip(df["Date de publication"],df["Intitule"], df["competences"],df['Lieu'],df["Salaire_minimum"],df["Salaire_maximum"],df['Type_poste'],df["Type de poste"])),columns =['Date_de_publication', 'Intitule',"Competences","Lieu","Salaire_minimum","Salaire_maximum","Type_poste","Société"])

df_clean.to_csv("df_clean.csv")




# II. Préprocessing des données
# 1. Count vectorize compétences
#count-vectorize pour les compétences qui tranforme le nombre de mots en 1 utiliser dans un array
vectorizer = CountVectorizer(tokenizer=lambda x: x.split(','))
X = vectorizer.fit_transform(df["competences"])
vectorizer.get_feature_names_out()
print(X)
print(vectorizer.get_feature_names_out())




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
df_poste2 = df_poste.drop(['Unnamed: 0'], axis=1).dropna().groupby(['Intitule']).median()
df_poste2.sort_values('Salaire_minimum', ascending=False, inplace=True)
df_poste2




# 4. Les compétences les mieux payés
df_competence=pd.read_csv('df_clean.csv') 
df_competence.columns
df_competence2 = df_competence.drop(['Unnamed: 0'], axis=1).dropna()
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


# 5. Le salaire moyen par compétence
dfcresults_mean = df_cresults.mean(axis=1)
dfcresults_mean2=dfcresults_mean.to_frame(name = 'Salaire moyen') 
df_salairemoyen = pd.concat([df_cresults, dfcresults_mean2.reindex(df_cresults.index)], axis=1)

# plot the figure

#transform salaire max and mini in numpy array
#df_array = df_salairemoyen[['Salaire_minimum', 'Salaire_maximum']]
#ndarray = df_array.to_numpy()

#x_salairemoyen = df_salairemoyen['Competence']
#y_salairemoyen = df_salairemoyen['Salaire moyen']
#xlabel = df_salairemoyen['Competence'] 

#plt.bar(x_salairemoyen, y_salairemoyen, width = 0.6, yerr = ndarray[[1,2]])
#plt.xticks(xlabel, rotation=90)
#plt.xlabel('Competence')
#plt.ylabel('Salaire moyen')
#plt.title('Salaire moyen par compétence')
#plt.show()









import seaborn as sns

# set edgecolor param (this is a global setting, so only set it once)
plt.rcParams["patch.force_edgecolor"] = True

# setup the dataframe
Delay = df_salairemoyen['Competence']

Time = df_salairemoyen['Salaire moyen']

df = pd.DataFrame({'Delay':Delay,'Time':Time})

# create a dict for the errors
df_error = df_salairemoyen[['Salaire_minimum', 'Salaire_maximum']]

df_errordict = df_error.T.to_dict()


fig, ax = plt.subplots(figsize=(8, 6))
sns.barplot(x='Delay', y='Time', data=df, ax=ax)

# add the lines for the errors 
for p in ax.patches:
    x = p.get_x()  # get the bottom left x corner of the bar
    w = p.get_width()  # get width of bar
    h = p.get_height()  # get height of bar
    min_y = df_errordict[h]['Salaire_minimum']  # use h to get min from dict z
    max_y = df_errordict[h]['Salaire_maximum']  # use h to get max from dict z
    plt.vlines(x+w/2, min_y, max_y, color='k')
    

# 6. Les types de contrat
#












