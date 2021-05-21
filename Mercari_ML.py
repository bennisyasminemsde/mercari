#!/usr/bin/env python
# coding: utf-8

# # **EHTP/MSDE2**
# #  Machine Learning

# ## Data Description :
# 
# Les fichiers consistent en une liste de listes de produits. Ces fichiers sont délimités par des tabulations.
# 
# - train_id ou test_id : l'identifiant de l'annonce.
# - name : le titre de l'annonce. (Notez que nous avons nettoyé les données pour supprimer le texte qui ressemble à des prix (par exemple 20 $) pour éviter les fuites. Ces prix supprimés sont représentés par [rm].
# - item_condition_id : l'état des articles fournis par le vendeur.
# - category_name : catégorie de l'annonce.
# - brand_name : marque du produit.
# - price : le prix auquel l'article a été vendu. C'est la variable cible que vous allez prédire. L'unité est USD. Cette colonne n'existe pas test.tsv car c'est ce que vous prévoyez.
# - shipping : 1 si les frais d'expédition sont payés par le vendeur et 0 par l'acheteur.
# - item_description : la description complète de l'article. Notez que nous avons nettoyé les données pour supprimer le texte qui ressemble à des prix pour éviter les fuites. Ces prix supprimés sont représentés par [rm].
# 
# 
# Source du dataset : https://www.kaggle.com/c/mercari-price-suggestion-challenge.
# 
# But de l'étude : Prédire le prix de vente d'une annonce en fonction des informations fournies par un utilisateur pour cette annonce. 

# ## 1- Data importing & describing

# In[1]:


import numpy as np
import pandas as pd
import seaborn as sns
import sklearn

from sklearn import preprocessing
from sklearn.preprocessing import LabelEncoder, MinMaxScaler, StandardScaler
from sklearn.model_selection import train_test_split

import matplotlib.pyplot as plt
get_ipython().run_line_magic('matplotlib', 'inline')

import math


# In[2]:


print("Loading data...")
train = pd.read_csv("/Users/yasminebennis/Desktop/MSDE/Module 6 /test/mercari-price-suggestion-challenge/train.tsv",sep='\t')
test = pd.read_csv("/Users/yasminebennis/Desktop/MSDE/Module 6 /test/mercari-price-suggestion-challenge/test.tsv",sep='\t')
Y_test = pd.read_csv("/Users/yasminebennis/Desktop/MSDE/Module 6 /test/mercari-price-suggestion-challenge/sample_submission.csv")
print('Train shape :', train.shape)
print('Test shape :', test.shape)
print('Y_test shape :', Y_test.shape)


# Le train set est composé de 8 colonnes et 1 482 535 enregistrements.
# Le test set est composé de 7 colonnes et 693 359 enregistrements. 

# In[3]:


train.columns


# In[4]:


test.columns


# In[5]:


Y_test.columns


# In[6]:


train.head()


# In[6]:


#On veut ajouter la colonne relative à la variable cible (Y_test) au dataframe test afin de faciliter notre étude et d'appliquer des modification à notre test set
test = pd.merge(test, Y_test)


# #### General info

# In[8]:


train.info()


# #### Valeurs nulles

# In[7]:


train.isnull().sum(axis=0)


# In[8]:


total_rows = train.shape[0]
null_values = train.isnull().sum(axis=0)
print("Null brands : ", round(null_values['brand_name']/total_rows*100,2),'%')
print("Null categories : ", round(null_values['category_name']/total_rows*100,2),'%')
print("Null item description : ", round(null_values['item_description']/total_rows*100,4),'%')


# ## 2- Data Cleaning -part 1-

# In[9]:


#On convertit nos données textuelles en lettre minuscules.
#Train set
train["name"] = train["name"].str.lower()
train["brand_name"] = train["brand_name"].str.lower()
train["item_description"] = train["item_description"].str.lower()


# In[10]:


#Test set
test["name"] = test["name"].str.lower()
test["brand_name"] = test["brand_name"].str.lower()
test["item_description"] = test["item_description"].str.lower()


# In[11]:


#On supprime les lignes dont le nom est [rm], ce qui signifie que le nom a été supprimé car il comportait un prix.
#Train set
index_name_rm = train[train['name']=='[rm]'].index
train.drop(index_name_rm , inplace=True)


# In[12]:


#Test set
index_name_rm = test[test['name']=='[rm]'].index
test.drop(index_name_rm , inplace=True)


# In[13]:


#On supprime les lignes dont la description est [rm], ce qui signifie que la description a été supprimée car il comportait un prix.
#Train set
index_item_description_rm = train[train['item_description']=='[rm]'].index
train.drop(index_item_description_rm , inplace=True)


# In[14]:


#Test set
index_item_description_rm = test[test['item_description']=='[rm]'].index
test.drop(index_item_description_rm , inplace=True)


# In[15]:


#On remplace toutes les valeurs nulles par 'Missing'
#Train set
train['brand_name'] = train['brand_name'].fillna(value='Missing')
train['item_description'] = train['item_description'].fillna(value='Missing')


# In[16]:


#Test set
test['brand_name'] = test['brand_name'].fillna(value='Missing')
test['item_description'] = test['item_description'].fillna(value='Missing')


# ## 3- Variables quantitatives.

# ### 3.1 - 'train_id'

# In[17]:


#'train_id' et 'test_id' deviennent les index respectifs de nos dataframes ‘train’ et ‘test’
#Train set
train = train.set_index('train_id')
#Test set
test = test.set_index('test_id')
#On vérifie que 'train_id' est bien l'index du dataframe
train.head()


# In[20]:


#On vérifie que 'test_id' est bien l'index du dataframe
test.head()


# ### 3.2 - 'item_condition_id'

# In[18]:


train.item_condition_id.value_counts()


# In[19]:


total_rows = train.shape[0]
categories_counts = train['item_condition_id'].value_counts()
print("La condition '1' représente :", round(categories_counts[1]/total_rows*100,2),'%')
print("La condition '2' représente :", round(categories_counts[2]/total_rows*100,2),'%')
print("La condition '3' représente :", round(categories_counts[3]/total_rows*100,2),'%')
print("La condition '4' représente :", round(categories_counts[4]/total_rows*100,2),'%')
print("La condition '5' représente :", round(categories_counts[5]/total_rows*100,2),'%')


# In[20]:


train['item_condition_id'].value_counts().plot.bar()


# ### 3.3 - 'shipping'

# In[24]:


train.shipping.value_counts()


# In[25]:


train.shipping.value_counts()
train.shipping .value_counts().plot.pie(autopct="%.1f%%")


# 55.3 % des livraisons sont payées par l’acheteur et 44.7 % des livraisons sont payées par le vendeur.

# ### 3.4 - 'price' : variable cible / target

# In[26]:


train.price.describe().apply(lambda x: format(x, 'f'))


# Les prix sont compris entre 0 $\$$ et 2009 $\$$. Leur moyenne est 26 $\$$, et leur médiane est 17 $\$$ (c'est à dire 50% des prix sont plus petit que cette valeur)  

# In[452]:


print ("Skew is:", train.price.skew())
sns.displot(train.price, kde=True)
plt.show()


# In[453]:


##print 90 to 100 percentile values with step size of 1. 
for i in range(90,101):
    print(i,"ème centile est :",np.percentile(train['price'].values, i))


# Nous avons que 99% des prix sont inférieurs 170 $\$$

# In[454]:


sns.boxplot(y='price', data=train, showfliers=False)
plt.show()


# In[455]:


#On applique à la colonne 'price' la fonction f(x)=log(x+1)
print ("Skew is:", np.log1p(train['price']).skew())
sns.displot(np.log1p(train['price']))


# In[182]:


#On applique à la colonne 'log_price' la fonction f(x)=log(x+1) juste pour avoir une idée de ce que ca peut donner
print ("Skew is:", np.log1p(train['log_price']).skew())
sns.displot(np.log1p(train['log_price']))


# In[21]:


train['log_price'] = np.log1p(train['price'])
test['log_price'] = np.log1p(test['price'])


# In[28]:


train.log_price.describe().apply(lambda x: format(x, 'f'))


# In[29]:


##print 90 to 100 percentile values with step size of 1. 
for i in range(90,101):
    print(i,"ème centile est :",np.percentile(train['log_price'].values, i))


# Nous avons que 99% des log_prix sont inférieurs 5.14 $\$$

# ## 4- Variables qualitatives.

# ### 4.1 - 'name'

# In[30]:


train.name.describe()


# In[31]:


train.name.value_counts().head(10)


# ### 4.2 - 'category_name'

# In[32]:


train.category_name.describe()


# In[33]:


train.category_name.value_counts().head(10)


# ### 4.3 - 'brand_name'

# In[34]:


train.brand_name.describe()


# In[35]:


train.brand_name.value_counts().head(10)


# In[36]:


unique_brands=train['brand_name'].value_counts()
print("Number of Unique Brands: {}".format(len(unique_brands)))
plt.figure(figsize=(15,5))
sns.barplot(unique_brands.index[1:11],unique_brands[1:11])
plt.title('Top 10 Brands vs Number of Items Of Each Brand')
plt.xlabel('Brand Names')
plt.ylabel('Count')
plt.plot()
plt.show()


# ### 4.4 - 'item_description'

# In[37]:


train.item_description.describe()


# In[38]:


train.item_description.value_counts().head(10)


# ## 5- Pre processing.

# ##### Transformation des variables qualitatives en variables quantitatives.

# ### 5.1 - 'name'

# In[22]:


def col_len(item):
    return len(item)


# In[23]:


train['name_len']=train['name'].apply(col_len)
test['name_len']=test['name'].apply(col_len)


# ### 5.2 - 'category_name'

# On remarque que la colonne "categorie_name" est composé de trois éléments séparé par un '/', on cherche donc à étudier avec plus de détails cette variable.

# In[24]:


def category_split(category_name):
    try:
        return category_name.split('/')
    except:
        return ['Missing', 'Missing', 'Missing']


# In[25]:


train['cat1'], train['cat2'], train['cat3'] = zip(*train['category_name'].apply(lambda x: category_split(x)))
test['cat1'], test['cat2'], test['cat3'] = zip(*test['category_name'].apply(lambda x: category_split(x)))


# In[26]:


print('Train Set')
print("Le nombre d éléments uniques dans la categorie 1 : ", len(train['cat1'].unique()))
print("Le nombre d éléments uniques dans la categorie 2 : ", len(train['cat2'].unique()))
print("Le nombre d éléments uniques dans la categorie 3 : ", len(train['cat3'].unique()))


# In[27]:


print('Test Set')
print("Le nombre d éléments uniques dans la categorie 1 : ", len(test['cat1'].unique()))
print("Le nombre d éléments uniques dans la categorie 2 : ", len(test['cat2'].unique()))
print("Le nombre d éléments uniques dans la categorie 3 : ", len(test['cat3'].unique()))


# In[28]:


#Train set
train['category_name'] = train['category_name'].fillna(value='Missing')
#Test set
test['category_name'] = test['category_name'].fillna(value='Missing')


# In[49]:


train.cat1.describe() 
train.cat1.value_counts().plot.pie(autopct="%.1f%%")


# 44.8% des produits concernent les femmes, et seulement 6.3% concernent les hommes.

# In[50]:


train.cat1.value_counts() 


# In[51]:


train.cat2.describe()
train.cat2.value_counts().head(15).plot.bar()


# In[52]:


train.cat3.describe()
train.cat3.value_counts().head(15).plot.bar()


# In[29]:


#On labélise les différentes sous-catégories
#Encoding with LabelEncoder
#For train set
def numeric_train(data,to):
    if train[data].dtype == type(object):
        le = preprocessing.LabelEncoder()
        train[to] = le.fit_transform(train[data].astype(str))

numeric_train('cat1','cat1_label')
numeric_train('cat2','cat2_label')
numeric_train('cat3','cat3_label')
        
#For test set
def numeric_test(data,to):
    if test[data].dtype == type(object):
        le = preprocessing.LabelEncoder()
        test[to] = le.fit_transform(test[data].astype(str))        

numeric_test('cat1','cat1_label')
numeric_test('cat2','cat2_label')
numeric_test('cat3','cat3_label')


# In[30]:


train.head(5)


# ### 5.3 - 'brand_name'

# In[31]:


#Train set
liste_brand_train = train.brand_name.unique()
liste_brand_train = [x for x in liste_brand_train if pd.isnull(x) == False]

#Test set
liste_brand_test = test.brand_name.unique()
liste_brand_test = [x for x in liste_brand_test if pd.isnull(x) == False]


# In[32]:


len(liste_brand_train)


# In[33]:


len(liste_brand_test)


# In[34]:


def replace_train(phrase) :
    l_phrase = phrase.split(' ')
    for brand in liste_brand_train :
        if brand in l_phrase :
            return brand
    return 'Missing'


# In[35]:


def replace_test(phrase) :
    l_phrase = phrase.split(' ')
    for brand in liste_brand_test :
        if brand in l_phrase :
            return brand
    return 'Missing'


# In[60]:


train[train['brand_name']=='Missing']['name']


# In[603]:


#on doit séléctionner train[train['brand_name']=='Missing']['name']
#a ne pas executer
train[train['brand_name']=='Missing']['brand_name']=train[train['brand_name']=='Missing']['name'].apply(replace_train)
train[train['brand_name']=='Missing']['brand_name']=train[train['brand_name']=='Missing']['name'].apply(replace_test)


# In[36]:


def yes_or_no(element):
    if element  == 'Missing' :
        return 0
    return 1


# In[37]:


#On crée une nouvelle colonne 'is_branded'
#Train set
train['is_branded']=train['brand_name'].apply(yes_or_no)

#Test set
test['is_branded']=test['brand_name'].apply(yes_or_no)


# In[64]:


train.is_branded.value_counts().plot.pie(autopct="%.1f%%")


# ### 5.4 - 'item_description'

# In[38]:


#On crée une nouvelle colonne item_description_len
#Train set
train['item_description_len']=train['item_description'].apply(col_len)
#Test set
test['item_description_len']=test['item_description'].apply(col_len)


# In[66]:


import re
import string
import nltk
from tqdm import tqdm  
from nltk.corpus import stopwords
from nltk.stem import PorterStemmer
from nltk.stem.wordnet import WordNetLemmatizer
from wordcloud import WordCloud, STOPWORDS, ImageColorGenerator

from os import path
from PIL import Image


# In[521]:


nltk.download('stopwords')


# In[50]:


#i) Remplacer les mots contractés de langue anglais par leurs formes complètes
def decontracted(phrase):
    phrase = re.sub(r"won't","will not", phrase)
    phrase = re.sub(r"can't","can not", phrase)
    
    phrase = re.sub(r"n\'t","not", phrase)
    phrase = re.sub(r"\'re","are", phrase)
    phrase = re.sub(r"\'s","is", phrase)
    phrase = re.sub(r"\'d","would", phrase)
    phrase = re.sub(r"\'t","not", phrase)
    phrase = re.sub(r"\'ll","will", phrase)
    phrase = re.sub(r"\'ve","have", phrase)
    phrase = re.sub(r"\'m","am", phrase)
    #phrase = re.sub("\S*\d\S*", " ", phrase).strip()
    #phrase = re.sub('[^A-Za-z0-9]+', ' ', phrase)
    return phrase


# In[51]:


stopwords.words("english")


# In[52]:


def stopwords_count(data) :
    count_stopwords=[]
    for i in tqdm(train['item_description']):
        count=0
        for j in i.split(' '):
            if j in stopwords: count+=1
        count_stopwords.append(count)
    return count_stopwords


# In[ ]:


#ii) Remplacer les chaînes de caractères littérales comme \ r, \\, \ n par des chaînes vides.
preprocessed_total_train=[]
#changer le nom preprocessed_total_train en total_item_description
for sentance in tqdm(train['item_description'].values):
    sent = decontracted(sentance)
    sent = sent.replace('\\r', ' ')
    sent = sent.replace('\\"', ' ')
    sent = sent.replace('\\n', ' ')
    sent = re.sub('[^A-Za-z0-9]+', ' ', sent)
    stop_words = set(stopwords.words("english"))
    sent = ' '.join(e for e in sent.split() if e.lower() not in stop_words)
    preprocessed_total_train.append(sent.lower().strip())

preprocessed_total_train[2000]


# In[ ]:


def text_preprocessing(data):
    preprocessed_total = []
    for sentance in tqdm(data['item_description'].values):
        sent = decontracted(sentance)
        sent = sent.replace('\\r', ' ')
        sent = sent.replace('\\"', ' ')
        sent = sent.replace('\\n', ' ')
        sent = re.sub('[^A-Za-z0-9]+', ' ', sent)
        stop_words = set(stopwords.words("english"))
        sent = ' '.join(e for e in sent.split() if e.lower() not in stop_words)  #removing stop words
        preprocessed_total.append(sent.lower().strip())
    return preprocessed_total


# In[ ]:


train['item_description']=text_preprocessing(train)
#cv_data['item_description']=text_preprocessing(cv_data)
#test['item_description']=text_preprocessing(test)


# #### Words Cloud

# In[53]:


#stopwords=set(stopwords.words('english'))


# In[54]:


stopwords=set(STOPWORDS)
word_cloud = WordCloud(width = 600, height = 600,background_color ='white', stopwords=stopwords,min_font_size = 10).generate("1 ".join(train['item_description']))
plt.figure(figsize = (15, 10))
plt.imshow(word_cloud)
plt.axis('off')                                             
plt.show()


# In[ ]:


word_count={}
for sentence in tqdm(train['item_description']):
    for word in sentence.split(' '):
        if len(word)>=3:
            if word not in word_count:
                word_count[word]=1
            else :
                word_count[word]+=1


# In[ ]:


word_count


# In[ ]:


import collections
n_print=25
word_counter=collections.Counter(word_count)
words=[]
counter=[]
for word, count in word_counter.most_common(n_print):
    words.append(word)
    counter.append(count)


# In[ ]:


plt.figure(figsize=(10,7))
sns.barplot(counter,words)
plt.title("25 Most Frequent Words in Item-Description")
plt.xlabel('Frequency')
plt.ylabel('Words')
plt.show()


# #### Sentiment Score

# In[75]:


from nltk.sentiment.vader import SentimentIntensityAnalyzer
import nltk
nltk.download('vader_lexicon')

sid = SentimentIntensityAnalyzer()

train_sentiment = []; 
for sentence in tqdm(preprocessed_total_train):
    for_sentiment = sentence
    ss = sid.polarity_scores(for_sentiment)
    train_sentiment.append(ss)


# In[ ]:


negative=[]
neutral=[]
positive=[]
compounding=[]
for i in train_sentiment:
    
    for polarity,score in i.items():
        if(polarity=='neg'):
            negative.append(score)
        if(polarity=='neu'):
            neutral.append(score)
        if(polarity=='pos'):
            positive.append(score)
        if(polarity=='compound'):
            compounding.append(score)


# In[ ]:


train['negative']=negative
train['neutral']=neutral
train['positive']=positive
train['compound']=compounding


# In[ ]:


train.head(10)


# ## 6- Data Cleaning -part 2-

# In[39]:


train.isnull().sum()


# #### 'category_name'

# In[40]:


len(train[train['category_name']=='Missing'])


# In[41]:


#Train set
index_cat_name_missing = train[train['category_name']=='Missing'].index
train.drop(index_cat_name_missing , inplace=True)


# In[42]:


#Test set
index_cat_name_missing = test[test['category_name']=='Missing'].index
test.drop(index_cat_name_missing , inplace=True)


# In[43]:


len(train[train['category_name']=='Missing'])


# #### 'price'

# In[44]:


len(train[train.price<=0])


# In[45]:


train = train[train.price > 0]
test = test[test.price > 0]


# #### 'item_description'

# In[46]:


len(train[train['item_description']=='Missing'])


# In[47]:


#Train set
index_item_descr_missing = train[train['item_description']=='Missing'].index
train.drop(index_item_descr_missing , inplace=True)


# In[48]:


#Test set
index_item_descr_missing = test[test['item_description']=='Missing'].index
test.drop(index_item_descr_missing , inplace=True)


# In[49]:


len(train[train['item_description']=='Missing'])


# ## 7- Analyse bivariée des données

# In[50]:


#item_condition_id vs log_price
plt.figure(figsize=(15,5))
sns.boxplot(x='item_condition_id',y='log_price', data=train, showfliers=True)
plt.show()


# In[51]:


#shipping vs log_price
sns.boxplot(x='shipping',y='log_price', data=train, showfliers=True)
plt.show()
#condition 5:better price


# In[52]:


train[train['shipping']==0]['log_price'].describe()


# In[53]:


train[train['shipping']==1]['log_price'].describe()


# In[54]:


#is_branded vs log_price
sns.boxplot(x='is_branded',y='log_price', data=train, showfliers=True)
plt.show()
#condition 5:better price


# In[55]:


is_branded=train.loc[train['is_branded']==1,'log_price']
no_branded=train.loc[train['is_branded']==0,'log_price']
sns.distplot(is_branded,label='is_branded = 1')
sns.distplot(no_branded,label='is_branded = 0')
plt.title('Prix des produits de marques comparé aux prix des produits sans marques.')
plt.xlabel("log(price+1)")
plt.grid()
plt.legend()
plt.show()


# In[56]:


#cat1 vs log_price
plt.figure(figsize=(7,10))
sns.boxplot(y=train['cat1'],x=train['log_price'])
plt.title('Boxplot des variables cat1 et log_price')
plt.grid()
plt.show()


# ## 8- Correlation et choix des variables

# In[57]:


columns = list(train.columns)
plt.figure(figsize = (10, 10))
sns.heatmap(train[columns].corr(), annot = True, linewidth = 0.5)
plt.show()


# #### Feature selection

# In[63]:


#On sélectionne seulement les varibles quantitatives
train_ = train[['item_condition_id','shipping','is_branded','name_len','item_description_len', 'cat1_label','cat2_label','cat3_label','log_price']]
test_ = test[['item_condition_id','shipping','is_branded','name_len','item_description_len', 'cat1_label','cat2_label','cat3_label','log_price']]


# In[64]:


print("------------------------")
X_train = train_.drop('log_price', axis=1)
Y_train = train_['log_price']
print("Size of train dataset", X_train.shape)
print("------------------------")
X_test = test_.drop('log_price', axis=1)
Y_test = test_['log_price']
print("Size of test dataset", X_test.shape)


# In[84]:


# Such a large set of features can cause overfitting and also slow computing
# Use feature selection to select the most important features
# On crée un data frame avec les 6 variables les plus importantes
# On va utiliser le dataframe obtenu à partir de 'sk.learn_selection' dans la partie "amélioration du modèle"
import sklearn.feature_selection

select = sklearn.feature_selection.SelectKBest(k=6)
selected_features = select.fit(X_train, Y_train)
indices_selected = selected_features.get_support(indices=True)
colnames_selected = [X_train.columns[i] for i in indices_selected]

X_train_selected = X_train[colnames_selected]
X_test_selected = X_test[colnames_selected]


# In[85]:


X_train_selected
#Les colonnes supprimés sont 'cat2_label' et 'cat3_label'


# ## 9- Application des algorithmes ML

# In[86]:


from sklearn.linear_model import LinearRegression, SGDRegressor, Lasso, Ridge, ElasticNet
from sklearn.svm import LinearSVR, SVR
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score
from sklearn.tree import DecisionTreeRegressor, DecisionTreeClassifier
from sklearn.ensemble import RandomForestRegressor

from sklearn.preprocessing import PolynomialFeatures, StandardScaler, MinMaxScaler
from sklearn import linear_model, metrics, preprocessing
 
import time


# In[87]:


print("------------------------")
X_train = train_.drop('log_price', axis=1)
Y_train = train_['log_price']
print("Size of train dataset", X_train.shape)
print("------------------------")
X_test = test_.drop('log_price', axis=1)
Y_test = test_['log_price']
print("Size of test dataset", X_test.shape)


# In[88]:


#Creation de la fonction rmsle
def rmsle(y, y_pred):
    return np.sqrt(np.mean(np.power(np.log1p(y) - np.log1p(y_pred), 2)))


# Critères d'évaluation du modèle :
# - Mean absolute error (MAE) : moyenne arithmétique des valeurs absolues des écarts entre les prévisions du modèle et les observations.
# - Mean squared error (MSE) : moyenne arithmétique des carrés des écarts entre les prévisions du modèle et les observations.
# - Residual sum of squares (RMSE) : erreur quadratique moyenne, ou la racine carré du MSE.
# - Root Mean Squared Logarithmic Error (RMSLE) :  RMSE des valeurs prédites et des vraies valeurs en leur appliquant la fonction log.
# - R2-score : coefficient de détermination, utilisé pour juger de la qualité d’une régression linéaire. Cette valeur croît avec l’adéquation de la régression au modèle, vaut au maximum 1 et peut être négative . 

# ### 9.1 - Algorithme 1 : Multiple linear regression

# In[89]:


mreg = LinearRegression()
model = mreg.fit(X_train, Y_train)


# In[90]:


# The coefficients
print("------------------------")
print ('Coefficients : ', model.coef_)
print("------------------------")
print ('Intercept : ',model.intercept_)


# #### Prediction 

# In[91]:


Yp_test = model.predict(X_test)


# #### Evaluation

# In[92]:


#Critères d'évaluation
MAE1 = np.mean(np.absolute(Yp_test - Y_test))
MSE1 = mean_squared_error(Y_test,Yp_test)
RMSE1 = np.mean((Yp_test - Y_test) ** 2)
RMSLE1 = rmsle(Y_test, Yp_test)
R2_score1 = r2_score(Yp_test , Y_test)
yp1 = [MAE1,MSE1,RMSE1,RMSLE1,R2_score1]
print("MAE: %.2f" % MAE1)
print("MSE : %.2f" % MSE1)
print("RMSE: %.2f" % RMSE1)
print("RMSLE : %.2f" % RMSLE1)
print("R2-score: %.2f" % R2_score1)


# #### Features scaling

# In[93]:


# Initialise the Scaler
#Standardisation
scaler = StandardScaler()
  
# To scale data
X_train_fitted = scaler.fit_transform(X_train)
X_test_fitted = scaler.fit_transform(X_test)


# In[94]:


model_fitted = mreg.fit(X_train_fitted, Y_train)
Yp_test_fitted = model_fitted.predict(X_test_fitted)


# In[95]:


#Les coefficients
print("------------------------")
print ('Coefficients : ', model_fitted.coef_)
print("------------------------")
print ('Intercept : ',model_fitted.intercept_)


# In[96]:


#Critères d'évaluation
MAE1_fitted = np.mean(np.absolute(Yp_test_fitted - Y_test))
MSE1_fitted = mean_squared_error(Y_test,Yp_test_fitted)
RMSE1_fitted = np.mean((Yp_test_fitted - Y_test) ** 2)
RMSLE1_fitted = rmsle(Y_test, Yp_test_fitted)
R2_score1_fitted = r2_score(Yp_test_fitted , Y_test)
yp1_fitted = [MAE1_fitted,MSE1_fitted,RMSE1_fitted,RMSLE1_fitted,R2_score1_fitted]
print("MAE_fitted : %.2f" % MAE1_fitted)
print("MSE_fitted : %.2f" % MSE1_fitted)
print("RMSE_fitted: %.2f" % RMSE1_fitted)
print("RMSLE_fitted : %.2f" % RMSLE1_fitted)
print("R2-score_fitted : %.2f" % R2_score1_fitted)


# In[97]:


#Initialise the Scaler
#MinMax scaler / Normalisation
scaler = MinMaxScaler()

#To scale data
X_train_minmax = scaler.fit_transform(X_train)
X_test_minmax = scaler.fit_transform(X_test)


# In[98]:


model_minmax = mreg.fit(X_train_minmax, Y_train)
Yp_test_minmax = model_fitted.predict(X_test_minmax)


# In[99]:


#Les coefficients
print("------------------------")
print ('Coefficients : ', model_minmax.coef_)
print("------------------------")
print ('Intercept : ',model_minmax.intercept_)


# In[100]:


#Critères d'évaluation
MAE1_minmax = np.mean(np.absolute(Yp_test_minmax - Y_test))
MSE1_minmax = mean_squared_error(Y_test,Yp_test_minmax)
RMSE1_minmax = np.mean((Yp_test_minmax - Y_test) ** 2)
RMSLE1_minmax = rmsle(Y_test, Yp_test_minmax)
R2_score1_minmax = r2_score(Yp_test_minmax, Y_test)
yp1_minmax = [MAE1_minmax,MSE1_minmax,RMSE1_minmax,RMSLE1_minmax,R2_score1_minmax]
print("MAE_fitted : %.2f" % MAE1_minmax)
print("MSE_fitted : %.2f" % MSE1_minmax)
print("RMSE_fitted: %.2f" % RMSE1_minmax)
print("RMSLE_fitted : %.2f" % RMSLE1_minmax)
print("R2-score_fitted : %.2f" % R2_score1_minmax)


# In[101]:


import statsmodels.regression.linear_model as sm


# In[102]:


model_sm = sm.OLS(Y_train, X_train)
results_sm = model_sm.fit()
print(results_sm.summary())


# ### 9.2 - Algorithme 2 : Polynomial regression

# In[103]:


poly = PolynomialFeatures(degree=3)
X_train_poly = poly.fit_transform(X_train)
X_train_poly


# In[104]:


mreg_poly = linear_model.LinearRegression()
mreg_poly.fit(X_train_poly, Y_train)
mreg_poly.score(X_train_poly, Y_train)


# In[105]:


# Les coefficients
print("------------------------")
print ('Coefficients : ', mreg_poly.coef_)
print("------------------------")
print ('Intercept : ', mreg_poly.intercept_)


# #### Prediction

# In[106]:


X_test_poly = poly.fit_transform(X_test)
Yp_test = mreg_poly.predict(X_test_poly)


# #### Evaluation

# In[107]:


#Critères d'évaluation
MAE2 = np.mean(np.absolute(Yp_test - Y_test))
MSE2 = mean_squared_error(Y_test,Yp_test)
RMSE2 = np.mean((Yp_test - Y_test) ** 2)
RMSLE2 = rmsle(Y_test, Yp_test)
R2_score2 = r2_score(Yp_test , Y_test)
yp2 = [MAE2,MSE2,RMSE2,RMSLE2,R2_score2]
print("MAE: %.2f" % MAE2)
print("MSE : %.2f" % MSE2)
print("RMSE: %.2f" % RMSE2)
print("RMSLE : %.2f" % RMSLE2)
print("R2-score: %.2f" % R2_score2)


# ### 9.3 - Algorithme 3 : Ridge regression
# Ridge Regression is a technique used when the data suffers from multicollinearity ( independent variables are highly correlated)

# In[108]:


lm_ridge = Ridge(solver='lsqr', fit_intercept=False)
model = lm_ridge.fit(X_train, Y_train)


# In[109]:


# Les coefficients
print("------------------------")
print ('Coefficients : ', model.coef_)
print("------------------------")
print ('Intercept : ', model.intercept_)


# #### Prediction

# In[110]:


Yp_test = model.predict(X_test)


# #### Evaluation

# In[111]:


#Critères d'évaluation
MAE3 = np.mean(np.absolute(Yp_test - Y_test))
MSE3 = mean_squared_error(Y_test,Yp_test)
RMSE3 = np.mean((Yp_test - Y_test) ** 2)
RMSLE3 = rmsle(Y_test, Yp_test)
R2_score3 = r2_score(Yp_test , Y_test)
yp3 = [MAE3,MSE3,RMSE3,RMSLE3,R2_score3]
print("MAE: %.2f" % MAE3)
print("MSE : %.2f" % MSE3)
print("RMSE: %.2f" % RMSE3)
print("RMSLE : %.2f" % RMSLE3)
print("R2-score: %.2f" % R2_score3)


# ### 9.4 - Algorithme 4 : Arbre de décision

# In[112]:


mercari_tree_4 = DecisionTreeRegressor(criterion="mse", random_state=0, max_depth = 4) 
mercari_tree = DecisionTreeRegressor()
mercari_tree_4.fit(X_train,Y_train)
mercari_tree.fit(X_train,Y_train)


# In[113]:


mercari_tree_4.score(X_train,Y_train)


# In[114]:


#Plus le score est proche de 1, plus l'algorithme est performant
mercari_tree.score(X_train,Y_train)


# #### Prediction

# In[115]:


Yp_test = mercari_tree.predict(X_test)
Yp_test_4 = mercari_tree_4.predict(X_test)


# #### Evaluation

# In[116]:


#Critères d'évaluation
#mercari_tree
MAE4 = np.mean(np.absolute(Yp_test - Y_test))
MSE4 = mean_squared_error(Y_test,Yp_test)
RMSE4 = np.mean((Yp_test - Y_test) ** 2)
RMSLE4 = rmsle(Y_test, Yp_test)
R2_score4 = r2_score(Yp_test , Y_test)
yp4 = [MAE4,MSE4,RMSE4,RMSLE4,R2_score4]
print("MAE: %.2f" % MAE4)
print("MSE : %.2f" % MSE4)
print("RMSE: %.2f" % RMSE4)
print("RMSLE : %.2f" % RMSLE4)
print("R2-score: %.2f" % R2_score4)


# In[117]:


#Critères d'évaluation
#mercari_tree_4
MAE4_4 = np.mean(np.absolute(Yp_test_4 - Y_test))
MSE4_4 = mean_squared_error(Y_test,Yp_test_4)
RMSE4_4 = np.mean((Yp_test_4 - Y_test) ** 2)
RMSLE4_4 = rmsle(Y_test, Yp_test_4)
R2_score4_4 = r2_score(Yp_test_4 , Y_test)
yp4_4 = [MAE4_4,MSE4_4,RMSE4_4,RMSLE4_4,R2_score4_4]
print("MAE: %.2f" % MAE4_4)
print("MSE : %.2f" % MSE4_4)
print("RMSE: %.2f" % RMSE4_4)
print("RMSLE : %.2f" % RMSLE4_4)
print("R2-score: %.2f" % R2_score4_4)


# In[118]:


import matplotlib as mpl
mpl.rcParams['agg.path.chunksize'] = 10000
#affichage des résultats
#générer les nombre aléatoire
generateur = np.random.RandomState(1)
#tableau trié contenant 80 nombre généré aléatoirement
x = np.sort(5 * generateur.rand(200, 1), axis = 0)
#applatir le tableau en utilisant ravel
y = np.sin(x).ravel()
#ajout d'un bruit
y[::1] += 0.2 * (0.5 - generateur.rand(200))

plt.figure()
plt.scatter(x, y, c = "red", label = "données")
plt.plot(X_test, Yp_test, color = "blue", label = "profondeurArbre = 0",
linewidth = 2)
plt.xlabel("données")
plt.ylabel("cible")
plt.title("Arbre de régression 0")
plt.legend()
plt.show()


# In[256]:


import matplotlib as mpl
mpl.rcParams['agg.path.chunksize'] = 10000
#affichage des résultats
#générer les nombre aléatoire
generateur = np.random.RandomState(1)
#tableau trié contenant 80 nombre généré aléatoirement
x = np.sort(5 * generateur.rand(200, 1), axis = 0)
#applatir le tableau en utilisant ravel
y = np.sin(x).ravel()
#ajout d'un bruit
y[::1] += 0.2 * (0.5 - generateur.rand(200))

plt.figure()
plt.scatter(x, y, c = "red", label = "données")
plt.plot(X_test, Yp_test_4, color = "blue", label = "profondeurArbre = 4",
linewidth = 2)
plt.xlabel("données")
plt.ylabel("cible")
plt.title("Arbre de régression 4")
plt.legend()
plt.show()


# ### 9.4 - Algorithme 5 : Random Forest

# In[147]:


from sklearn.ensemble import RandomForestRegressor
RF_regr = RandomForestRegressor()
RF_regr.fit(X_train, Y_train)

RF_regr_16 = RandomForestRegressor(n_estimators= 300, max_features= 'sqrt', n_jobs= -1, max_depth=16, min_samples_split=5, min_samples_leaf=5)
RF_regr_16.fit(X_train, Y_train)


# In[148]:


RF_regr_16 = RandomForestRegressor(n_estimators= 300, max_features= 'sqrt', n_jobs= -1, max_depth=16, min_samples_split=5, min_samples_leaf=5)
RF_regr_16.fit(X_train, Y_train)


# #### Prediction

# In[149]:


Yp_test = RF_regr.predict(X_test)
Yp_test_16 = RF_regr_16.predict(X_test)


# #### Evaluation

# In[150]:


#Critères d'évaluation
MAE5 = np.mean(np.absolute(Yp_test - Y_test))
MSE5 = mean_squared_error(Y_test,Yp_test)
RMSE5 = np.mean((Yp_test - Y_test) ** 2)
RMSLE5 = rmsle(Y_test, Yp_test)
R2_score5 = r2_score(Yp_test , Y_test)
yp5 = [MAE5,MSE5,RMSE5,RMSLE5,R2_score5]
print("MAE: %.2f" % MAE5)
print("MSE : %.2f" % MSE5)
print("RMSE: %.2f" % RMSE5)
print("RMSLE : %.2f" % RMSLE5)
print("R2-score: %.2f" % R2_score5)


# In[151]:


#Critères d'évaluation
MAE5_16 = np.mean(np.absolute(Yp_test_16 - Y_test))
MSE5_16 = mean_squared_error(Y_test,Yp_test_16)
RMSE5_16 = np.mean((Yp_test_16 - Y_test) ** 2)
RMSLE5_16 = rmsle(Y_test, Yp_test_16)
R2_score5_16 = r2_score(Yp_test_16 , Y_test)
yp5_16 = [MAE5_16,MSE5_16,RMSE5_16,RMSLE5_16,R2_score5_16]
print("MAE: %.2f" % MAE5_16)
print("MSE : %.2f" % MSE5_16)
print("RMSE: %.2f" % RMSE5_16)
print("RMSLE : %.2f" % RMSLE5_16)
print("R2-score: %.2f" % R2_score5_16)


# ## 10- Choix du modèle

# In[152]:


results = pd.DataFrame(columns = ['MAE', 'MSE', 'RMSE','RMSLE','R2-score'], index = ['Multiple linear regression','MLR standardisation','MLR normalisation', 'Polynomial regression', 'Ridge regression', 'Decision Tree md0','Decision Tree md4' ,'Random Forest', 'Random Forest 16'])


# In[153]:


results.iloc[0]=yp1
results.iloc[1]=yp1_fitted
results.iloc[2]=yp1_minmax
results.iloc[3]=yp2
results.iloc[4]=yp3
results.iloc[5]=yp4
results.iloc[6]=yp4_4
results.iloc[7]=yp5
results.iloc[8]=yp5_16


# In[154]:


results


# ## 11- Amélioration du modèle

# #### Améiloration 1

# In[180]:


#On applique à ce modèle une regression linéaire multiple
mreg = LinearRegression()
model_fs = mreg.fit(X_train_selected, Y_train)
Yp_test_fs = model_fs.predict(X_test_selected)
RMSLE_fs = rmsle(Y_test, Yp_test_fs)
R2_score_fs = r2_score(Y_test , Yp_test_fs)
#R2_score_fs = r2_score(Yp_test_fs , Y_test)

yp_fs = [RMSLE_fs, R2_score_fs]
print("RMSLE_fs : %.2f" % RMSLE_fs)
print("R2-score_fs : %.2f" % R2_score_fs)
#On obtient la même valeur de RMSLE quand on remplace X_train par X_train_selected 
#Donc autant garder X_train_selected à la place de X_train afin de garder seulement les données nécessaire.


# In[182]:


X_train_selected.head(3)


# #### Amélioration 2

# In[183]:


#Nous avons plusieurs catégories
train.cat1.value_counts()


# In[184]:


train_ = train[['item_condition_id','shipping','is_branded','name_len','item_description_len','cat1','cat1_label','log_price']]
test_ = test[['item_condition_id','shipping','is_branded','name_len','item_description_len', 'cat1', 'cat1_label','log_price']]

train_women=train_[train_['cat1']=='Women']
X_train_women = train_women.drop(['log_price','cat1'], axis=1)
Y_train_women = train_women['log_price']

train_beauty=train_[train_['cat1']=='Beauty']
X_train_beauty = train_beauty.drop(['log_price','cat1'], axis=1)
Y_train_beauty = train_beauty['log_price']

train_kids=train_[train_['cat1']=='Kids']
X_train_kids = train_kids.drop(['log_price','cat1'], axis=1)
Y_train_kids = train_kids['log_price']

train_electronics=train_[train_['cat1']=='Electronics']
X_train_electronics = train_electronics.drop(['log_price','cat1'], axis=1)
Y_train_electronics = train_electronics['log_price']

train_men=train_[train_['cat1']=='Men']
X_train_men = train_men.drop(['log_price','cat1'], axis=1)
Y_train_men = train_men['log_price']

train_home=train_[train_['cat1']=='Home']
X_train_home = train_home.drop(['log_price','cat1'], axis=1)
Y_train_home = train_home['log_price']

train_vintage_collectibles=train_[train_['cat1']=='Vintage & Collectibles']
X_train_vintage_collectibles = train_vintage_collectibles.drop(['log_price','cat1'], axis=1)
Y_train_vintage_collectibles = train_vintage_collectibles['log_price']

train_other=train_[train_['cat1']=='Other']
X_train_other = train_other.drop(['log_price','cat1'], axis=1)
Y_train_other = train_other['log_price']

train_handmade=train_[train_['cat1']=='Handmade']
X_train_handmade = train_handmade.drop(['log_price','cat1'], axis=1)
Y_train_handmade = train_handmade['log_price']

train_sports_outdoors=train_[train_['cat1']=='Sports & Outdoors']
X_train_sports_outdoors = train_sports_outdoors.drop(['log_price','cat1'], axis=1)
Y_train_sports_outdoors = train_sports_outdoors['log_price']


# In[185]:


test_women=test_[test_['cat1']=='Women']
X_test_women = test_women.drop(['log_price','cat1'], axis=1)
Y_test_women = test_women['log_price']

test_beauty=test_[test_['cat1']=='Beauty']
X_test_beauty = test_beauty.drop(['log_price','cat1'], axis=1)
Y_test_beauty = test_beauty['log_price']

test_kids=test_[test_['cat1']=='Kids']
X_test_kids = test_kids.drop(['log_price','cat1'], axis=1)
Y_test_kids = test_kids['log_price']

test_electronics=test_[test_['cat1']=='Electronics']
X_test_electronics = test_electronics.drop(['log_price','cat1'], axis=1)
Y_test_electronics = test_electronics['log_price']

test_men=test_[test_['cat1']=='Men']
X_test_men = test_men.drop(['log_price','cat1'], axis=1)
Y_test_men = test_men['log_price']

test_home=test_[test_['cat1']=='Home']
X_test_home = test_home.drop(['log_price','cat1'], axis=1)
Y_test_home = test_home['log_price']

test_vintage_collectibles=test_[test_['cat1']=='Vintage & Collectibles']
X_test_vintage_collectibles = test_vintage_collectibles.drop(['log_price','cat1'], axis=1)
Y_test_vintage_collectibles = test_vintage_collectibles['log_price']

test_other=test_[test_['cat1']=='Other']
X_test_other = test_other.drop(['log_price','cat1'], axis=1)
Y_test_other = test_other['log_price']

test_handmade=test_[test_['cat1']=='Handmade']
X_test_handmade = test_handmade.drop(['log_price','cat1'], axis=1)
Y_test_handmade = test_handmade['log_price']

test_sports_outdoors=test_[test_['cat1']=='Sports & Outdoors']
X_test_sports_outdoors = test_sports_outdoors.drop(['log_price','cat1'], axis=1)
Y_test_sports_outdoors = test_sports_outdoors['log_price']


# In[186]:


#Cat1 Women
mreg = LinearRegression()
model = mreg.fit(X_train_women, Y_train_women)
Yp_test_women = model.predict(X_test_women)
RMSLE1 = rmsle(Y_test_women, Yp_test_women)
R2_score1 = r2_score(Y_test_women , Yp_test_women)
### on na pas le meme resultat quand on calcule r2_score(Yp_test_women , Y_test_women)
ypc1 = [RMSLE1, R2_score1]
print("RMSLE : %.2f" % RMSLE1)
print("R2-score: %.2f" % R2_score1)


# In[187]:


#Cat1 Beauty
model = mreg.fit(X_train_beauty, Y_train_beauty)
Yp_test_beauty = model.predict(X_test_beauty)
RMSLE2 = rmsle(Y_test_beauty, Yp_test_beauty)
R2_score2 = r2_score(Y_test_beauty , Yp_test_beauty)
ypc2 = [RMSLE2, R2_score2]
print("RMSLE : %.2f" % RMSLE2)
print("R2-score: %.2f" % R2_score2)


# In[188]:


#Cat1 Kids 
model = mreg.fit(X_train_kids, Y_train_kids)
Yp_test_kids = model.predict(X_test_kids)
RMSLE3 = rmsle(Y_test_kids, Yp_test_kids)
R2_score3 = r2_score(Y_test_kids , Yp_test_kids)
ypc3 = [RMSLE3, R2_score3]
print("RMSLE : %.2f" % RMSLE3)
print("R2-score: %.2f" % R2_score3)


# In[189]:


#Cat1 Electronics 
model = mreg.fit(X_train_electronics, Y_train_electronics)
Yp_test_electronics = model.predict(X_test_electronics)
RMSLE4 = rmsle(Y_test_electronics, Yp_test_electronics)
R2_score4 = r2_score(Y_test_electronics , Yp_test_electronics)
ypc4 = [RMSLE4, R2_score4]
print("RMSLE : %.2f" % RMSLE4)
print("R2-score: %.2f" % R2_score4)


# In[190]:


#Cat1 Men 
model = mreg.fit(X_train_men, Y_train_men)
Yp_test_men = model.predict(X_test_men)
RMSLE5 = rmsle(Y_test_men, Yp_test_men)
R2_score5 = r2_score(Y_test_men , Yp_test_men)
ypc5 = [RMSLE5, R2_score5]
print("RMSLE : %.2f" % RMSLE5)
print("R2-score: %.2f" % R2_score5)


# In[191]:


#Cat1 Home 
model = mreg.fit(X_train_home, Y_train_home)
Yp_test_home = model.predict(X_test_home)
RMSLE6 = rmsle(Y_test_home, Yp_test_home)
R2_score6 = r2_score(Y_test_home , Yp_test_home)
ypc6 = [RMSLE6, R2_score6]
print("RMSLE : %.2f" % RMSLE6)
print("R2-score: %.2f" % R2_score6)


# In[192]:


#Cat1 Vintage_collectibles 
model = mreg.fit(X_train_vintage_collectibles, Y_train_vintage_collectibles)
Yp_test_vintage_collectibles = model.predict(X_test_vintage_collectibles)
RMSLE7 = rmsle(Y_test_vintage_collectibles, Yp_test_vintage_collectibles)
R2_score7 = r2_score(Y_test_vintage_collectibles , Yp_test_vintage_collectibles)
ypc7 = [RMSLE7, R2_score7]
print("RMSLE : %.2f" % RMSLE7)
print("R2-score: %.2f" % R2_score7)


# In[193]:


#Cat1 Other
model = mreg.fit(X_train_other, Y_train_other)
Yp_test_other = model.predict(X_test_other)
RMSLE8 = rmsle(Y_test_other, Yp_test_other)
R2_score8 = r2_score(Y_test_other , Yp_test_other)
ypc8 = [RMSLE8, R2_score8]
print("RMSLE : %.2f" % RMSLE8)
print("R2-score: %.2f" % R2_score8)


# In[194]:


#Cat1 Handmade
model = mreg.fit(X_train_handmade, Y_train_handmade)
Yp_test_handmade = model.predict(X_test_handmade)
RMSLE9 = rmsle(Y_test_handmade, Yp_test_handmade)
R2_score9 = r2_score(Y_test_handmade , Yp_test_handmade)
ypc9= [RMSLE9, R2_score9]
print("RMSLE : %.2f" % RMSLE9)
print("R2-score: %.2f" % R2_score9)


# In[195]:


#Cat1 Sports & outdoors
model = mreg.fit(X_train_sports_outdoors, Y_train_sports_outdoors)
Yp_test_sports_outdoors = model.predict(X_test_sports_outdoors)
RMSLE10 = rmsle(Y_test_sports_outdoors, Yp_test_sports_outdoors)
R2_score10 = r2_score(Y_test_sports_outdoors , Yp_test_sports_outdoors)
ypc10 = [RMSLE10, R2_score10]
print("RMSLE : %.2f" % RMSLE10)
print("R2-score: %.2f" % R2_score10)


# In[196]:


results_cat = pd.DataFrame(columns = ['RMSLE','R2-score'], index = ['Women', 'Beauty', 'Kids', 'Electronics','Men','Home', 'Vintage & Collectibles', 'Other', 'Handmade','Sports & Outdoors'])
results_cat.iloc[0]=ypc1
results_cat.iloc[1]=ypc2
results_cat.iloc[2]=ypc3
results_cat.iloc[3]=ypc4
results_cat.iloc[4]=ypc5
results_cat.iloc[5]=ypc6
results_cat.iloc[6]=ypc7
results_cat.iloc[7]=ypc8
results_cat.iloc[8]=ypc9
results_cat.iloc[9]=ypc10
results_cat


# In[197]:


train_ = train[['item_condition_id','shipping','is_branded','name_len','item_description_len','cat1','cat1_label','log_price']]
test_ = test[['item_condition_id','shipping','is_branded','name_len','item_description_len', 'cat1', 'cat1_label','log_price']]

train_bis=train_[(train_['cat1']== 'Beauty') | (train_['cat1']== 'Kids') |(train_['cat1']== 'Electronics') | (train_['cat1']== 'Men') | (train_['cat1']== 'Handmade')]
X_train_bis = train_bis.drop(['log_price','cat1'], axis=1)
Y_train_bis= train_bis['log_price']

test_bis=test_[(test_['cat1']== 'Beauty') | (test_['cat1']== 'Kids') |(test_['cat1']== 'Electronics') | (test_['cat1']== 'Men') | (test_['cat1']== 'Handmade')]
X_test_bis = test_bis.drop(['log_price','cat1'], axis=1)
Y_test_bis = test_bis['log_price']


# In[198]:


mreg_bis = LinearRegression()
model_bis = mreg_bis.fit(X_train_bis, Y_train_bis)


# In[199]:


# The coefficients
print("------------------------")
print ('Coefficients : ', model_bis.coef_)
print("------------------------")
print ('Intercept : ',model_bis.intercept_)


# In[200]:


Yp_test_bis = model.predict(X_test_bis)


# In[201]:


#Critères d'évaluation
MAE1_bis = np.mean(np.absolute(Yp_test_bis - Y_test_bis))
MSE1_bis = mean_squared_error(Y_test_bis,Yp_test_bis)
RMSE1_bis = np.mean((Yp_test_bis - Y_test_bis) ** 2)
RMSLE1_bis = rmsle(Y_test_bis, Yp_test_bis)
R2_score1_bis = r2_score(Yp_test_bis , Y_test_bis)
yp1_bis = [MAE1_bis,MSE1_bis,RMSE1_bis,RMSLE1_bis,R2_score1_bis]
print("MAE : %.2f" % MAE1_bis)
print("MSE : %.2f" % MSE1_bis)
print("RMSE: %.2f" % RMSE1_bis)
print("RMSLE : %.2f" % RMSLE1_bis)
print("R2-score : %.2f" % R2_score1_bis)


# #### Amélioration 2

# In[174]:


#On rappelle que nous avions obtenu les dataframe suivants suite à l'étape feature selection
X_train_selected.head(3)


# In[179]:


#On applique à ce modèle une regression linéaire multiple
mreg = LinearRegression()
model_fs = mreg.fit(X_train_selected, Y_train)
Yp_test_fs = model_fs.predict(X_test_selected)
RMSLE_fs = rmsle(Y_test, Yp_test_fs)
R2_score_fs = r2_score(Y_test , Yp_test_fs)
#R2_score_fs = r2_score(Yp_test_fs , Y_test)

yp_fs = [RMSLE_fs, R2_score_fs]
print("RMSLE_fs : %.2f" % RMSLE_fs)
print("R2-score_fs : %.2f" % R2_score_fs)
#On obtient la même valeur de RMSLE quand on remplace X_train par X_train_selected 
#Donc autant garder X_train_selected à la place de X_train afin de garder seulement les données nécessaire.


# In[177]:


yp1


# ## 12- Pycaret
# 

# In[134]:


conda create pycaret python=3.6


# In[129]:


get_ipython().system('pip install CMake')


# In[146]:


from pycaret.datasets import get_data


# In[144]:


get_ipython().system('pip install pycaret')


# In[145]:


get_ipython().system('pip list')


# In[127]:


from pycaret.datasets import get_data


# ## 13- Flask

# In[202]:


import pickle


# In[215]:


mreg = LinearRegression()
model_fs = mreg.fit(X_train_selected, Y_train)


# In[216]:


pickle.dump(mreg, open('modelreg.pkl','wb'))


# In[226]:


from flask import Flask

app = Flask(__name__)   
@app.route('/') 

def sample_fun():      
    return "Home.html"  
if __name__ == "__main__":      

    app.run(host ='0.0.0.0', port = 5002, debug = True)


# In[224]:


#from werkzeug.utils import secure_filename
#from werkzeug.datastructures import  FileStorage
from werkzeug.local import ContextVar


# In[225]:


from werkzeug.local import Local, LocalManager


# In[ ]:


filename = 'mercari_model.sav'
f=open(filename,'wb')

