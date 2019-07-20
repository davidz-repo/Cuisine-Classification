
# coding: utf-8

# # Machine Learning Engineer Nanodegree
# ## Kaggle- What's Cooking
# Qixiang Zhang  
# Jul 3rd, 2018

# ## Keras Deep Learning

# In[3]:


##### EXPLORE #########==================
# data exploring and basic libraries
import random
import re
import numpy as np
import pandas as pd
from collections import Counter
from collections import deque as dq

# NLP preprocessing
from nltk.stem.wordnet import WordNetLemmatizer
from nltk import word_tokenize as TK
from nltk import pos_tag
from sklearn.feature_extraction.text import TfidfVectorizer

##### MODELING ######===================
# from time import time
# train test split
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import LabelEncoder

# deep learning
import tensorflow as tf
from keras.models import Sequential
from keras.layers import Dense, Dropout, Flatten, PReLU
from keras.optimizers import SGD
from keras.callbacks import EarlyStopping


# In[4]:


# load the data - test data
rawdf_te = pd.read_json(path_or_buf='raw_data/test.json').set_index('id')
rawdf_tr = pd.read_json(path_or_buf='raw_data/train.json').set_index('id')


# ### Preprocess (regular expression + lemmatizing)

# In[5]:


# substitute the matched pattern
def sub_match(pattern, sub_pattern, ingredients):
    for i in ingredients.index.values:
        for j in range(len(ingredients[i])):
            ingredients[i][j] = re.sub(pattern, sub_pattern, ingredients[i][j].strip())
            ingredients[i][j] = ingredients[i][j].strip()
    re.purge()
    return ingredients

def regex_sub_match(series):
    # remove all units
    p0 = re.compile(r'\s*(oz|ounc|ounce|pound|lb|inch|inches|kg|to)\s*[^a-z]')
    series = sub_match(p0, ' ', series)
    # remove all digits
    p1 = re.compile(r'\d+')
    series = sub_match(p1, ' ', series)
    # remove all the non-letter characters
    p2 = re.compile('[^\w]')
    series = sub_match(p2, ' ', series)
    return series


# In[6]:


# copy the series from the dataframe
ingredients_tr = rawdf_tr['ingredients']
# do the test.json while at it
ingredients_te = rawdf_te['ingredients']


# In[7]:


# regex train data
ingredients_tr = regex_sub_match(ingredients_tr)
# regex test.json data
ingredients_te = regex_sub_match(ingredients_te)


# In[8]:


# declare instance from WordNetLemmatizer
lemmatizer = WordNetLemmatizer()

# remove all the words that are not nouns -- keep the essential ingredients
def lemma(series):
    for i in series.index.values:
        for j in range(len(series[i])):
            # get rid of all extra spaces
            series[i][j] = series[i][j].strip()
            # Tokenize a string to split off punctuation other than periods
            token = TK(series[i][j])
            # set all the plural nouns into singular nouns
            for k in range(len(token)):
                token[k] = lemmatizer.lemmatize(token[k])
            token = ' '.join(token)
            # write them back
            series[i][j] = token
    return series


# In[9]:


# lemmatize the train data
ingredients_tr = lemma(ingredients_tr)
# lemmatize test.json
ingredients_te = lemma(ingredients_te)


# In[10]:


# copy back to the dataframe
rawdf_tr['ingredients_lemma'] = ingredients_tr
rawdf_tr['ingredients_lemma_string'] = [' '.join(_).strip() for _ in rawdf_tr['ingredients_lemma']]
# do the same for the test.json dataset
rawdf_te['ingredients_lemma'] = ingredients_te
rawdf_te['ingredients_lemma_string'] = [' '.join(_).strip() for _ in rawdf_te['ingredients_lemma']]


# In[12]:


# basically train_test_split customized to input cuisine name, outputs are 2 lists of indicies for train and test for the cuisine
def tt_split(cuisine):
    cuisine_population = rawdf_tr.loc[(rawdf_tr['cuisine'] == cuisine)].index.values
    train, valid = train_test_split(cuisine_population, test_size=0.15, random_state=0)
    train = train.tolist()
    valid = valid.tolist()
    return train, valid

cuisine_list = rawdf_tr['cuisine'].unique().tolist()
# split the training data into 85-15
ix_train = [] # 85% for training (and validation)
ix_valid = [] # 15% for hold-out test
for _ in cuisine_list:
    temp_train, temp_valid = tt_split(_)
    ix_train += temp_train
    ix_valid += temp_valid

# DataFrame for training and validation
traindf = rawdf_tr[['cuisine', 'ingredients_lemma_string']].loc[ix_train].reset_index(drop=True)
print('traindf: ', traindf.shape)
validdf = rawdf_tr[['cuisine', 'ingredients_lemma_string']].loc[ix_valid].reset_index(drop=True)
print('validdf: ', validdf.shape)
    
# 85% for training and validation ===================
# X_train
X_train_ls = traindf['ingredients_lemma_string']
vectorizertr = TfidfVectorizer(stop_words='english', analyzer="word", max_df=0.65, min_df=2, binary=True)
X_train = vectorizertr.fit_transform(X_train_ls)

# y_train
y_train = traindf['cuisine']
le = LabelEncoder()
y_train_ec = le.fit_transform(y_train)
# 1-hot encoding for keras input deep learning
y_train_1h = pd.get_dummies(y_train_ec)

# save the 15% data for hold-out test ===============
# X_pred
X_valid_ls = validdf['ingredients_lemma_string']
vectorizerts = TfidfVectorizer(stop_words='english')
X_valid = vectorizertr.transform(X_valid_ls)

# y_true
y_valid = validdf['cuisine']
y_valid_ec = le.fit_transform(y_valid)
# 1-hot encoding for keras input deep learning
y_valid_1h = pd.get_dummies(y_valid_ec)

# prediction test dataframe ==========================
testdf = rawdf_te[['ingredients_lemma_string']]
print(testdf.shape)
testdf.head(n=2)
# predicting =================
# X_test
X_test_ls = testdf['ingredients_lemma_string']
vectorizerts = TfidfVectorizer(stop_words='english')
X_test = vectorizertr.transform(X_test_ls)


# ## Neural Network using Keras

# In[36]:


# define the layers
model = Sequential()
model.add(Dense(1024, input_shape=(2182,), activation='relu'))
model.add(Dropout(0.5))
model.add(Dense(144, activation='tanh'))
model.add(Dropout(0.67))
model.add(Dense(20, activation='softmax'))
model.compile(loss='categorical_crossentropy', optimizer='adam', metrics=['accuracy'])


# In[37]:


# parameters for neural network
epochs = 10
early_stopping = EarlyStopping(monitor='val_loss', patience=1)


# In[38]:


# fit the model
model.fit(X_train,y_train_1h,
          batch_size=20,
          epochs=epochs,
          verbose=2,
          callbacks=[early_stopping],
          validation_data=(X_valid, y_valid_1h),
          shuffle=True)


# In[39]:


# make prediction based on the model
y_test_nn = le.inverse_transform(model.predict_classes(X_test))

# save the output to a csv file
submit_df = pd.DataFrame()
submit_df['id'] = testdf.index.values
submit_df['cuisine'] = y_test_nn
submit_df.to_csv('Neural_Network.csv', index=False)

