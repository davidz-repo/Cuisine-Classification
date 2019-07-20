
# coding: utf-8

# # Machine Learning Engineer Nanodegree
# ## Kaggle- What's Cooking
# Qixiang Zhang  
# Jul 3rd, 2018

# ## Grid Search and validation for each model's best estimator

# In[ ]:


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
from sklearn.model_selection import StratifiedKFold
from sklearn.preprocessing import LabelEncoder

# model eval
from sklearn.metrics import accuracy_score
from sklearn.model_selection import GridSearchCV

# basic multi-class classification models
from sklearn.ensemble import RandomForestClassifier
from sklearn.naive_bayes import MultinomialNB
from sklearn.linear_model import LogisticRegression
from sklearn.neural_network import MLPClassifier

# binary class classification models
from sklearn.svm import SVC
from sklearn.linear_model import SGDClassifier

# One vs All wrapper
from sklearn.multiclass import OneVsRestClassifier

# additional multi-class classification models
from xgboost import XGBClassifier
from lightgbm import LGBMClassifier


# In[ ]:


# load the data - train data
rawdf_tr = pd.read_json(path_or_buf='raw_data/train.json').set_index('id')


# ### Preprocess (regular expression + lemmatizing)

# In[ ]:


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


# In[ ]:


# copy the series from the dataframe
ingredients_tr = rawdf_tr['ingredients']


# In[ ]:


# regex both train and test data
ingredients_tr = regex_sub_match(ingredients_tr)


# In[ ]:


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


# In[ ]:


# lemmatize both train and test data
ingredients_tr = lemma(ingredients_tr)


# In[ ]:


# copy back to the dataframe
rawdf_tr['ingredients_lemma'] = ingredients_tr
rawdf_tr['ingredients_lemma_string'] = [' '.join(_).strip() for _ in rawdf_tr['ingredients_lemma']]


# # GridSearch

# ### Split the training dataset (train.json) into 85% training/validation and 15% testing

# In[ ]:


# basically train_test_split customized to input cuisine name, outputs are 2 lists of indicies for train and test for the cuisine
def tt_split(cuisine):
    cuisine_population = rawdf_tr.loc[(rawdf_tr['cuisine'] == cuisine)].index.values
    train, test = train_test_split(cuisine_population, test_size=0.15, random_state=0)
    train = train.tolist()
    test = test.tolist()
    return train, test


# In[ ]:


cuisine_list = rawdf_tr['cuisine'].unique().tolist()
# split the training data into 85-15
ix_train = [] # 85% for training (and validation)
ix_valid = [] # 15% for hold-out test
for _ in cuisine_list:
    temp_train, temp_valid = tt_split(_)
    ix_train += temp_train
    ix_valid += temp_valid


# In[ ]:


# check if the data are split correctly
print('top 3 weights', Counter(rawdf_tr['cuisine'].loc[ix_train]).most_common(3))

# DataFrame for training
traindf = rawdf_tr[['cuisine', 'ingredients_lemma_string']].loc[ix_train].reset_index(drop=True)
print('traindf shape: ',traindf.shape)
# DataFrame for validation
validdf = rawdf_tr[['cuisine', 'ingredients_lemma_string']].loc[ix_valid].reset_index(drop=True)
print('validdf shape: ',validdf.shape)
print('')

# weights check
total_recipes_tr = len(rawdf_tr)
cuisine_weights = {}
for i in cuisine_list:
    cuisine_weights[i] = float(dq(rawdf_tr['cuisine']).count(i) / total_recipes_tr)
# check train weights
print('TRAINING')
print('Weight\t Recipe\t Cuisine\n')
for _ in (Counter(traindf['cuisine']).most_common()):print(round(_[1]/traindf.cuisine.count()*100, 2),'%\t',_[1],'\t', _[0])
# check validation weights

print('\nVALIDATION')
print('Weight\t Recipe\t Cuisine\n')
for _ in (Counter(validdf['cuisine']).most_common()):print(round(_[1]/validdf.cuisine.count()*100, 2),'%\t',_[1],'\t', _[0])


# In[ ]:


#### X_train & X_pred TF-IDF vectorizer

# 85% for training and validation ===================
# X_train
X_train_ls = traindf['ingredients_lemma_string']
vectorizertr = TfidfVectorizer(stop_words='english', analyzer="word", max_df=0.65, min_df=2, binary=True)
X_train = vectorizertr.fit_transform(X_train_ls)

# y_train
y_train = traindf['cuisine']
# for xgboost the labels need to be labeled with encoder
le = LabelEncoder()
y_train_ec = le.fit_transform(y_train)

# 15% data for the hold-out validation ===============
# X_pred
X_valid_ls = validdf['ingredients_lemma_string']
vectorizerts = TfidfVectorizer(stop_words='english')
X_valid = vectorizertr.transform(X_valid_ls)

# y_valid as true y for validation
y_valid = validdf['cuisine']
y_valid_ec = le.fit_transform(y_valid)


# ## Define GridSearch Functions

# In[ ]:


#### GridSearchCV with StratifiedKFold to find the best parameters for each model

def grid_cv_clf(clf, parameter_dict, X_train, y_train):
    # model input such as SVC()
    classifier = clf
    # stratifiedKFold to maintain class ratio
    cv_sets = StratifiedKFold(n_splits=5, random_state=0, shuffle=True)
    # parameters {'kernel': ['rbf', 'linear'], 'C': [0.001, 0.1, 0.5, 1.0]} for SVC()
    params = parameter_dict
    # scoring method using accuracy
    scoring = 'average_precision'
    # grid search and cross validate
    grid = GridSearchCV(estimator = classifier,
                        param_grid = params,
                        scoring = scoring,
                        n_jobs = 2,
                        cv = cv_sets,
                        verbose = 2).fit(X_train, y_train)
    # return the best estimator
    return grid.best_estimator_


# ### Random Forest (sklearn.ensemble.[RandomForestClassifier](http://scikit-learn.org/stable/modules/generated/sklearn.ensemble.RandomForestClassifier.html))

# In[ ]:


# random forest
param_rf = {'n_estimators':[500, 750, 1000],
            'n_jobs':[2],
            'oob_score':[True],
            'criterion':['gini', 'entropy'],
            'max_features': [3, 5, 7],
            'random_state':[0],
            'verbose': [1]}
clf_rf = grid_cv_clf(RandomForestClassifier(), param_rf, X_train, y_train).fit(X_train, y_train)

# use accuracy as the metric
score_ac_rf = clf_rf.score(X_valid, y_valid)

print(clf_rf, '\naccuracy: %.2f' % (score_ac_rf*100))
# accuracy: 75.95


# ### Naive Bayes (sklearn.naive_bayes.[MultinomialNB](http://scikit-learn.org/stable/modules/generated/sklearn.naive_bayes.MultinomialNB.html))
# 
# According to this [article](https://medium.com/syncedreview/applying-multinomial-naive-bayes-to-nlp-problems-a-practical-explanation-4f5271768ebf), the multinomial Naive Bayes classifier is suitable for classification with discrete features (e.g., word counts for text classification).

# In[ ]:


param_mnb = {'alpha': [0.01, 0.02, 0.035, 0.04, 0.1, 0.5, 1]}

clf_mnb = grid_cv_clf(MultinomialNB(), param_mnb, X_train, y_train).fit(X_train, y_train)

y_pred_mnb = clf_mnb.predict(X_pred)

# use accuracy as the metric
score_ac_mnb = clf_mnb.score(X_pred, y_true)

print(clf_mnb, '\naccuracy: %.2f' % (score_ac_mnb*100))
# accuracy: 74.24


# ### Logistic Regression (sklearn.linear_model.[LogisticRegression](http://scikit-learn.org/stable/modules/generated/sklearn.linear_model.LogisticRegression.html#sklearn.linear_model.LogisticRegression))

# In[ ]:


param_lr = {'multi_class': ['multinomial'],
            'C':[0.1, 1, 5, 10, 100],
            'solver': ['lbfgs','newton-cg', 'sag', 'saga'],
            'random_state': [0]}

clf_lr = grid_cv_clf(LogisticRegression(), param_lr, X_train, y_train).fit(X_train,y_train)

y_pred_lr = clf_lr.predict(X_pred)

# use accuracy as the metric
score_ac_lr = clf_lr.score(X_pred, y_true)

print(clf_lr, '\naccuracy: %.2f' % (score_ac_lr*100))
# accuracy: 78.96


# ### Neural Network by sklearn (sklearn.neural_network.[MLPClassifier](http://scikit-learn.org/stable/modules/generated/sklearn.neural_network.MLPClassifier.html#sklearn.neural_network.MLPClassifier))

# In[ ]:


param_mlp = {'activation':['logistic', 'relu'],
             'solver': ['lbfgs', 'sgd'],
             'learning_rate':['constant', 'adaptive', 'invscaling'],
             'random_state': [0],
             'early_stopping': [True],
             'verbose': [True]}

clf_mlp = grid_cv_clf(MLPClassifier(), param_mlp, X_train, y_train).fit(X_train, y_train)

y_pred_mlp = clf_mlp.predict(X_pred)

# use accuracy as the metric
score_ac_mlp = clf_mlp.score(X_pred, y_true)

print(clf_mlp, '\naccuracy: %.2f' % (score_ac_mlp*100))
# accuracy: 78.11


# ### sklearn.multiclass.[OneVsRestClassifier](http://scikit-learn.org/stable/modules/generated/sklearn.multiclass.OneVsRestClassifier.html#sklearn.multiclass.OneVsRestClassifier)
# 
# ### Logistic Regression (sklearn.linear_model.[LogisticRegression](http://scikit-learn.org/stable/modules/generated/sklearn.linear_model.LogisticRegression.html#sklearn.linear_model.LogisticRegression))

# In[ ]:


param_ovcr_lr = {'C':[0.001, 0.1, 1, 5, 10],
                 'multi_class': ['ovr'],
                 'solver': ['lbfgs', 'sag', 'saga'],
                 'random_state': [0]}

clf_ovrc_lr = OVRC(grid_cv_clf(LogisticRegression(), param_ovcr_lr,X_train,y_train)).fit(X_train, y_train)

y_pred_ovrc_lr = clf_ovrc_lr.predict(X_pred)

# use accuracy as the metric
score_ac_ovrc_lr = clf_ovrc_lr.score(X_pred, y_true)

print(clf_ovrc_lr, '\naccuracy: %.2f' % (score_ac_ovrc_lr*100))
# accuracy: 79.58


# ### Support Vector Classification (sklearn.svm.[SVC¶](http://scikit-learn.org/stable/modules/generated/sklearn.svm.SVC.html#sklearn.svm.SVC))

# In[ ]:


param_ovrc_svm = {'C': [0.01, 0.1, 1, 3.25, 10, 100],
                  'gamma': [1, 50],
                  'coef0': [0, 1, 2],
                  'cache_size': [200, 500],
                  'kernel': ['rbf', 'poly', 'sigmoid'],
                  'random_state': [0]}

clf_ovrc_svm = OVRC(grid_cv_clf(SVC(), param_ovrc_svm, X_train, y_train)).fit(X_train, y_train)

y_pred_ovrc_svm = clf_ovrc_svm.predict(X_pred)

# use accuracy as the metric
score_ac_ovrc_svm = clf_ovrc_svm.score(X_pred, y_true)

print(clf_ovrc_svm, '\naccuracy: %.2f' % (score_ac_ovrc_svm*100))
# accuracy: 80.82


# ### SGD (sklearn.linear_model.[SGDClassifier](http://scikit-learn.org/stable/modules/generated/sklearn.linear_model.SGDClassifier.html#sklearn.linear_model.SGDClassifier))

# In[ ]:


param_ovcr_sgd = {'loss':['perceptron', 'modified_huber'],
                  'learning_rate':['constant', 'optimal', 'invscaling'],
                  'penalty': ['l2'],
                  'verbose':[2],
                  'n_jobs': [2],
                  'random_state': [0]}

clf_ovrc_sgd = OVRC(grid_cv_clf(SGDClassifier(), param_ovcr_sgd ,X_train,y_train)).fit(X_train, y_train)

y_pred_ovrc_sgd = clf_ovrc_sgd.predict(X_pred)

# use accuracy as the metric
score_ac_ovrc_sgd = clf_ovrc_sgd.score(X_pred, y_true)

print(clf_ovrc_sgd, '\naccuracy: %.2f' % (score_ac_ovrc_sgd*100))
# accuracy: 77.97


# ### [XGBoost](https://xgboost.readthedocs.io/en/latest/python/python_api.html)

# In[ ]:


param_xgb = {'n_jobs':[2],
             'learning_rate': [0.001, 0.01],
             'gamma':[0.1, 1],
             'subsample':[0.8],
             'max_depth': [6, 8, 12],
             'random_state':[0],
             'n_estimators': [750, 1000]}

clf_xgb = grid_cv_clf(XGBClassifier(), param_xgb, X_train, y_train_ec).fit(X_train, y_train_ec)

y_pred_xgb = clf_xgb.predict(X_pred)
y_pred_xgb = le.inverse_transform(y_pred_xgb)
score_ac_xgb = accuracy_score(y_pred_xgb, y_true)

print(clf_xgb, '\naccuracy: %.2f' % (score_ac_xgb*100))
# accuracy: 77.10


# ### [LightGBM](https://lightgbm.readthedocs.io/en/latest/_modules/lightgbm/sklearn.html#LGBMClassifier)

# In[ ]:


param_gbm = {'n_jobs':[2],
             'objective': ['multiclass'],
             'boosting_type':['gbdt'],
             'learning_rate': [0.01, 0.05],
             'gamma':[1],
             'subsample':[0.8],
             'max_depth': [6],
             'random_state':[0],
             'n_estimators': [500, 1000]}

clf_gbm = grid_cv_clf(LGBMClassifier(), param_gbm, X_train, y_train_ec).fit(X_train, y_train_ec)

y_pred_gbm = clf_gbm.predict(X_pred)
y_pred_gbm = le.inverse_transform(y_pred_gbm)
score_ac_gbm = accuracy_score(y_pred_gbm, y_true)

print(clf_gbm, '\naccuracy: %.2f' % (score_ac_gbm*100))
# 10% - accuracy: 75.37

