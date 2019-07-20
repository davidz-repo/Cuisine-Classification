### What's Cooking | Kaggle
In this project, I explore different models in supervised learning (i.e. svm, Logistic Regression, random forest, xgb, etc) and design a convolutional neural network (using Keras and TensorFlow) to predict cuisines using given lists of ingredients.

[See the final report](report/report_capstone.md)

### Raw dataset
You can download the data [here](https://www.kaggle.com/c/whats-cooking/data)

![Cover](https://caspiannews.com/media/caspian_news/all_original_photos/1528831479_7183783_1528831390_5761793SFF-Foto-2018-001web2.jpg)

### Software Requirements (to be updated)
* Python 3.6 with anaconda from this [link](anaconda.com/download)
* TensorFlow with Keras - (GPU version for better performance)
* Important Python packages (see the requirements.txt)
  - nltk - install all the packages and Data
    ```
    sudo pip install -U nltk
    sudo pip install -U numpy
    >>> import nltk
    >>> nltk.download() # make sure you also download the WordNet
    ```
  - xgboost, lightgbm - install on conda environment:
    ```
    conda install py-xgboost
    conda install -c conda-forge lightgbm
    ```
## Key Files
* requirements.txt
* kitchen.yaml (if you needed)
* raw_data/
  - train.json
  - test.json
* notebooks/
  - whats_cooking_udacity_mlnd_capstone.ipynb
  - chef_keras_deep_cooking.ipynb
  - best_estimators.ipynb
  - sensitivity.ipynb
  - visual.ipynb
* report/
  - report_capstone.pdf



## Training Data
* Total of 39774 recipes
* Total of 20 types of cuisines including Greek, Southern US, Filipino, Indian, Jamaican, Spanish, Italian, Mexican, Chinese, British, Thai, Vietnamese,
Cajun Creole, Brazilian, French, Japanese, Irish, Korean, Moroccan, and Russian
* Total of 6714 unique ingredients

## Techniques learned and used:
* regular expression
* GridSearchCV
* tfidfVectorizer
* MultilabelBinarizer
* lemmatize
* StratifiedKFold
* deep learning by keras
