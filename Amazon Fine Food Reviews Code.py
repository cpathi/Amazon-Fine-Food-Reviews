# -*- coding: utf-8 -*-
"""
Created on Fri Nov  9 11:22:30 2018

@author: Chakri
"""

import numpy as np
import matplotlib.pyplot as plt
import pandas as pd

dataset = pd.read_csv('Reviews.csv')

dataset.shape
dataset.columns
dataset.dtypes

#Drop Missing Values
dataset.isnull().count()
dataset = dataset.dropna()
dataset.shape

#Drop duplicates
dataset = dataset.drop_duplicates(subset={"UserId","ProfileName","Time","Text","ProductId"})
dataset.count()
 
#Drop where Num<Denom
dataset = dataset[dataset['HelpfulnessNumerator'] <= dataset['HelpfulnessDenominator']]
dataset.count()

dataset.Score[dataset.Score<=3]=0   
dataset.Score[dataset.Score>=4]=1
dataset.shape

dataset=dataset.reset_index(drop=True)

#Start text cleaning from here.
import re
import nltk
nltk.download('stopwords')
from nltk.corpus import stopwords
from nltk.stem.porter import PorterStemmer
corpus = []
for i in range(0, 567205):
    review = re.sub('[^a-zA-Z]', ' ', dataset['Text'][i])
    review = review.lower()
    review = review.split()
    ps = PorterStemmer()
    review = [ps.stem(word) for word in review if not word in set(stopwords.words('english'))]
    review = ' '.join(review)
    corpus.append(review)        
    
# Creating the Bag of Words model
from sklearn.feature_extraction.text import CountVectorizer
cv = CountVectorizer(max_features=2500)
X = cv.fit_transform(corpus).toarray()
y = dataset.iloc[:, 6].values

# Splitting the dataset into the Training set and Test set
from sklearn.cross_validation import train_test_split
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size = 0.20, random_state = 0)

X_train_summary.describe()
X_test_summary.describe()
y_train_summary.describe()
y_test_summary.describe()

# Fitting Decision Tree Classification to the Training set
from sklearn.tree import DecisionTreeClassifier
classifier = DecisionTreeClassifier(criterion = 'entropy', random_state = 0)
classifier.fit(X_train, y_train)

# Predicting the Test set results
y_pred = classifier.predict(X_test)

# Making the Confusion Matrix
from sklearn.metrics import confusion_matrix
cm = confusion_matrix(y_test, y_pred)
from sklearn.metrics import accuracy_score, classification_report
accuracy_score(y_test, y_pred)
classification_report(y_test, y_pred)

# Repeating the processing with 5-fold cross validation 
from sklearn.model_selection import cross_val_score
cv_scores=cross_val_score(classifier,X,y,cv=5)

#ROC curve code for Decision Tree Model
import sklearn.metrics
probs = classifier.predict_proba(X_test)
preds = probs[:,1]
fpr, tpr, threshold = metrics.roc_curve(y_test, preds)
roc_auc = metrics.auc(fpr, tpr)

import matplotlib.pyplot as plt
plt.title('Receiver Operating Characteristic')
plt.plot(fpr, tpr, 'b', label = 'AUC = %0.2f' % roc_auc)
plt.legend(loc = 'lower right')
plt.plot([0, 1], [0, 1],'r--')
plt.xlim([0, 1])
plt.ylim([0, 1])
plt.ylabel('True Positive Rate')
plt.xlabel('False Positive Rate')
plt.show()

# Fitting Random Forest Classification to the Training set
from sklearn.ensemble import RandomForestClassifier
RFclassifier = RandomForestClassifier(n_estimators = 10, criterion = 'entropy', random_state = 0)
RFclassifier.fit(X_train, y_train)

# Predicting the Test set results
y_pred = RFclassifier.predict(X_test)

# Making the Confusion Matrix
from sklearn.metrics import confusion_matrix
cm = confusion_matrix(y_test, y_pred)
accuracy_score(y_test, y_pred)
classification_report(y_test, y_pred)
from sklearn.metrics import accuracy_score, classification_report
accuracy_score(y_test, y_pred)

#ROC curve code for Random Forest Model
import sklearn.metrics
probs = RFclassifier.predict_proba(X_test)
preds = probs[:,1]
fpr, tpr, threshold = metrics.roc_curve(y_test, preds)
roc_auc = metrics.auc(fpr, tpr)

import matplotlib.pyplot as plt
plt.title('Receiver Operating Characteristic')
plt.plot(fpr, tpr, 'b', label = 'AUC = %0.2f' % roc_auc)
plt.legend(loc = 'lower right')
plt.plot([0, 1], [0, 1],'r--')
plt.xlim([0, 1])
plt.ylim([0, 1])
plt.ylabel('True Positive Rate')
plt.xlabel('False Positive Rate')
plt.show()

