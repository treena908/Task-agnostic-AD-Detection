from cmath import sqrt

import  pandas as pd
import numpy as np
import sys

from sklearn.ensemble import RandomForestRegressor
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.linear_model import LinearRegression
from sklearn import tree
from sklearn.model_selection import train_test_split, LeaveOneOut, KFold,cross_validate,RepeatedKFold
from sklearn.metrics import roc_curve, auc,mean_absolute_error, mean_squared_error,confusion_matrix, r2_score,accuracy_score,make_scorer,precision_score,recall_score,f1_score
from sklearn.feature_selection import RFE, SelectFromModel
import matplotlib.pyplot as plt
import seaborn as sns
import statsmodels.api as sm
import pickle
import scipy.stats
from sklearn.pipeline import Pipeline
from sklearn.ensemble import RandomForestClassifier

from sklearn.svm import SVR,SVC
from nltk.tokenize import sent_tokenize, word_tokenize
from sklearn.preprocessing import StandardScaler


def average_score_on_cross_val_classification(clf, x, y, scoring, cv):
    """
    Evaluates a given model/estimator using cross-validation
    and returns a dict containing the absolute vlues of the average (mean) scores
    for classification models.

    clf: scikit-learn classification model
    X: features (no labels)
    y: labels
    scoring: a dictionary of scoring metrics
    cv: cross-validation strategy
    """
    # Score metrics on cross-validated dataset
    scores_dict = cross_validate(clf, x, y, scoring=scoring, cv=cv, n_jobs=-1)

    # return the average scores for each metric
    return {metric: round(np.mean(scores), 5) for metric, scores in scores_dict.items()}
def prepare_data(df,option):
  
  
   
  print(len(df.loc[df['label']==1])) 
  print(len(df.loc[df['label']==0]))  
  if 'svc' in model or 'rfc' in model:
    y=df['label']
  else:
    y=df['mmse']

  x = df.drop(['label','file','mmse'],axis=1)
  
  
  return x,y
def write_excel(name,result):

  

  df = pd.DataFrame(data=result, index=[0])

  df = (df.T)


  df.to_excel('result/'+name+'.xlsx')
def read_data(name,model,cv):
  df = pd.read_pickle('data/'+name+'.pickle')
  # unigram = pd.read_pickle('data/'+uni+'.pickle')
  # bigram = pd.read_pickle('data/'+bi+'.pickle')
  # trigram = pd.read_pickle('data/'+tri+'.pickle')
  # topic = pd.read_pickle('data/'+topic+'.pickle')
  
  x,y=prepare_data(df,model)
  
  
  
  if 'svc' in model or 'rfc' in model:
    run_experiment_classification(x,y,model,cv)
  else:
    run_experiment_regression(x,y,model,cv)
def run_experiment_regression(x,y,model,cv):
  scoring = {
           'rmse': make_scorer(mean_squared_error),
           'mae':make_scorer(mean_absolute_error)
          
           
           }
  if cv=='loo':
    CV=LeaveOneOut()  
  elif cv=='rkf':
    CV = RepeatedKFold(n_splits=5, n_repeats=3, random_state=1)

  else: 
      
    CV = KFold(n_splits=5,shuffle=True)  
  if model=='svr_poly':
    estimator=SVR(kernel='poly', C=100, gamma='auto', degree=3)
  if model=='svr_rbf':   
    estimator=SVR( kernel='rbf', C=100, gaSmma=0.1)
  if model=='rfr':
    estimator=RandomForestRegressor(n_estimators=100,bootstrap=True,max_depth= 10)
  
  pipe_SVC = Pipeline([('scaler', StandardScaler()), ('clf', estimator)])
           
  result=average_score_on_cross_val_classification(pipe_SVC, x, y, scoring=scoring, cv=CV)
  
  print(result)
  write_excel(model+'_'+cv,result)
def run_experiment_classification(x,y,model,cv):
  scoring = {'accuracy': 'accuracy',
           'precision': make_scorer(precision_score),
           'recall': make_scorer(recall_score),
           'f1': make_scorer(f1_score)
           
           }
  if cv=='loo':
    CV=LeaveOneOut()  
  elif cv=='rkf':
    CV = RepeatedKFold(n_splits=5, n_repeats=3, random_state=1)

  else: 
      
    CV = KFold(n_splits=5,shuffle=True)  
  if model=='svc_poly':
    estimator=SVC(kernel='poly', C=100, gamma='auto', degree=3,probability=True)
  if model=='svc_rbf':   
    estimator=SVC( kernel='rbf', C=100, gaSmma=0.1,probability=True)
  if model=='rfc':
    estimator=RandomForestClassifier(n_estimators=100,criterion='entropy',bootstrap=True,max_depth= 10)
  
  pipe_SVC = Pipeline([('scaler', StandardScaler()), ('clf', estimator)])
           
  result=average_score_on_cross_val_classification(pipe_SVC, x, y, scoring=scoring, cv=CV)
  
  print(result)
  write_excel(model+'_'+cv,result)

#write param
model=['svc_poly','svc_rbf','rfc','svr_poly','svr_rbf','rfr']
option=2 #1 means classification, else regression
model_name=model[2]
cv=['loo','kf','rkf']
read_data('data_w_feature',model_name,cv[2]) 
  
  