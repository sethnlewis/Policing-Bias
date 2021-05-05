import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
import os, sys

from custom_functions import *
from datetime import time
from sklearn.pipeline import make_pipeline, Pipeline
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import OneHotEncoder, LabelEncoder
from sklearn.compose import make_column_selector, make_column_transformer
from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import KFold, cross_val_score
from sklearn.neighbors import KNeighborsClassifier
from sklearn.preprocessing import StandardScaler

from sklearn.metrics import f1_score, accuracy_score



# class ModelHistory:
#     def __init__(self):
#         self.cols = ['Model', 'n_features', 'CV', 'Features',  'F1 Score', 'Accuracy', 'Notes']
#         self.history = pd.DataFrame(columns = self.cols)
    
#     def add_results(self, model, x, y, cv=False, verbose=False, return_history=False, random_state=None):
#         model_type = str(model)
#         features = x.columns
#         n_features = len(features)        
#         notes = input()
        
#         if cv:
#             kfold = KFold(n_splits=cv, shuffle=True, random_state=random_state)
#             cvs_acc = cross_val_score(model, x, y, cv=kfold, scoring='accuracy')
#             cvs_f1 = cross_val_score(model, x, y, cv=kfold, scoring='f1')  
            
#             acc = cvs_acc.mean()
#             f1 = cvs_f1.mean()
#         else:
#             acc = accuracy_score(y, model.predict(x))
#             f1 = f1_score(y, model.predict(x))



#         new_line = [model_type, n_features, cv, features, f1, acc, notes]
#         new_line_df = pd.DataFrame([new_line], columns=self.cols)
        
#         self.history = self.history.append([new_line_df])
#         self.history.reset_index(inplace=True, drop=True)
        
#         if verbose:
#             display(self.history)
#         if return_history:
#             return self.history

        
        
class ModelHistoryV2:
    def __init__(self):
        self.cols = ['Model', 'n_features', 'Features', 'F1 Score', 'Accuracy', 'Notes']
        #self.random_state = random_state
        self.history = pd.DataFrame(columns = self.cols)
    
    def add_model(self, model, x, y, display_results=False, return_history=False, notes=None):
        model_type = str(model)
        features = x.columns
        n_features = len(features)        
        acc = accuracy_score(y, model.predict(x))
        f1 = f1_score(y, model.predict(x))
        
        if notes == None:
            notes = input()

        new_line = [model_type, n_features, features, f1, acc, notes]
        new_line_df = pd.DataFrame([new_line], columns=self.cols)
        
        self.history = self.history.append([new_line_df])
        self.history.reset_index(inplace=True, drop=True)
        
        if display_results:
            display(self.history)
        if return_history:
            return self.history



def bar_plot(data, x_axis, y_axis, agg_type, verbose=False):
    
    if type(y_axis) != type(['abc']):
        y_axis = [y_axis]

    if agg_type == 'sum':
        grouped = data.groupby(by=x_axis).sum()
    elif agg_type == 'mean':
        grouped = data.groupby(by=x_axis).mean()
    else:
        print ('INVALID AGG_TYPE')
        return 'ERROR'
        
    grouped = pd.DataFrame(grouped).reset_index()
    grouped = grouped.loc[:, [x_axis]+y_axis]

    if len(y_axis) == 1:
        plt.figure()
        grouped.plot(x=x_axis, kind='barh')
        plt.title(f'{x_axis} vs. {y_axis} ({agg_type.upper()})', size=15)
        plt.show()
    else:
        if len(y_axis) < 4:
            fig, axes = plt.subplots(nrows=1, ncols=len(y_axis), figsize=(len(y_axis)*8, 4))
        else:
            fig, axes = plt.subplots(nrows=2, ncols=int(np.ceil(len(y_axis)/2)), figsize=(int(np.ceil(len(y_axis)/2))*8, 16))
        for col, ax in zip(y_axis, axes.flatten()):
            cols = grouped[[x_axis, col]]
            
            cols.plot(x=x_axis, kind='bar', ax=ax)
            ax.set_title(f'{x_axis} vs. {col} ({agg_type.upper()})')#, size=15)
            plt.subplots_adjust(wspace=.5, hspace=.5)
        display(fig)
        
    plt.xlabel(x_axis, size=15)
    plt.ylabel('Average %', size=15)
    plt.close()
    
    if verbose:
        if agg_type == 'sum':
            print(data.groupby(by=x_axis).sum()[y_axis].reset_index().sort_values(by=y_axis[0], ascending=False))
        elif agg_type == 'mean':
            print(data.groupby(by=x_axis).mean()[y_axis].reset_index().sort_values(by=y_axis[0], ascending=False))
        else:
            print ('INVALID AGG_TYPE')
            return 'ERROR'
        
        print('--------------------------------------------------------------------\n\n')
    return



# IN PROGRESS!!!!!
def forward_selected(model, data, response):
    """Linear model designed by forward selection.

    SOURCE: https://planspace.org/20150423-forward_selection_with_statsmodels/
    
    Parameters:
    -----------
    data : pandas DataFrame with all possible predictors and response

    response: string, name of response column in data

    Returns:
    --------
    model: an "optimal" fitted statsmodels linear model
           with an intercept
           selected by forward selection
           evaluated by adjusted R-squared
    """
    remaining = set(data.columns)
    remaining.remove(response)
    selected = []
    current_score, best_new_score = 0.0, 0.0
    while remaining and current_score == best_new_score:
        scores_with_candidates = []
        for candidate in remaining:
            #formula = "{} ~ {} + 1".format(response,
            #                               ' + '.join(selected + [candidate]))
            #score = smf.ols(formula, data).fit().rsquared_adj
            
            #model = 
            
            #score = 
            
            scores_with_candidates.append((score, candidate))
        scores_with_candidates.sort()
        best_new_score, best_candidate = scores_with_candidates.pop()
        if current_score < best_new_score:
            remaining.remove(best_candidate)
            selected.append(best_candidate)
            current_score = best_new_score
    formula = "{} ~ {} + 1".format(response,
                                   ' + '.join(selected))
    model = smf.ols(formula, data).fit()
    return model
