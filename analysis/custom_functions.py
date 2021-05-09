import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
import os, sys
import shap

from custom_functions import *
from datetime import time
from imblearn.pipeline import make_pipeline, Pipeline
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import OneHotEncoder, LabelEncoder
from sklearn.compose import make_column_selector, make_column_transformer
from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import KFold, cross_val_score
from sklearn.neighbors import KNeighborsClassifier
from sklearn.preprocessing import StandardScaler

from sklearn.metrics import f1_score, accuracy_score


    
# Andrew's function
def plot_importances(grid_search, X):
    try: 
        best_pipe = grid_search.best_estimator_
    except:
        best_pipe = grid_search

    ohe_names = best_pipe[0].transformers_[0][1].\
                get_feature_names(X.select_dtypes('object').columns)
    numeric_names = best_pipe[0].transformers_[1][2]

    feature_names = list(ohe_names)+numeric_names
    importances = best_pipe[1].feature_importances_
    importances_sorted = sorted(list(zip(feature_names, importances)), key=lambda x: x[1])

    x = [val[0] for val in importances_sorted]
    y = [val[1] for val in importances_sorted]

    plt.figure(figsize=(8,12))
    plt.barh(x, y)
    plt.xticks(rotation=90);

    

def train_test_scores(grid_search, return_results=False):
    train_score = grid_search.cv_results_['mean_train_score'].mean() 
    test_score = grid_search.cv_results_['mean_test_score'].mean()
    print('Train: {}\nTest: {}'.format(train_score, test_score))
    if return_results:
        return train_score, test_score

        
        
class ModelHistory:
    def __init__(self):
        self.cols = ['Model', 'n_features', 'Features', 'F1 Score', 'Accuracy', 'Notes']
        self.history = pd.DataFrame(columns = self.cols)
    
    def add_model(self, model, x, y, display_results=False, notes=None):
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

    def get_results(self):
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









# USED EXCLUSIVELY FOR "produce_shap_plot" FUNCTION
def get_df(df_test, df_train, ohe):
    df_train_cat = df_train.select_dtypes('object')
    df_test_cat = df_test.select_dtypes('object')
    
    ohe.fit(df_train_cat)
    df_train_cat_ohe = ohe.transform(df_train_cat)
    df_test_cat_ohe = ohe.transform(df_test_cat)
    
    df_train_num = df_train.select_dtypes('number')
    df_test_num = df_test.select_dtypes('number')
    
    names_ohe = ohe.get_feature_names(df_train_cat[df_train_cat.columns].columns)
    
    df_train_expanded = pd.DataFrame(df_train_cat_ohe, columns=names_ohe)
    df_test_expanded = pd.DataFrame(df_test_cat_ohe, columns=names_ohe)
    
    df_train_expanded[df_train_num.columns] = df_train_num
    df_test_expanded[df_test_num.columns] = df_test_num
    
    
    # Scale dataframe
    ss = StandardScaler()
    ss.fit(df_train_expanded)
    
    df_train_expanded_scaled = ss.transform(df_train_expanded)
    df_test_expanded_scaled = ss.transform(df_test_expanded)
    
    df_train_expanded_scaled = pd.DataFrame(df_train_expanded_scaled, columns=df_train_expanded.columns)
    df_test_expanded_scaled = pd.DataFrame(df_test_expanded_scaled, columns=df_test_expanded.columns)
    
    return df_train_expanded_scaled, df_test_expanded_scaled
def produce_shap_plot(df, target, ohe, pipe, df_train_for_fitting_only=False, target_train_for_fitting_only=False):
    try: 
        if df_train_for_fitting_only == False:
            df_train_for_fitting_only = df.copy()
            target_train_for_fitting_only = target.copy()
    except:
        pass
    
    df_train, df_test = get_df(df, df_train_for_fitting_only, ohe)
    
    model = pipe.steps[1][1]
    model.fit(df_train, target_train_for_fitting_only)
    pred = model.predict(df_test)#, output_margin=True)
    explainer = shap.TreeExplainer(model)
    shap_values = explainer.shap_values(df_test)
    #np.abs(shap_values.sum(1) + explainer.expected_value - pred).max()
    shap.summary_plot(shap_values, df_test)