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
from sklearn.neighbors import KNeighborsClassifier
from sklearn.preprocessing import StandardScaler





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