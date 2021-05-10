import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns

# ------ ESTIMATE SUBJECT AGE ------

years_in_category = {'1 - 17': [10, 17],
'18 - 25': [18, 25],
'26 - 35': [26, 35],
'36 - 45': [36, 45],
'46 - 55': [46, 55],
'56 and Above': [56, 70]}

def predict_ages(df, UNKNOWN):
    avg, std, x, y = calculate_mean_and_std(df, UNKNOWN)    
    ages = []
    for age_categorical in df['Subject Age Group']:
        age = predict_age_from_category(age_categorical, avg, std, UNKNOWN)
        ages.append(int(round(age,0)))
    return ages, x, y


def calculate_mean_and_std(df, UNKNOWN):
    xy = {}
    age_group_counts = dict(df['Subject Age Group'].value_counts())
    
    for age_group in df['Subject Age Group'].replace(UNKNOWN, np.nan).dropna().unique():
        age_range_list = set_range(age_group)
        age_range_ct = age_group_counts[age_group]
        
        for item in age_range_list:
            xy[item] = int(age_range_ct/len(age_range_list))
    
    x = np.array(list(xy.keys()))
    y = np.array(list(xy.values()))
    avg = x.mean()
    std = x.std()
    return avg, std, x, y


def set_range(age_range):
    age_range = years_in_category[age_range]
    start = age_range[0]
    stop = age_range[1]
    return np.linspace(start, stop, stop-start+1)

def predict_age_from_category(bucket, avg, std, UNKNOWN):
    if bucket == UNKNOWN:
        year_range = np.array(list(years_in_category.values())).flatten()
        min_age = year_range.min()
        max_age = year_range.max()
    else:
        year_range = years_in_category[bucket]
        min_age = year_range[0]
        max_age = year_range[1]
    age = 0
    while ((age <= min_age-0.5) or (age > max_age+0.5)):
        age = np.random.normal(avg, std)
    return age


def estimate_age_from_categorical(df, UNKNOWN): 
    
    plt.figure()
    ages, x, y = predict_ages(df, UNKNOWN)
    sns.histplot(ages, bins=60, kde=True, label='Estimated Ages within Bins');
    plt.scatter(x, y, c='r', s=10, label='Age Bins')
    plt.legend()
    plt.title('Subject Ages (Est)');
    return ages