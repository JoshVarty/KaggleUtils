import pandas as pd
from scipy import stats
import numpy as np

import matplotlib.pyplot as plt
import seaborn as sns

pd.set_option('display.max_columns', 500)
sns.set(font_scale=2)


def getStatsForDataframe(df):
    # https://www.kaggle.com/artgor/is-this-malware-eda-fe-and-lgb-updated
    stats = []
    for col in df.columns:
        stats.append((col, df[col].nunique(), df[col].isnull().sum() * 100 / df.shape[0], df[col].value_counts(normalize=True, dropna=False).values[0] * 100, df[col].dtype))
        
    stats_df = pd.DataFrame(stats, columns=['Feature', 'Unique_values', '%Missing', '%Biggest', 'type'])
    return stats_df.sort_values('%Missing', ascending=False)

def findPossibleOutliers(df, z_threshold=4):
    numeric_df = df._get_numeric_data()

    possible_outliers = []

    for col in numeric_df.columns:
        z_score = np.abs((df[col] - df[col].mean())/df[col].std(ddof=0))
        #If any item's z_score exceeds our threshold, it is a candidate outlier
        if np.any(z_score > z_threshold):
            possible_outliers.append(col)

    return possible_outliers


def plot_category_percent_of_target(df, col, target, numberToShow=20):
    fig, ax = plt.subplots(1, 1, figsize=(10, 7))
    cat_percent = df[[col, target]].groupby(col, as_index=False).mean()
    cat_size = df[col].value_counts().reset_index(drop=False)
    cat_size.columns = [col, 'count']
    cat_percent = cat_percent.merge(cat_size, on=col, how='left')
    cat_percent[target] = cat_percent[target].fillna(0)
    cat_percent = cat_percent.sort_values(by='count', ascending=False)[:numberToShow]
    sns.barplot(ax=ax, x=target, y=col, data=cat_percent, order=cat_percent[col])

    for i, p in enumerate(ax.patches):
        ax.annotate('{}'.format(cat_percent['count'].values[i]), (p.get_width(), p.get_y()+0.5), fontsize=15)

    plt.xlabel('% of ' + target + '(target)')
    plt.ylabel(col)
    plt.show()


def findProblematicColumns(df):
    
    stats = getStatsForDataframe(df)
    problematic_columns = []
    
    for i in range(len(stats)):
        currentRow = stats.iloc[i]
        
        # If one feature is dominated by a single value, it's a problem
        if currentRow['Unique_values'] == 1:
            problematic_columns.append(currentRow['Feature'])
            
        # If one feature is dominated by a single value, it's a problem
        if currentRow['%Biggest'] > 99.9:
            problematic_columns.append(currentRow['Feature'])
            
        # If one feature has entirely unique values it's (probably) a problem. 
        # (Exception might be if there's a data leak in hashes)
        if currentRow['Unique_values'] == len(df):
            problematic_columns.append(currentRow['Feature'])
            
    return problematic_columns
