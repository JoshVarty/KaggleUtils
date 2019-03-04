import pandas as pd
from scipy import stats
import numpy as np



def getStatsForDataframe(df):
    # https://www.kaggle.com/artgor/is-this-malware-eda-fe-and-lgb-updated
    stats = []
    for col in df.columns:
        stats.append((col, df[col].nunique(), df[col].isnull().sum() * 100 / df.shape[0], df[col].value_counts(normalize=True, dropna=False).values[0] * 100, df[col].dtype))
        
    stats_df = pd.DataFrame(stats, columns=['Feature', 'Unique_values', 'Percentage of missing values', 'Percentage of values in the biggest category', 'type'])
    return stats_df.sort_values('Percentage of missing values', ascending=False)

def findPossibleOutliers(df, z_threshold=4):
    numeric_df = df._get_numeric_data()

    possible_outliers = []

    for col in numeric_df.columns:
        z_score = np.abs((df[col] - df[col].mean())/df[col].std(ddof=0))
        #If any item's z_score exceeds our threshold, it is a candidate outlier
        if np.any(z_score > z_threshold):
            possible_outliers.append(col)

    return possible_outliers