import pandas as pd

def getStatsForDataframe(df):
    # https://www.kaggle.com/artgor/is-this-malware-eda-fe-and-lgb-updated
    stats = []
    for col in df.columns:
        stats.append((col, df[col].nunique(), df[col].isnull().sum() * 100 / df.shape[0], df[col].value_counts(normalize=True, dropna=False).values[0] * 100, df[col].dtype))
        
    stats_df = pd.DataFrame(stats, columns=['Feature', 'Unique_values', 'Percentage of missing values', 'Percentage of values in the biggest category', 'type'])
    return stats_df.sort_values('Percentage of missing values', ascending=False)
