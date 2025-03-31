import pandas as pd
import numpy as np
import seaborn as sns
import matplotlib.pyplot as plt
from sklearn.preprocessing import StandardScaler


ROUTE = '/home/adiez/Desktop/Deep Learning/DL - Assignment 2/data/1M/'

# DATA.CSV
df_data = pd.read_csv(ROUTE + 'structured/data.csv')
print(df_data.isna().sum()) # Total number of NAs per column
#Normalize the timestamp column 
scaler = StandardScaler()
df_data['timestamp'] = scaler.fit_transform(df_data['timestamp'].values.reshape(-1, 1))
# Plot the distribution of ratings
rating_counts = df_data['rating'].value_counts().sort_index()  
sns.barplot(x=rating_counts.index, y=rating_counts.values)
plt.show()
# Check for duplicates
print(df_data.duplicated().sum()) # No duplicates
#Save changes
df_data.to_csv(ROUTE + 'processed/data.csv', index=False)

# 