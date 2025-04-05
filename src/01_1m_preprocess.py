import pandas as pd
import numpy as np
import seaborn as sns
import matplotlib.pyplot as plt
from sklearn.preprocessing import StandardScaler
from sklearn.preprocessing import LabelEncoder

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

# ITEM.CSV
df_ratings = pd.read_csv(ROUTE + 'structured/data.csv')
df_ratings.to_csv(ROUTE + 'processed/data.csv', index=False)

# USER.CSV
df_user = pd.read_csv(ROUTE + 'structured/user.csv')
# One-hot encode the gender column
df_user = pd.get_dummies(df_user, columns=['gender'])
# One-hot encode the occupation column
df_user = pd.get_dummies(df_user, columns=['occupation'],prefix="Occupation")
#Check percentage of unique zip codes
print(df_user['zip_code'].nunique()/len(df_user))
#Use label encoding from sklearn for zip_code
df_user['zip_code'] = LabelEncoder().fit_transform(df_user['zip_code'])
# age is binned into bins, so one-hot encode the age column
df_user = pd.get_dummies(df_user, columns=['age'],prefix="Age")
#Save changes
df_user.to_csv(ROUTE + 'processed/user.csv', index=False)
