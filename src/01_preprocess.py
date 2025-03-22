import pandas as pd
import numpy as np
import seaborn as sns
import matplotlib.pyplot as plt
from sklearn.preprocessing import StandardScaler


ROUTE = 'data/100k/'

# U.DATA
df_data = pd.read_csv(ROUTE + 'structured/data.csv')
df_data['timestamp'] = pd.to_datetime(df_data['timestamp'])
print(df_data.isna().sum()) # Total number of NAs per column
# Plot the distribution of ratings
rating_counts = df_data['rating'].value_counts().sort_index()  
sns.barplot(x=rating_counts.index, y=rating_counts.values)
plt.show()

# U.ITEM
df_item = pd.read_csv(ROUTE + 'structured/item.csv')
print(df_item.isna().sum()) # Total number of NAs per column
# Delete video_release_date column as it has a 100% of NAs
df_item = df_item.drop(columns='video_release_date')
# Delete the rest of the columns with NAs
df_item = df_item.dropna()
# Check for duplicates
print(df_item.duplicated().sum())
# Plot the distribution of genres
genre_counts = df_item.iloc[:, 6:].sum()
sns.barplot(x=genre_counts.values, y=genre_counts.index)
plt.ylabel('Genre')
plt.xlabel('Count')
plt.show()
# Save changes
df_item.to_csv(ROUTE + 'processed/item.csv', index=False)

# U.USER
df_user = pd.read_csv(ROUTE + 'structured/user.csv')
print(df_user.isna().sum()) # Total number of NAs per column
# Check for duplicates
print(df_user.duplicated().sum())
# Check percentage of uniquue ZIP codes
zip_code_unique_percentage = (df_user['zip_code'].nunique() / len(df_user)) * 100
print(zip_code_unique_percentage) # 84.3 percent of ZIP codes are unique
# Drop ZIP code column
df_user = df_user.drop(columns='zip_code')
# Normalize the age column
scaler = StandardScaler()
df_user['age'] = scaler.fit_transform(df_user['age'].values.reshape(-1, 1))
# Check the distribution of ages
sns.histplot(df_user['age'])
plt.show()
# Plot the distribution of occupations
occupation_counts = df_user['occupation'].value_counts()
sns.barplot(x=occupation_counts.values, y=occupation_counts.index)
plt.show()
# One hot encode for gender
df_user = pd.get_dummies(df_user, columns=['gender', 'occupation'])
# Save changes
df_user.to_csv(ROUTE + 'processed/user.csv', index=False)