import pandas as pd



ROUTE = '/home/adiez/Desktop/Deep Learning/DL - Assignment 2/data/1M/'

# ratings.dat
df_data = pd.read_csv(ROUTE + 'raw/ratings.dat', sep='::', header=None)
df_data.columns = ['user_id', 'item_id', 'rating', 'timestamp']
df_data.to_csv(ROUTE + 'structured/data.csv', index=False)


# movies.dat
df_item = pd.read_csv(ROUTE + 'raw/movies.dat', sep='::', header=None, encoding='latin-1')
df_item.columns = ['item_id', 'title', 'genre']
df_item['genre'] = df_item['genre'].str.split('|')
df_item = df_item.explode('genre')
df_item['genre'] = df_item['genre'].str.strip()

# Create one-hot encoding for genres
genre_dummies = pd.get_dummies(df_item['genre'], prefix='genre')

# Group by item_id and aggregate the dummy columns
# This will combine the genres for each movie
genre_dummies = genre_dummies.groupby(df_item['item_id']).max()
# Merge the genre dummies back with the original dataframe
df_item = df_item.groupby(['item_id', 'title']).first().reset_index()
df_item = df_item.merge(genre_dummies, on='item_id')
# Drop the original genre column if you don't need it anymore
df_item = df_item.drop('genre', axis=1)
df_item.to_csv(ROUTE + 'structured/item.csv', index=False)

# users.dat
df_user = pd.read_csv(ROUTE + 'raw/users.dat', sep='::', header=None)
df_user.columns = ['user_id', 'gender', 'age','occupation', 'zip_code']

df_user.to_csv(ROUTE + 'structured/user.csv', index=False)
