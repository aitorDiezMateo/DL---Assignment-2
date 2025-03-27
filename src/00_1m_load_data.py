import pandas as pd



ROUTE = '/home/adiez/Desktop/Deep Learning/DL - Assignment 2/data/1M/'

# ratings.dat
df_data = pd.read_csv(ROUTE + 'raw/ratings.dat', sep='::', header=None)
df_data.columns = ['user_id', 'item_id', 'rating', 'timestamp']
df_data.to_csv(ROUTE + 'structured/data.csv', index=False)


# movies.dat
df_item = pd.read_csv(ROUTE + 'raw/movies.dat', sep='::', header=None, encoding='latin-1')
df_item.columns = ['item_id', 'title', 'Action', 'Adventure', 'Animation', 'Children', 'Comedy', 'Crime', 'Documentary', 'Drama', 'Fantasy', 'Film-Noir', 'Horror', 'Musical', 'Mystery', 'Romance', 'Sci-Fi', 'Thriller', 'War', 'Western']
df_item.to_csv(ROUTE + 'structured/item.csv', index=False)

# users.dat
df_user = pd.read_csv(ROUTE + 'raw/users.dat', sep='::', header=None)
df_user.columns = ['user_id', 'gender', 'age','occupation', 'zip_code']
df_user.to_csv(ROUTE + 'structured/user.csv', index=False)