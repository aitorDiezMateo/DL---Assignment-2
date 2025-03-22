import pandas as pd


ROUTE = 'data/100k/'

# U.DATA
df_data = pd.read_csv(ROUTE + 'raw/u.data', sep='\t', header=None)
df_data.columns = ['user_id', 'item_id', 'rating', 'timestamp']
df_data.to_csv(ROUTE + 'structured/data.csv', index=False)


# U.ITEM
df_item = pd.read_csv(ROUTE + 'raw/u.item', sep='|', header=None, encoding='latin-1')
df_item.columns = ['item_id', 'title', 'release_date', 'video_release_date', 'IMDb_URL', 'unknown', 'Action', 'Adventure', 'Animation', 'Children', 'Comedy', 'Crime', 'Documentary', 'Drama', 'Fantasy', 'Film-Noir', 'Horror', 'Musical', 'Mystery', 'Romance', 'Sci-Fi', 'Thriller', 'War', 'Western']
df_item['release_date'] = pd.to_datetime(df_item['release_date'], errors='raise')
df_item['release_date'] = df_item['release_date'].astype(int) / 10**9
df_item.to_csv(ROUTE + 'structured/item.csv', index=False)

# U.USER
df_user = pd.read_csv(ROUTE + 'raw/u.user', sep='|', header=None)
df_user.columns = ['user_id', 'age', 'gender', 'occupation', 'zip_code']
df_user.to_csv(ROUTE + 'structured/user.csv', index=False)