# %%

import pandas as pd

# %%

df = pd.read_csv('transactional-sample.csv')
df
# %%

df_1 = (df.sort_values('merchant_id').groupby('merchant_id')).agg(
    number_of_transactions = ('merchant_id','size'),
    media_amount = ('transaction_amount', 'mean')).reset_index()

df_1.sort_values('number_of_transactions', ascending=False)

# %%
