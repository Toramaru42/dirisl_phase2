# %%
#A/Bテスト結果データに対するデータの前処理
import pandas as pd
import numpy as np
# %%
df_ABtest = pd.read_csv(r"C:\Users\toraL\dirisl_phase2\data\raw\abtest_results_for_coupon_sending.csv", encoding='utf-8')
print(df_ABtest.head())

# %%
print(df_ABtest.info())
# %%
# 0,1に変換
sex_mapping = {'woman':0, 'man':1}
df_ABtest['sex'] = df_ABtest['sex'].map(sex_mapping)

mens_product_purchase_mapping = {'purchase':1, 'no_purchase':0}
df_ABtest['mens_product_purchase'] = df_ABtest['mens_product_purchase'].map(mens_product_purchase_mapping)

womens_product_purchase_mapping = {'purchase':1, 'no_purchase':0}
df_ABtest['womens_product_purchase'] = df_ABtest['womens_product_purchase'].map(mens_product_purchase_mapping)

newbie_mapping = {'Yes':1, 'No':0}
df_ABtest['newbie'] = df_ABtest['newbie'].map(newbie_mapping)

segment_mapping = {'emailed':1, 'no_emailed':0}
df_ABtest['segment'] = df_ABtest['segment'].map(segment_mapping)

visit_mapping = {'visited':1, 'no_visited':0}
df_ABtest['visit'] = df_ABtest['visit'].map(visit_mapping)

conversion_mapping = {'conversioned':1, 'no_conversioned':0}
df_ABtest['conversion'] = df_ABtest['conversion'].map(conversion_mapping)

print(df_ABtest.head())

# %%
# 標準化
from sklearn import preprocessing
ss = preprocessing.StandardScaler()

df_ABtest['recency'] = np.log(df_ABtest['recency']) 
df_ABtest['recency'] = ss.fit_transform(df_ABtest[['recency']])
df_ABtest['history'] = np.log(df_ABtest['history'])
df_ABtest['history'] = ss.fit_transform(df_ABtest[['history']])
print(df_ABtest.head())

# %%
# ダミー変数化
dummies = pd.get_dummies(df_ABtest[['area_classification', 'channel']], drop_first=True, dtype=int)
df_ABtest2 = pd.concat([df_ABtest.drop(['area_classification', 'channel'], axis=1), dummies], axis=1)
print(df_ABtest2.head())
# %%
# 型の確認
print(df_ABtest2.info())
# %%
