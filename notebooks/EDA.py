# %%
# 
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt

# %% ク
# クーポン付きのデータの取り込み
df = pd.read_csv(r"C:\Users\toraL\dirisl_phase2\data\raw\abtest_results_for_coupon_sending.csv", encoding='utf-8')
print(df.head())

# %% j
# japaneze_matplotlibの動作テスト
import japanize_matplotlib
plt.figure(figsize=(6, 4))
plt.plot([1, 2, 3], [4, 5, 6], label="テストライン")
plt.title("日本語テスト")
plt.legend()
plt.show()
# %%
# EDA
import ydata_profiling 
import japanize_matplotlib
profile = ydata_profiling.ProfileReport(df)
profile.to_notebook_iframe()
# %%
# EDAをhtmlで保持
profile.to_file("dirislphase2_EDAreport.html")
# %%
# EDAをpdfで保持
profile.to_file("dirislphase2_EDAreport.pdf")
# %%
# デバック用プログラム 
import sys
print(sys.path)

# %%
print(df.info())
# %%
# データセットを加工して相関が分かりやすいようにして再度EDA
df2 = df.copy()
print(df2.head())
# %%
#from sklearn.preprocessing import LabelEncoder

#sex_le = LabelEncoder()
#sex_le.classes_ = ['man', 'woman']
#sex_le.fit(df2['sex']) 
#df2['sex'] = sex_le.transform(df2['sex'])

sex_mapping = {'woman':1, 'man':0}
df2['sex'] = df2['sex'].map(sex_mapping)

mens_product_purchase_mapping = {'purchase':1, 'no_purchase':0}
df2['mens_product_purchase'] = df2['mens_product_purchase'].map(mens_product_purchase_mapping)

womens_product_purchase_mapping = {'purchase':1, 'no_purchase':0}
df2['womens_product_purchase'] = df2['womens_product_purchase'].map(womens_product_purchase_mapping)

area_classification_mapping = {'urban':2, 'suburban':1, 'ural':0}
df2['area_classification'] = df2['area_classification'].map(area_classification_mapping)

newbie_mapping = {'Yes':1, 'No':0}
df2['newbie'] = df2['newbie'].map(newbie_mapping)

channel_mapping = {'multi_device':2, 'smartphone':1, 'pc':0}
df2['channel'] = df2['channel'].map(channel_mapping)

segment_mapping = {'emailed':1, 'no_emailed':0}
df2['segment'] = df2['segment'].map(segment_mapping)

visit_mapping = {'visited':1, 'no_visited':0}
df2['visit'] = df2['visit'].map(visit_mapping)

conversion_mapping = {'conversioned':1, 'no_conversioned':0}
df2['conversion'] = df2['conversion'].map(conversion_mapping)

# %%
print(df2.head())
# %%
# EDA
import ydata_profiling 
import japanize_matplotlib
profile = ydata_profiling.ProfileReport(df2)
profile.to_notebook_iframe()
# %%
# visitとsegmentの相関
#df2 = df.copy()
import seaborn as sns
# 数値データの相関係数を計算
correlation = df2[['sex', 'mens_product_purchase', 'womens_product_purchase']].corr()

# 相関係数のヒートマップを作成
plt.figure(figsize=(10, 8))
sns.heatmap(correlation, annot=True, cmap='coolwarm', vmin=-1, vmax=1, center=0)
plt.title('変数間の相関係数')
plt.show()

# %%
correlation2 = df2[['segment', 'visit', 'conversion']].corr()

# 相関係数のヒートマップを作成
plt.figure(figsize=(10, 8))
sns.heatmap(correlation2, annot=True, cmap='coolwarm', vmin=-1, vmax=1, center=0)
plt.title('変数間の相関係数')
plt.show()

# %%
# 現状
df_real = df = pd.read_csv(r"C:\Users\toraL\dirisl_phase2\data\raw\all_results_for_coupon_sending.csv", encoding='utf-8')
print(df.head())