# %%
# モジュールのimport
import os
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
import shap
from datetime import datetime
from joblib import dump, load
from classdefinition import Dataframeflag
from classdefinition import DataframeLogScaler 
from sklearn import preprocessing
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import roc_curve, roc_auc_score, auc, precision_recall_curve, make_scorer, average_precision_score
from sklearn.model_selection import GridSearchCV, StratifiedKFold
from sklearn.cluster import KMeans, DBSCAN
from sklearn.ensemble import RandomForestRegressor
from sklearn.inspection import permutation_importance
from sklearn.mixture import GaussianMixture
from mpl_toolkits.mplot3d import Axes3D
import plotly.express as px
# %%
# A/Bテストデータ加工←今度自動化したい
df_ABtest = pd.read_csv(r"C:\Users\toraL\dirisl_phase2\data\raw\abtest_results_for_coupon_sending.csv", encoding='utf-8')

mappings = {
    'sex': {'woman':0, 'man':1},
    'mens_product_purchase':{'purchase':1, 'no_purchase':0},
    'womens_product_purchase':{'purchase':1, 'no_purchase':0},
    'newbie':{'Yes':1, 'No':0},
    'segment':{'emailed':1, 'no_emailed':0},
    'visit':{'visited':1, 'no_visited':0},
    'conversion':{'conversioned':1, 'no_conversioned':0}
}

flagment = Dataframeflag(mappings)
df_ABtest = flagment.transform(df_ABtest)

columns_to_transform = ['recency', 'history']
log_scaler = DataframeLogScaler(columns_to_transform)
df_ABtest = log_scaler.transform(df_ABtest)

dummies = pd.get_dummies(df_ABtest[['area_classification', 'channel']], drop_first=True, dtype=int)
df_ABtest2 = pd.concat([df_ABtest.drop(['area_classification', 'channel'], axis=1), dummies], axis=1)
df_ABtest2 = df_ABtest2.drop(columns=['Unnamed: 0', 'womens_product_purchase', 'conversion', 'spend'])

print(df_ABtest2.head())
# %%
# 全体データにおいても同様に
df_sending = pd.read_csv(r"C:\Users\toraL\dirisl_phase2\data\raw\all_results_for_coupon_sending.csv", encoding='utf-8')

mappings = {
    'sex': {'woman':0, 'man':1},
    'mens_product_purchase':{'purchase':1, 'no_purchase':0},
    'womens_product_purchase':{'purchase':1, 'no_purchase':0},
    'newbie':{'Yes':1, 'No':0},
    'segment':{'emailed':1, 'no_emailed':0},
    'visit':{'visited':1, 'no_visited':0},
    'conversion':{'conversioned':1, 'no_conversioned':0}
}

flagment = Dataframeflag(mappings)
df_sending = flagment.transform(df_sending)

columns_to_transform = ['recency', 'history']
log_scaler = DataframeLogScaler(columns_to_transform)
df_sending = log_scaler.transform(df_sending)

dummies = pd.get_dummies(df_sending[['area_classification', 'channel']], drop_first=True, dtype=int)
df_sending = pd.concat([df_sending.drop(['area_classification', 'channel'], axis=1), dummies], axis=1)
df_sending = df_sending.drop(columns=['Unnamed: 0', 'womens_product_purchase', 'conversion', 'spend','segment'])


print(df_sending.head())

# %%
# 目的変数毎にデータを分割
df_ABtest2_coupon = df_ABtest2.loc[df_ABtest2['segment']== 1]
df_ABtest2_no_coupon = df_ABtest2.loc[df_ABtest2['segment']== 0]
df_ABtest2_coupon = df_ABtest2_coupon.drop(columns='segment')
df_ABtest2_no_coupon = df_ABtest2_no_coupon.drop(columns='segment')

print(df_ABtest2_coupon.head())
# %%
#クーポン配布グループに対して学習を行う
#データ分割
X = df_ABtest2_coupon.drop('visit', axis=1)
y = df_ABtest2_coupon['visit']

X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, shuffle=True, stratify=y)

#training
Reg_1 = LogisticRegression()
Reg_1.fit(X_train,y_train)
# inference
y_pred_prob = Reg_1.predict_proba(X_test)[:,1]
print(y_pred_prob)
# 保存先ディレクトリ
save_dir = "models"
os.makedirs(save_dir, exist_ok=True)
# パラメーターの保存
timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
model_path = f"models/cpmodel_{timestamp}.joblib"

dump(Reg_1, model_path)
print(f"モデルを {model_path} に保存しました。")
# permutation importance算出
treatment_perm = permutation_importance(Reg_1, X_test ,y_test, n_repeats=10, random_state=42)
# %%
# ROC曲線の値の生成：fpr、tpr、閾値
fpr, tpr, thresholds = roc_curve(y_test, y_pred_prob)
#auc値算出
print("AUC:{}".format(roc_auc_score(y_test, y_pred_prob)))
# ROC曲線のプロット
plt.plot([0, 1], [0, 1], 'k--')
plt.plot(fpr,tpr,label='roc curve (AUC = %0.3f)' % auc(fpr,tpr))
plt.legend()
plt.xlabel('False Positive Rate')
plt.ylabel('True Positive Rate')
plt.title('ROC Curve')
plt.show()
# %%
# PR-ROC曲線の図示
precision, recall, thresholds = precision_recall_curve(y_true=y_test, y_score=y_pred_prob)
pr_auc = auc(recall, precision)

plt.plot(recall,precision,label='precision_recall_curve (AUC = %0.3f)' % pr_auc)
plt.plot([0,1], [1,1], linestyle='--', label='ideal line')
plt.legend()
plt.xlabel('recall')
plt.ylabel('precision')
plt.show()

# %%
# データ分布を見る
# ヒストグラムを描画
plt.figure(figsize=(6, 4))
plt.hist(y_pred_prob[y_test == 0], bins=20, alpha=0.6, color='blue', label='0')
plt.hist(y_pred_prob[y_test == 1], bins=20, alpha=0.6, color='orange', label='1')

plt.xlabel("Predicted Probability")
plt.ylabel("Frequency")
plt.legend()
plt.grid(True)
plt.show()
# %%
# 全体データでの推論
loaded_model_Reg_1 = load(r'C:\Users\toraL\dirisl_phase2\notebooks\models\cpmodel_20250306_143907.joblib')

X = df_sending.drop('visit', axis=1)
y = df_sending['visit']

y_pred_prob_cp = loaded_model_Reg_1.predict_proba(X)[:,1]
#予測確率の保存
save_dir = "results"
os.makedirs(save_dir, exist_ok=True)
timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
result_path = f"results/cpresult_{timestamp}.joblib"

dump(y_pred_prob_cp, result_path)
print(f"結果を {result_path} に保存しました。")
#AUC値
print("ROC-AUC:{}".format(roc_auc_score(y, y_pred_prob_cp)))
precision, recall, thresholds = precision_recall_curve(y,y_pred_prob_cp) 
pr_auc_cp = auc(recall, precision)
print("PR-AUC:{}".format(pr_auc_cp))
# %%
#同様にクーポン送付していないグループのモデル作成を行う
X = df_ABtest2_no_coupon.drop('visit', axis=1)
y = df_ABtest2_no_coupon['visit']

X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, shuffle=True, stratify=y)

#training
Reg_0 = LogisticRegression(class_weight="balanced")
Reg_0.fit(X_train,y_train)
# inference
y_pred_prob = Reg_0.predict_proba(X_test)[:,1]
print(y_pred_prob)
# 保存先ディレクトリ
#save_dir = "models"
#os.makedirs(save_dir, exist_ok=True)
# パラメーターの保存
timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
model_path = f"models/nocpmodel_{timestamp}.joblib"

dump(Reg_0, model_path)
print(f"モデルを {model_path} に保存しました。")
# permutation importance算出
control_perm = permutation_importance(Reg_0, X_test ,y_test, n_repeats=10, random_state=42)
# %%
# permutation importanceをまとめてみる
permutation_df = pd.DataFrame({
    'Feature': X_test.columns,
    'Treatment Importance': treatment_perm.importances_mean,
    'Control Importance': control_perm.importances_mean
}).sort_values(by='Treatment Importance', ascending=False)

print(permutation_df)
# %%
plt.figure(figsize=(10,6))
sns.barplot(x=permutation_df['Treatment Importance'], y=permutation_df['Feature'], color='blue', label='Treatment Model', alpha=0.6)
sns.barplot(x=permutation_df['Control Importance'], y=permutation_df['Feature'], color='red', label='Control Model', alpha=0.6)
plt.xlabel('Permutation Importance')
plt.ylabel('Feature')
plt.title('Permutation Importance')
plt.legend()
plt.show()
# %%
# ROC曲線の値の生成：fpr、tpr、閾値
fpr, tpr, thresholds = roc_curve(y_test, y_pred_prob)
#auc値算出
print("AUC:{}".format(roc_auc_score(y_test, y_pred_prob)))
# ROC曲線のプロット
plt.plot([0, 1], [0, 1], 'k--')
plt.plot(fpr,tpr,label='roc curve (AUC = %0.3f)' % auc(fpr,tpr))
plt.legend()
plt.xlabel('False Positive Rate')
plt.ylabel('True Positive Rate')
plt.title('ROC Curve')
plt.show()
# %%
# PR-ROC曲線の図示
precision, recall, thresholds = precision_recall_curve(y_true=y_test, y_score=y_pred_prob)
pr_auc = auc(recall, precision)

plt.plot(recall,precision,label='precision_recall_curve (AUC = %0.3f)' % pr_auc)
plt.plot([0,1], [1,1], linestyle='--', label='ideal line')
plt.legend()
plt.xlabel('recall')
plt.ylabel('precision')
plt.show()
# %%
# データ分布を見る
# ヒストグラムを描画
plt.figure(figsize=(6, 4))
plt.hist(y_pred_prob[y_test == 0], bins=20, alpha=0.6, color='blue', label='0')
plt.hist(y_pred_prob[y_test == 1], bins=20, alpha=0.6, color='orange', label='1')

plt.xlabel("Predicted Probability")
plt.ylabel("Frequency")
plt.legend()
plt.grid(True)
plt.show()
# %%
# 未送付グループのモデル精度が悪いので、ハイパーパラメータチューニングで改善を試みる
# やらない方向でいく
param_grid = {
    'C': [0.1, 1, 10, 100, 1000],  
    'penalty': ['l2'],  
    'solver': ['liblinear', 'saga'],  
    'class_weight': [None, 'balanced']  
}

scorer = make_scorer(average_precision_score, needs_proba=True)
cv = StratifiedKFold(n_splits=5, shuffle=True, random_state=42)
grid_search = GridSearchCV(LogisticRegression(), param_grid, cv=cv, scoring=scorer, n_jobs=-1)
grid_search.fit(X_train, y_train)
best_params = grid_search.best_params_
print("Best parameters:", best_params)

best_model = grid_search.best_estimator_
print(best_model.coef_)
y_pred_prob2 = best_model.predict_proba(X_test)[:,1]
print(y_pred_prob2)

timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
model_path = f"models/nocpmodel_{timestamp}.joblib"

dump(best_model, model_path)
print(f"モデルを {model_path} に保存しました。")

# %%
# ROC曲線の値の生成：fpr、tpr、閾値
fpr, tpr, thresholds = roc_curve(y_test, y_pred_prob2)
#auc値算出
print("AUC:{}".format(roc_auc_score(y_test, y_pred_prob2)))
# ROC曲線のプロット
plt.plot([0, 1], [0, 1], 'k--')
plt.plot(fpr,tpr,label='roc curve (AUC = %0.3f)' % auc(fpr,tpr))
plt.legend()
plt.xlabel('False Positive Rate')
plt.ylabel('True Positive Rate')
plt.title('ROC Curve')
plt.show()
# %%
# PR-ROC曲線の図示
precision, recall, thresholds = precision_recall_curve(y_true=y_test, y_score=y_pred_prob2)
pr_auc = auc(recall, precision)

plt.plot(recall,precision,label='precision_recall_curve (AUC = %0.3f)' % pr_auc)
plt.plot([0,1], [1,1], linestyle='--', label='ideal line')
plt.legend()
plt.xlabel('recall')
plt.ylabel('precision')
plt.show()
# %%
# 全体データでの推論
loaded_model_Reg_0 = load(r'C:\Users\toraL\dirisl_phase2\notebooks\models\nocpmodel_20250306_143724.joblib')

X = df_sending.drop('visit', axis=1)
y = df_sending['visit']

y_pred_prob_nocp = loaded_model_Reg_0.predict_proba(X)[:,1]
#予測確率の保存
#save_dir = "results"
#os.makedirs(save_dir, exist_ok=True)
timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
result_path = f"results/nocpresult_{timestamp}.joblib"

dump(y_pred_prob_nocp, result_path)
print(f"結果を {result_path} に保存しました。")
#AUC値
print("ROC-AUC:{}".format(roc_auc_score(y, y_pred_prob_nocp)))
precision, recall, thresholds = precision_recall_curve(y,y_pred_prob_nocp) 
pr_auc_nocp = auc(recall, precision)
print("PR-AUC:{}".format(pr_auc_nocp))


# %%
# Upliftingscore算出
tau = y_pred_prob_cp /y_pred_prob_nocp

sns.histplot(tau, bins=30, kde=True)
plt.xlabel('Upliftingscore')
plt.title('Distribution of Uplift Score')
plt.show()
# %%
df_sending['UpliftingScore'] = tau
print(df_sending.loc[df_sending['UpliftingScore']>=1].count())
print(df_sending.loc[df_sending['UpliftingScore']>=2].count())
# %%
df_sending1 = df_sending.copy()
df_sending_tau1up = df_sending1.loc[df_sending1['UpliftingScore']>=1]
df_sending_tau1down = df_sending1.loc[df_sending1['UpliftingScore']<=1]
#print(df_sending_tau1.shape)
# %%
# アップリフトスコアの上位・下位 をグループ分け
df_sending1.loc[df_sending1['UpliftingScore']>=1, 'uplift_group'] = 'High Uplift'
df_sending1.loc[df_sending1['UpliftingScore']<=1, 'uplift_group'] = 'Low Uplift'

# 可視化（例: 年齢ごとの分布）
plt.figure(figsize=(8, 6))
sns.violinplot(x='uplift_group', y='history', data=df_sending1, order=['Low Uplift', 'High Uplift'], inner="quartile", palette={'Low Uplift': 'blue', 'High Uplift': 'red'})
plt.xlabel('Uplift Group')
plt.ylabel('history')
plt.title('history Distribution Across Uplift Groups')
plt.show()
# %%
plt.figure(figsize=(8, 6))
sns.violinplot(x='uplift_group',
                y='recency', 
                data=df_sending1, 
                order=['Low Uplift', 'High Uplift'], 
                inner="quartile",
                palette={'Low Uplift': 'blue', 'High Uplift': 'red'})
plt.xlabel('Uplift Group')
plt.ylabel('Recency')
plt.title('Recency Distribution Across Uplift Groups')
plt.show()
# %%
plt.figure(figsize=(8, 6))
sns.violinplot(x='uplift_group', y='newbie', data=df_sending1, order=['Low Uplift', 'High Uplift'], inner="quartile", palette={'Low Uplift': 'blue', 'High Uplift': 'red'})
plt.xlabel('Uplift Group')
plt.ylabel('Newbie')
plt.title('Newbie Distribution Across Uplift Groups')
plt.show()
# %%
# Shapで可視化
X = df_sending.drop(['visit', 'UpliftingScore'], axis=1)
y = df_sending['UpliftingScore']

upliftscore_model = RandomForestRegressor(n_estimators=50, random_state=42)
upliftscore_model.fit(X,y)

X_sample = X.sample(n=10000, random_state=42)

explainer = shap.TreeExplainer(upliftscore_model, X)
shap_values = explainer(X_sample)

shap.summary_plot(shap_values, X_sample)
# %%
#高スコア層のクラスタリング
kmeans = KMeans(n_clusters=3, random_state=42)
df_sending_tau1up['cluster'] = kmeans.fit_predict(df_sending_tau1up)

sns.pairplot(df_sending_tau1up, hue="cluster", diag_kind="kde")
plt.show()
# %%
kmeans = KMeans(n_clusters=6, random_state=42)
df_sending1['cluster'] = kmeans.fit_predict(df_sending1)

sns.pairplot(df_sending1, hue="cluster", diag_kind="kde")
plt.show()
# %%
print(df_sending.shape)
# %%
X = df_sending1[['recency', 'history']]
kmeans = KMeans(n_clusters=6, random_state=42)
df_sending1['cluster'] = kmeans.fit_predict(X)

sns.scatterplot(x='recency',y='history', hue="cluster", data=df_sending1, palette='Set1', s=100)
plt.xlabel('Recency')
plt.ylabel('Frequency')
plt.title('K-means Clustering of Customers')
plt.legend(title='Cluster')
plt.show()
# %%
X = df_sending1[['recency', 'UpliftingScore']]
kmeans = KMeans(n_clusters=6, random_state=42)
df_sending1['cluster'] = kmeans.fit_predict(X)

sns.scatterplot(x='recency',y='UpliftingScore', hue="cluster", data=df_sending1, palette='Set1', s=100)
plt.xlabel('Recency')
plt.ylabel('UpliftingScore')
plt.title('K-means Clustering of Customers')
plt.legend(title='Cluster')
plt.show()
# %%
X = df_sending1[['recency', 'history']]
gmm = GaussianMixture(n_components=5, random_state=42)
df_sending1['cluster'] = gmm.fit_predict(X)

# クラスタごとに可視化
plt.figure(figsize=(8,6))
sns.scatterplot(x='recency', y='history', hue='cluster', data=df_sending1, palette='Set1', s=100)
plt.xlabel('Recency')
plt.ylabel('History')
plt.title('Gaussian Mixture Model Clustering')
plt.legend(title='Cluster')
plt.show()
# %%
X = df_sending1[['history', 'UpliftingScore']]
kmeans = KMeans(n_clusters=6, random_state=42)
df_sending1['cluster'] = kmeans.fit_predict(X)

sns.scatterplot(x='history',y='UpliftingScore', hue="cluster", data=df_sending1, palette='Set1', s=100)
plt.xlabel('History')
plt.ylabel('UpliftingScore')
plt.title('K-means Clustering of Customers')
plt.legend(title='Cluster')
plt.show()
# %%
X = df_sending1[['recency', 'history', 'UpliftingScore']]

kmeans = KMeans(n_clusters=6, random_state=42)
df_sending1['cluster'] = kmeans.fit_predict(X)

fig = plt.figure(figsize=(8,6))
ax = fig.add_subplot(111, projection='3d')
sc = ax.scatter(df_sending1['recency'], df_sending1['history'], df_sending1['UpliftingScore'], 
                c=df_sending1['cluster'], cmap='Set1', s=50)
ax.set_xlabel('Recency')
ax.set_ylabel('History')
ax.set_zlabel('UpliftingScore')
plt.title('K-means Clustering (3D)')
plt.colorbar(sc)
plt.show()
# %%
X = df_sending1[['recency', 'newbie', 'UpliftingScore']]

kmeans = KMeans(n_clusters=6, random_state=42)
df_sending1['cluster'] = kmeans.fit_predict(X)

plt.ion()
fig = plt.figure(figsize=(8,6))
ax = fig.add_subplot(111, projection='3d')
sc = ax.scatter(df_sending1['recency'], df_sending1['newbie'], df_sending1['UpliftingScore'], 
                c=df_sending1['cluster'], cmap='Set1', s=50)
ax.set_xlabel('Recency')
ax.set_ylabel('Newbie')
ax.set_zlabel('UpliftingScore')
plt.title('K-means Clustering (3D)')
plt.colorbar(sc)
plt.show()
# %%
X = df_sending1[['recency', 'history', 'UpliftingScore']]

kmeans = KMeans(n_clusters=6, random_state=42)
df_sending1['cluster'] = kmeans.fit_predict(X)

fig = px.scatter_3d(df_sending1, 
                    x='recency', 
                    y='history', 
                    z='UpliftingScore', 
                    color=df_sending1['cluster'].astype(str),  # クラスタごとに色分け
                    title='K-means Clustering (3D)')
fig.show()
# %%
X = df_sending1[['recency', 'newbie', 'UpliftingScore']]

kmeans = KMeans(n_clusters=6, random_state=42)
df_sending1['cluster'] = kmeans.fit_predict(X)

fig = px.scatter_3d(df_sending1, 
                    x='recency', 
                    y='newbie', 
                    z='UpliftingScore', 
                    color=df_sending1['cluster'].astype(str),  # クラスタごとに色分け
                    title='K-means Clustering (3D)')
fig.show()
# %%
X = df_sending1[['history', 'newbie', 'UpliftingScore']]

kmeans = KMeans(n_clusters=6, random_state=42)
df_sending1['cluster'] = kmeans.fit_predict(X)

fig = px.scatter_3d(df_sending1, 
                    x='history', 
                    y='newbie', 
                    z='UpliftingScore', 
                    color=df_sending1['cluster'].astype(str),  # クラスタごとに色分け
                    title='K-means Clustering (3D)')
fig.show()
# %%
