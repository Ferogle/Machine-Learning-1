import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from sklearn.preprocessing import LabelEncoder, StandardScaler
from sklearn.ensemble import RandomForestClassifier
from sklearn.decomposition import PCA, TruncatedSVD
from sklearn.model_selection import train_test_split
from statsmodels.stats.outliers_influence import variance_inflation_factor
import seaborn as sns
np.random.seed(5805)

df=pd.read_csv(r'C:\Users\SrikarUmmineni\PycharmProjects\pythonProject\Information Visualization\Airlines.csv')
print(df.columns)
print(df.head().to_string())
print(df.dtypes)

print("Missing values in the dataset")
print(df.isna().sum().sum() + df.isnull().sum().sum())

print("Duplicated values in the dataset")
print(df.duplicated().sum())

value_counts = df['Delay'].value_counts()
print("Value counts of Delay variable")
print(value_counts)

# Down sampling to make class '0' observations equal to class '1' observations
num_obs_to_remove = value_counts[0]-value_counts[1]
indices_of_0 = df[df['Delay']==0].index
indices_to_remove = np.random.choice(indices_of_0, num_obs_to_remove, replace=False)
df=df.drop(indices_to_remove)

print("After down sampling, the number of observations for each target class")
print(df['Delay'].value_counts())


# Label encoding
encode_features = ['Airline', 'AirportFrom', 'AirportTo']
le=LabelEncoder()
for i in encode_features:
    df[i] = le.fit_transform(df[i])
print(df.head().to_string())

print("Values in each category of all Airlines")
df_Airline = df[['Airline', 'id']].groupby('Airline').count()
print(df_Airline)
print("Values in each category of source Airport")
df_AirportFrom = df[['AirportFrom', 'id']].groupby('AirportFrom').count()
print(df_AirportFrom)
print("Values in each category of destination airport")
df_AirportTo = df[['AirportTo', 'id']].groupby('AirportTo').count()
print(df_AirportTo)

df.drop('id', axis=1, inplace=True)
X = df.drop('Delay',axis=1)
y = df['Delay']
sc = StandardScaler()
for i in X.columns:
    X[i] = sc.fit_transform(X[[i]])

X_train,X_test,y_train,y_test=train_test_split(X, y, test_size=0.2, random_state=5805, shuffle=True)

# Dimensionality reduction
# Random forest analysis
model_rf = RandomForestClassifier(random_state=5805)
model_rf.fit(X_train,y_train)
importances = model_rf.feature_importances_
print("Feature importances from Random forest method")
indices = np.argsort(importances)
sortedImportance = importances[indices]
sorted_features = X_train.columns[indices]
plt.figure(figsize=(12,8))
plt.barh(range(len(sortedImportance)), sortedImportance)
plt.yticks(range(len(sorted_features)), sorted_features)
plt.title('Feature importances by RandomForestClassifier')
plt.xlabel('Feature Importance')
plt.ylabel('Features')
plt.show()

# Principal component analysis
pca = PCA()
pca.fit(X_train)
print(pca.explained_variance_ratio_)
cum_var=np.round(np.cumsum(sorted(pca.explained_variance_ratio_,reverse=True))*100,
decimals=3)
labels=[i for i in range(1,len(cum_var)+1)]
plt.bar(x=labels,height=cum_var,alpha=0.5)
x_point=7
y_point=95
plt.axvline(x=x_point, linestyle='--', label=f'x = {x_point}')
plt.axhline(y=y_point, linestyle='--', label=f'y = {y_point}')
plt.xlabel("Number of components")
plt.ylabel("Percentage contribution to variance")
plt.title("Cumulative explained variance of each component from PCA")
plt.show()

# Singular Value Decomposition anaysis
tsvd = TruncatedSVD(n_components=7)
tsvd_result = tsvd.fit(X_train)
print("Explained variance ratio from SVD ", tsvd.explained_variance_ratio_)
print(f"Singular values of features {tsvd.singular_values_}")
cum_evr = np.cumsum(100*tsvd.explained_variance_ratio_)
labels=[i for i in range(1,len(cum_evr)+1)]
plt.bar(x=labels,height=cum_evr,alpha=0.5)
x_point=7
y_point=95
plt.axvline(x=x_point, linestyle='--', label=f'x = {x_point}')
plt.axhline(y=y_point, linestyle='--', label=f'y = {y_point}')
plt.xlabel("Number of components")
plt.ylabel("Percentage contribution to variance")
plt.title("Cumulative variance ratio of each component from SVD")
plt.show()

# Variance Inflation Factor
vif_fea = [variance_inflation_factor(X_train.values, i) for i in range(X_train.shape[1])]
for i,j in zip(df.columns, vif_fea):
    print(i,round(j,2))

cov_matrix = X_train.cov()
plt.figure(figsize=(10, 8))
sns.heatmap(cov_matrix, annot=True,fmt='.2f', cmap='coolwarm', square=True, xticklabels=X_train.columns, yticklabels=X_train.columns)
plt.title("Covariance Matrix Heatmap")
plt.show()

corr_matrix = X_train.corr()
plt.figure(figsize=(10, 8))  # Adjust the figure size as needed
sns.heatmap(corr_matrix, annot=True, cmap='coolwarm', fmt=".2f", square=True)
plt.title("Correlation Coefficient Heatmap")
plt.show()

print(df['Delay'].value_counts())
sns.set_style('whitegrid')
sns.countplot(data=df,x='Delay')
plt.title('Countplot of # of obs. for each class of Delay')
plt.xlabel("Whether flight delayed")
plt.ylabel("# of samples")
plt.show()

sns.boxplot(data=X_train, x='Length')
plt.title("Box plot of Length feature to detect outliers")
plt.show()

# Outlier detection and removal using distance based method
Q1 = X_train['Length'].quantile(0.25)
Q3 = X_train['Length'].quantile(0.75)
IQR = Q3 - Q1
lower_bound = Q1 - 1.5 * IQR
upper_bound = Q3 + 1.5 * IQR
outlier_indices = X_train.index[(X_train['Length']< lower_bound) | (X_train['Length'] > upper_bound)].tolist()
X_train=X_train.drop(outlier_indices)
print(len(X_train))

sns.boxplot(data=X_train, x='Length')
plt.title("Box plot of Length feature after detecting outliers")
plt.show()