from sklearn.datasets import make_regression,make_classification,make_blobs
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import seaborn as sbn

np.set_printoptions(precision=2)
pd.set_option('display.float_format', '{:.2f}'.format)
# Q1 Create a regression data with 1000 samples and 100 features where all are informative and target is 1
X,Y=make_regression(n_samples=1000,n_features=100,n_informative=100,n_targets=1,random_state=5805)
print("First 5 rows of X (feature matrix)")
print(X[:5])
print("First 5 rows of Y (target matrix)")
print(Y[:5])
# print(X.shape)
# print(Y.shape)
# print(X.T.shape)
# print(Y.T.shape)
# Q2
features=["feature"+str(i) for i in range(1,101)]
features.append("target")
Y=np.array([Y])
print(X.shape)
print(Y.T.shape)
print("X tail")
print(X[:6])
print("Y tail")
print(Y[:6])
# Q2. Concatenate features and target matrices
X_df = np.concatenate([X.T,Y],axis=0)
df=pd.DataFrame(data=X_df.T,columns=features)
print(df.tail().to_string())

# Q3. Slice the dataframe and take only first columns
df_sliced = df.iloc[:,:5]
print("Last 5 observations of sliced dataframe")
print(df_sliced.tail().to_string())

# Q4. Print pair wise covariance matrix and correlation matrix of the sliced dataframe
print("Covariance matrix")
print(df_sliced.cov())
print("Correlation matrix")
print(df_sliced.corr())

# Q5. Print the pairwise bivariate distributions of features in sliced dataframe
sbn.pairplot(df_sliced,kind="kde")
plt.show()

# Q6. Plot a scatterplot between feature1 and target
plt.title("Target vs. Feature1")
plt.xlabel("Feature1")
plt.ylabel("Target")
plt.scatter(x=df['feature1'],y=df['target'])
plt.show()

# Q7. Generating synthetic dataset using make_classification
X1,Y1=make_classification(n_samples=1000,n_features=100,n_informative=100,random_state=5805,n_classes=4,n_redundant=0,n_repeated=0)
Y1=np.array([Y1])
X1_df = np.concatenate([X1.T,Y1],axis=0)
df1=pd.DataFrame(data=X1_df.T,columns=features)
print("Synthetic data using make_classification")
print(df1.head().to_string())
print(df1.tail().to_string())

# Q8. Pairwise plot of first 5 features of previous dataset
sbn.pairplot(df1[['feature1','feature2','feature3','feature4','feature5']],kind='kde')
plt.show()

# Q9. Generate isotropic Gaussian blobs
X2,Y2=make_blobs(n_samples=5000,centers=4,n_features=2,random_state=5805)
Y2=np.array([Y2])
X2_df = np.concatenate([X2.T,Y2],axis=0)
df2=pd.DataFrame(data=X2_df.T,columns=['feature1','feature2','target'])
print("Synthetic isotropic Gaussian blobs")
print(df2.head().to_string())
print(df2.tail().to_string())

# Q10. Scatter plot of feature1 and feature2
sbn.scatterplot(x=df2['feature1'],y=df2['feature2'],hue=df2['target'])
plt.show()