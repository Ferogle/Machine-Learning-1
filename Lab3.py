import pandas as pd
import numpy as np
import seaborn as sns
import statsmodels.api as sm
from sklearn.metrics import mean_squared_error
from sklearn.preprocessing import StandardScaler
from sklearn.decomposition import PCA
from sklearn.ensemble import RandomForestRegressor
from sklearn.model_selection import train_test_split
from sklearn.model_selection import GridSearchCV
from sklearn.linear_model import LinearRegression
from sklearn.preprocessing import PolynomialFeatures
from sklearn.pipeline import Pipeline
from prettytable import PrettyTable
pd.set_option('display.float_format', lambda x: f'{x:.3f}')
sns.set_style("whitegrid")
np.random.seed(5805)

df=pd.read_csv("https://raw.githubusercontent.com/rjafari979/Information-Visualization-Data-Analytics-Dataset-/main/Carseats.csv")

# Q1 a.
import matplotlib.pyplot as plt
df1=df[['ShelveLoc','Sales','US']]
df1=df1.groupby(['ShelveLoc','US']).sum().reset_index()
print(df1.head())
plt.title("ShelveLoc vs. Sales in US")
plt.xlabel("Sales")
plt.ylabel("ShelveLoc")
sns.barplot(data=df1,x='Sales',y='ShelveLoc',hue='US')
plt.grid()
plt.show()

# Q1 b.
df=pd.get_dummies(df, columns=['ShelveLoc'],drop_first=True)
df['Urban']=df['Urban'].map({'Yes':1,'No':0})
df['US']=df['US'].map({'Yes':1,'No':0})
print(df.head().to_string())

mean_sales=np.mean(df['Sales'])
std_sales=np.std(df['Sales'])
X_price_orig=df['Price']
Y_sales_orig=df['Sales']

# Q1 c.
ss=StandardScaler()
features=['Sales','CompPrice','Income','Advertising','Population','Price','Age','Education']
for i in features:
    df[i]=ss.fit_transform(df[[i]])
# print(df.head().to_string())
X=df.drop('Sales',axis=1)
X=sm.add_constant(X)
Y=df['Sales']
X_train,X_test,Y_train,Y_test=train_test_split(X,Y,test_size=0.2,random_state=5805,shuffle=True)
print("X_train")
print(X_train.round(3).head().to_string())
print("X_test")
print(X_test.round(3).head().to_string())
print("Y_train")
print(Y_train.round(3).head().to_string())
print("Y_test")
print(Y_test.round(3).head().to_string())

pt=PrettyTable()
pt.title="Dropping features according to p-values and Adj R-squared"
pt.field_names=['Dropped feature','AIC','BIC','Adj R-squared','p-value']

# Q2 a.
# Using all features
X_orig=X_train.copy()
model=sm.OLS(Y_train,X_train).fit()
print(model.summary())
pt.add_row(['None',round(model.aic,3),round(model.bic,3),round(model.rsquared_adj,3),'None'])

# Dropping 'Population' feature
pvalue=round(model.pvalues[list(X_train.columns).index('Population')],3)
X_train.drop(['Population'],axis=1,inplace= True)
model = sm.OLS(Y_train,X_train).fit()
print(model.summary())
pt.add_row(['Population',round(model.aic,3),round(model.bic,3),round(model.rsquared_adj,3),round(pvalue,3)])

# Dropping 'Education' feature
pvalue=round(model.pvalues[list(X_train.columns).index('Education')],3)
X_train.drop(['Education'],axis=1, inplace= True)
model = sm.OLS(Y_train,X_train).fit()
print(model.summary())
pt.add_row(['Education',round(model.aic,3),round(model.bic,3),round(model.rsquared_adj,3),round(pvalue,3)])

# Dropping 'US' feature
pvalue=round(model.pvalues[list(X_train.columns).index('US')],3)
X_train.drop(['US'],axis=1, inplace= True)
model = sm.OLS(Y_train,X_train).fit()
print(model.summary())
pt.add_row(['US',round(model.aic,3),round(model.bic,3),round(model.rsquared_adj,3),round(pvalue,3)])

# Dropping 'Urban' feature
pvalue=round(model.pvalues[list(X_train.columns).index('Urban')],3)
X_train.drop(['Urban'],axis=1,inplace= True)
model = sm.OLS(Y_train,X_train).fit()
print(model.summary())
pt.add_row(['Urban',round(model.aic,3),round(model.bic,3),round(model.rsquared_adj,3),round(pvalue,3)])
print(pt)

dropped_features=['Population','Education','US','Urban']
print("The dropped features are",dropped_features)
print("features in final dataset are", X_train.columns)

pt=PrettyTable()
pt_2_4=PrettyTable()
pt_2_4.title="Comparing Backward Stepwise and Random Forest"
pt_2_4.field_names=["Score","Backward Stepwise","Random Forest"]
pt.title="Test set vs. Predicted set using Backward Stepwise"
pt.field_names=['Y_test','Y_predicted']
print(X_train.shape)
X_test_final=X_test[list(X_train.columns)]
model_ols=sm.OLS(Y_train,X_train).fit()
Y_pred=model_ols.predict(X_test_final)
Y_test_inv = (Y_test*std_sales)+mean_sales
Y_pred_inv_bs = (Y_pred*std_sales)+mean_sales
print("The coefficients of the OLS model using the final set of features")
print(model_ols.params)
for i,j in zip(Y_test_inv,Y_pred_inv_bs):
    pt.add_row([round(i,3),round(j,3)])
print(pt)

# Q2.c
index=np.arange(len(Y_test))
plt.plot(index,Y_test_inv)
plt.plot(index,Y_pred_inv_bs)
plt.title("Actual values vs. Predicted using Backward stepwise regression feature selection and OLS")
plt.xlabel("# of samples")
plt.ylabel("Sales")
plt.legend(['Actual test set values','Predicted values of test set'])
plt.show()

# Q2.d
mse_ols=round(mean_squared_error(Y_test_inv,Y_pred_inv_bs),3)
print(f"Mean squared error with backward stepwise regression {mse_ols}")

# Q3.
pca=PCA()
pca.fit(X_orig)
print("PCA X_orig features", X_orig.columns)

# Q3.a
print(pca.explained_variance_ratio_)
cum_var=np.round(np.cumsum(sorted(pca.explained_variance_ratio_,reverse=True))*100,decimals=3)
print(cum_var)

# Q3. b & c
labels=[i for i in range(1,len(cum_var)+1)]
plt.bar(x=labels,height=cum_var,alpha=0.5)
plt.xlabel("Number of features")
plt.ylabel("Percentage contribution to variance")
plt.title("Cumulative percentage contribution of each feature to total variance")
plt.axhline(y=90)
plt.axvline(x=7)
plt.show()

# Q4.a
model_rf=RandomForestRegressor(random_state=5805)
model_rf.fit(X_orig, Y_train)
features=X_orig.columns
importances=model_rf.feature_importances_
indices=np.argsort(importances)
importances=np.array(importances)[indices]
features=np.array(features)[indices]
print(importances)
print(features)
plt.title("Feature importances")
plt.barh(range(len(importances)),importances,align='center')
plt.yticks(range(len(importances)),features)
plt.xlabel("Relative importance")
plt.tight_layout()
plt.show()

# Q4.b & c
# Assuming threshold 0.05
dropped_features=['const','US','Urban','Education','Population']
print("Assuming a threshold of 0.05, the eliminated features are", dropped_features)
X_train_drop=X_orig.drop(dropped_features,axis=1)
X_test_drop=X_test.drop(dropped_features,axis=1)
print("The features selected in the final dataset", X_train_drop.columns)
model_rf_ols=sm.OLS(Y_train,X_train_drop).fit()
print(model_rf_ols.summary())

# Q4.d
Y_test_pred = model_rf_ols.predict(X_test_drop)
Y_pred_inv = (Y_test_pred*std_sales)+mean_sales
plt.title("Prediction with Random Forest")
plt.plot(index,Y_test_inv)
plt.plot(index,Y_pred_inv)
plt.xlabel("# of samples")
plt.ylabel("Sales")
plt.legend(['Actual values of test set','Predicted values with test set'])
plt.show()

# Q4.e
mse_rf=round(mean_squared_error(Y_test_inv,Y_pred_inv),3)
print(f"Mean squared error using random forest analysis {mse_rf}")

# Q5
pt_2_4.add_row(["R-squared",round(model_ols.rsquared,3),round(model_rf_ols.rsquared,3)])
pt_2_4.add_row(["Adj. R-squared",round(model_ols.rsquared_adj,3),round(model_rf_ols.rsquared_adj,3)])
pt_2_4.add_row(["AIC",round(model_ols.aic,3),round(model_rf_ols.aic,3)])
pt_2_4.add_row(["BIC",round(model_ols.bic,3),round(model_rf_ols.bic,3)])
pt_2_4.add_row(["MSE",mse_ols,mse_rf])
print(pt_2_4)

###############################################################################################################

# Q6.
Y_prediction=model_ols.get_prediction(X_test_final)
ci_pred_frame=Y_prediction.summary_frame(alpha=0.05)
ci_upper = ci_pred_frame.obs_ci_upper
ci_lower = ci_pred_frame.obs_ci_lower

plt.plot(index,Y_pred_inv_bs,label="Predicted",color='blue')
plt.fill_between(index,ci_upper*std_sales+mean_sales,ci_lower*std_sales+mean_sales,color='blue',alpha=0.3,label="Confidence Interval")
plt.xlabel("Number of observations")
plt.ylabel("Sales")
plt.title("Confidence interval of Predicted values")
plt.legend()
plt.show()

###################################################################################################################

# Q7.a
X_price_poly=X["Price"]
Y_sales_poly=Y.copy()
Y__test_sales_poly_inv = (Y_sales_poly*std_sales)+mean_sales
param_grid = {'polynomial_features__degree': [i for i in range(1,16)]}
pipeline = Pipeline([ ('polynomial_features',PolynomialFeatures()), ('linear_regression',LinearRegression())])
grid_search=GridSearchCV(estimator=pipeline,param_grid=param_grid,scoring='neg_root_mean_squared_error')
grid_search.fit(X_price_poly.values.reshape(-1,1),Y_sales_poly)
best_degree=grid_search.best_params_['polynomial_features__degree']

# Q7.b
print(f"The optimal order for n from GridSearchCV is {best_degree}")

# Q7.c
cv_results = grid_search.cv_results_
rmse_scores=-cv_results['mean_test_score']
print(rmse_scores)
plt.plot(param_grid['polynomial_features__degree'],rmse_scores)
plt.title("RMSE vs. Degree of Polynomial feature")
plt.xlabel("Degree of polynomial feature")
plt.ylabel("RMSE")
plt.grid()
plt.show()

# Q7.d
poly=PolynomialFeatures(degree=best_degree)
poly_features=poly.fit_transform(X_price_poly.values.reshape((-1,1)))
poly_feature_names=poly.get_feature_names_out(['Price'])
poly_df=pd.DataFrame(poly_features, columns=poly_feature_names)
print(poly_df.head().to_string())
X_poly_train,X_poly_test,Y_poly_train,Y_poly_test=train_test_split(poly_df,Y_sales_orig,test_size=0.2,random_state=5805)
model_poly=sm.OLS(Y_poly_train,X_poly_train).fit()
print(model_poly.summary())

index=np.arange(len(Y_poly_test))
Y_poly_pred=model_poly.predict(X_poly_test)
plt.title("Predicted Sales vs. Actual Values using GridSearchCV and OLS")
plt.plot(index,Y_poly_test,label="Actual test set values")
plt.plot(index,Y_poly_pred,label="Predicted values from test set")
plt.xlabel("# of samples")
plt.ylabel("Sales")
plt.legend()
plt.show()

# Q7.e
print(f"Mean square error using polynomial features in linear regression {round(mean_squared_error(Y_poly_pred,Y_poly_test),3)}")
