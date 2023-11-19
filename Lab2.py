import pandas as pd
import seaborn as sbn
import numpy as np
import matplotlib.pyplot as plt
from prettytable import PrettyTable
pd.set_option('display.precision', 3)
def covXY(X,Y):
    n=len(X)
    meanX=np.mean(X)
    meanY=np.mean(Y)
    return np.sum((X-meanX)*(Y-meanY))/(n-1)

#Q1 a.
df=pd.read_csv("https://raw.githubusercontent.com/rjafari979/Information-Visualization-Data-Analytics-Dataset-/main/stock%20prices.csv")
missing_values=df.isnull().sum().sum()
print(f"Number of missing values :{missing_values}")
missing_features=df.isnull().any()
missing_feature_names=df.columns[missing_features].to_list()
print(f"Features with missing values: {missing_feature_names}")
print(f"Missing values in each feature:")
for i in missing_feature_names:
    print(i,df[i].isna().sum())
# Q1 b
df.fillna(df.mean(numeric_only=True),inplace=True)

# Q1 c
missing_values_filled=df.isnull().sum().sum()
print(f"Missing values in cleaned dataframe: {missing_values_filled}")

# Q2 a
print("Unique companies in the dataset")
print(df['symbol'].unique())

# Q2 c
df_sliced=df.loc[df['symbol'].isin(['GOOGL','AAPL'])]
print("Dataframe with only GOOGL and AAPL in symbol")
print(df_sliced.head().to_string())
plt.figure(figsize=(12,8))
plt.title("Close vs. Date for GOOGLE and APPLE companies")
df_sliced['date']=pd.to_datetime(df_sliced['date'])
pt=sbn.lineplot(df_sliced,x='date',y='close',hue='symbol')
plt.legend()
plt.show()

# Q3
df_agg=df.groupby('symbol').sum()
print("Aggregated Dataframe")
print(df_agg.head().to_string())
print(f"Cleaned dataset has {df.shape[0]} objects whereas cleaned and aggregated dataset has {df_agg.shape[0]}")

# Q4
df_scv=df[['symbol','close','volume']]
df_cv=df_scv.groupby('symbol').aggregate(close_mean=('close', np.mean),close_var=('close',np.var),volume_mean=('volume',np.mean),volume_var=('volume',np.var)).round(2)
name=df_cv.reset_index('symbol').iloc[np.argmax(df_cv['close_var'])]['symbol']
print(f"The company which has maximum variation in the close cost is {name}")

# Q5
df_goog = df[df['symbol']=='GOOGL']
df_goog = df_goog[df_goog['date']>'2015-01-01']
print("Dataframe with only GOOGL and closing cost after 2015-01-01")
print(df_goog.head())

# Q6.
df_groll=pd.DataFrame()
df_groll['date']=pd.to_datetime(df_goog['date'])
df_groll['close']=df_goog['close'].rolling(30).mean()
print("Number of missing observations when rolling window is applied:",df_groll['close'].isna().sum())
df_goog['date']=pd.to_datetime(df_goog['date'])
df_groll['date']=pd.to_datetime(df_groll['date'])
df_groll['date']=df_groll['date']
plt.figure(figsize=(12,8))
plt.title("Original Close and Rolling Window Mean Close for GOOGLE")
sbn.lineplot(data=df_goog,x='date',y='close')
sbn.lineplot(data=df_groll,x='date',y='close')
plt.grid()
plt.show()

# Q7.
df_goog['price_category']=pd.cut(df_goog['close'],bins=5,labels=['very low','low','normal','high','very high'])
print(df_goog.to_string())
plt.title("Count plot of Close vs. Price Ranges")
sbn.countplot(data=df_goog,x="price_category")
plt.xlabel("Price Range")
plt.tight_layout()
plt.grid(axis='y')
plt.show()

# Q8.
plt.title("Histogram of GOOGL's Close")
sbn.histplot(data=df_goog,x="close",bins=5)
plt.show()

# Q9.
df_goog['price_category']=pd.qcut(df_goog['close'],q=5,precision=0,labels=["very low",'low','normal','high','very high'])
print("Datafram with qcut")
print(df_goog.to_string())
plt.title('Count plot with equal frequency price range')
sbn.countplot(x='price_category',data=df_goog)
plt.grid(axis='y')
plt.tight_layout()
plt.show()

# Q10.
features=['open','high','low','close','volume']
N=len(features)
pt=PrettyTable()
pt.title="Covariance matrix using definition"
pt.field_names = [""]+features
for i in range(N):
    row=[features[i]]
    for j in range(N):
        row.append(round(covXY(df_goog[features[i]],df_goog[features[j]]),3))
    pt.add_row(row)
    row=[]
print(pt)

# Q11
print("Covariance matrix")
print(df_goog[features].cov())
