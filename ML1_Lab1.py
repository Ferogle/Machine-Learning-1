import seaborn as sbn
import numpy as np
import matplotlib.pyplot as plt
from scipy import stats
#Q1 Print names in such a way that it is visible at once on the console
names = sbn.get_dataset_names()
for i in range(len(names)):
    print(names[i], end=" ")
    if i%5==0:
        print()
print()

# Q2 Print name of each dataset along with their shape
datasets=["diamonds","iris","tips","penguins","titanic"]

df_arr=[]
# load each dataset into a dataframe and store it in an array
for i in datasets:
    df_arr.append(sbn.load_dataset(i))
    print(df_arr[-1].shape)  # print shape of the last loaded dataset

# Q3 Print the count,mean,std,25%,50%,75% amd max of numerical features of the dataset using desribe() method of pandas
print(df_arr[-1].describe())
    # Missing values are calculated with isnull() and then summing the truth values
print("Yes there are missing values\nThe number of missing values in the dataset are ",df_arr[-1].isnull().sum().sum())
print(df_arr[-1].head())

# Q4. Dataframe with numercial features
    # Select numerical features from the original dataset which are age,sibsp,parch,fare
df_titanic_num = df_arr[-1][['age','sibsp','parch','fare']].copy()
    # Print first 5 rows of original dataset
print(df_arr[-1].head().to_string())
    # Print first 5 rows of new dataset with only numerical features
print(df_titanic_num.head().to_string())

# Q5 Calculate percentage of missing values
    # Removing those observations which has missing attribute values using dropna()
df_cleaned=df_titanic_num.dropna()
print(df_cleaned.shape)
print(df_titanic_num.shape)
print("The number of missing observations are: ",df_titanic_num.shape[0]-df_cleaned.shape[0])
    # Calculate percentage of the removed observations
percentage=(((df_titanic_num.shape[0]-df_cleaned.shape[0])*100)/df_titanic_num.shape[0])
print(f"The percentage of observations that were eliminated as part of cleaning the dataset: {percentage:.2f}")

# Q8. Calculate and print AM, GM and HM
print("Arithmetic Means of numerical features")
ameans=np.mean(df_cleaned,axis=0)
for i in range(len(df_cleaned.columns)):
    print(f"{df_cleaned.columns[i]} {ameans[i]:.2f}")
print("Geometric Means of numercial features")
gmeans=stats.gmean(df_cleaned,axis=0)
    # Print GM of each column along with the column name
for i in range(len(df_cleaned.columns)):
    print(f"{df_cleaned.columns[i]} {gmeans[i]:.2f}")

print("Harmonic means of numerical features")
hmeans=stats.hmean(df_cleaned,axis=0)
    # Print HM of each column along with the column name
for i in range(len(df_cleaned.columns)):
    print(f"{df_cleaned.columns[i]} {hmeans[i]:.2f}")

# Q9. Histogram plot of age and fare attributes
plt.hist(df_cleaned['age'])
plt.title("Histogram plot of Age")
plt.xlabel('Age')
plt.ylabel('Number of observations')
plt.show()
plt.hist(df_cleaned['fare'])
plt.title("Histogram plot of Fare")
plt.xlabel('Fare($)')
plt.ylabel('NUmber of observations')
plt.show()

# Q10 Shows the pairwise bivariate distribution plot using kernel density estimation
sbn.pairplot(df_cleaned, diag_kind='kde', markers='o')
plt.show()