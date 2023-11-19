import pandas as pd
import matplotlib.pyplot as plt
from statistics import median
import numpy as np

class pre_processing:
    c_DF=pd.DataFrame()
    def __init__(self,a_DF=None):
        self.c_DF=a_DF
    def Normalized(self):
        w_norm_df=pd.DataFrame()
        for i in self.c_DF.columns:
            w_norm_df[i]=(self.c_DF[i] - self.c_DF[i].min()) / (self.c_DF[i].max() - self.c_DF[i].min())
        return w_norm_df
    def Standardized(self):
        w_std_df=pd.DataFrame()
        for i in self.c_DF.columns:
            w_std_df[i]=(self.c_DF[i] - self.c_DF[i].mean())/self.c_DF[i].std()
        return w_std_df
    def IQR(self):
        w_iqr_df=pd.DataFrame()
        for i in self.c_DF.columns:
            medn=median(self.c_DF[i])
            q3,q1=np.percentile(self.c_DF[i],[75,25])
            iqr=q3-q1
            w_iqr_df[i]=(self.c_DF[i]-medn)/iqr
        return w_iqr_df
    def Show_original(self):
        plt.figure()
        self.c_DF.plot()
        plt.title("Plotting all features of original dataframe")
        plt.grid(axis='y')
        plt.show()
    def Show_normalized(self):
        norm_df=self.Normalized()
        plt.figure()
        norm_df.plot()
        plt.title("Plotting all features of Normalized dataframe")
        plt.grid(axis='y')
        plt.show()
    def Show_standardized(self):
        stand_df=self.Standardized()
        plt.figure()
        stand_df.plot()
        plt.title("Plotting all features of Standardized dataframe")
        plt.grid(axis='y')
        plt.show()
    def Show_IQR(self):
        iqr_df=self.IQR()
        plt.figure()
        iqr_df.plot()
        plt.title("Plotting all features of dataframe scaled with median and IQR")
        plt.grid(axis='y')
        plt.show()
