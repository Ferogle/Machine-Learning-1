import matplotlib.pyplot as plt
from matplotlib.lines import Line2D
import pandas as pd
from prettytable import PrettyTable
from HW2_Q1 import pre_processing
from scipy.spatial.distance import minkowski
from pandas_datareader import data
import numpy as np
import yfinance as yf
yf.pdr_override()
np.random.seed(5808)
# Q2
df_appl=data.get_data_yahoo('AAPL',start="2000-01-01",end="2022-09-25")
pp=pre_processing(df_appl)
pp.Show_original()
pp.Show_normalized()
pp.Show_standardized()
pp.Show_IQR()

# Q3.
p_values = [0.5, 1.0, 1.5, 2, 2.5, 3.0, 3.5, 4.0, 4.5, 5.0, 5.5, 6.0]
X, Y = np.meshgrid(np.linspace(-1, 1, num=1000), np.linspace(-1, 1, num=1000))
colors=['red','blue','green','orange','violet','cyan','magenta','yellow','black','pink','brown','olive']
k=0
legend_handles=[]
for p in p_values:
    mdis = ((np.abs((X))**p) + (np.abs((Y))**p))**(1./p)
    cs=plt.contour(X,Y,mdis,[1],colors=colors[k],linestyles='dotted')
    legend_handles.append(Line2D([0],[0], color=colors[k], linestyle='dotted', label=str(p)+" norm"))
    k+=1
plt.title("L-norm")
plt.axis('equal')
plt.xlim(-1,1)
plt.ylim(-1,1)
plt.gca().set_aspect("equal",adjustable='box')
plt.tight_layout()
plt.legend(handles=legend_handles)
plt.grid()
plt.show()

# Q6a
def covX(X):
    mnX=np.mean(X,axis=0)
    n=len(X)
    X=X-mnX
    return ((X.T) @ X)/(n-1)
x=np.random.normal(1,np.sqrt(2),1000)
y=x+np.random.normal(2,np.sqrt(3),1000)
X=np.array([x,y]).T
ecovX=covX(X)
pt=PrettyTable()
pt.title="Estimated covariance Matrix"
pt.field_names=['',"x","y"]
pt.add_row(["x",round(ecovX[0][0],2),round(ecovX[0][1],2)])
pt.add_row(["y",round(ecovX[1][0],2),round(ecovX[1][1],2)])
print(pt)

# Q6b.
evl,evc=np.linalg.eig(ecovX)
pt=PrettyTable()
pt.title="Eigen value and Eigen vector of covariance matrix"
pt.field_names=["","Lambda1","Lambda2"]
pt.add_row(["Eigen value", evl[0].round(2),evl[1].round(2)])
pt.add_row(["Eigen vector", evc[0].round(2),evc[1].round(2)])
print(pt)
#
# # Q6c.
origin=[0,0]
# plt.figure(figsize=(12,12))
plt.scatter(x,y)
QV1=plt.quiver(*origin,*evc[:,0],color=['r'],scale=evl[1]*5,label="Min eigen vector")
QV2=plt.quiver(*origin,*evc[:,1],color=['y'],scale=evl[0]*5,label="Max eigen vector")
plt.title("Scatter plot and Eigen vectors of x & y")
plt.xlabel("x values")
plt.ylabel("y values")
plt.legend(handles=[QV1,QV2])
plt.axis('equal')
plt.grid()
plt.show()

# Q6.d
singular_values=np.linalg.svd(X,compute_uv=False)
pt=PrettyTable()
pt.title="Singular values of feature matrix X"
pt.field_names=['Singular value 1','Singular value 2']
pt.add_row([singular_values[0].round(2),singular_values[1].round(2)])
print(pt)

# Q6.d verification of relation between singular values and eigen values
x1=np.random.normal(0,np.sqrt(2),1000)
y1=x+np.random.normal(0,np.sqrt(3),1000)
X1=np.array([x1,y1]).T
ecovX1=covX(X1)
evl1,evc1=np.linalg.eig(ecovX1)
sin_val=np.linalg.svd(X1,compute_uv=False)
plt.scatter(x1,y1)
QV1=plt.quiver(*origin,*evc[:,0],color=['r'],scale=evl[1]*5,label="Min eigen vector")
QV2=plt.quiver(*origin,*evc[:,1],color=['y'],scale=evl[0]*5,label="Max eigen vector")
plt.title("Scatter plot and Eigen vectors of x1 & y1")
plt.xlabel("x1 values")
plt.ylabel("y1 values")
plt.legend(handles=[QV1,QV2])
plt.axis('equal')
plt.grid()
plt.show()


# Q6e
Xcorr=np.corrcoef(X[:,0],X[:,1])
print(Xcorr.round(2))


# Q7.
# x_time=np.array(list(i for i in range(1,501)))
x_time=np.array(list(i for i in range(-4,5,1)))
y_time=x_time**3
y_time=pd.DataFrame(y_time,columns=['original'])
y_time['diff_order_1']=y_time['original'].diff()
y_time['diff_order_2']=y_time['diff_order_1'].diff()
y_time['diff_order_3']=y_time['diff_order_2'].diff()
y_time.plot()
plt.title("y, dy, d2y, d3y plots")
plt.xlabel("x(t)")
plt.ylabel("y(t)")
plt.legend()
plt.grid()
plt.show()



