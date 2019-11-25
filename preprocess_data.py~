import sys
import math
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import matplotlib.lines as mlines
pi= 3.14159265359

def input_data(filename):
    data = pd.read_csv(filename, sep='\s+', header = 0)
    return data
def boundary_1(x, mu_0,mu_1, Sigma_c0, Sigma_c1,P_w0=0.5,P_w1=0.5):
    sigma_sq=0.5*(np.linalg.det(mu_0)+np.linalg.det(mu_1))
    c0=0.5*(mu_0+mu_1)-(sigma_sq/np.linalg.det(mu_0-mu_1)^2)*ln(P_w0/P_w1)

data_tr=input_data('synth.tr')
data_tr_c0=data_tr.loc[data_tr.yc==0]# splits data into 2 classes
data_tr_c1=data_tr.loc[data_tr.yc==1]

n_c0=count_row=data_tr_c0.shape[0]#total number of data points calculation
n_c1=count_row=data_tr_c1.shape[0]

mu_c0=(1/n_c0)*data_tr_c0['xs'].sum()#MLE for mu calculation
mu_c1=(1/n_c1)*data_tr_c1['xs'].sum()


print("mu_c0=",mu_c0,"\t")
print("mu_c1=",mu_c1,"\t")

sigma_c0=math.sqrt((1/n_c0)*((data_tr_c0['xs']-mu_c0)**2).sum())#MLE for sigma calculation
sigma_c1=math.sqrt((1/n_c1)*((data_tr_c1['xs']-mu_c1)**2).sum())

print("sigma_c0=",sigma_c0,"\t")
print("sigma_c1=",sigma_c1,"\t")

t=np.arange(-1.5,1.0,0.01)
mu_c0=t*(mu_c0)/t
mu_c1=t*(mu_c1)/t
sigma_c0=t*(sigma_c0)/t
sigma_c1=t*(sigma_c1)/t
plt.scatter(data_tr_c0['xs'],data_tr_c0['ys'],c='DarkBlue')# plot data
plt.scatter(data_tr_c1['xs'],data_tr_c1['ys'],c='DarkRed')

plt.plot(t,gauss_dist(t,mu_c0,sigma_c0),c='DarkBlue')#plot MLE
plt.plot(t,gauss_dist(t,mu_c1,sigma_c1),c='DarkRed')

plt.xlabel('xs')
plt.ylabel('ys')
plt.xlim(-1.5,1.0)
c_1_leg= mlines.Line2D([], [], color='DarkBlue', marker='',markersize=15)
c_2_leg= mlines.Line2D([], [], color='DarkRed', marker='',markersize=15)
plt.legend((c_1_leg,c_2_leg),('class 0', 'class 1'))
plt.title('synth.tr data with maximum likelihood estimators')
plt.show()

