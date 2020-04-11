import numpy as np
import matplotlib.pyplot as plt
import pandas as pd
from scipy.optimize import curve_fit

data = pd.read_csv('Shenzhen.csv')
t=np.arange(1,data['time'].shape[0]+1,1)
P=data['I']
P=np.array(P)

# 定义逻辑增长函数
def logistic_increase_function(t,K,P0,r):
    t0=1
    r=0.2
#   r值越大，模型越快收敛到K，r值越小，越慢收敛到K
    exp_value=np.exp(r*(t-t0))
    return (K*exp_value*P0)/(K+(exp_value-1)*P0)

# 用最小二乘法估计拟合
popt, pcov = curve_fit(logistic_increase_function, t[:35], P[:35])
#获取popt里面是拟合系数
print("K:",popt[0],"P0:",popt[1],"r:",popt[2])

#拟合后预测的P值
P_t=np.arange(1,100,1)
P_predict = logistic_increase_function(P_t,popt[0],popt[1],popt[2])
#未来预测
future=np.arange(data['time'].shape[0]-40,data['time'].shape[0]+40,2)
future_predict=logistic_increase_function(future,popt[0],popt[1],popt[2])

# 画出预测图
plot1 = plt.plot(t[:35], P[:35], 'o',label="Ialready")

plot2 = plt.plot(future, future_predict, 'o',label='Ipredict')
plot4 = plt.plot(t[36:],P[36:],'o',label="I after")
plot3 = plt.plot(P_t, P_predict, 'r',label='predict curve')

plt.xlabel('days')
plt.ylabel('I number')
plt.legend(loc=0)
plt.show()
