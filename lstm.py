import numpy as np
import matplotlib.pyplot as plt
from keras.models import Sequential
from keras.layers import Dense
from keras.layers import LSTM
import pandas as pd
from keras.models import Sequential, load_model
from sklearn.preprocessing import StandardScaler
from sklearn.preprocessing import MinMaxScaler

# 读取数据
dataframe = pd.read_csv('SARS.csv',usecols=[1])

for i in range(dataframe['total'].shape[0]):
    if dataframe['total'][i] == 0:
        j=i+1
        while(dataframe['total'][j]==0):
            j+=1
        dataframe['total'][i]=(dataframe['total'][i-1]+dataframe['total'][j])//2

dataset = dataframe.values

dataset = dataset.astype('float32')
#归一化
scaler = MinMaxScaler(feature_range=(0, 1))
dataset = scaler.fit_transform(dataset)

def create_dataset(dataset, timestep ):
    dataX, dataY = [], []
    for i in range(len(dataset)-timestep -1):
        a = dataset[i:(i+timestep )]
        dataX.append(a)
        dataY.append(dataset[i + timestep ])
    return np.array(dataX),np.array(dataY)
#训练数据太少 timestep 取2
timestep  = 1
trainX,trainY  = create_dataset(dataset,timestep )

trainX = np.reshape(trainX, (trainX.shape[0], trainX.shape[1], 1))
# create and fit the LSTM network
model = Sequential()
model.add(LSTM(4, input_shape=(None,1)))
model.add(Dense(1))
model.compile(loss='mean_squared_error', optimizer='adam')
model.fit(trainX, trainY, epochs=100, batch_size=1, verbose=2)
model.save("LSTM.h5")

trainPredict = model.predict(trainX)
#反归一化
trainPredict = scaler.inverse_transform(trainPredict)
trainY = scaler.inverse_transform(trainY)


plt.figure(figsize=(10, 8))
plt.plot(trainY[1:])
plt.plot(trainPredict)
plt.title('训练集上的结果')
plt.xlabel('天数')
plt.ylabel('感染人群数目')
plt.legend(['train', 'trainPredict'], loc='best')
plt.show()



# 读取新型冠状病毒数据进行测试
dataframe = pd.read_csv('Shenzhen.csv',usecols=[1])
dataset = dataframe.values
dataset = dataset.astype('float32')
#归一化
scaler = MinMaxScaler(feature_range=(0, 1))
dataset = scaler.fit_transform(dataset)

testdata,TEXTY  = create_dataset(dataset,1)
res= model.predict(testdata)
# 结果反归一化
res = scaler.inverse_transform(res)
TEXTY = scaler.inverse_transform(TEXTY)


# 画图
plt.figure(figsize=(10, 8))
plt.plot(TEXTY[1:])
plt.plot(res)
plt.xlabel('天数')
plt.ylabel('感染人群数目')
plt.title('分析疫情的结果')
plt.legend(['train', 'trainPredict'], loc='best')
plt.show()
