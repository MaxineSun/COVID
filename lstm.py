import numpy as np
import matplotlib.pyplot as plt
from keras.models import Sequential
from keras.layers import Dense
from keras.layers import LSTM
import pandas as pd
from keras.models import Sequential, load_model
from sklearn.preprocessing import StandardScaler
from sklearn.preprocessing import MinMaxScaler
from sklearn.metrics import r2_score

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
timestep  = 3
trainX,trainY  = create_dataset(dataset,timestep )

trainX = np.reshape(trainX, (trainX.shape[0], trainX.shape[1], 1))
# create and fit the LSTM network
model = Sequential()
model.add(LSTM(4, input_shape=(None,1)))
model.add(Dense(1))
model.compile(loss='mean_squared_error', optimizer='adam')
model.fit(trainX, trainY, epochs=40, batch_size=1, verbose=2)
model.save("LSTM.h5")


# 读取新型冠状病毒数据进行测试
dataframeI = pd.read_csv('Italy.csv',usecols=[1])
dataframeS = pd.read_csv('Spain.csv',usecols=[1])
dataframeU = pd.read_csv('USA.csv',usecols=[1])
datasetI = dataframeI.values
datasetS = dataframeS.values
datasetU = dataframeU.values
datasetI = datasetI.astype('float32')
datasetS = datasetS.astype('float32')
datasetU = datasetU.astype('float32')
#归一化
scaler = MinMaxScaler(feature_range=(0, 1))
datasetI = scaler.fit_transform(datasetI)
datasetS = scaler.fit_transform(datasetS)
datasetU = scaler.fit_transform(datasetU)

testdataI,TEXTYI  = create_dataset(datasetI,3)
resI= model.predict(testdataI)
# 结果反归一化
resI = scaler.inverse_transform(resI)
TEXTYI = scaler.inverse_transform(TEXTYI)
scoreI = r2_score(TEXTYI, resI, multioutput='raw_values')
print(scoreI)

testdataS,TEXTYS  = create_dataset(datasetS,3)
resS= model.predict(testdataS)
# 结果反归一化
resS = scaler.inverse_transform(resS)
TEXTYS = scaler.inverse_transform(TEXTYS)
scoreS = r2_score(TEXTYS, resS, multioutput='raw_values')
print(scoreS)

testdataU,TEXTYU  = create_dataset(datasetU,3)
resU= model.predict(testdataU)
# 结果反归一化
resU = scaler.inverse_transform(resU)
TEXTYU = scaler.inverse_transform(TEXTYU)
scoreU = r2_score(TEXTYU, resU, multioutput='raw_values')
print(scoreU)

# 画图
plt.figure(figsize=(10, 8))
plt.plot(TEXTYI)
plt.plot(resI)
plt.xlabel('date(days)')
plt.ylabel('number of comfirmed(ones)')
plt.legend(['Actual', 'Predicted'], loc='best')
plt.show()

plt.figure(figsize=(10, 8))
plt.plot(TEXTYS)
plt.plot(resS)
plt.xlabel('date(days)')
plt.ylabel('number of comfirmed(ones)')
plt.legend(['Actual', 'Predicted'], loc='best')
plt.show()

plt.figure(figsize=(10, 8))
plt.plot(TEXTYU)
plt.plot(resU)
plt.xlabel('date(days)')
plt.ylabel('number of comfirmed(ones)')
plt.legend(['Actual', 'Predicted'], loc='best')
plt.show()
