
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

testdataI,TEXTYI  = create_dataset(datasetI,7)
resI= model.predict(testdataI)
# 结果反归一化
resI = scaler.inverse_transform(resI)
TEXTYI = scaler.inverse_transform(TEXTYI)


testdataS,TEXTYS  = create_dataset(datasetS,7)
resS= model.predict(testdataS)
# 结果反归一化
resS = scaler.inverse_transform(resS)
TEXTYS = scaler.inverse_transform(TEXTYS)


testdataU,TEXTYU  = create_dataset(datasetU,7)
resU= model.predict(testdataU)
# 结果反归一化
resU = scaler.inverse_transform(resU)
TEXTYU = scaler.inverse_transform(TEXTYU)

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
