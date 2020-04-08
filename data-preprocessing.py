import csv
import matplotlib.pyplot as plt
import numpy as np

csvFile = open("DXYArea.csv","r")
reader = csv.reader(csvFile)
headers = next(reader)

provinceName = []
provinceEnglishName = []
province_zipCode = []
cityName = []
cityEnglishName = []
city_zipCode = []
province_confirmedCount = []
province_suspectedCount = []
province_curedCount = []
province_deadCount = []
city_confirmedCount = []
city_suspectedCount = []
city_curedCount = []
city_deadCount = []
updateTime = []

for row in reader:
    provinceName.append(row[0])
    provinceEnglishName.append(row[1])
    province_zipCode.append(row[2])
    cityName.append(row[3])
    cityEnglishName.append(row[4])
    city_zipCode.append(row[5])
    province_confirmedCount.append(row[6])
    province_suspectedCount.append(row[7])
    province_curedCount.append(row[8])
    province_deadCount.append(row[9])
    city_confirmedCount.append(row[10])
    city_suspectedCount.append(row[11])
    city_curedCount.append(row[12])
    city_deadCount.append(row[13])
    updateTime.append(row[14])
csvFile.close()

Guangzhou = np.where(np.array(cityEnglishName) == 'Guangzhou')
Guangzhou_confirmedCount = np.array(city_confirmedCount)[Guangzhou]
Guangzhou_confirmedCount = list(map(int,Guangzhou_confirmedCount))
Guangzhou_curedCount = np.array(city_curedCount)[Guangzhou]
Guangzhou_curedCount = list(map(int,Guangzhou_curedCount))
Guangzhou_deadCount = np.array(city_deadCount)[Guangzhou]
Guangzhou_deadCount = list(map(int,Guangzhou_deadCount))
# Guangzhou_remainedCount = Guangzhou_confirmedCount - Guangzhou_curedCount
# Guangzhou_remainedCount = Guangzhou_remainedCount - Guangzhou_deadCount
Guangzhou_updateTime = np.array(updateTime)[Guangzhou]
# print(type(Guangzhou))
# print(type(Guangzhou_confirmedCount))
# print(Guangzhou_updateTime)
# print(Guangzhou_confirmedCount)

#plt.plot((np.arange(len(Guangzhou_confirmedCount),0,-1)),[Guangzhou_confirmedCount[i] - Guangzhou_curedCount[i] - Guangzhou_deadCount[i] for i in range(0,len(Guangzhou_confirmedCount))])
plt.plot(,[Guangzhou_confirmedCount[i] for i in range(0,len(Guangzhou_confirmedCount))])
plt.show()
