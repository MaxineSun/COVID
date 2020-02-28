import csv
csvFile = open("DXYArea.csv","r")
reader = csv.reader(csvFile)

headers = next(reader)
for i in range(15):
    name = headers[i]
    locals()[headers[i]]=[]
    for row in reader:
        locals()[headers[i]].append(row[i])
        #print(locals()[headers[i]])
print(locals()[headers[0]])
csvFile.close()
