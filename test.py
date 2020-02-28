a = [e,b,c,d]
for i in range(4):
    name = a[i]
    locals()[a[i]]=i
print(locals()['c'])
