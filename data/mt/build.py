de =[]
en =[]
for line in file("europarl.de.de"):
    de.append(line.strip('\n'))
for line in file("europarl.de.en"):
    en.append(line.strip('\n'))
f = open("europarl.en-de","w")

for i in range(30000):
    f.write(en[i]+"\t"+de[i]+'\n')
f.close()

