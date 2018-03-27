i=0
with open('atis_train.tsv') as f:
    for line in f.readlines():
        w = open(str(i)+'.atis.tsv','w')
        i = i +1
        w.write(line)
        w.close()


