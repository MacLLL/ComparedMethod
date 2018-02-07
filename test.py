basePath = "/home/luy100/ForVLDB/ComparedMethod/Datasets"


if __name__ == '__main__':
    groundtruth = open(basePath + '/glove/glove.twitter.27B.100d.2000queryGT', 'r').readlines()

    for line in groundtruth:
        line=line.replace("[","").replace("]\n","").replace(" ","").split(",")
        print(line)
        one=[]
        for item in line:
            one.append(int(item))
        print(set(one))
        break



