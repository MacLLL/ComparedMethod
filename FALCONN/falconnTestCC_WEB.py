from __future__ import print_function
import numpy as np
import falconn
import timeit
import math
import os
from operator import itemgetter
import sklearn.preprocessing

def getSortedDirList(path):

    l1 = os.listdir(path)
    if l1.__contains__('.DS_Store'):
        l1.remove('.DS_Store')
    # print(l1)
    l2 = []
    for item in l1:
        tmp = [item.split("_"), item]
        l2.append(tmp)
    # print(l2)
    for item in l2:
        item[0][1] = int(item[0][1])
    l3 = sorted(l2, key=itemgetter(0, 1))
    # print(l3)
    sortedDir = []
    for item in l3:
        sortedDir.append(item[1])
    # print(sortedDir)
    return sortedDir

def getAllVideoDenseVector(mainPath,categoryNum,descriptorNum):
    bigList = os.listdir(mainPath)
    if bigList.__contains__('.DS_Store'):
        bigList.remove('.DS_Store')
    # sorted the bigList
    bigList = [int(item) for item in bigList]
    bigList = sorted(bigList)
    bigList = [str(item) for item in bigList]
    # print(bigList)
    # print(len(bigList))
    allDenseVectors = []
    dirNum=bigList[categoryNum]
        # print(dirNum)
    tmpdir=getSortedDirList(mainPath+"/"+dirNum)
    # print(tmpdir)
    descriptor = ["HSV_blue", "HSV_green", "HSV_red"]
    for item in tmpdir:
        # c=[]
        # for des in descriptor:
        c = np.load("/Users/eternity/PycharmProjects/transferVector/CC_WEB_VIDEO_d_256/videoDescriptor_256HSV/" +
                        dirNum + "/" + item + "/" + descriptor[descriptorNum] + ".npy")
            # c.append(c1)
        # c=np.reshape(c,(1,768))
        # c=np.reshape(c,768)
        allDenseVectors.append(c)
    print("finish dir {0}".format(dirNum))
    print("length is {0}".format(len(allDenseVectors)))

    return sklearn.preprocessing.normalize(allDenseVectors)

def getQueryAndGroundTruth(categoryNum):
    file=open('Test_ES.rst','r')
    s=file.readlines()[categoryNum].replace("[",'').replace("]",'').replace(' ','')
    totalNum,ENum,SNum,Eset,Sset=s.split("#")
    #Todo 做一下Set的切分，现在还是string
    Eset=Eset.split(",")
    Sset=Sset.split(",")
    Eset=[int(x) for x in Eset]
    Sset=[int(x) for x in Sset]
    # Eset=[x.replace('[','') for x in Eset]
    # Eset=[x.replace(']','') for x in Eset]
    # Sset=[x.replace('[','') for x in Sset]
    # Sset=[x.replace(']','') for x in Sset]

    return (int(totalNum),int(ENum),int(SNum),Eset,Sset)

if __name__ == '__main__':
    # categoryNum from 0 -23
    categoryNum = 0
    allDenseVector_HSV_blue = getAllVideoDenseVector(
        "/Users/eternity/PycharmProjects/transferVector/CC_WEB_VIDEO_d_256/videoDescriptor_256HSV",
        categoryNum, 0)
    allDenseVector_HSV_green = getAllVideoDenseVector(
        "/Users/eternity/PycharmProjects/transferVector/CC_WEB_VIDEO_d_256/videoDescriptor_256HSV",
        categoryNum, 1)
    allDenseVector_HSV_red = getAllVideoDenseVector(
        "/Users/eternity/PycharmProjects/transferVector/CC_WEB_VIDEO_d_256/videoDescriptor_256HSV",
        categoryNum, 2)
    (totalNum, ENum, SNum, Eset, Sset) = getQueryAndGroundTruth(categoryNum)

    number_of_queries=len(Sset)
    number_of_tables=100
    dataset=allDenseVector_HSV_blue
    QueryIndexWithVector_blue = [[int(i), allDenseVector_HSV_blue[int(i)]] for i in Sset]
    queries=[ x[1] for x in QueryIndexWithVector_blue]
    print('Centering the dataset and queries')
    center = np.mean(dataset, axis=0)
    dataset -= center
    queries -= center
    print('Done')
    # assert dataset.dtype == np.float32

    params_cp = falconn.LSHConstructionParameters()


