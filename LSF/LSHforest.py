from sklearn.neighbors import LSHForest
import os
import numpy as np
from operator import itemgetter
import time
import random
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
        c = np.load("/home/luy100/python_workspace/CC_WEB_VIDEO_d_256/videoDescriptor_256HSV/" +
                        dirNum + "/" + item + "/" + descriptor[descriptorNum] + ".npy")
            # c.append(c1)
        # c=np.reshape(c,(1,768))
        # c=np.reshape(c,768)
        allDenseVectors.append(c)
    print("finish dir {0}".format(dirNum))
    print("length is {0}".format(len(allDenseVectors)))

    return sklearn.preprocessing.normalize(allDenseVectors)

def nomalization(denseVector):
    minimum=0.0
    maximum=0.0
    for i in range(len(denseVector)):
        minimum=min(denseVector[i])
        maximum=max(denseVector[i])
        # print("min={0},max={1}".format(minimum,maximum))
        for j in range(len(denseVector[i])):
            denseVector[i][j]=(denseVector[i][j]-minimum)/(maximum-minimum)

    return denseVector
def getGroundTruth(filePath, number):
    file=open(filePath+"appendTop20NNNomalizedForAllDenseVectors",'r')
    groundTruth=file.readlines()
    for i in range(len(groundTruth)):
        groundTruth[i]=groundTruth[i].split(",")
        groundTruth[i][0]=groundTruth[i][0].replace("[","")
        groundTruth[i][-1]=groundTruth[i][-1].replace("]\n","")
        groundTruth[i]=[ float(x.replace(" ","")) for x in groundTruth[i]]
        # print(groundTruth[i])
    return groundTruth


def calculateEuclideanDistance(vector1,vector2):
    length1=len(vector1)
    length2=len(vector2)
    if(length1!=length2):
        print("two vectors dimension are different, cannot calculate the distance")
        return
    sum=0.0
    for i in range(length1):
        sum+=(vector1[i]-vector2[i])**2
    return sum**0.5

# ratio from 0 to 1, 1 is the best.
def calculateRatio(indices,sampleVectors=[],sampleIndices=[], allDenseVector=[],groundTruth=[]):
    sampleGroundTruth=[ groundTruth[x][:10] for x in sampleIndices]

    sampleDistances=[]
    for i in range(len(indices)):
        oneDistances=[]
        for j in range(len(indices[i])):
            oneDistances.append(calculateEuclideanDistance(sampleVectors[i],allDenseVector[indices[i][j]]))
        sampleDistances.append(sorted(oneDistances))

    print(sampleDistances)
    print(sampleGroundTruth)

    RatioSum=0.0
    for i in range(len(sampleDistances)):

        singleRatio=0.0
        tmp = 0.0
        for j in range(len(sampleDistances[i])):

            if j == 0:
                if sampleDistances[i][j]==0.0:
                    tmp += 1.0
                else:
                    tmp += 0.0
            elif j != 0.0:
                if sampleDistances[i][j]==0.0:
                    tmp += 1.0
                else:
                    tmp = tmp + (sampleGroundTruth[i][j-1])/(sampleDistances[i][j])
        singleRatio= tmp / 10.0
        RatioSum += singleRatio

    return RatioSum/len(sampleDistances)

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

    #categoryNum from 0 -23
    categoryNum=22
    allDenseVector_HSV_blue=getAllVideoDenseVector("/home/luy100/python_workspace/CC_WEB_VIDEO_d_256/videoDescriptor_256HSV",
                                          categoryNum,0)
    allDenseVector_HSV_green=getAllVideoDenseVector("/home/luy100/python_workspace/CC_WEB_VIDEO_d_256/videoDescriptor_256HSV",
                                          categoryNum,1)
    allDenseVector_HSV_red=getAllVideoDenseVector("/home/luy100/python_workspace/CC_WEB_VIDEO_d_256/videoDescriptor_256HSV",
                                          categoryNum,2)
    (totalNum, ENum, SNum, Eset, Sset)=getQueryAndGroundTruth(categoryNum)

    # print(Sset[0]+Sset[1])
    #Do own designed normalization for descriptor
    # allDenseVector=nomalization(allDenseVector)
    # X_train=allNormalizedDenseVector
    # X_test = allNormalizedDenseVector[:100]

    # tmp=[ [i,allDenseVector[i]] for i in range(len(allDenseVector))]

    QueryIndexWithVector_blue=[ [int(i),allDenseVector_HSV_blue[int(i)]] for i in Sset]
    QueryIndexWithVector_green=[ [int(i),allDenseVector_HSV_green[int(i)]] for i in Sset]
    QueryIndexWithVector_red=[ [int(i),allDenseVector_HSV_red[int(i)]] for i in Sset]
    # print(QueryIndexWithVector_red)
    QueryVectors_blue=[ x[1] for x in QueryIndexWithVector_blue]
    QueryVectors_green=[ x[1] for x in QueryIndexWithVector_green]
    QueryVectors_red=[ x[1] for x in QueryIndexWithVector_red]
    X_train_blue = allDenseVector_HSV_blue
    X_train_green=allDenseVector_HSV_green
    X_train_red=allDenseVector_HSV_red
    X_test_blue = QueryVectors_blue
    X_test_green = QueryVectors_green
    X_test_red = QueryVectors_red

    estimateList=[100]
    #
    test_num=1
    for estimateNum in estimateList:
        time_sum = 0.0
        recallSum=0.0
        lshf_blue = LSHForest(n_estimators=estimateNum, random_state=42, n_candidates=SNum+1)
        lshf_green= LSHForest(n_estimators=estimateNum, random_state=42, n_candidates=SNum+1)
        lshf_red= LSHForest(n_estimators=estimateNum, random_state=42, n_candidates=SNum+1)
        lshf_blue.fit(X_train_blue)
        lshf_green.fit(X_train_green)
        lshf_red.fit(X_train_red)
        startMillis = int(round(time.time() * 1000))
        distances1, indices1 = lshf_blue.kneighbors(X_test_blue, n_neighbors=SNum)
        distances2, indices2 = lshf_green.kneighbors(X_test_green, n_neighbors=SNum)
        distances3, indices3 = lshf_red.kneighbors(X_test_red, n_neighbors=SNum)
        indices=[]
        for i in range(len(indices1)):
            indices.append(set(indices1[i]).union(set(indices2[i])).union(set(indices3[i])))
        endMillis = int(round(time.time() * 1000))
        time_sum += (endMillis-startMillis)
        print(time_sum/SNum)
        # print(indices3)
        # print(type(indices[0]))

        for OneResult in indices:
             recallSum += len(set(OneResult).intersection(set(Sset)))/float(SNum)
        print("recall={0}".format(recallSum/SNum))




            # groundTruth = getGroundTruth("/Users/eternity/PycharmProjects/transferVector/", 10000)
            # ratioSum += calculateRatio(indices, sampleVectors, sampleIndices, allDenseVector, groundTruth)
        # print("for table#{0}, run test time is {1}ms".format(estimateNum,time_sum / (test_num * 100.0)))
        # print("average ratio is {0}\n".format(ratioSum/test_num))

    # for i in range(len(distances)):
    #     print(distances[i])
    #     print(indices[i])
    #print out the distances and correspond indices
    # print(distances)
    # print(indices)



    # print("run test time is {0}ms".format(time_sum/(test_num*100.0)))

    #calculate the ratio
    # groundTruth=getGroundTruth("/Users/eternity/PycharmProjects/transferVector/",10000)
    # oneRatio=calculateRatio(indices,sampleVectors,sampleIndices,allDenseVector,groundTruth)
    # print(oneRatio)


