from __future__ import print_function
import numpy as np
import falconn
import timeit
import math
import os
from operator import itemgetter
import sklearn.preprocessing

basePath = "/home/luy100/ForVLDB/ComparedMethod/Datasets"

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
        c = np.load(basePath+ "/LastTest/CC_WEB_VIDEO_d_256/videoDescriptor_256HSV/" +
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
    allrecall=[]
    alltime=[]
    categoryNum = 0
    while categoryNum < 24:
        print("For category {}----------------".format(categoryNum))
        allDenseVector_HSV_blue = getAllVideoDenseVector(
            basePath+"/LastTest/CC_WEB_VIDEO_d_256/videoDescriptor_256HSV",
            categoryNum, 0)
        allDenseVector_HSV_green = getAllVideoDenseVector(
            basePath+"/LastTest/CC_WEB_VIDEO_d_256/videoDescriptor_256HSV",
            categoryNum, 1)
        allDenseVector_HSV_red = getAllVideoDenseVector(
            basePath+"/LastTest/CC_WEB_VIDEO_d_256/videoDescriptor_256HSV",
            categoryNum, 2)
        (totalNum, ENum, SNum, Eset, Sset) = getQueryAndGroundTruth(categoryNum)

        number_of_queries=len(Sset)
        number_of_tables=100
        dataset_blue=allDenseVector_HSV_blue
        dataset_green=allDenseVector_HSV_green
        dataset_red=allDenseVector_HSV_red
        QueryIndexWithVector_blue = [[int(i), allDenseVector_HSV_blue[int(i)]] for i in Sset]
        QueryIndexWithVector_green = [[int(i), allDenseVector_HSV_green[int(i)]] for i in Sset]
        QueryIndexWithVector_red = [[int(i), allDenseVector_HSV_red[int(i)]] for i in Sset]
        queries_blue=[ x[1] for x in QueryIndexWithVector_blue]
        queries_green=[ x[1] for x in QueryIndexWithVector_green]
        queries_red=[ x[1] for x in QueryIndexWithVector_red]

        # print('Centering the dataset and queries')
        # center = np.mean(dataset, axis=0)
        # dataset -= center
        # queries -= center
        # print('Done')
        #assert dataset.dtype == np.float32
        number_of_probes = [900]
        #

        params_cp_blue = falconn.LSHConstructionParameters()
        params_cp_blue.dimension = len(dataset_blue[0])
        params_cp_blue.lsh_family = falconn.LSHFamily.CrossPolytope
        params_cp_blue.distance_function = falconn.DistanceFunction.EuclideanSquared
        params_cp_blue.l = number_of_tables
        params_cp_blue.num_rotations = 1
        params_cp_blue.seed = 666666
        params_cp_blue.num_setup_threads = 1
        params_cp_blue.storage_hash_table = falconn.StorageHashTable.BitPackedFlatHashTable
        falconn.compute_number_of_hash_functions(20, params_cp_blue)

        print('Constructing the LSH table')
        t1 = timeit.default_timer()
        table_blue = falconn.LSHIndex(params_cp_blue)
        table_blue.setup(dataset_blue)
        t2 = timeit.default_timer()
        query_object_blue= table_blue.construct_query_object()
        print('Done')
        print('Construction time: {}'.format((t2 - t1)))

        params_cp_green = falconn.LSHConstructionParameters()
        params_cp_green.dimension = len(dataset_green[0])
        params_cp_green.lsh_family = falconn.LSHFamily.CrossPolytope
        params_cp_green.distance_function = falconn.DistanceFunction.EuclideanSquared
        params_cp_green.l = number_of_tables
        params_cp_green.num_rotations = 1
        params_cp_green.seed = 666666
        params_cp_green.num_setup_threads = 1
        params_cp_green.storage_hash_table = falconn.StorageHashTable.BitPackedFlatHashTable
        falconn.compute_number_of_hash_functions(20, params_cp_green)

        print('Constructing the LSH table')
        t1 = timeit.default_timer()
        table_green = falconn.LSHIndex(params_cp_green)
        table_green.setup(dataset_green)
        t2 = timeit.default_timer()
        query_object_green = table_green.construct_query_object()
        print('Done')
        print('Construction time: {}'.format((t2 - t1)))

        params_cp_red = falconn.LSHConstructionParameters()
        params_cp_red.dimension = len(dataset_red[0])
        params_cp_red.lsh_family = falconn.LSHFamily.CrossPolytope
        params_cp_red.distance_function = falconn.DistanceFunction.EuclideanSquared
        params_cp_red.l = number_of_tables
        params_cp_red.num_rotations = 1
        params_cp_red.seed = 666666
        params_cp_red.num_setup_threads = 1
        params_cp_red.storage_hash_table = falconn.StorageHashTable.BitPackedFlatHashTable
        falconn.compute_number_of_hash_functions(20, params_cp_red)

        print('Constructing the LSH table')
        t1 = timeit.default_timer()
        table_red = falconn.LSHIndex(params_cp_red)
        table_red.setup(dataset_red)
        t2 = timeit.default_timer()
        query_object_red = table_blue.construct_query_object()
        print('Done')
        print('Construction time: {}'.format((t2 - t1)))

        for num_p in number_of_probes:
            query_object_blue.set_num_probes(num_p)
            query_object_green.set_num_probes(num_p)
            query_object_red.set_num_probes(num_p)

            print("num of probes {}".format(num_p))

            score = 0.0
            t1 = timeit.default_timer()

            for i in range(SNum):
                result = set(query_object_blue.find_k_nearest_neighbors(queries_blue[i],SNum)).\
                    union(query_object_green.find_k_nearest_neighbors(queries_green[i],SNum)).\
                    union(query_object_red.find_k_nearest_neighbors(queries_red[i],SNum))
                score += len(result.intersection(set(Sset)))
            t2 = timeit.default_timer()
            print('Query time: {} per query'.format((t2 - t1) * 1000 / float(
                len(queries_red))))
            print("the recall is {}".format(score / SNum / float(len(queries_red))))
        categoryNum += 1
        allrecall.append(score / SNum / float(len(queries_red)))
        alltime.append((t2 - t1) * 1000 / float(
                len(queries_red)))
    print("-------------------")
    print("max recall is {}".format(np.max(allrecall)))
    print("min recall is {}".format(np.min(allrecall)))
    print("average recall is {}".format(np.mean(allrecall)))
    print("average time is is {}".format(np.mean(alltime)))

