from sklearn.neighbors import LSHForest

import numpy as np
import h5py
import timeit
import time

# This file is to test the performance of hnsw on dataset SIFT

basePath = "/home/luy100/ForVLDB/ComparedMethod/Datasets"

if __name__ == '__main__':
    dataset_file_utl = basePath + '/nytimes/nytimes-256-angular.hdf5'

    print('Reading the dataset')
    dataset_file = h5py.File(dataset_file_utl, 'r')
    a_group_key = list(dataset_file.keys())
    print(a_group_key)

    neighbors = list(dataset_file[a_group_key[1]])
    testDataset = np.array(dataset_file[a_group_key[2]])
    trainDataset = np.array(dataset_file[a_group_key[3]])

    print('Done')

    number_of_queries = 2000
    topk=10

    print('Generating queries')
    # np.random.seed(666666)
    # np.random.shuffle(trainDataset)
    queries = testDataset[:number_of_queries]
    npGroundTruth = np.array(neighbors)
    groundTruth = npGroundTruth[:number_of_queries, :topk]
    print('Done')

    estimateList = [2,5,10,15,20,30,40,50]

    test_num = 1
    for estimateNum in estimateList:
        time_sum = 0.0
        recallSum = 0.0
        lshf_nytimes = LSHForest(n_estimators=estimateNum, random_state=42, n_candidates=500)
        lshf_nytimes.fit(trainDataset)
        startMillis = int(round(time.time() * 1000))
        distances1, indices1 = lshf_nytimes.kneighbors(queries, n_neighbors=topk)
        indices = []
        # for i in range(len(indices1)):
        #     indices.append(set(indices1[i]).union(set(indices2[i])).union(set(indices3[i])))
        endMillis = int(round(time.time() * 1000))
        time_sum += (endMillis - startMillis)
        print(time_sum / number_of_queries)
        # print(indices3)
        # print(type(indices[0]))

        for (i,OneResult) in enumerate(indices1):
            recallSum += len(set(OneResult).intersection(set(groundTruth[i]))) / number_of_queries
        print("recall={0}".format(recallSum / topk))