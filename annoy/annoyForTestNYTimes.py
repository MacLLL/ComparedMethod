from annoy import AnnoyIndex
import random
import timeit
import h5py
import numpy as np

# This file is to test the performance of falconn on dataset SIFT

basePath = "/home/luy100/ComparedMethod/Datasets"

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

    number_of_trees = 60
    search_k = [1000,2000,4000,8000,16000,32000,64000,128000,256000]
    topk = 10

    print('Generating queries')
    # np.random.seed(666666)
    # np.random.shuffle(trainDataset)
    queries = testDataset[:number_of_queries]
    npGroundTruth = np.array(neighbors)
    groundTruth = npGroundTruth[:number_of_queries, :topk]
    print('Done')

    print('Centering the dataset and queries')
    center = np.mean(trainDataset, axis=0)
    trainDataset -= center
    queries -= center
    print('Done')

    print('Constructing the index')
    t1 = timeit.default_timer()
    # set the parameters
    dimension = len(trainDataset[0])
    index = AnnoyIndex(dimension, 'angular')

    for (i, object) in enumerate(trainDataset):
        index.add_item(i, object)
    index.build(number_of_trees)
    t2 = timeit.default_timer()
    print('Done')
    print('Construction time: {}'.format((t2 - t1)))

    print("start querying")
    for k in search_k:
        score = 0.0
        t1 = timeit.default_timer()
        for (i, query) in enumerate(queries):
            score += len(set(index.get_nns_by_vector(query, topk, search_k=k)).intersection(set(groundTruth[i])))
        t2 = timeit.default_timer()
        print("for search_k = {}".format(k))
        print('Query time: {} per query'.format((t2 - t1) * 1000 / float(
            len(queries))))
        print("the recall is {}".format(score / topk / float(len(queries))))
