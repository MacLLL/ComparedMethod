import nmslib
import numpy as np
import h5py
import timeit

# This file is to test the performance of hnsw on dataset SIFT

basePath = "/home/luy100/ComparedMethod/Datasets"

if __name__ == '__main__':
    dataset_file_utl = basePath + '/sift/sift-128-euclidean.hdf5'

    print('Reading the dataset')
    dataset_file = h5py.File(dataset_file_utl, 'r')
    a_group_key = list(dataset_file.keys())
    print(a_group_key)

    neighbors = list(dataset_file[a_group_key[1]])
    testDataset = np.array(dataset_file[a_group_key[2]])
    trainDataset = np.array(dataset_file[a_group_key[3]])
    print('Done')

    number_of_queries = 2000

    # number_of_trees = 120
    # search_k = [1000, 2000, 4000, 8000, 16000, 32000, 64000]
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
    # initialize a new index, using a HNSW index on Cosine Similarity
    index = nmslib.init(method='hnsw', space='cosinesimil')
    index.addDataPointBatch(trainDataset)
    index.createIndex({'post': 2}, print_progress=False)
    t2 = timeit.default_timer()
    print('Done')
    print('Construction time: {}'.format((t2 - t1)))

    print("start querying")
    score=0.0
    t1 = timeit.default_timer()
    # get all nearest neighbours for all the datapoint
    # using a pool of 4 threads to compute
    neighbors = index.knnQueryBatch(queries, k=topk, num_threads=4)
    count =0

    for (i, oneResult) in enumerate(neighbors):
        score += len(set(oneResult[0]).intersection(set(groundTruth[i])))

    t2 = timeit.default_timer()
    print('Query time: {} per query'.format((t2 - t1) * 1000 / float(
        len(queries))))
    print("the recall is {}".format(score / topk / float(len(queries))))