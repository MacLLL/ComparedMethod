import nmslib
import numpy as np
import h5py
import timeit


# This file is to test the performance of hnsw on dataset glove100d

basePath = "/home/luy100/ComparedMethod/Datasets"

if __name__ == '__main__':
    dataset_file_utl = basePath + '/glove/glove.twitter.27B.100d.npy'

    number_of_queries = 2000
    topk = 10
    print('Reading the dataset')
    dataset = np.load(dataset_file_utl)
    print('Done')

    assert dataset.dtype == np.float32

    print('Normalizing the dataset')
    dataset /= np.linalg.norm(dataset, axis=1).reshape(-1, 1)
    print('Done')

    print('Generating queries')
    np.random.seed(666666)
    np.random.shuffle(dataset)
    queries = dataset[:number_of_queries]
    dataset=dataset[number_of_queries:]
    print('Done')

    print('Centering the dataset and queries')
    center = np.mean(dataset, axis=0)
    dataset -= center
    queries -= center
    print('Done')

    print("read the groundtruth")
    groundTruth = []
    lines = open(basePath + '/glove/glove.twitter.27B.100d.2000queryGT', 'r').readlines()
    for line in lines:
        line = line.replace("[", "").replace("]\n", "").replace(" ", "").split(",")
        one = []
        for item in line:
            one.append(int(item))
        groundTruth.append(set(one))
    print("Done")

    print('Constructing the index')
    t1 = timeit.default_timer()
    # initialize a new index, using a HNSW index on Cosine Similarity
    index = nmslib.init(method='hnsw', space='cosinesimil')
    index.addDataPointBatch(dataset[:-1])
    index.createIndex({'post': 2}, print_progress=False)
    t2 = timeit.default_timer()
    print('Done')
    print('Construction time: {}'.format((t2 - t1)))

    print("starting insert one object")
    t1 = timeit.default_timer()
    index.addDataPoint(len(dataset)-1,dataset[-1])
    t2 =timeit.default_timer()
    print("insert one object cost {}ms".format((t2-t1)*1000))

    print("start querying")
    score = 0.0
    t1 = timeit.default_timer()
    # get all nearest neighbours for all the datapoint
    # using a pool of 4 threads to compute
    neighbors = index.knnQueryBatch(queries, k=topk, num_threads=4)
    count = 0

    for (i, oneResult) in enumerate(neighbors):
        score += len(set(oneResult[0]).intersection(set(groundTruth[i])))

    t2 = timeit.default_timer()
    print('Query time: {} per query'.format((t2 - t1) * 1000 / float(
        len(queries))))
    print("the recall is {}".format(score / topk / float(len(queries))))