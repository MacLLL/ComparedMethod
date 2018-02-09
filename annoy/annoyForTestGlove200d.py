from annoy import AnnoyIndex
import random
import timeit
import h5py
import numpy as np

# This file is to test the performance of falconn on dataset SIFT

basePath = "/home/luy100/ComparedMethod/Datasets"

if __name__ == '__main__':
    dataset_file_utl = basePath + '/glove/glove.twitter.27B.200d.npy'

    number_of_queries = 2000

    number_of_trees = 40
    search_k = [1000, 2000, 4000, 8000, 16000, 32000, 64000, 128000, 256000,512000]
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
    print('Done')


    # t1 = timeit.default_timer()
    # ID = 0
    # with open(basePath + '/glove/glove.twitter.27B.200d.2000queryGT', 'w') as thefile:
    #     for query in queries:
    #         answers = np.dot(dataset, query).argsort()[-10:][::-1]
    #         answers = [x for x in answers]
    #         thefile.write("%s\n" % answers)
    #         ID += 1
    #         if ID % 100 == 0:
    #             print(ID)
    # t2 = timeit.default_timer()
    # print("Done")
    # print('Linear scan time: {} per query'.format((t2 - t1) / float(
    #     len(queries))))

    print('Centering the dataset and queries')
    center = np.mean(dataset, axis=0)
    dataset -= center
    queries -= center
    print('Done')

    print("read the groundtruth")
    groundTruth = []
    lines = open(basePath + '/glove/glove.twitter.27B.200d.2000queryGT', 'r').readlines()
    for line in lines:
        line = line.replace("[", "").replace("]\n", "").replace(" ", "").split(",")
        one = []
        for item in line:
            one.append(int(item))
        groundTruth.append(set(one))
    print("Done")

    print('Constructing the index')
    t1 = timeit.default_timer()
    # set the parameters
    dimension = len(dataset[0])
    index = AnnoyIndex(dimension, 'angular')

    for (i, object) in enumerate(dataset):
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
