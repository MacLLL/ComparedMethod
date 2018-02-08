import falconn
import numpy as np
import timeit

# This file is to test the performance of falconn on dataset Glove100d

basePath = "/home/luy100/ForVLDB/ComparedMethod/Datasets"

if __name__ == '__main__':
    dataset_file_utl = basePath + '/glove/glove.twitter.27B.100d.npy'

    number_of_queries = 2000

    number_of_probes = 5000
    number_of_tables = 180

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
    # with open(basePath + '/glove/glove.twitter.27B.100d.2000queryGT', 'w') as thefile:
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
    lines = open(basePath + '/glove/glove.twitter.27B.100d.2000queryGT', 'r').readlines()
    for line in lines:
        line = line.replace("[", "").replace("]\n", "").replace(" ", "").split(",")
        one = []
        for item in line:
            one.append(int(item))
        groundTruth.append(set(one))
    print("Done")

    params_cp = falconn.LSHConstructionParameters()
    params_cp.dimension = len(dataset[0])
    params_cp.lsh_family = falconn.LSHFamily.CrossPolytope
    params_cp.distance_function = falconn.DistanceFunction.EuclideanSquared
    params_cp.l = number_of_tables
    params_cp.num_rotations = 1
    params_cp.seed = 666666
    params_cp.num_setup_threads = 4
    params_cp.storage_hash_table = falconn.StorageHashTable.BitPackedFlatHashTable
    falconn.compute_number_of_hash_functions(20, params_cp)

    print('Constructing the LSH table')
    t1 = timeit.default_timer()
    table = falconn.LSHIndex(params_cp)
    table.setup(dataset)
    t2 = timeit.default_timer()
    query_object = table.construct_query_object()
    print('Done')
    print('Construction time: {}'.format((t2 - t1)))



    query_object.set_num_probes(number_of_probes)

    score = 0.0
    t1 = timeit.default_timer()
    for (i, query) in enumerate(queries):
        score += len(set(query_object.find_k_nearest_neighbors(query, 10)).intersection(groundTruth[i]))
    t2 = timeit.default_timer()
    print('Query time: {} per query'.format((t2 - t1)*1000 / float(
            len(queries))))
    print("the recall is {}".format(score / 10.0 / float(len(queries))))
