import falconn
import numpy as np
import timeit
import h5py

# This file is to test the performance of falconn on dataset Mnist

basePath = "/home/luy100/ForVLDB/ComparedMethod/Datasets"

if __name__ == '__main__':
    dataset_file_utl = basePath + '/mnist/mnist-784-euclidean.hdf5'

    print('Reading the dataset')
    dataset_file = h5py.File(dataset_file_utl, 'r')
    a_group_key = list(dataset_file.keys())
    print(a_group_key)

    # distances=list(dataset_file[a_group_key[0]])
    neighbors = list(dataset_file[a_group_key[1]])
    testDataset = np.array(list(dataset_file[a_group_key[2]]))
    trainDataset = np.array(list(dataset_file[a_group_key[3]]))
    print('Done')

    number_of_queries = 2000

    number_of_probes = 6400
    number_of_tables = 20

    print('Generating queries')
    # np.random.seed(666666)
    # np.random.shuffle(trainDataset)
    queries = testDataset[:number_of_queries]
    npGroundTruth=np.array(neighbors)
    groundTruth = npGroundTruth[:number_of_queries,:]
    print('Done')

    print('Centering the dataset and queries')
    center = np.mean(trainDataset, axis=0)
    trainDataset -= center
    queries -= center
    print('Done')


    params_cp = falconn.LSHConstructionParameters()
    params_cp.dimension = len(trainDataset[0])
    params_cp.lsh_family = falconn.LSHFamily.CrossPolytope
    params_cp.distance_function = falconn.DistanceFunction.EuclideanSquared
    params_cp.l = number_of_tables
    params_cp.num_rotations = 2
    params_cp.seed = 666666
    params_cp.num_setup_threads = 4
    params_cp.storage_hash_table = falconn.StorageHashTable.BitPackedFlatHashTable
    falconn.compute_number_of_hash_functions(16, params_cp)

    print('Constructing the LSH table')
    t1 = timeit.default_timer()
    table = falconn.LSHIndex(params_cp)
    table.setup(trainDataset)
    t2 = timeit.default_timer()
    query_object = table.construct_query_object()
    print('Done')
    print('Construction time: {}'.format((t2 - t1)))

    query_object.set_num_probes(number_of_probes)

    # print(groundTruth[0])
    # print(query_object.find_k_nearest_neighbors(queries[0],10))


    score = 0.0
    t1 = timeit.default_timer()
    for (i, query) in enumerate(queries):
        score += len(set(query_object.find_k_nearest_neighbors(query, 100)).intersection(set(groundTruth[i])))
    t2 = timeit.default_timer()
    print('Query time: {} per query'.format((t2 - t1) * 1000 / float(
        len(queries))))
    print("the recall is {}".format(score / 100.0 / float(len(queries))))



