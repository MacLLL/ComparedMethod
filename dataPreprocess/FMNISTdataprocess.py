import numpy as np
import h5py
import timeit
import sklearn as sk

# This file is to test the performance of hnsw on dataset SIFT

basePath = "/home/luy100/ForVLDB/ComparedMethod/Datasets"

if __name__ == '__main__':
    dataset_file_utl = basePath + '/fmnist/fashion-mnist-784-euclidean.hdf5'
    print('Reading the dataset')
    dataset_file = h5py.File(dataset_file_utl, 'r')
    a_group_key = list(dataset_file.keys())
    print(a_group_key)

    # neighbors = list(dataset_file[a_group_key[1]])
    # testDataset = np.array(dataset_file[a_group_key[2]])
    trainDataset = np.array(dataset_file[a_group_key[3]])
    print('Done')
    # print('Normalizing the dataset')
    # trainDataset /= np.linalg.norm(trainDataset, axis=1).reshape(-1, 1)
    # print('Done')

    assert trainDataset.dtype == np.float32
    number_of_queries = 2000

    topk = 100

    print('Generating queries')
    # np.random.seed(666666)
    # np.random.shuffle(trainDataset)
    queries = trainDataset[:number_of_queries]
    # queries = testDataset[:number_of_queries]
    # npGroundTruth = np.array(neighbors)
    # groundTruth = npGroundTruth[:number_of_queries, :topk]
    print('Done')

    # print('Centering the dataset and queries')
    # center = np.mean(trainDataset, axis=0)
    # trainDataset -= center
    # queries -= center
    # print('Done')

    ID = 0
    with open(basePath + '/fmnist/fashion-mnist-784d-DenseVector.txt', 'w') as thefile:
        for i in range(len(trainDataset)):
            values = []
            for item in range(len(trainDataset[i])):
                values.append(trainDataset[i][item])
            DenseVector = [ID, values]
            thefile.write("%s\n" % DenseVector)
            ID += 1
            if ID % 10000 == 0:
                print("%d vectors transfered" % ID)

    # ID = 0
    # with open(basePath + '/fmnist/fashion-mnist-784d-SparseVector.txt', 'w') as thefile:
    #     for i in range(len(trainDataset)):
    #         values = []
    #         dim=len(trainDataset[i])
    #         indices = []
    #         for item in range(len(trainDataset[i])):
    #             if trainDataset[i][item] !=0:
    #                 values.append(trainDataset[i][item])
    #                 indices.append(item)
    # #             values.append(trainDataset[i][item])
    #         SparseVector = [ID,dim,indices,values]
    #         thefile.write("%s\n" % SparseVector)
    #         ID += 1
    #         if ID % 10000 == 0:
    #             print("%d vectors transfered" % ID)

    t1 = timeit.default_timer()
    ID = 0
    with open(basePath + '/fmnist/fashion-mnist-784d-top100-2000queryGT', 'w') as thefile:
        for query in queries:
            answers = np.dot(trainDataset, query).argsort()[-100:][::-1]
            answers = [x for x in answers]
            thefile.write("%s\n" % answers)
            ID += 1
            if ID % 100 == 0:
                print(ID)
    t2 = timeit.default_timer()
    print("Done")
    print('Linear scan time: {} per query'.format((t2 - t1) / float(
        len(queries))))


    # print(queries[-1])
    # print(trainDataset[groundTruth[-1][0]-1])
    # print(groundTruth[-1])
    # print(len(groundTruth))
    # print(trainDataset[-1])
    # print(len(trainDataset))