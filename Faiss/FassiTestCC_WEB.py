import faiss
import numpy as np
import timeit
import h5py


basePath = "/home/luy100/ForVLDB/ComparedMethod/Datasets"

if __name__ == '__main__':
    dataset_file_utl = basePath + '/fmnist/fashion-mnist-784-euclidean.hdf5'

    print('Reading the dataset')
    dataset_file = h5py.File(dataset_file_utl, 'r')
    a_group_key = list(dataset_file.keys())
    print(a_group_key)

    # distances=list(dataset_file[a_group_key[0]])
    neighbors = list(dataset_file[a_group_key[1]])
    testDataset = np.array(dataset_file[a_group_key[2]])
    trainDataset = np.array(dataset_file[a_group_key[3]])
    print('Done')

    topk = 10


    d = 784  # dimension
    nb = trainDataset.size  # database size
    nq = 2000  # nb of queries
    # np.random.seed(1234)  # make reproducible
    # xb = np.random.random((nb, d)).astype('float32')
    # xb[:, 0] += np.arange(nb) / 1000.
    # xq = np.random.random((nq, d)).astype('float32')
    # xq[:, 0] += np.arange(nq) / 1000.

    print('Generating queries')
    # np.random.seed(666666)
    # np.random.shuffle(trainDataset)
    queries = testDataset[:nq]
    npGroundTruth = np.array(neighbors)
    groundTruth = npGroundTruth[:nq, :topk]
    print('Done')


    n_bits=2*d
    index = faiss.IndexLSH(d,n_bits)  # build the index
    print(index.is_trained)
    index.add(trainDataset)  # add vectors to the index
    print(index.ntotal)
     # we want to see 4 nearest neighbors
    t1 = timeit.default_timer()
    D, I = index.search(queries, topk)  # sanity check
    t2 = timeit.default_timer()
    # print(I)
    # print(D)
    # D, I = index.search(xq, k)  # actual search
    # print(I[:5])  # neighbors of the 5 first queries
    # print(I[-5:])

    score = 0.0

    for (i, query) in enumerate(I):
        score += len(set(I[i]).intersection(set(groundTruth[i])))
    print('Query time: {} per query'.format((t2 - t1) * 1000 / float(
        len(queries))))
    print("the recall is {}".format(score / topk / float(len(queries))))