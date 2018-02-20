import random
import timeit
import h5py
import numpy as np

# This file is to test the performance of annoy on dataset glove100d

basePath = "/home/luy100/ForVLDB/ComparedMethod/Datasets"
if __name__ == '__main__':
    dataset_file_utl = basePath + '/glove/glove.twitter.27B.100d.npy'
    number_of_queries = 2000

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

    print('Centering the dataset and queries')
    center = np.mean(dataset, axis=0)
    dataset -= center
    queries -= center
    print('Done')

    ID=0
    with open(basePath + '/glove/partition/glove.twitter.27B.100d.DenseVector.txt', 'w') as thefile:
        for i in range(len(dataset)):
            values=[]
            for item in range(len(dataset[i])):
                values.append(dataset[i][item])
            DenseVector=[ID,values]
            thefile.write("%s\n" % DenseVector)
            ID+=1
            if ID % 10000 ==0:
                print("%d vectors transfered" % ID)

    # ID = 0
    # with open(basePath + '/glove/partition/glove.twitter.27B.100d.2000queryGT.top100', 'w') as thefile:
    #     for query in queries:
    #         answers = np.dot(dataset, query).argsort()[-100:][::-1]
    #         answers = [x for x in answers]
    #         thefile.write("%s\n" % answers)
    #         ID += 1
    #         if ID % 100 == 0:
    #             print(ID)
    # t2 = timeit.default_timer()
    # print("Done")