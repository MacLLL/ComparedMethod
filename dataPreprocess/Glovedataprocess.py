import numpy as np
import timeit

# This file is to test the performance of falconn on dataset Glove100d

basePath = "/home/luy100/ForVLDB/ComparedMethod/Datasets"

if __name__ == '__main__':
    dataset_file_utl = basePath + '/glove/glove.twitter.27B.100d.npy'

    number_of_queries = 100

    # number_of_probes = 5000
    # number_of_tables = 180

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
    dataset=dataset[:150000]
    # dataset = dataset[:200000]
    queries = dataset[:number_of_queries]
    print('Done')

    print('Centering the dataset and queries')
    center = np.mean(dataset, axis=0)
    dataset -= center
    queries -= center
    print('Done')

    ID = 0
    with open(basePath + '/glove/glove-100d-DenseVector-150k.txt', 'w') as thefile:
        for i in range(len(dataset)):
            values = []
            for item in range(len(dataset[i])):
                values.append(dataset[i][item])
            DenseVector = [ID, values]
            thefile.write("%s\n" % DenseVector)
            ID += 1
            if ID % 10000 == 0:
                print("%d vectors transfered" % ID)

    t1 = timeit.default_timer()
    ID = 0
    with open(basePath + '/glove/glove-100d-top100-100queryGT-150k', 'w') as thefile:
        for query in queries:
            answers = np.dot(dataset, query).argsort()[-100:][::-1]
            answers = [x for x in answers]
            thefile.write("%s\n" % answers)
            ID += 1
            if ID % 100 == 0:
                print(ID)
    t2 = timeit.default_timer()
    print("Done")
    print('Linear scan time: {} per query'.format((t2 - t1) / float(
        len(queries))))