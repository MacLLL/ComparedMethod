import numpy as np
import timeit
import threading


class TraditionalLSH:
    """
    Parameters:
        m -------> hash functions number in one hash table
        l -------> number of hash tables
        n -------> total number of dataset

    Hash function parameters:
        w -------> the width
        b -------> a real number chosen uniformly from the range [0,w]
        a -------> a vector chosen from p-stable distribution, for example, N(0,1)
        d -------> the dimension
    """
    m, l, w, b, a, d, n, P1, W, h = 0, 0, 0, 0, 0, 0, 0, 0, 0, 0
    allhv = None
    index = None
    trainData = None

    def __init__(self, m, l, d):
        self.m, self.l, self.d, self.w, self.P1, self.W = m, l, d, 4, 8191, 1
        self.b = np.random.random_sample((self.l, self.m)) * self.w
        if self.d == 0:
            print("please initialize the dimension d")
        self.generate_hash_function()

    """
    generate p_stable hash functions,l*m*d
    """

    def generate_hash_function(self):
        mu, sigma = 1, 1
        self.a = [np.random.normal(mu, sigma, (self.m, self.d)) for i in range(self.l)]

    """
    hash the high dimensional data， basic p-stable hash: h(v)=floor((a.v+b)/w)
    Parameters:
    data --------> the points matrix n rows and d columns
    Return:
    hashvalues --> a l*n*m 
    """

    def hash(self, data):
        (self.n, self.d) = np.shape(data)
        # finish todo: generate l * m* n b
        hashvalues = [np.floor((np.dot(data, np.transpose(self.a[i])) +
                                np.reshape(list(self.b[i]) * self.n, (self.n, self.m))) / self.w) for i in
                      range(self.l)]
        self.allhv = hashvalues[0]
        return hashvalues

    def oneHash(self,query):
        hv=np.floor((np.dot(query,np.transpose(self.a[0][0])) + self.b[0]) / self.w)
        return hv
    """
    Use the basic index methods: sum(h*hashvalues) mod P1, h is a weight, we can use several set of h.
    Parameters:
    data -------> the points matrix n row and d columns
    P1 -------> number of buckets: large prime number 6700417
    h -------> set of weights, m dimensional guassian distribution 
    W -------> the number of h for index, can be 1 or 2
    
    """

    def indexBasic(self, data):
        # generate the first hashvalues, l*n*m
        firhv = self.hash(data)
        # generate the h for second index, l*W*m
        self.h = [np.floor(np.random.random_sample((self.W, self.m)) * 10) for i in range(self.l)]
        finalhv = [np.dot(firhv[i], np.transpose(self.h[i])) for i in range(self.l)]
        # it's a l*n*W
        if self.W == 1:
            finalhv = np.reshape(finalhv, (self.l, self.n))
        # print(finalhv)
        flatIndex = [None] * self.l
        for tableI in range(self.l):
            flatIndex[tableI] = [None] * self.P1
            for (key, indexID) in enumerate(finalhv[tableI]):
                tmp = []
                if flatIndex[tableI][int(indexID)] is None:
                    tmp.append(key)
                    flatIndex[tableI][int(indexID)] = tmp
                else:
                    tmp = flatIndex[tableI][int(indexID)]
                    tmp.append(key)
                    flatIndex[tableI][int(indexID)] = tmp
        # print(flatIndex)
        return flatIndex

    """
    fit the trainData in index
    """

    def fit(self, trainData, mode='basic'):
        if mode == 'basic':
            # todo: finish the indexbasic
            self.trainData = trainData
            self.index = self.indexBasic(trainData)

    """
    Search the similarity, the most basic one, without multi probes
    Parameters:
    k -------> top k approximate nearest neighbor
    threshold -----> the similarity threshold for whether
    """

    def query(self, k, queryData, threshold=0):
        queryNum = len(queryData)
        t1=timeit.default_timer()
        hv = [np.floor((np.dot(queryData, np.transpose(self.a[i])) +
                        np.reshape(list(self.b[i]) * queryNum, (queryNum, self.m))) / self.w) for i in range(self.l)]
        finalhv = [np.dot(hv[i], np.transpose(self.h[i])) for i in range(self.l)]
        t2=timeit.default_timer()
        print("hash time is {}".format(t2-t1))
        # print(finalhv)
        # resultKeys = []
        # # print(self.index)
        # for (tableI, onehv) in enumerate(finalhv):
        #     tmprk=set()
        #     for hashkey in onehv:
        #         tmprk=tmprk.union(set(self.index[tableI][int(hashkey)]))
        #     resultKeys.append(tmprk)
        t1=timeit.default_timer()
        candidatesKeys = [set()] * queryNum
        for (tableI, onehv) in enumerate(finalhv):
            for (queryID, hashkey) in enumerate(onehv):
                candidatesKeys[queryID] = set(candidatesKeys[queryID]).union(set(self.index[tableI][int(hashkey)]))
        t2=timeit.default_timer()
        print("get candidates time is {}".format(t2-t1))
        # print(candidatesKeys)
        # print(len(candidatesKeys[0]))
        results=[]

        #filter the candidates todo: generate the results by using matrix
        t1=timeit.default_timer()
        for (queryID,oneQueryCandidatesKey) in enumerate(candidatesKeys):
            oneCandidatesMatrix = [self.trainData[key] for key in oneQueryCandidatesKey]
            tmpAnswers = np.dot(oneCandidatesMatrix, queryData[queryID]).argsort()[-k:][::-1]
            # todo: filter by hashvalues, don't improve the performance
            # oneCandidatesMatrix = [self.allhv[key] for key in oneQueryCandidatesKey]
            # tmpAnswers = np.dot(oneCandidatesMatrix, self.oneHash(queryData[queryID])).argsort()[-k:][::-1]
            oneQueryCandidatesKey=list(oneQueryCandidatesKey)
            # print(tmpAnswers)
            # print(len(tmpAnswers)))
            answers = [ oneQueryCandidatesKey[key] for key in tmpAnswers]
            results.append(answers)
        t2=timeit.default_timer()
        print("filter time is {}".format(t2-t1))
        return results


def ratio(groundtruth,results,query):
    pass

basePath = "/home/luy100/ForVLDB/ComparedMethod/Datasets"
if __name__ == '__main__':
    dataset_file_utl = basePath + '/glove/glove.twitter.27B.100d.npy'
    print('Reading the dataset')
    trainData = np.load(dataset_file_utl)
    print('Done')
    assert trainData.dtype == np.float32

    print('Normalizing the dataset')
    trainData /= np.linalg.norm(trainData, axis=1).reshape(-1, 1)
    print('Done')

    print('Generating queries')
    np.random.shuffle(trainData)
    # trainData = trainData[:100000]
    # dataset = dataset[:200000]
    queries = trainData[:2]
    print('Done')

    print('Centering the dataset and queries')
    center = np.mean(trainData, axis=0)
    trainData -= center
    queries -= center
    print('Done')

    highestsn=0
    highestR=0
    # trainData = np.random.random_sample((10000, 100)) * -1
    gt = []
    gt.append(np.dot(trainData, trainData[0]).argsort()[-10:][::-1])
    gt.append(np.dot(trainData, trainData[1]).argsort()[-10:][::-1])


    # for sn in range(1000):
    np.random.seed(241)
    t = TraditionalLSH(16, 10, 100)
    t1 = timeit.default_timer()
    t.fit(trainData)
    t2 = timeit.default_timer()
    print(t2 - t1)
    t1 = timeit.default_timer()
    results=t.query(10, [trainData[0], trainData[1]], 0)
    # print(results)
    t2 = timeit.default_timer()
    print(t2 - t1)
    # print(gt)
    # R=len(set(gt[0]).intersection(set(results[0]))) + len(set(gt[1]).intersection(set(results[1])))
    print(len(set(gt[0]).intersection(set(results[0]))))
    print(len(set(gt[1]).intersection(set(results[1]))))
        # if R > highestR:
        #     highestR = R
        #     highestsn = sn
    # print(highestR)
    # print(highestsn)


