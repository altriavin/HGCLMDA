import scipy.sparse as sp
from scipy.sparse import coo_matrix, csr_matrix
import torch
import numpy as np
import pickle
from Params import args


class DataHandler:
    def __init__(self):
        predir = ""
        if args.data == "mDA1":
            predir = "Data/mRNA-drug-large/mRNA_drug_not_Mutation_p-value_0.05/"
        elif args.data == "mDA2":
            predir = "Data/mRNA-drug-large/mRNA_drug_not_Mutation_p-value_0.005/"
        elif args.data == "mDA3":
            predir = "Data/mRNA-drug-large/mRNA_drug_not_Mutation_p-value_0.0005/"
        elif args.data == "mDA_":
            predir = "Data/mRNA-drug-large/mRNA_drug_association_90/"

        self.predir = predir
        self.trnfile = predir + 'train_data.pkl'
        self.tstfile = predir + 'test_data.pkl'
        self.validfile = predir + "valid_data.pkl"

    def LoadData(self):
        if args.percent > 1e-8:
            print('noised')
            with open(self.predir + 'noise_%.2f' % args.percent, 'rb') as fs:
                trnMat = (pickle.load(fs) != 0).astype(np.float32)
        else:
            print(f"loading data from: {self.trnfile}")

            with open(self.trnfile, 'rb') as fs:
                trnMat = (pickle.load(fs) != 0).astype(np.float32)
        # test set
        with open(self.tstfile, 'rb') as fs:
            tstMat = pickle.load(fs)
            # tstMat = (pickle.load(fs) != 0).astype(np.float32)
        tstLocs = [None] * tstMat.shape[0]
        tstMRNAs = set()
        for i in range(len(tstMat.data)):
            row = tstMat.row[i]
            col = tstMat.col[i]
            if tstLocs[row] is None:
                tstLocs[row] = list()
            tstLocs[row].append(col)
            tstMRNAs.add(row)
        tstMRNAs = np.array(list(tstMRNAs))

        with open(self.validfile, "rb") as fs:
            validMat = pickle.load(fs)
        validLocs = [None] * validMat.shape[0]
        validMRNAs = set()
        for i in range(len(validMat.data)):
            row = validMat.row[i]
            col = validMat.col[i]
            if validLocs[row] is None:
                validLocs[row] = list()
            validLocs[row].append(col)
            validMRNAs.add(row)
        validMRNAs = np.array(list(validMRNAs))

        self.trnMat = trnMat
        self.tstLocs = tstLocs
        self.tstMRNAs = tstMRNAs
        self.validLocs = validLocs
        self.validMRNAs = validMRNAs
        args.mRNA, args.drug = self.trnMat.shape

        # print(f'mRNA: {args.mRNA}; drug: {args.drug}')

        self.prepareGlobalData()

    def prepareGlobalData(self):
        adj = self.trnMat
        adj = (adj != 0).astype(np.float32)
        self.labelP = np.squeeze(np.array(np.sum(adj, axis=0)))
        tpadj = transpose(adj)
        adjNorm = np.reshape(np.array(np.sum(adj, axis=1)), [-1])
        tpadjNorm = np.reshape(np.array(np.sum(tpadj, axis=1)), [-1])
        for i in range(adj.shape[0]):
            for j in range(adj.indptr[i], adj.indptr[i + 1]):
                adj.data[j] /= adjNorm[i]
        for i in range(tpadj.shape[0]):
            for j in range(tpadj.indptr[i], tpadj.indptr[i + 1]):
                tpadj.data[j] /= tpadjNorm[i]
        self.adj = adj
        self.tpadj = tpadj


def transpose(mat):
    coomat = coo_matrix(mat)
    return csr_matrix(coomat.transpose())


def negSamp(temLabel, sampSize, nodeNum):
    negset = [None] * sampSize
    cur = 0
    while cur < sampSize:
        rdmDrug = np.random.choice(nodeNum)
        if temLabel[rdmDrug] == 0:
            negset[cur] = rdmDrug
            cur += 1
    return negset


def transToLsts(mat, mask=False, norm=False):
    shape = torch.Size(mat.shape)
    mat = sp.coo_matrix(mat)
    indices = torch.from_numpy(np.vstack((mat.row, mat.col)).astype(np.int64))
    data = mat.data

    if norm:
        rowD = np.squeeze(np.array(1 / (np.sqrt(np.sum(mat, axis=1) + 1e-8) + 1e-8)))
        colD = np.squeeze(np.array(1 / (np.sqrt(np.sum(mat, axis=0) + 1e-8) + 1e-8)))
        for i in range(len(mat.data)):
            row = indices[0, i]
            col = indices[1, i]
            data[i] = data[i] * rowD[row] * colD[col]
        # half mask
    if mask:
        spMask = (np.random.uniform(size=data.shape) > 0.5) * 1.0
        data = data * spMask

    if indices.shape[0] == 0:
        indices = np.array([[0, 0]], dtype=np.int32)
        data = np.array([0.0], np.float32)

    data = torch.from_numpy(data)
    # a =torch.sparse.FloatTensor(indices, values, shape).to(torch.float32).cuda()
    return indices, data, shape
