import numpy as np
import torch
import torch.nn as nn
from Params import args

torch.manual_seed(666)


def LeakyRelu(data):
    # global leaky
    ret = torch.maximum(args.leaky * data, data)
    return ret


class FC(nn.Module):
    def __init__(self, inputDim, outDim, Bias=False, actFunc=None):
        super(FC, self).__init__()
        initializer = nn.init.xavier_normal_
        self.W_fc = nn.Parameter(initializer(torch.empty(inputDim, outDim).to(args.device)))

    def forward(self, inp, droprate=0):
        # W = self.W_fc.weight
        fc1 = inp @ self.W_fc
        ret = fc1
        ret = LeakyRelu(ret)
        return ret


class hyperPropagate(nn.Module):
    def __init__(self, inputdim):
        super(hyperPropagate, self).__init__()
        self.inputdim = inputdim
        self.fc1 = FC(self.inputdim, args.hyperNum, actFunc='leakyRelu').to(args.device)
        self.fc2 = FC(self.inputdim, args.hyperNum, actFunc='leakyRelu').to(args.device)
        self.fc3 = FC(self.inputdim, args.hyperNum, actFunc='leakyRelu').to(args.device)
        # self.actFunc = nn.LeakyReLU(negative_slope=args.leaky)

    def forward(self, lats, adj):
        lat1 = LeakyRelu(
            torch.transpose(adj, 0, 1) @ lats)  # shape adj:mRNA,hyperNum lats:mRNA,latdim lat1:hypernum,latdim
        lat2 = torch.transpose(self.fc1(torch.transpose(lat1, 0, 1)), 0, 1) + lat1  # shape hypernum,latdim
        lat3 = torch.transpose(self.fc2(torch.transpose(lat2, 0, 1)), 0, 1) + lat2
        lat4 = torch.transpose(self.fc3(torch.transpose(lat3, 0, 1)), 0, 1) + lat3
        ret = adj @ lat4
        ret = LeakyRelu(ret)
        return ret


class weight_trans(nn.Module):
    def __init__(self):
        super(weight_trans, self).__init__()
        initializer = nn.init.xavier_normal_
        self.W = nn.Parameter(initializer(torch.empty(args.latdim, args.latdim).to(args.device)))

    def forward(self, normalize):
        ret = normalize @ self.W
        return ret


class HGCLMDA(nn.Module):
    def __init__(self, adj_py, tpAdj_py):
        super(HGCLMDA, self).__init__()
        initializer = nn.init.xavier_normal_
        self.mEmbed0 = nn.Parameter(initializer(torch.empty(args.mRNA, args.latdim).to(args.device)))
        self.dEmbed0 = nn.Parameter(initializer(torch.empty(args.drug, args.latdim).to(args.device)))
        self.mhyper = nn.Parameter(initializer(torch.empty(args.latdim, args.hyperNum).to(args.device)))
        self.dhyper = nn.Parameter(initializer(torch.empty(args.latdim, args.hyperNum).to(args.device)))

        self.adj = adj_py.to(args.device)  # shape mRNA,drug
        self.tpadj = tpAdj_py.to(args.device)  # shape drug,mRNA

        self.hyperMLat_layers = nn.ModuleList()
        self.hyperDLat_layers = nn.ModuleList()
        self.weight_layers = nn.ModuleList()

        for i in range(args.gnn_layer):
            self.hyperMLat_layers.append(hyperPropagate(args.hyperNum))  # shape hyperNum,hyperNum
            self.hyperDLat_layers.append(hyperPropagate(args.hyperNum))  # shape hyperNum,hyperNum
            self.weight_layers.append(weight_trans())

    def messagePropagate(self, lats, adj):
        return LeakyRelu(torch.sparse.mm(adj, lats))

    def calcSSL(self, hyperLat, gnnLat):
        # print(f'args.temp: {args.temp}')
        posScore = torch.exp(torch.sum(hyperLat * gnnLat, dim=1) / args.temp)
        negScore = torch.sum(torch.exp(gnnLat @ torch.transpose(hyperLat, 0, 1) / args.temp), dim=1)
        uLoss = torch.sum(-torch.log(posScore / (negScore + 1e-8) + 1e-8))
        return uLoss

    def Regularize(self, reg, method='L2'):
        ret = 0.0
        for i in range(len(reg)):
            ret += torch.sum(torch.square(reg[i]))
        return ret

    def edgeDropout(self, mat, drop):
        def dropOneMat(mat):
            indices = mat._indices().cpu()
            values = mat._values().cpu()
            shape = mat.shape
            newVals = nn.functional.dropout(values, p=drop)
            return torch.sparse.FloatTensor(indices, newVals, shape).to(torch.float32).to(args.device)

        return dropOneMat(mat)

    def forward_test(self):
        mEmbed0 = self.mEmbed0
        dEmbed0 = self.dEmbed0
        mhyper = self.mhyper
        dhyper = self.dhyper

        mmhyper = mEmbed0 @ mhyper  # shape mRNA,hyperNum
        ddhyper = dEmbed0 @ dhyper  # shape drug,hyperNum

        mlats = [mEmbed0]
        dlats = [dEmbed0]

        for i in range(args.gnn_layer):
            mlat = self.messagePropagate(dlats[-1], self.edgeDropout(self.adj, drop=0))
            dlat = self.messagePropagate(mlats[-1], self.edgeDropout(self.tpadj, drop=0))
            hyperMLat = self.hyperMLat_layers[i](mlats[-1], nn.functional.dropout(mmhyper, p=0))
            hyperDLat = self.hyperDLat_layers[i](dlats[-1], nn.functional.dropout(ddhyper, p=0))

            mlats.append(mlat + hyperMLat + mlats[-1])
            dlats.append(dlat + hyperDLat + dlats[-1])

        mlat = sum(mlats)
        dlat = sum(dlats)
        return mlat, dlat

    def forward(self, mids, dids, droprate=args.droprate):
        mEmbed0 = self.mEmbed0
        dEmbed0 = self.dEmbed0
        mhyper = self.mhyper
        dhyper = self.dhyper
        gnnMLats = []
        gnnDLats = []
        hyperMLats = []
        hyperDLats = []

        mlats = [mEmbed0]
        dlats = [dEmbed0]
        for i in range(args.gnn_layer):
            mlat = self.messagePropagate(dlats[-1], self.edgeDropout(self.adj, drop=droprate))
            dlat = self.messagePropagate(mlats[-1], self.edgeDropout(self.tpadj, drop=droprate))
            hyperMLat = self.hyperMLat_layers[i](mlats[-1], nn.functional.dropout(mEmbed0 @ mhyper,
                                                                                  p=droprate))  # / (1 - droprate))
            hyperDLat = self.hyperDLat_layers[i](dlats[-1], nn.functional.dropout(dEmbed0 @ dhyper,
                                                                                  p=droprate))  # / (1 - droprate) )

            gnnMLats.append(mlat)
            gnnDLats.append(dlat)
            hyperMLats.append(hyperMLat)
            hyperDLats.append(hyperDLat)

            mlats.append(mlat + hyperMLat + mlats[-1])
            dlats.append(dlat + hyperDLat + dlats[-1])

        mlat = sum(mlats)
        dlat = sum(dlats)

        pckmlat = torch.index_select(mlat, 0, mids)
        pckdlat = torch.index_select(dlat, 0, dids)
        preds = torch.sum(pckmlat * pckdlat, dim=-1)

        sslloss = 0
        uniqmids = torch.unique(mids)
        uniqdids = torch.unique(dids)

        for i in range(len(hyperMLats)):
            pckhyperMLat = self.weight_layers[i](
                torch.nn.functional.normalize(torch.index_select(hyperMLats[i], 0, uniqmids), p=2, dim=1))  # @ self.weight_layers[i].weight
            pckGnnmlat = torch.nn.functional.normalize(torch.index_select(gnnMLats[i], 0, uniqmids), p=2, dim=1)
            pckhyperDLat = self.weight_layers[i](
                torch.nn.functional.normalize(torch.index_select(hyperDLats[i], 0, uniqdids), p=2,
                                              dim=1))  # @ self.weight_layers[i].weight
            pckGnndlat = torch.nn.functional.normalize(torch.index_select(gnnDLats[i], 0, uniqdids), p=2, dim=1)
            uLoss = self.calcSSL(pckhyperMLat, pckGnnmlat)
            iLoss = self.calcSSL(pckhyperDLat, pckGnndlat)
            sslloss += uLoss + iLoss

        return preds, sslloss, self.Regularize([mEmbed0, dEmbed0, mhyper, dhyper])