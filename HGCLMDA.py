from random import shuffle

import torch
import numpy as np

from Model import HGCLMDA
from DataHandler import DataHandler, negSamp, transToLsts, transpose
from Params import args

torch.manual_seed(666)
np.random.seed(666)


class hgclmda():
    def __init__(self, handler):
        self.handler = handler
        self.handler.LoadData()

        adj = handler.trnMat
        idx, data, shape = transToLsts(adj, norm=True)
        self.adj_py = torch.sparse.FloatTensor(idx, data, shape).to(torch.float32).to(args.device)
        idx, data, shape = transToLsts(transpose(adj), norm=True)
        self.tpAdj_py = torch.sparse.FloatTensor(idx, data, shape).to(torch.float32).to(args.device)

        self.curepoch = 0

    def preparemodel(self):
        self.model = HGCLMDA(self.adj_py, self.tpAdj_py).to(args.device)
        self.opt = torch.optim.Adam(params=self.model.parameters(), lr=args.lr)
        self.scheduler = torch.optim.lr_scheduler.ExponentialLR(optimizer=self.opt, gamma=args.decayRate)
        for name, param in self.model.named_parameters():
            if param.requires_grad:
                print(name, param.shape)

    def sampleTrainBatch(self, batIds, labelMat):
        temLabel = labelMat[batIds].toarray()
        batch = len(batIds)
        temlen = batch * 2 * args.sampNum
        uLocs = [None] * temlen
        iLocs = [None] * temlen
        cur = 0
        for i in range(batch):
            posset = np.reshape(np.argwhere(temLabel[i] != 0), [-1])
            sampNum = min(args.sampNum, len(posset))
            if sampNum == 0:
                poslocs = [np.random.choice(args.item)]
                neglocs = [poslocs[0]]
            else:
                poslocs = np.random.choice(posset, sampNum)
                neglocs = negSamp(temLabel[i], sampNum, args.item)
            for j in range(sampNum):
                posloc = poslocs[j]
                negloc = neglocs[j]
                uLocs[cur] = uLocs[cur + temlen // 2] = batIds[i]
                iLocs[cur] = posloc
                iLocs[cur + temlen // 2] = negloc
                cur += 1
        uLocsa = uLocs[:cur] + uLocs[temlen // 2: temlen // 2 + cur]
        iLocsa = iLocs[:cur] + iLocs[temlen // 2: temlen // 2 + cur]

        return torch.Tensor(uLocsa).to(args.device), torch.Tensor(iLocsa).to(args.device)

    def trainEpoch(self):
        args.actFunc = 'leakyRelu'

        sfIds = np.random.permutation(args.user)
        epochLoss, epochPreLoss, epochsslloss, epochregloss = [0] * 4
        num = len(sfIds)
        steps = int(np.ceil(num / args.batch))
        self.model.train()

        for i in range(steps):
            st = i * args.batch
            ed = min((i + 1) * args.batch, num)
            batIds = sfIds[st: ed]

            uLocs, iLocs = self.sampleTrainBatch(batIds, self.handler.trnMat)

            preds, sslloss, regularize = self.model(uLocs.long(), iLocs.long())
            sampNum = uLocs.shape[0] // 2
            posPred = preds[:sampNum]
            negPred = preds[sampNum:sampNum * 2]

            preLoss = torch.sum(
                torch.maximum(torch.Tensor([0.0]).to(args.device), 1.0 - (posPred - negPred))) / args.batch
            sslloss = args.ssl_reg * sslloss
            regLoss = args.reg * regularize

            loss = preLoss + regLoss + sslloss

            self.opt.zero_grad()
            loss.backward()
            self.opt.step()
            if i % args.decay_step == 0:
                self.scheduler.step()

            epochLoss += loss
            epochPreLoss += preLoss
            epochregloss += args.reg * regularize
            epochsslloss += args.ssl_reg * sslloss

        ret = dict()
        ret['Loss'] = epochLoss / steps
        ret['preLoss'] = epochPreLoss / steps
        ret['sslLoss'] = epochsslloss / steps
        ret['regLoss'] = epochregloss / steps

        return ret

    def run(self):
        self.preparemodel()
        stloc = 0
        for ep in range(stloc, args.epoch):
            reses = self.trainEpoch()
            self.curepoch = ep
            print(f'Loss: {reses["Loss"]};\n'
                  f'preLoss: {reses["preLoss"]};\n'
                  f'sslLoss: {reses["sslLoss"]};\n'
                  f'regLoss: {reses["regLoss"]}.')


if __name__ == '__main__':
    handler = DataHandler()
    handler.LoadData()
    model = hgclmda(handler)
    model.run()