import os
os.environ["CUDA_VISIBLE_DEVICES"] = '0'

import math
import torch.optim as optim
import numpy as np
from SW_CVGCN import *
import torch.utils.data as Data
from DMSGCN import *


class MyDataSet(Data.Dataset):
  def __init__(self, x, y, z):
    super(MyDataSet, self).__init__()
    self.x = x
    self.y = y
    self.z = z
  def __len__(self):
    return self.x.shape[0]

  def __getitem__(self, idx):
    return self.x[idx], self.y[idx], self.z[idx]

def slidingwindow(data,ts_len,window_size=3,step_size=1):
    subj_num = data.shape[0]
    window_datas = []
    for subj in range(subj_num):
        subj_ts_len = int(ts_len[subj,0])
        valid_data = data[subj,:subj_ts_len,:]
        for window in range((subj_ts_len-window_size+1)//step_size):
            window_data = valid_data[step_size*window:step_size*window+window_size,:]
            window_datas.append(window_data)
    window_datas = np.array(window_datas)
    win_num = (ts_len-window_size+1)//step_size
    return window_datas,win_num


num_domains = 17
num_subj = 100
roi = 200
BOLD_len = np.random.randint(100,300,size=(num_subj,1))
BOLD_series = np.random.random((num_subj,300,roi))
sites = np.random.randint(0,num_domains,size=(num_subj))
x = BOLD_series
k = 5
knn_graph_g = np.random.random((roi,roi))
knn_graph_g = knn_graph_g.swapaxes(0, 1)
knn_graph_g = knn_graph_g[:, 1:k + 1]
knn_graph_c = np.random.random((roi,roi))
knn_graph_c = knn_graph_c.swapaxes(0, 1)
knn_graph_c = knn_graph_c[:, 1:k + 1]
y = np.random.randint(0,2,size = (num_subj,1))
y = y.astype(float)
lr = 0.0005
epoch_SWDGCN = 20
epoch_MSGCN = 10
batchsize = 16
criterion = nn.BCEWithLogitsLoss()#(reduction='none')
window_size = 3
hide = 256
alpha = 0.5

for site in range(num_domains):
    ###处理输入和标签数据类型使能训练
    print('site id:', site)
    train_ind = np.where(sites != site)[0]
    test_ind = np.where(sites == site)[0]
    x_train = x[train_ind,:,:]
    x_test = x[test_ind,:,:]
    y_train = y[train_ind]
    y_test = y[test_ind]
    BOLD_len_train = BOLD_len[train_ind]
    BOLD_len_test = BOLD_len[test_ind]
    assert x_train.shape[0]+x_test.shape[0] == num_subj
    knn_graph1 = torch.from_numpy(knn_graph_g).cuda().to(torch.uint8)
    knn_graph2 = torch.from_numpy(knn_graph_c).cuda().to(torch.uint8)
    model = time_distribulate_DGCNN_cls(window_size, k, [3, 4, 8, 8],hide).cuda()  # [8, 16, 32, 64, 64]
    optimizer = optim.Adam(model.parameters(),
                           lr=lr, weight_decay=0)
    loader1 = Data.DataLoader(MyDataSet(x_train, y_train, BOLD_len_train), batch_size=batchsize, shuffle=False)
    loader2 = Data.DataLoader(MyDataSet(x_test, y_test, BOLD_len_test), batch_size=batchsize, shuffle=False)
    for epoch in range(epoch_SWDGCN):
        # model.train()
        ts_feature_train_all = torch.empty(0,hide).cuda()
        ts_feature_test_all = torch.empty(0,hide).cuda()
        for x1, y1, l1 in loader1:
            optimizer.zero_grad()
            x11,w_num1 = slidingwindow(x1.numpy(),l1.numpy(),window_size=window_size)
            x1 = torch.from_numpy(x11).to(torch.float32).cuda()
            w_num1 = torch.from_numpy(w_num1).to(torch.int).type(torch.long).cuda()
            y1 = y1.cuda()
            out_train,ts_feature_train = model(x1, w_num1, knn_graph1,knn_graph2) #
            loss_train = criterion(out_train, y1)
            # print(loss_train)
            loss_train.backward()
            optimizer.step()
            ts_feature_train_all = torch.cat((ts_feature_train_all,ts_feature_train),dim=0)
        for x2, y2, l2 in loader2:
            x22, w_num2 = slidingwindow(x2.numpy(), l2.numpy(),window_size=window_size)
            x2 = torch.from_numpy(x22).to(torch.float32).cuda()
            w_num2 = torch.from_numpy(w_num2).to(torch.int).type(torch.long).cuda()
            out_test,ts_feature_test = model(x2,w_num2, knn_graph1,knn_graph2)
            ts_feature_test_all = torch.cat((ts_feature_test_all, ts_feature_test), dim=0)

    len_train = train_ind.shape[0]
    sites_train = sites[train_ind]
    sites_test = sites[test_ind]
    sites_ms = np.concatenate((sites_train,sites_test),0)
    y_mc = np.concatenate((y_train,y_test),0)
    ts_feature = torch.cat((ts_feature_train_all,ts_feature_test_all),0).data
    pheno_graph, site_graph, label_graph = multi_site_graph_LOCV(sites_ms, site, num_domains, y_mc)
    in_pheno_graph = pheno_graph[site, :, :]
    for i in range(num_subj):
        in_pheno_graph[i, i] = 0
    out_pheno_graph = np.concatenate((pheno_graph[:site, :, :], pheno_graph[site + 1:, :, :]), axis=0)
    in_pheno_graph = torch.from_numpy(in_pheno_graph).to(torch.float32).cuda()
    out_pheno_graph = torch.from_numpy(out_pheno_graph).to(torch.float32).cuda()
    mcgcn = GCN_fc_LOCV(hide).cuda()
    optimizer2 = optim.Adam(mcgcn.parameters(),
                           lr=lr, weight_decay=0)
    y_train = torch.from_numpy(y_train).to(torch.float32).squeeze()
    y_test = torch.from_numpy(y_test).to(torch.float32).squeeze()
    for epoch in range(epoch_MSGCN):
        mcgcn.train()
        optimizer2.zero_grad()
        x3 = ts_feature.cuda()
        y3 = y_train.cuda()
        out_all = mcgcn(x3,alpha,in_pheno_graph,out_pheno_graph,k)
        loss = criterion(out_all[:len_train], y3)
        loss.backward()
        optimizer2.step()