import numpy as  np
import torch
import torch.nn as nn
from torch.autograd import Function
import torch.nn.init as init
import torch.nn.functional as F
from scipy.spatial import distance


def cal_fea_adj(x,alpha):
    x_copy = x.detach().cpu().numpy()
    bs = np.shape(x_copy)[0]
    distv = distance.pdist(x_copy, metric='correlation')
    dist0 = distance.squareform(distv)
    sigma = np.mean(dist0)
    inter_adj = np.exp(- dist0 ** 2 / (2 * sigma ** 2))
    inter_adj = (inter_adj-np.eye(bs))*alpha+np.eye(bs)
    return inter_adj



#### LOSO CV ####
def multi_site_graph_LOCV(sites,target_site,site_num,label):
    subj_num = np.size(sites)
    site_graph1 = np.zeros((site_num,subj_num,subj_num))
    all_idx = np.array(range(subj_num))
    target_idx = all_idx[sites==target_site]
    for i in range(site_num):
        i_idx = all_idx[sites ==i]
        for j in target_idx:
            for k in i_idx:
                site_graph1[i,j,k] = 1
                site_graph1[i, k, j] = 1
    site_graph2 = np.zeros((subj_num,subj_num))
    train_idx = all_idx[sites!=target_site]
    for i in train_idx:
        for j in train_idx:
            site_graph2[i,j] = 1
    site_graph = np.concatenate((site_graph1, np.expand_dims(site_graph2, axis=0)), axis=0)
    for i in target_idx:
        label[i] = 0.5
    label_graph = np.zeros((subj_num,subj_num))
    for i in range(subj_num):
        for j in range(subj_num):
            if label[i] == label[j]:
                label_graph[i,j] = 1
            elif (label[i] != label[j]) & (label[i] == 0.5):
                label_graph[i,j] = 0.5
            elif (label[i] != label[j]) & (label[j] == 0.5):
                label_graph[i,j] = 0.5
    pheno_graph = np.repeat(np.expand_dims(label_graph,axis=0),site_num+1,axis=0)*site_graph



    return pheno_graph,site_graph,label_graph



class GCN_fc_LOCV(nn.Module):
    def __init__(self, hid):
        super(GCN_fc_LOCV, self).__init__()
        self.coefficient = torch.nn.Parameter(torch.Tensor([0]))
        self.coefficient0 = torch.nn.Parameter(torch.Tensor([0]))
        self.coefficient1 = torch.nn.Parameter(torch.Tensor([0]))
        self.coefficient2 = torch.nn.Parameter(torch.Tensor([0]))
        self.coefficient3 = torch.nn.Parameter(torch.Tensor([0]))
        self.coefficient4 = torch.nn.Parameter(torch.Tensor([0]))
        self.coefficient5 = torch.nn.Parameter(torch.Tensor([0]))
        self.coefficient6 = torch.nn.Parameter(torch.Tensor([0]))
        self.coefficient7 = torch.nn.Parameter(torch.Tensor([0]))
        self.coefficient8 = torch.nn.Parameter(torch.Tensor([0]))
        self.coefficient9 = torch.nn.Parameter(torch.Tensor([0]))
        self.coefficient10 = torch.nn.Parameter(torch.Tensor([0]))
        self.coefficient11 = torch.nn.Parameter(torch.Tensor([0]))
        self.coefficient12 = torch.nn.Parameter(torch.Tensor([0]))
        self.coefficient13 = torch.nn.Parameter(torch.Tensor([0]))
        self.coefficient14 = torch.nn.Parameter(torch.Tensor([0]))
        self.coefficient15 = torch.nn.Parameter(torch.Tensor([0]))
        self.coefficient16 = torch.nn.Parameter(torch.Tensor([0]))
        self.fc1 = nn.Linear(hid,32)
        self.fc2 = nn.Linear(32,1)
        self.relu = nn.LeakyReLU(negative_slope=0.2)
        self.sig = nn.Sigmoid()

    def forward(self, x,alpha,in_pheno_graph,out_pheno_graph,k):
        fea_graph = cal_fea_adj(x, alpha)
        fea_graph=torch.from_numpy(fea_graph).to(torch.float32).cuda()
        eye = torch.eye(100).cuda()
        pheno_graph1 = ( self.coefficient0*out_pheno_graph[0,:,:]+
                         self.coefficient1*out_pheno_graph[1,:,:]+
                         self.coefficient2*out_pheno_graph[2,:,:]+
                         self.coefficient3*out_pheno_graph[3,:,:]+
                         self.coefficient4*out_pheno_graph[4,:,:]+
                         self.coefficient5*out_pheno_graph[5,:,:]+
                         self.coefficient6*out_pheno_graph[6,:,:]+
                         self.coefficient7*out_pheno_graph[7,:,:]+
                         self.coefficient8*out_pheno_graph[8,:,:]+
                         self.coefficient9*out_pheno_graph[9,:,:]+
                         self.coefficient10*out_pheno_graph[10,:,:]+
                         self.coefficient11*out_pheno_graph[11,:,:]+
                         self.coefficient12*out_pheno_graph[12,:,:]+
                         self.coefficient13*out_pheno_graph[13,:,:]+
                         self.coefficient14*out_pheno_graph[14,:,:]+
                         self.coefficient15*out_pheno_graph[15,:,:]+
                         self.coefficient16*out_pheno_graph[16,:,:]+
                         eye+self.coefficient*in_pheno_graph)
        adj = fea_graph*pheno_graph1
        indices_to_remove = adj < torch.topk(adj, k=k)[0][..., -1, None]
        adj[indices_to_remove] = 0
        x1 = torch.mm(adj,x)
        out = self.fc2(self.relu(self.fc1(x1)))
        return out.squeeze()


#### 10 fold ####

def multi_site_graph_10CV(site,train_idx,test_idx,label):
    subj_num = np.size(site)
    test_in_graph = np.zeros((subj_num, subj_num))
    test_out_graph = np.zeros((subj_num, subj_num))
    train_out_graph = np.zeros((subj_num,subj_num))
    for i in test_idx:
        for j in train_idx:
            if site[i] == site[j]:
                test_in_graph[i, j] = 1
                test_in_graph[j, i] = 1
            else:
                test_out_graph[i, j] = 1
                test_out_graph[j, i] = 1
    for i in train_idx:
        for j in train_idx:
            if (label[i] == label[j])&(site[i]!=site[j]):
                train_out_graph[i,j] = 1

    return test_in_graph,test_out_graph,train_out_graph

class GCN_fc_10CV(nn.Module):
    def __init__(self, hid):
        super(GCN_fc_10CV, self).__init__()
        self.coefficient0 = torch.nn.Parameter(torch.Tensor([0]))
        self.coefficient1 = torch.nn.Parameter(torch.Tensor([0]))
        self.coefficient2 = torch.nn.Parameter(torch.Tensor([0]))
        self.fc2 = nn.Linear(hid,1)
        self.relu = nn.LeakyReLU(negative_slope=0.2)
        self.sig = nn.Sigmoid()

    def forward(self, x,alpha,test_in_graph,test_out_graph,train_out_graph,k):
        fea_graph = cal_fea_adj(x, alpha)
        fea_graph=torch.from_numpy(fea_graph).to(torch.float32).cuda()
        eye = torch.eye(100).cuda()
        pheno_graph1 = ( self.coefficient0*test_in_graph+
                         self.coefficient1 * test_out_graph +
                         self.coefficient2 * train_out_graph +
                         eye)
        adj = fea_graph*pheno_graph1
        indices_to_remove = adj < torch.topk(adj, k=k)[0][..., -1, None]
        adj[indices_to_remove] = 0
        x1 = torch.mm(adj,x)
        out = self.fc2(x1)
        return out.squeeze()