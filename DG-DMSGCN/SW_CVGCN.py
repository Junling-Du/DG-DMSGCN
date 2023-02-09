import torch
import torch.nn as nn
from torch.autograd import Function
import torch.nn.init as init
import torch.nn.functional as F




def get_graph_feature(x, k, idx=None):
    batch_size = x.size(0)
    num_points = x.size(2)
    x = x.view(batch_size, -1, num_points)
    device = torch.device('cuda')

    idx_base = torch.arange(0, batch_size, device=device).view(-1, 1, 1) * num_points

    idx = idx + idx_base

    idx = idx.view(-1)

    _, num_dims, _ = x.size()

    x = x.transpose(2,
                    1).contiguous()  # (batch_size, num_points, num_dims)  -> (batch_size*num_points, num_dims) #   batch_size * num_points * k + range(0, batch_size*num_points)
    # print(x.size)
    feature = x.view(batch_size * num_points, -1)[idx, :]
    feature = feature.view(batch_size, num_points, k, num_dims)
    x = x.view(batch_size, num_points, 1, num_dims).repeat(1, 1, k, 1)

    feature = torch.cat((feature - x, x), dim=3).permute(0, 3, 1, 2).contiguous()

    return feature  # (batch_size, 2*num_dims, num_points, k)


class DGCNN_cls(nn.Module):
    def __init__(self, inp_dim, k, kernels):
        super(DGCNN_cls, self).__init__()
        self.k = k

        self.bn1 = nn.BatchNorm2d(kernels[0])
        self.bn2 = nn.BatchNorm2d(kernels[1])
        self.bn3 = nn.BatchNorm2d(kernels[2])
        self.bn4 = nn.BatchNorm1d(kernels[3])

        self.conv1 = nn.Sequential(nn.Conv2d(2*inp_dim, kernels[0], kernel_size=1, bias=False),
                                   self.bn1,
                                   nn.LeakyReLU(negative_slope=0.2))
        self.conv2 = nn.Sequential(nn.Conv2d(2*kernels[0], kernels[1], kernel_size=1, bias=False),
                                   self.bn2,
                                   nn.LeakyReLU(negative_slope=0.2))
        self.conv3 = nn.Sequential(nn.Conv2d(2*kernels[1] , kernels[2], kernel_size=1, bias=False),
                                   self.bn3,
                                   nn.LeakyReLU(negative_slope=0.2))
        self.conv4 = nn.Sequential(nn.Conv1d((kernels[0]+kernels[1]+kernels[2]), kernels[3], kernel_size=1, bias=False),
                                   self.bn4,
                                   nn.LeakyReLU(negative_slope=0.2))


    def forward(self, x, idx = None):
        batch_size = x.size(0)
        if idx is not None:
            idx = idx.unsqueeze(0).repeat(batch_size,1,1)
        x = get_graph_feature(x, k=self.k, idx=idx)
        x = self.conv1(x)
        x1 = x.max(dim=-1, keepdim=False)[0]

        x = get_graph_feature(x1, k=self.k, idx=idx)
        x = self.conv2(x)
        x2 = x.max(dim=-1, keepdim=False)[0]

        x = get_graph_feature(x2, k=self.k, idx=idx)
        x = self.conv3(x)
        x3 = x.max(dim=-1, keepdim=False)[0]

        x = torch.cat((x1,x2,x3), dim=1)

        x = self.conv4(x)

        return x

class time_distribulate_DGCNN_cls(nn.Module):
    def __init__(self, inp_dim, k, kernel,hide):
        super(time_distribulate_DGCNN_cls, self).__init__()
        self.DGCNN = DGCNN_cls(inp_dim, k, kernel)
        self.fc1 = nn.Linear(800*kernel[3],hide)
        self.fc2 = nn.Linear(hide,1)
        self.relu = nn.LeakyReLU(negative_slope=0.2)
        self.average = nn.AvgPool1d(312,stride=None)

    def forward(self, x,tslen,idx1,idx2):
        feature1 = self.DGCNN(x,idx1)
        feature2 = self.DGCNN(x, idx2)
        feature = torch.cat((feature1,feature2),1)
        time_point,fea_dim,roi = feature.shape[0],feature.shape[1],feature.shape[2]
        bs = tslen.shape[0]
        assert torch.sum(tslen) == time_point
        tsend = 0
        subj_features1 = torch.ones(bs, fea_dim, roi).cuda()
        subj_features2 = torch.ones(bs, fea_dim, roi).cuda()
        for i in range(bs):
            subj_feature = feature[tsend:tsend+tslen[i,0],:,:]
            subj_features1[i] = torch.mean(subj_feature,dim=0).squeeze()
            subj_features2[i] = torch.var(subj_feature, dim=0,unbiased=False).squeeze()
            tsend = tsend+tslen[i,0]
        subj_features = torch.cat((subj_features1,subj_features2),dim=1)
        out_flatten = torch.flatten(subj_features,1,-1)
        out_feature1 = self.relu(self.fc1(out_flatten))
        out_roi =self.fc2(out_feature1)
        return out_roi, out_feature1