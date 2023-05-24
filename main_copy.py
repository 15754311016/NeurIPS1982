import time
import argparse
import numpy as np
# a
import torch
# import torch.nn.functional as F
import torch.optim as optim
from sklearn import preprocessing
import torch.nn as nn
import torch.nn as nn
import torch
from torch.autograd import Variable
import torch.nn.functional as F
import torch.nn.init
import pdb
import torchvision.transforms as transforms
import numpy as np
import torch.optim as optim
import os
from utils import *
from torch.utils.data import DataLoader, TensorDataset
os.environ["CUDA_VISIBLE_DEVICES"] = "0"

parser = argparse.ArgumentParser()
parser.add_argument('--lr', type=float, default=0.01)
parser.add_argument('--weight_decay', type=float, default=1e-4)
parser.add_argument('--weight_wass', type=float, default=1e-4)
parser.add_argument('--weight_reconstruct', type=float, default=1e-4)
parser.add_argument('--weight_reg', type=float, default=1e-4)
parser.add_argument('--dropout', type=float, default=0.1)
parser.add_argument('--n_in', type=int, default=1)
parser.add_argument('--epochs', type=int, default=1000)
parser.add_argument('--hid_dim', type=int, default=50)
parser.add_argument('--reg', type=str, default='mi')
args = parser.parse_args()
Tensor = torch.cuda.FloatTensor
LongTensor = torch.cuda.LongTensor
loss = nn.MSELoss()
np.random.seed(43)
torch.manual_seed(31)
from sklearn.neighbors import KernelDensity


def kernel(x, sigma=1):
    return (1/np.sqrt(2*np.pi*sigma**2))*torch.exp(-x**2/(2*sigma**2))

def kde(x):
    n = len(x)
    # print(n)
    # h = bandwidth(x)
    h = 1
    kde = torch.zeros_like(x)
    for i in range(n):
        # print(i)
        kde += kernel((x - x[i])/h, sigma=1)
    kde /= (n*h)
    return kde

# Define the mutual information function
def mutual_info(x, y, n=50):
    x=x[:n]
    y=y[:n]
    # print("y shape:", y.shape)
    kdex = kde(x)
    kdey =kde(y)
    hx = -torch.sum(kdex*torch.log(kdex))
    hy = -torch.sum(kdey*torch.log(kdey))
    xy = torch.cat((x.unsqueeze(1), y.unsqueeze(1)), dim=1)
    hxy = -torch.sum(kde(xy)*torch.log(kde(xy)))
    mi = hx + hy - hxy
    return mi
class Net(nn.Module):
    def __init__(self, dim, dropout, n_in=2):
        super(Net, self).__init__()
        hid_dim=args.hid_dim
        self.n_in = n_in
        self.dropout = dropout
        self.layer = nn.Linear(dim,hid_dim).cuda()
        self.layer_c = nn.Linear(hid_dim,hid_dim).cuda()

        self.layer_zm_0 = nn.Linear(dim,hid_dim).cuda()
        self.layer_zm_1 = nn.Linear(dim,hid_dim).cuda()
        self.layer_zc_0 = nn.Linear(dim,hid_dim).cuda()
        self.layer_zc_1 = nn.Linear(dim,hid_dim).cuda()

        self.layer_zm_0_out = nn.ModuleList(nn.Linear(hid_dim,hid_dim).cuda() for i in range(n_in))
        self.layer_zm_1_out = nn.ModuleList(nn.Linear(hid_dim,hid_dim).cuda() for i in range(n_in))
        self.layer_zc_0_out = nn.ModuleList(nn.Linear(hid_dim,hid_dim).cuda() for i in range(n_in))
        self.layer_zc_1_out = nn.ModuleList(nn.Linear(hid_dim,hid_dim).cuda() for i in range(n_in))
        # self.layer_zm_0 = nn.ModuleList([nn.Linear(dim,hid_dim).cuda()])
        # self.layer_zm_0.extend(nn.Linear(hid_dim,hid_dim).cuda() for i in range(n_in))
        # self.layer_zm_1 = nn.ModuleList([nn.Linear(dim,hid_dim).cuda()])
        # self.layer_zm_1.extend(nn.Linear(hid_dim,hid_dim).cuda() for i in range(n_in))
        # self.layer_zc_0 = nn.ModuleList([nn.Linear(dim,hid_dim).cuda()])
        # self.layer_zc_0.extend(nn.Linear(hid_dim,hid_dim).cuda() for i in range(n_in))
        # self.layer_zc_1 = nn.ModuleList([nn.Linear(dim,hid_dim).cuda()])
        # self.layer_zc_1.extend(nn.Linear(hid_dim,hid_dim).cuda() for i in range(n_in))
        # self.layer_zm_1 = nn.ModuleList(nn.Linear(dim,hid_dim).cuda() for i in range(n_in))
        # self.layer_zc_0 = nn.ModuleList(nn.Linear(dim,hid_dim).cuda() for i in range(n_in))
        # self.layer_zc_1 = nn.ModuleList(nn.Linear(dim,hid_dim).cuda() for i in range(n_in))
        self.layer_y0_final = nn.Linear(hid_dim*2, hid_dim*2)
        self.layer_y1_final = nn.Linear(hid_dim*2,hid_dim*2)
        self.layer_y1 = nn.Linear(hid_dim*2,1)
        self.layer_y0 = nn.Linear(hid_dim*2,1)
        self.decode=nn.Sequential(nn.Linear(3*hid_dim, hid_dim), nn.ReLU(), nn.Linear(hid_dim, dim))
        self.lin_mi_0 = nn.Linear(2 * hid_dim, hid_dim)
        self.lin_mi_1 = nn.Linear(hid_dim, 1)
        nn.init.xavier_normal_(self.layer_y0.weight)
        nn.init.xavier_normal_(self.layer_y1.weight)
        #nn.init.constant_(self.layer_y1.bias,0)
        #nn.init.constant_(self.layer_y0.bias,0)
        # for m in self.children():
        #     nn.init.xavier_normal_(m.weight.data, gain=0.1)

    def forward(self, x, t):
        rep = F.elu(self.layer(x))
        rep_c = F.elu(self.layer_c(rep))
        #rep_c = F.dropout(rep_c, self.dropout, training=self.training)
        # rep_zm = F.elu(self.layer_zm(x))
        # rep_zc = F.elu(self.layer_zc(x))
        rep_zm_1 = F.elu(self.layer_zm_1(x))
        rep_zm_0 = F.elu(self.layer_zm_0(x))
        rep_zc_1 = F.elu(self.layer_zc_1(x))
        rep_zc_0 = F.elu(self.layer_zc_0(x))
        for i in range(self.n_in):
            rep_zm_0 = F.elu(self.layer_zm_0_out[i](rep_zm_0))
            rep_zm_1 = F.elu(self.layer_zm_1_out[i](rep_zm_1))
            rep_zc_0 = F.elu(self.layer_zc_0_out[i](rep_zc_0))
            rep_zc_1 = F.elu(self.layer_zc_1_out[i](rep_zc_1))
        # rep_zm_0 = F.dropout(rep_zm_0, self.dropout,training=self.training)
        # rep_zm_1 = F.dropout(rep_zm_1, self.dropout,training=self.training)
        # rep_zc_0 = F.dropout(rep_zc_0, self.dropout,training=self.training)
        # rep_zc_1 = F.dropout(rep_zc_1, self.dropout,training=self.training)
        rep_for_y0 = torch.cat((rep_c,rep_zm_0),1)
        rep_for_y1 = torch.cat((rep_c,rep_zm_1),1)
        rep_for_y0 = F.relu(self.layer_y0_final(rep_for_y0))
        rep_for_y1 = F.relu(self.layer_y1_final(rep_for_y1))
        y_0 = self.layer_y0(rep_for_y0).view(-1)
        y_1 = self.layer_y1(rep_for_y1).view(-1)
        y = torch.where(t>0,y_1,y_0)
        rep_zm = torch.where(t.reshape(-1,1)>0, rep_zm_1, rep_zm_0)
        rep_zc = torch.where(t.reshape(-1,1)>0, rep_zc_1, rep_zc_0)
        rep_all = torch.cat((rep_c, rep_zc, rep_zm),1)
        x_construct = self.decode(rep_all)
        if args.reg =='mi':
            loss = mutual_info(rep_zm, rep_zc)+mutual_info(rep_c, rep_zc)+mutual_info(rep_c, rep_zm)
        return y,rep_c,x_construct, loss


def train(net, x, y1, y0, t, X_test, T_test, Y1_test, Y0_test, epoch_num, args):
    yf_train = torch.where(t>0,y1,y0)
    criterion = nn.MSELoss()
    re_loss = nn.MSELoss()
    # optimizer = optim.SGD([{'params': net.parameters()}], lr=args.lr, weight_decay=args.weight_decay, momentum=0.8)
    optimizer = optim.Adam([{'params': net.parameters()}], lr=args.lr, weight_decay=args.weight_decay)
    scheduler=torch.optim.lr_scheduler.StepLR(optimizer, 500, gamma=0.8, last_epoch=-1, verbose=False)
    for epoch in range(epoch_num):
            net.train()
            optimizer.zero_grad()
            yf_pre, rep_c, x_construct, mi = net(x, t)
            rep_t1 = rep_c[(t > 0).nonzero()]
            rep_t0 = rep_c[(t < 1).nonzero()]
            dist, _ = wasserstein(rep_t1, rep_t0, cuda=True)
            recons_loss = re_loss(x_construct, x)
            loss_train = criterion(yf_train, yf_pre) + args.weight_wass * dist + args.weight_reconstruct*recons_loss + args.weight_reg*mi
            loss_train.backward()
            optimizer.step()
            scheduler.step()
            yf_pre,_,_,_ = net(x, t)
            ycf_pre,_,_,_ = net(x, 1-t)
            y1_pred, y0_pred = torch.where(t > 0, yf_pre, ycf_pre), torch.where(t > 0, ycf_pre, yf_pre)
            pehe_train = torch.sqrt(loss((y1_pred - y0_pred), (y1 - y0)))
            net.eval()
            yf_pre_test, _,_,_ = net(X_test, T_test)
            ycf_pre_test, _,_,_ = net(X_test, 1-T_test)
            y1_pred_test, y0_pred_test = torch.where(T_test > 0, yf_pre_test, ycf_pre_test), torch.where(T_test > 0, ycf_pre_test, yf_pre_test)
            pehe_test = torch.sqrt(loss((y1_pred_test - y0_pred_test), (Y1_test - Y0_test)))
            #pehe_test = torch.sqrt(torch.mean(((y1_pred_test - y0_pred_test)-(Y1_test - Y0_test))**2))
            if epoch % 1 ==0:
                print("epoch:{}".format(epoch+1),'loss:{}'.format(loss_train.item()),
                "pehe_train:{}".format(pehe_train.item()), "pehe_test:{}".format(pehe_test.item())
                )

if __name__ == "__main__":
    seed_torch = 21
    for i in range(10):
        np.random.seed(i)
        torch.manual_seed(seed_torch)
        #params = {'lr': 0.0001, "weight_decay": 1e-4}
        #X, T, Y1, Y0 = load_syn('./dataset/syn/4_0 copy.csv')
        # X, T, Y1, Y0 = load_syn('./data/Syn_1_1/24_24_24/4_0.csv')
        # X, T, Y1, Y0 = load_syn('./data/Syn_1_1/16_16_16/4_0.csv')
        d_c = 8
        d_zc = 8
        d_zm= 8
        X, T, Y1, Y0 = load_syn('./data/Syn_1_1/{}_{}_{}/4_0.csv'.format(d_c,d_zc,d_zm))
        #X, T, Y1, Y0 = load_syn('./data/Syn_1_1/100_100_100/4_0.csv')
        X, T, Y1, Y0 = X[:8000,:], T[:8000], Y1[:8000], Y0[:8000]
        X_train, T_train, Y1_train, Y0_train, X_test, T_test, Y1_test, Y0_test = split_L_U(X, T, Y1, Y0)
        #X_train_L, Y1_train_L, Y0_train_L, X_train_U, Y1_train_U, Y0_train_U, X_test, Y1_test, Y0_test, T_train_L, T_train_U, T_test = split_L_U(
        #    X, T, Y1, Y0)
        print("dim of X:", X.shape[1])
        net = Net(dim=X.shape[1], dropout=args.dropout, n_in=args.n_in).cuda()
        train(net, X_train, Y1_train, Y0_train, T_train, X_test, T_test, Y1_test, Y0_test, args.epochs, args=args)
        yf_pre_test, _,_,_ = net(X_test, T_test)
        ycf_pre_test, _,_,_ = net(X_test, 1-T_test)
        y1_pred_test, y0_pred_test = torch.where(T_test > 0, yf_pre_test, ycf_pre_test), torch.where(T_test > 0, ycf_pre_test, yf_pre_test)
        pehe_test = torch.sqrt(loss((y1_pred_test - y0_pred_test), (Y1_test - Y0_test)))
        #pehe_test = torch.sqrt(torch.mean(((y1_pred_test - y0_pred_test)-(Y1_test - Y0_test))**2))
        yf_test = torch.where(T_test>0, Y1_test, Y0_test)
        mse = loss(yf_pre_test, yf_test)
        print("test pehe:{}".format(pehe_test.item()),"test_mse:{}".format(mse.item()))
