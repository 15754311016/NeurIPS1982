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
import scipy.io as sio
import scipy.sparse as sp
import torchvision.transforms as transforms
import numpy as np
import torch.optim as optim
Tensor = torch.cuda.FloatTensor
LongTensor = torch.cuda.LongTensor
loss = nn.MSELoss()
np.random.seed(43)
torch.manual_seed(31)
def load_syn(path):
    data = np.loadtxt(open(path, "rb"), delimiter=",")
    X = data[:, 3:]
    #X = preprocessing.scale(X)
    t = data[:, 0:1].reshape(1, -1)
    Y0 = data[:, 1:2].reshape(1, -1)
    Y1 = data[:, 2:3].reshape(1, -1)
    Y1 = Tensor(np.squeeze(Y1))
    Y0 = Tensor(np.squeeze(Y0))
    t = LongTensor(np.squeeze(t))
    X = Tensor(X)
    return X, t, Y1, Y0

def split_L_U(X, T, Y1, Y0, train_ratio=0.8, test_ratio=0.2):
    N = X.shape[0]
    idx = np.random.permutation(N)
    n_train = int(N * train_ratio)
    n_test = int(N * test_ratio)
    n_test = int(N * test_ratio)
    idx_train, idx_test = idx[:n_train], idx[n_train:n_train + n_test]
    X_train = X[idx_train]
    T_train = T[idx_train]
    Y1_train, Y0_train = Y1[idx_train], Y0[idx_train]
    X_test = X[idx_test]
    T_test = T[idx_test]
    Y1_test, Y0_test = Y1[idx_test], Y0[idx_test]
    # X_test = X[idx_test]
    # T_test = T[idx_test]
    # Y1_test, Y0_test = Y1[idx_test], Y0[idx_test]
    return X_train, T_train, Y1_train, Y0_train, X_test, T_test, Y1_test, Y0_test

def load_semidata(path, name='BlogCatalog',dim="200100", exp_id='0', original_X=False):
    print(path + name  + dim+'/' + name + exp_id + '.mat')
    data = sio.loadmat(path + name + dim + '/' + name + exp_id + '.mat')
    # A = data['Network']  # csr matrix

    # try:
    # 	A = np.array(A.todense())
    # except:
    # 	pass
    # print("original X",original_X)
    if not original_X:
        X = data['X_100_post']
    # else:
        # X = data['Attributes']
    Y1 = data['Y1']
    Y0 = data['Y0']
    T = data['T']
    # X = normalize(X)
    X = X.todense()
    X = Tensor(X)

    Y1 = Tensor(np.squeeze(Y1))
    Y0 = Tensor(np.squeeze(Y0))
    T = LongTensor(np.squeeze(T))
    #print(type(X),X.shape)
    # print(type(A),A[1,0])
    # print(type(T),T.shape)
    # print(type(Y1),Y1.shape)
    # print(data['Attributes'].shape)
    # print(s)

    #print("A:",A.todense())
    return X, T, Y1, Y0
np.random.normal
def load_mimicdata():
    # print(path + name  + dim+'/' + name + exp_id + '.mat')
    # data = sio.loadmat('./data/mimic_vaso_50.mat')
    data = sio.loadmat('./data/mimic_vaso_t0.mat')
    # data = sio.loadmat('./data/mimic_vent_50.mat')
    # A = data['Network']  # csr matrix

    # try:
    # 	A = np.array(A.todense())
    # except:
    # 	pass
    # print("original X",original_X)
    # if not original_X:
    X = data['X']
    # else:
        # X = data['Attributes']
    # Y1 = data['Y1']
    Y = data['Y']
    T = data['T']
    # X = normalize(X)
    # X = X.todense()
    X = Tensor(X)

    # Y1 = Tensor(np.squeeze(Y1))
    Y = Tensor(np.squeeze(Y))
    T = LongTensor(np.squeeze(T))
    #print(type(X),X.shape)
    # print(type(A),A[1,0])
    # print(type(T),T.shape)
    # print(type(Y1),Y1.shape)
    # print(data['Attributes'].shape)
    # print(s)

    #print("A:",A.todense())
    return X, T, Y

def normalize(mx):
    """Row-normalize sparse matrix"""
    rowsum = np.array(mx.sum(1))
    r_inv = np.power(rowsum, -1).flatten()
    r_inv[np.isinf(r_inv)] = 0.
    r_mat_inv = sp.diags(r_inv)
    mx = r_mat_inv.dot(mx)
    return mx

def pdist(sample_1, sample_2, norm=2, eps=1e-5):
    """Compute the matrix of all squared pairwise distances.
    Arguments
    ---------
    sample_1 : torch.Tensor or Variable
        The first sample, should be of shape ``(n_1, d)``.
    sample_2 : torch.Tensor or Variable
        The second sample, should be of shape ``(n_2, d)``.
    norm : float
        The l_p norm to be used.
    Returns
    -------
    torch.Tensor or Variable
        Matrix of shape (n_1, n_2). The [i, j]-th entry is equal to
        ``|| sample_1[i, :] - sample_2[j, :] ||_p``."""
    norm = float(norm)
    n_1, n_2 = sample_1.size(0), sample_2.size(0)
    if norm == 2.:
        norms_1 = torch.sum(sample_1 ** 2, dim=1, keepdim=True)
        norms_2 = torch.sum(sample_2 ** 2, dim=1, keepdim=True)
        norms = (norms_1.expand(n_1, n_2) +
                 norms_2.transpose(0, 1).expand(n_1, n_2))
        distances_squared = norms - 2 * sample_1.mm(sample_2.t())
        return torch.sqrt(eps + torch.abs(distances_squared))
    else:
        dim = sample_1.size(1)
        expanded_1 = sample_1.unsqueeze(1).expand(n_1, n_2, dim)
        expanded_2 = sample_2.unsqueeze(0).expand(n_1, n_2, dim)
        differences = torch.abs(expanded_1 - expanded_2) ** norm
        inner = torch.sum(differences, dim=2, keepdim=False)
        return (eps + inner) ** (1. / norm)

def wasserstein(x, y, p=0.5, lam=10, its=10, sq=False, backpropT=False, cuda=False):
    """return W dist between x and y"""
    '''distance matrix M'''
    nx = x.shape[0]
    ny = y.shape[0]
    x = x.squeeze()
    y = y.squeeze()
    #    pdist = torch.nn.PairwiseDistance(p=2)

    M = pdist(x, y)  # distance_matrix(x,y,p=2)

    '''estimate lambda and delta'''
    M_mean = torch.mean(M)
    M_drop = F.dropout(M, 10.0 / (nx * ny))
    delta = torch.max(M_drop).detach().cpu()
    eff_lam = (lam / M_mean).detach().cpu()

    '''compute new distance matrix'''
    Mt = M
    row = delta * torch.ones(M[0:1, :].shape)
    col = torch.cat([delta * torch.ones(M[:, 0:1].shape), torch.zeros((1, 1))], 0)
    if cuda:
        row = row.cuda()
        col = col.cuda()
    Mt = torch.cat([M, row], 0)
    Mt = torch.cat([Mt, col], 1)

    '''compute marginal'''
    a = torch.cat([p * torch.ones((nx, 1)) / nx, (1 - p) * torch.ones((1, 1))], 0)
    b = torch.cat([(1 - p) * torch.ones((ny, 1)) / ny, p * torch.ones((1, 1))], 0)

    '''compute kernel'''
    Mlam = eff_lam * Mt
    temp_term = torch.ones(1) * 1e-6
    if cuda:
        temp_term = temp_term.cuda()
        a = a.cuda()
        b = b.cuda()
    K = torch.exp(-Mlam) + temp_term
    U = K * Mt
    ainvK = K / a

    u = a

    for i in range(its):
        u = 1.0 / (ainvK.matmul(b / torch.t(torch.t(u).matmul(K))))
        if cuda:
            u = u.cuda()
    v = b / (torch.t(torch.t(u).matmul(K)))
    if cuda:
        v = v.cuda()

    upper_t = u * (torch.t(v) * K).detach()

    E = upper_t * Mt
    D = 2 * torch.sum(E)

    if cuda:
        D = D.cuda()

    return D, Mlam