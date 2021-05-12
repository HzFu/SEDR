#
import networkx as nx
import numpy as np
import torch
import scipy.sparse as sp
from scipy.spatial import distance


# edgeList to edgeDict
def edgeList2edgeDict(edgeList, nodesize):
    graphdict = {}
    tdict = {}
    for edge in edgeList:
        end1 = edge[0]
        end2 = edge[1]
        tdict[end1] = ""
        tdict[end2] = ""
        if end1 in graphdict:
            tmplist = graphdict[end1]
        else:
            tmplist = []
        tmplist.append(end2)
        graphdict[end1] = tmplist

        # check and get full matrix
    for i in range(nodesize):
        if i not in tdict:
            graphdict[i] = []

    return graphdict


def sparse_mx_to_torch_sparse_tensor(sparse_mx):
    """Convert a scipy sparse matrix to a torch sparse tensor."""
    sparse_mx = sparse_mx.tocoo().astype(np.float32)
    indices = torch.from_numpy(np.vstack((sparse_mx.row, sparse_mx.col)).astype(np.int64))
    values = torch.from_numpy(sparse_mx.data)
    shape = torch.Size(sparse_mx.shape)
    return torch.sparse.FloatTensor(indices, values, shape)


# ====== Graph preprocessing
def preprocess_graph(adj):
    adj = sp.coo_matrix(adj)
    adj_ = adj + sp.eye(adj.shape[0])
    rowsum = np.array(adj_.sum(1))
    degree_mat_inv_sqrt = sp.diags(np.power(rowsum, -0.5).flatten())
    adj_normalized = adj_.dot(degree_mat_inv_sqrt).transpose().dot(degree_mat_inv_sqrt).tocoo()
    return sparse_mx_to_torch_sparse_tensor(adj_normalized)


# ====== Graph construction
def graph_construction(adata_Adj, cell_N):
    graphdict = edgeList2edgeDict(adata_Adj, cell_N)
    adj_org = nx.adjacency_matrix(nx.from_dict_of_lists(graphdict))

    # Store original adjacency matrix (without diagonal entries) for later
    adj_m1 = adj_org
    adj_m1 = adj_m1 - sp.dia_matrix((adj_m1.diagonal()[np.newaxis, :], [0]), shape=adj_m1.shape)
    adj_m1.eliminate_zeros()

    # Some preprocessing
    adj_norm_m1 = preprocess_graph(adj_m1)
    adj_label_m1 = adj_m1 + sp.eye(adj_m1.shape[0])
    adj_label_m1 = torch.FloatTensor(adj_label_m1.toarray())
    norm_m1 = adj_m1.shape[0] * adj_m1.shape[0] / float((adj_m1.shape[0] * adj_m1.shape[0] - adj_m1.sum()) * 2)

    graph_dict = {
        "adj_org": adj_org,
        "adj_norm": adj_norm_m1,
        "adj_label": adj_label_m1,
        "norm_value": norm_m1
    }
    return graph_dict


def graph_computing(Adj_coo, cell_num, distanceType='euclidean', k=10):
    edgeList = []
    for node_idx in range(cell_num):
        tmp = Adj_coo[node_idx, :].reshape(1, -1)
        distMat = distance.cdist(tmp, Adj_coo, distanceType)
        res = distMat.argsort()[:k + 1]
        tmpdist = distMat[0, res[0][1:k + 1]]
        boundary = np.mean(tmpdist) + np.std(tmpdist)
        for j in np.arange(1, k + 1):
            # TODO: check, only exclude large outliners
            # if (distMat[0,res[0][j]]<=mean+std) and (distMat[0,res[0][j]]>=mean-std):
            if distMat[0, res[0][j]] <= boundary:
                weight = 1.0
            else:
                weight = 0.0
            edgeList.append((node_idx, res[0][j], weight))

    return edgeList


