#
import os
import torch
import argparse
import warnings
import numpy as np
import pandas as pd
from src.graph_func import graph_construction, graph_computing
from src.utils_func import mk_dir, adata_preprocess, load_ST_file
from src.clustering_func import eval_clustering
from src.SEDR_train import SEDR_Train
from sklearn import metrics
import matplotlib.pyplot as plt
import scanpy as sc

warnings.filterwarnings('ignore')
torch.cuda.cudnn_enabled = False
np.random.seed(0)
torch.manual_seed(0)
torch.cuda.manual_seed(0)
device = 'cuda:0' if torch.cuda.is_available() else 'cpu'
print('===== Using device: ' + device)

# _________ File Path _________ #
# set DLPFC data path
data_root = '/media/hzfu/Data/BioMultiModal/data/spatial_data/DLPFC/'
# data_root = '/Users/hzfu/Documents/Project_code/DeepMSC_data/spatial_data/DLPFC/'
proj_list = ['151507', '151508', '151509', '151510', '151669', '151670',
             '151671', '151672', '151673', '151674', '151675', '151676']
             
# set saving result path
save_root = '../SEDR_tmp_result/DLPFC/'

# ============= Hyper-parameter Setting ============= #
parser = argparse.ArgumentParser()
parser.add_argument('--k', type=int, default=10,
                    help='parameter k in KNN graph (default: 10)')
parser.add_argument('--knn_distanceType', type=str, default='euclidean',
                    help='KNN graph distance type: euclidean/cosine/correlation (default: euclidean)')
parser.add_argument('--epochs', type=int, default=200,
                    help='Number of epochs to train.')
parser.add_argument('--gcn_lr', type=float, default=0.01,
                    help='Initial learning rate.')
parser.add_argument('--gcn_decay', type=float, default=0.01,
                    help='Initial decay rate.')

params = parser.parse_args()
params.device = device
# ______________ Network Setting _________
params.feat_hidden1 = 100
params.feat_hidden2 = 20
params.gcn_hidden1 = 32
params.gcn_hidden2 = 16
params.p_drop = 0.2
params.feat_w = 10

params.using_dec = True
params.dec_kl_w = 0.1
params.dec_cluster_n = 10
params.dec_interval = 20
params.dec_tol = 0.01

# ______________ Eval clustering Setting _________
params.eval_cluster_n = 10
params.eval_graph_n = 20
# cluster_type: Louvain, KMeans, SpectralClustering
params.eval_cluster_type = 'Louvain'

ari_score_list = []

for proj_idx in range(len(proj_list)):
    proj_name = proj_list[proj_idx]
    print('===== Project ' + str(proj_idx+1) + ' : ' + proj_name)
    file_fold = os.path.join(data_root, proj_name)

    # _________ Load file _________ #
    adata_h5 = load_ST_file(file_fold=file_fold)
    adata_X = adata_preprocess(adata_h5, min_cells=5, pca_n_comps=300) 
    Adj_coo = adata_h5.obsm['spatial']

    # ============= Project Parameter Setting ============= #
    params.cell_num = adata_X.shape[0]
    params.cell_feat_dim = adata_X.shape[1]
    params.save_path = mk_dir(os.path.join(save_root, proj_name))

    # _________ Graph Construction _________ #
    adj_graph = graph_computing(Adj_coo, params.cell_num, params.knn_distanceType, params.k)
    graph_dict = graph_construction(adj_graph, params.cell_num)
    print('==== Graph Construction Finished')

    # ----------- Node Feature training --------------
    sed_net = SEDR_Train(adata_X, graph_dict, params)
    if params.using_dec:
        sed_net.train_with_dec()
    else:
        sed_net.train_without_dec()
    sed_feat, _, deep_feat, gnn_feat = sed_net.process()

    np.savez(os.path.join(params.save_path, "SED_result.npz"),
             sed_feat=sed_feat, deep_feat=deep_feat, gnn_feat=gnn_feat)

    # ---------- Load manually annotation ---------------
    file_label = os.path.join(data_root, proj_name, "metadata.tsv")
    df_label = pd.read_csv(file_label, sep='\t')
    fig_label = np.array(df_label['layer_guess'].to_list())

    feat_dict = {
        'deep_feat': deep_feat,
        'sed_feat': sed_feat,
        'gnn_feat': gnn_feat,
    }

    eval_clustering(feat_dict, params.save_path, params, label=fig_label)

    df_result = pd.read_csv(os.path.join(params.save_path, params.eval_cluster_type+"_clustering_result.tsv"), sep='\t')
    result_label = np.array(df_result['sed_labels'].to_list())

    ari_score = metrics.adjusted_rand_score(fig_label, result_label)
    print('===== Project: ' + proj_name + " ARI score: "+str(ari_score))
    ari_score_list.append(ari_score) 

    # # ---------- Load manually annotation ---------------
    # df_label = pd.read_csv(os.path.join(data_root, proj_name, "metadata.tsv"), sep='\t')
    # gt_label = np.array(df_label['layer_guess'].to_list())
    # adata_h5.obs['gt_label'] = gt_label

    # df_result = pd.read_csv(os.path.join(params.save_path, params.eval_cluster_type+"_clustering_result.tsv"), sep='\t')
    # adata_h5.obs['sed_labels'] = pd.Categorical(df_result['sed_labels'])
    # sed_labels = np.array(df_result['sed_labels'].to_list())

    # plt.rcParams["figure.figsize"] = (6, 6)
    # sc.pl.spatial(adata_h5, img_key="hires", color=['gt_label', 'sed_labels'])
    # plt.savefig(os.path.join(params.save_path, params.eval_cluster_type + "_clustering.jpg"), dpi=150, bbox_inches='tight')
    # plt.close('all')


df = pd.DataFrame({'name': proj_list, 'score': ari_score_list})
df.to_csv(save_root+'tmp.csv')

print('===== Project: AVG ARI score: ' + str(np.mean(ari_score_list)))

