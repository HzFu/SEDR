#
import os
import torch
import argparse
import warnings
import numpy as np
import pandas as pd
from src.graph_func import graph_construction
from src.utils_func import mk_dir, adata_preprocess, load_ST_file
import anndata
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

# ################ Parameter setting
parser = argparse.ArgumentParser()
parser.add_argument('--k', type=int, default=10, help='parameter k in spatial graph')
parser.add_argument('--knn_distanceType', type=str, default='euclidean',
                    help='graph distance type: euclidean/cosine/correlation')
parser.add_argument('--epochs', type=int, default=200, help='Number of epochs to train.')
parser.add_argument('--cell_feat_dim', type=int, default=300, help='Dim of PCA')
parser.add_argument('--feat_hidden1', type=int, default=100, help='Dim of DNN hidden 1-layer.')
parser.add_argument('--feat_hidden2', type=int, default=20, help='Dim of DNN hidden 2-layer.')
parser.add_argument('--gcn_hidden1', type=int, default=32, help='Dim of GCN hidden 1-layer.')
parser.add_argument('--gcn_hidden2', type=int, default=8, help='Dim of GCN hidden 2-layer.')
parser.add_argument('--p_drop', type=float, default=0.2, help='Dropout rate.')
parser.add_argument('--using_dec', type=bool, default=True, help='Using DEC loss.')
parser.add_argument('--using_mask', type=bool, default=False, help='Using mask for multi-dataset.')
parser.add_argument('--feat_w', type=float, default=10, help='Weight of DNN loss.')
parser.add_argument('--gcn_w', type=float, default=0.1, help='Weight of GCN loss.')
parser.add_argument('--dec_kl_w', type=float, default=10, help='Weight of DEC loss.')
parser.add_argument('--gcn_lr', type=float, default=0.01, help='Initial GNN learning rate.')
parser.add_argument('--gcn_decay', type=float, default=0.01, help='Initial decay rate.')
parser.add_argument('--dec_cluster_n', type=int, default=10, help='DEC cluster number.')
parser.add_argument('--dec_interval', type=int, default=20, help='DEC interval nnumber.')
parser.add_argument('--dec_tol', type=float, default=0.00, help='DEC tol.')
# ______________ Eval clustering Setting _________
parser.add_argument('--eval_resolution', type=int, default=1, help='Eval cluster number.')
parser.add_argument('--eval_graph_n', type=int, default=20, help='Eval graph kN tol.')
parser.add_argument('--eval_cluster_type', type=str, default='Louvain', help='Louvain/KMeans/SpectralClustering')

params = parser.parse_args()
params.device = device

# _________ File Path _________ #
# set DLPFC data path
data_root = '/media/hzfu/Data/BioMultiModal/data/spatial_data/DLPFC/'
# data_root = '/Users/hzfu/Documents/Project_code/DeepMSC_data/spatial_data/DLPFC/'

# all DLPFC folder list
proj_list = ['151507', '151508', '151509', '151510', '151669', '151670',
             '151671', '151672', '151673', '151674', '151675', '151676']
# set saving result path
save_root = './SEDR_result/DLPFC/'

ari_score_list = []
for proj_idx in range(len(proj_list)):
    proj_name = proj_list[proj_idx]
    print('===== Project ' + str(proj_idx+1) + ' : ' + proj_name)
    file_fold = os.path.join(data_root, proj_name)

    # _________ Load file _________ #
    adata_h5 = load_ST_file(file_fold=file_fold)
    adata_h5.var_names_make_unique()

    adata_X = adata_preprocess(adata_h5, min_cells=5, pca_n_comps=300)
    graph_dict = graph_construction(adata_h5.obsm['spatial'], adata_h5.shape[0], params)
    params.save_path = mk_dir(os.path.join(save_root, proj_name))
    params.cell_num = adata_h5.shape[0]
    print('==== Graph Construction Finished')

    # ----------- Node Feature training --------------
    sed_net = SEDR_Train(adata_X, graph_dict, params)
    if params.using_dec:
        sed_net.train_with_dec()
    else:
        sed_net.train_without_dec()
    sed_feat, _, _, _ = sed_net.process()

    np.savez(os.path.join(params.save_path, "SED_result.npz"), sed_feat=sed_feat, deep_Dim=params.feat_hidden2)

    # ################## result plot
    adata_sed = anndata.AnnData(sed_feat)
    adata_sed.uns['spatial'] = adata_h5.uns['spatial']
    adata_sed.obsm['spatial'] = adata_h5.obsm['spatial']

    sc.pp.neighbors(adata_sed, n_neighbors=params.eval_graph_n)
    sc.tl.umap(adata_sed)
    sc.tl.leiden(adata_sed, key_added="SEDR_leiden", resolution=params.eval_resolution)

    # sc.pl.umap(adata_sed, color=["SEDR_leiden"])
    # plt.savefig(os.path.join(params.save_path, "SEDR_umap_plot.jpg"), bbox_inches='tight')
    sc.pl.spatial(adata_sed, img_key="hires", color=['SEDR_leiden'])
    plt.savefig(os.path.join(params.save_path, "SEDR_leiden_plot.jpg"), bbox_inches='tight')

    df_result = pd.DataFrame(adata_sed.obs['SEDR_leiden'], columns=['SEDR_leiden'])
    df_result.to_csv(os.path.join(params.save_path, "SEDR_leiden_result.tsv"), sep='\t', index=False)

    # #################### evaluation
    # ---------- Load manually annotation ---------------
    file_label = os.path.join(data_root, proj_name, "metadata.tsv")
    df_label = pd.read_csv(file_label, sep='\t')
    fig_label = np.array(df_label['layer_guess'].to_list())
    # ---------- Load clustering result ---------------
    df_result = pd.read_csv(os.path.join(params.save_path, "SEDR_leiden_result.tsv"), sep='\t')
    result_label = np.array(df_result['SEDR_leiden'].to_list())

    ari_score = metrics.adjusted_rand_score(fig_label, result_label)
    print('===== Project: ' + proj_name + " ARI score: "+str(ari_score))
    ari_score_list.append(ari_score)
    plt.close('all')

df = pd.DataFrame({'name': proj_list, 'score': ari_score_list})
df.to_csv(save_root+'total_ARI_score.csv')

print('===== Project: AVG ARI score: ' + str(np.mean(ari_score_list)))

