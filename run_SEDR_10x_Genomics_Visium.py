#
import os
import torch
import argparse
import warnings
import numpy as np
import anndata
import scanpy as sc
import matplotlib.pyplot as plt
import pandas as pd
from src.graph_func import graph_construction
from src.utils_func import mk_dir, adata_preprocess, load_visium_sge
from src.SEDR_train import SEDR_Train

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
parser.add_argument('--gcn_hidden2', type=int, default=4, help='Dim of GCN hidden 2-layer.')
parser.add_argument('--p_drop', type=float, default=0.2, help='Dropout rate.')
parser.add_argument('--using_mask', type=bool, default=False, help='Using mask for multi-dataset.')
parser.add_argument('--using_dec', type=bool, default=True, help='Using DEC loss.')
parser.add_argument('--feat_w', type=float, default=10, help='Weight of DNN loss.')
parser.add_argument('--gcn_w', type=float, default=0.1, help='Weight of GCN loss.')
parser.add_argument('--dec_kl_w', type=float, default=10, help='Weight of DEC loss.')
parser.add_argument('--gcn_lr', type=float, default=0.01, help='Initial GNN learning rate.')
parser.add_argument('--gcn_decay', type=float, default=0.01, help='Initial decay rate.')
parser.add_argument('--dec_cluster_n', type=int, default=8, help='DEC cluster number.')
parser.add_argument('--dec_interval', type=int, default=20, help='DEC interval nnumber.')
parser.add_argument('--dec_tol', type=float, default=0.00, help='DEC tol.')
# ______________ Eval clustering Setting _________
parser.add_argument('--eval_resolution', type=int, default=1, help='Eval cluster number.')
parser.add_argument('--eval_graph_n', type=int, default=20, help='Eval graph kN tol.')
parser.add_argument('--eval_cluster_type', type=str, default='Louvain', help='Louvain/KMeans/SpectralClustering')

params = parser.parse_args()
params.device = device

# _________ Root Path _________ #
# Visium Spatial Gene Expression data from 10x Genomics.
# Database: https://support.10xgenomics.com/spatial-gene-expression/datasets
# sample_id_list = [‘V1_Breast_Cancer_Block_A_Section_1’, ‘V1_Breast_Cancer_Block_A_Section_2’,
# ‘V1_Human_Heart’, ‘V1_Human_Lymph_Node’, ‘V1_Mouse_Kidney’, ‘V1_Adult_Mouse_Brain’,
# ‘V1_Mouse_Brain_Sagittal_Posterior’, ‘V1_Mouse_Brain_Sagittal_Posterior_Section_2’,
# ‘V1_Mouse_Brain_Sagittal_Anterior’, ‘V1_Mouse_Brain_Sagittal_Anterior_Section_2’,
# ‘V1_Human_Brain_Section_1’, ‘V1_Human_Brain_Section_2’,
# ‘V1_Adult_Mouse_Brain_Coronal_Section_1’,
# ‘V1_Adult_Mouse_Brain_Coronal_Section_2’,
# ‘Targeted_Visium_Human_Cerebellum_Neuroscience’, ‘Parent_Visium_Human_Cerebellum’,
# ‘Targeted_Visium_Human_SpinalCord_Neuroscience’, ‘Parent_Visium_Human_SpinalCord’,
# ‘Targeted_Visium_Human_Glioblastoma_Pan_Cancer’, ‘Parent_Visium_Human_Glioblastoma’,
# ‘Targeted_Visium_Human_BreastCancer_Immunology’, ‘Parent_Visium_Human_BreastCancer’,
# ‘Targeted_Visium_Human_OvarianCancer_Pan_Cancer’,
# ‘Targeted_Visium_Human_OvarianCancer_Immunology’, ‘Parent_Visium_Human_OvarianCancer’,
# ‘Targeted_Visium_Human_ColorectalCancer_GeneSignature’, ‘Parent_Visium_Human_ColorectalCancer]

# Data download folder
data_root = '/media/hzfu/Data/BioMultiModal/data/spatial_data/10x_Genomics_Visium/'
# data_root = '/Users/hzfu/Documents/Project_code/DeepMSC_data/spatial_data/10x_Genomics_Visium/'
proj_name = 'V1_Breast_Cancer_Block_A_Section_1'
save_fold = os.path.join('./SEDR_result/10x_Genomics_Visium/', proj_name)

# _________ Load file _________ #
adata_h5 = load_visium_sge(sample_id=proj_name, save_path=data_root)
adata_X = adata_preprocess(adata_h5, min_cells=5, pca_n_comps=300)
graph_dict = graph_construction(adata_h5.obsm['spatial'], adata_h5.shape[0], params)
params.cell_num = adata_h5.shape[0]
params.save_path = mk_dir(save_fold)
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

sc.pl.spatial(adata_sed, img_key="hires", color=['SEDR_leiden'])
plt.savefig(os.path.join(params.save_path, "SEDR_leiden_plot.pdf"), bbox_inches='tight', dpi=200)

df_result = pd.DataFrame(adata_sed.obs['SEDR_leiden'], columns=['SEDR_leiden'])
df_result.to_csv(os.path.join(params.save_path, "SEDR_leiden_n_"""+str(params.eval_resolution)+"_result.tsv"),
                 sep='\t', index=False)

