#
import os
import torch
import argparse
import warnings
import numpy as np
import pandas as pd
from src.utils_func import mk_dir, load_ST_file
from src.clustering_func import eval_clustering
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
data_root = '/media/hzfu/Data/BioMultiModal/data/spatial_data/DLPFC/'
# data_root = '/Users/hzfu/Documents/Project_code/DeepMSC_data/spatial_data/DLPFC/'
proj_list = ['151507', '151508', '151509', '151510', '151669', '151670',
             '151671', '151672', '151673', '151674', '151675', '151676']

save_root = '../SEDR_tmp_result/DLPFC/'

ari_score_list = []

parser = argparse.ArgumentParser()
params = parser.parse_args()
params.eval_cluster_n = 7
params.eval_graph_n = 20
# cluster_type: Louvain, KMeans, SpectralClustering
params.eval_cluster_type = 'SpectralClustering'

for proj_idx in range(len(proj_list)):
    proj_name = proj_list[proj_idx]
    print('===== Project ' + str(proj_idx+1) + ' : ' + proj_name)
    file_fold = os.path.join(data_root, proj_name)

    save_fold = mk_dir(os.path.join(save_root, proj_name))
    # _________ Load file _________ #
    adata_h5 = load_ST_file(file_fold=file_fold)
    sed_result = np.load(os.path.join(save_fold, "SED_result.npz"))

    # ---------- Load manually annotation ---------------
    df_label = pd.read_csv(os.path.join(data_root, proj_name, "metadata.tsv"), sep='\t')
    gt_label = np.array(df_label['layer_guess'].to_list())
    adata_h5.obs['gt_label'] = gt_label

    eval_clustering(sed_result, save_fold, params, label=gt_label)

    df_result = pd.read_csv(os.path.join(save_fold, params.eval_cluster_type+"_clustering_result.tsv"), sep='\t')
    adata_h5.obs['sed_labels'] = pd.Categorical(df_result['sed_labels'])
    sed_labels = np.array(df_result['sed_labels'].to_list())

    ari_score = metrics.adjusted_rand_score(gt_label, df_result['sed_labels'])
    print('===== Project: ' + proj_name + " ARI score: " + str(ari_score))
    ari_score_list.append(ari_score)

    plt.rcParams["figure.figsize"] = (6, 6)
    sc.pl.spatial(adata_h5, img_key="hires", color=['gt_label', 'sed_labels'])
    plt.savefig(os.path.join(save_fold, params.eval_cluster_type + "_clustering.jpg"), dpi=200, bbox_inches='tight')
    plt.close('all')

df = pd.DataFrame({'name': proj_list, 'score': ari_score_list})
df.to_csv(save_root+'tmp.csv')

print('===== Project: AVG ARI score: ' + str(np.mean(ari_score_list)))
