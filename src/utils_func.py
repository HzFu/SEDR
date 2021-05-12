#
import os
import scanpy as sc
import pandas as pd


def mk_dir(input_path):
    if not os.path.exists(input_path):
        os.makedirs(input_path)
    return input_path


def adata_preprocess(i_adata, min_cells=3, pca_n_comps=200):
    sc.pp.filter_genes(i_adata, min_cells=min_cells)
    i_adata = sc.pp.normalize_total(
        i_adata, target_sum=1, exclude_highly_expressed=True, inplace=False)['X']
    i_adata = sc.pp.scale(i_adata)
    i_adata = sc.pp.pca(i_adata, n_comps=pca_n_comps)
    return i_adata


def load_ST_file(file_fold, count_file='filtered_feature_bc_matrix.h5', load_images=True, file_Adj=None):
    adata_h5 = sc.read_visium(file_fold, load_images=load_images, count_file=count_file)
    adata_h5.var_names_make_unique()

    if load_images is False:
        if file_Adj is None:
            file_Adj = os.path.join(file_fold, "spatial/tissue_positions_list.csv")

        positions = pd.read_csv(file_Adj, header=None)
        positions.columns = [
            'barcode',
            'in_tissue',
            'array_row',
            'array_col',
            'pxl_col_in_fullres',
            'pxl_row_in_fullres',
        ]
        positions.index = positions['barcode']
        adata_h5.obs = adata_h5.obs.join(positions, how="left")
        adata_h5.obsm['spatial'] = adata_h5.obs[['pxl_row_in_fullres', 'pxl_col_in_fullres']].to_numpy()
        adata_h5.obs.drop(columns=['barcode', 'pxl_row_in_fullres', 'pxl_col_in_fullres'], inplace=True)

    print('adata: (' + str(adata_h5.shape[0]) + ', ' + str(adata_h5.shape[1]) + ')')
    return adata_h5







