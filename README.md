# SEDR: Unsupervised Spatial Embedded Deep Representation of Spatial Transcriptomics

## About:


__SEDR__ (spatial embedded deep representation)  learns a low-dimensional latent representation of gene expression embedded with spatial information for spatial transcriptomics analysis.  SEDR method consists of two main components, a deep autoencoder network for learning a gene representation, and a variational graph autoencoder network for embedding the spatial information.  SEDR has been applied on the 10x Genomics Visium spatial transcriptomics dataset as well as Stereo-seq dataset, and demonstrated its ability to achieve better representation for various follow-up analysis tasks including clustering, visualization, trajectory inference and batch effect correction.

![](figure/frame.jpg)




## Installation:
 
SEDR is implemented in the pytorch framework (tested on Ubuntu 18.04, MacOS catalina with Python 3.8). Please run SEDR on CUDA if possible. The following packages are required to be able to run everything in this repository (included are the versions we used):

```bash
anndata==0.7.6 
bokeh==2.3.1
h5py==2.10.0
imageio==2.9.0   
leidenalg==0.8.4 
matplotlib==3.3.4 
numpy==1.20.2
python-igraph==0.9.1
python-louvain==0.15 
scanpy==1.7.2
scikit-image==0.18.1
scikit-learn==0.24.1
scikit-network==0.23.1
scipy==1.6.3
seaborn==0.11.1
torch==1.8.1
torch-geometric==1.7.0 
torchvision==0.9.1
tqdm==4.59.0
umap-learn==0.5.1 
```

(**Recommended**) Use python virutal environment with conda（<https://anaconda.org/>）
```shell
conda create -n SEDR_Env python=3.8 pip
conda activate SEDR_Env
pip install -r requirements.txt
```


## Quick Start:

SDER utilizes anndata (based on Scanpy) as input. We provide three examples:

1. `run_SEDR_10x_Genomics_Visium.py`: The demo code for *'Visium 10x Genomics'* data.
   
   ![](figure/Visium_result.jpg)

2. `run_SEDR_DLPFC_data.py`: The demo code for *'LIBD human dorsolateral prefrontal cortex (DLPFC, <http://research.libd.org/spatialLIBD/>)'* data .
   
   ![](figure/DLPFC_result.jpg)

3. `run_UBC_DLPFC_data.py`: The demo code for unsupervised batch correction for multiple *DLPFC* data. 
   
   ![](figure/UBC_SEDR_plot.jpg)


## Citation:

This repository contains the source code for the paper:

Huazhu Fu, Hang Xu, Kelvin Chong, Mengwei Li, Hong Kai Lee, Kok Siong Ang, Ao Chen, Ling Shao, Longqi Liu, and Jinmiao Chen, **"Unsupervised Spatial Embedded Deep Representation of Spatial Transcriptomics"**,  *bioRxiv* (2021). <https://>

**BibTeX**

```latex
@article{Fu2021SEDR,
	author = {Huazhu Fu, Hang Xu, Kelvin Chong, Mengwei Li, Hong Kai Lee, Kok Siong Ang, Ao Chen, Ling Shao, Longqi Liu, and Jinmiao Chen},
	title = {Unsupervised Spatial Embedded Deep Representation of Spatial Transcriptomics},
	year = {2021}, 
	journal = {bioRxiv}
}
```