setwd("~/Dropbox/Proj_code/SED_code/R_code")

library(Seurat)
library(SeuratData)
library(ggplot2)
library(patchwork)
library(dplyr)
library(colormap)
library(mclust) 
library(ggmap)

options(bitmapType = 'cairo')

prj_name <- '151673'

dir.output <- file.path('/Users/hzfu/Documents/Project_code/DeepMSC_data/spatial_data/DLPFC/', prj_name)
dir.GSCoutput <- file.path('../tmp_result/DLPFC/', prj_name)

sp_data <- readRDS(file.path(dir.output, 'Seurat_final.rds'))
result <- read.table(file.path(dir.GSCoutput,  'clustering_result.tsv'), sep='\t', header=TRUE) 

sp_data@meta.data$sed_labels <- as.character(result$sed_labels)   
sp_data@meta.data$deep_labels <- as.character(result$deep_labels)    
sp_data@meta.data$layer_guess <- as.character(result$layer_guess)    
sp_data@meta.data$gnn_labels <- as.character(result$gnn_labels)   

p2 <- SpatialPlot(sp_data, features = 'nFeature_Spatial', label.box = FALSE, alpha = c(0, 0)) 
p3 <- SpatialDimPlot(sp_data, label = TRUE, cols = colormap(colormaps$jet,nshades=length(table(sp_data@meta.data$seurat_clusters))), label.size = 2) + ggtitle('Seurat')
p4 <- SpatialDimPlot(sp_data, label = TRUE, cols = colormap(colormaps$jet,nshades=length(table(sp_data@meta.data$sed_labels))), group.by = 'sed_labels', label.size=2) + ggtitle('SED') 
p5 <- SpatialDimPlot(sp_data, label = TRUE, cols = colormap(colormaps$jet,nshades=length(table(sp_data@meta.data$layer_guess))), group.by = 'layer_guess', label.size=2) + ggtitle('layer_guess')

p6 <- SpatialDimPlot(sp_data, label = TRUE, cols = colormap(colormaps$jet,nshades=length(table(sp_data@meta.data$deep_labels))), group.by = 'deep_labels', label.size=2) + ggtitle('DNN') 
p7 <- SpatialDimPlot(sp_data, label = TRUE, cols = colormap(colormaps$jet,nshades=length(table(sp_data@meta.data$gnn_labels))), group.by = 'gnn_labels', label.size=2) + ggtitle('GNN')
(p3+p4)/(p5+p2) 
ggsave(file.path(dir.GSCoutput, '/comparison.spatial.jpg'), width=6, height=6)

adjustedRandIndex(sp_data@meta.data$layer_guess, sp_data@meta.data$sed_labels)
