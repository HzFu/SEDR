# args = commandArgs(trailingOnly=TRUE)
# sample.name <- args[1]

library(Seurat)
library(SeuratData)
library(ggplot2)
library(patchwork)
library(dplyr)




options(bitmapType = 'cairo')

dir.root <- file.path('/Users/hzfu/Documents/Project_code/DeepMSC_data/spatial_data/DLPFC/')
dir.data <- file.path(dir.root, '151676/') 
dir.output <- dir.data
# if(!dir.exists(file.path(dir.output))){
#   dir.create(file.path(dir.output), recursive = TRUE)
# }

### load data
sp_data <- Load10X_Spatial(dir.data, filename = "filtered_feature_bc_matrix.h5")


### Data processing
# plot1 <- VlnPlot(sp_data, features = "nCount_Spatial", pt.size = 0.1) + NoLegend()
# plot2 <- SpatialFeaturePlot(sp_data, features = "nCount_Spatial") + theme(legend.position = "right")
# wrap_plots(plot1, plot2)
# ggsave(paste(dir.output, './Seurat.QC.png', sep=""), width = 10, height=5)

# sctransform
sp_data <- SCTransform(sp_data, assay = "Spatial", verbose = FALSE)


# write.table(sp_data@assays$SCT@scale.data, file.path(dir.output, './rna.scaled_data.tsv'), sep='\t', quote=FALSE)
write.table(sp_data@images$slice1@coordinates, file.path(dir.output, './spatial.coordinates.tsv'), sep='\t', quote=FALSE)


### Dimensionality reduction, clustering, and visualization
sp_data <- RunPCA(sp_data, assay = "SCT", verbose = FALSE)
sp_data <- FindNeighbors(sp_data, reduction = "pca", dims = 1:30)
sp_data <- FindClusters(sp_data, verbose = FALSE)
sp_data <- RunUMAP(sp_data, reduction = "pca", dims = 1:30)

# p1 <- DimPlot(sp_data, reduction = "umap", label = TRUE)
# p2 <- SpatialDimPlot(sp_data, label = TRUE, label.size = 3)
# p1 + p2
# 
# ggsave(paste(dir.output, './Seurat.cell_cluster.png', sep=""), width=10, height=5)


write.table(sp_data@meta.data, file=file.path(dir.output, 'Seurat_metadata.tsv'), sep='\t', quote=FALSE)
saveRDS(sp_data, file=file.path(dir.output, 'Seurat_final.rds'))
