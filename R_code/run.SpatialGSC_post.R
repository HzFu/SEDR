setwd("~/Dropbox/Proj_code/SED_code/R_code")

library(Seurat)
library(SeuratData)
library(ggplot2)
library(patchwork)
library(dplyr)
library(colormap)
options(bitmapType = 'cairo')

prj_name <- 'Targeted_Visium_Human_ColorectalCancer_GeneSignature'

dir.output <- file.path('/Users/hzfu/Documents/Project_code/DeepMSC_data/spatial_data/astar/', prj_name)
dir.GSCoutput <- file.path('../tmp_result/', prj_name)

sp_data <- readRDS(file.path(dir.output, 'Seurat_final.rds'))
result <- read.table(file.path(dir.GSCoutput,  'clustering_result.tsv'), sep='\t', header=TRUE)

sp_data@meta.data$sed_labels <- as.character(result$sed_labels)   
sp_data@meta.data$deep_labels <- as.character(result$deep_labels)    

p2 <- SpatialPlot(sp_data, features = 'nFeature_Spatial', label.box = FALSE, alpha = c(0, 0))
p3 <- SpatialDimPlot(sp_data, label = TRUE, cols = colormap(colormaps$jet,nshades=length(table(sp_data@meta.data$seurat_clusters))), label.size = 2) + ggtitle('Seurat')
p4 <- SpatialDimPlot(sp_data, label = TRUE, cols = colormap(colormaps$jet,nshades=length(table(sp_data@meta.data$sed_labels))), group.by = 'sed_labels', label.size=2) + ggtitle('SED') 
p5 <- SpatialDimPlot(sp_data, label = TRUE, cols = colormap(colormaps$jet,nshades=length(table(sp_data@meta.data$deep_labels))), group.by = 'deep_labels', label.size=2) + ggtitle('Deep') 
p3+p4+p2 
ggsave(file.path(dir.GSCoutput, 'comparison.spatial.jpg'), width=9, height=3)

 
