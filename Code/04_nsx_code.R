# sudo rstudio-server start
.libPaths(c("/home/hth/R/x86_64-pc-linux-gnu-library/4.4", .libPaths()))
options(stringsAsFactors = FALSE)
library(monocle)
library(dplyr)
library(data.table)
library(tibble)

# 设置工作目录为Data目录（相对于Code目录）
setwd("../Data")

cat("正在读取数据...\n")
LUAD_data = fread("LUAD_intersection_genes.csv", header = TRUE)

# 读取临床信息文件
cat("正在读取临床信息...\n")
LUAD_cli = fread("LUAD_clinical.csv")

# 提取tissue_type信息
tissue_type_data = LUAD_cli[, .(sample, `tissue_type.samples`)]
names(tissue_type_data) = c("Sample_ID", "tissue_type")

# 转置数据并处理
LUAD = as.data.frame(t(column_to_rownames(LUAD_data, "Sample_ID")))
stage_simple = LUAD["Stage_Simple", , drop = F]
cluster = LUAD["Cluster", , drop = F]
stage_simple[is.na(stage_simple)] <- "unknown"

# 为每个样本匹配tissue_type信息
sample_names = colnames(LUAD)  # 转置后的列名就是原始的Sample_ID
tissue_type_matched = character(length(sample_names))

# 创建tissue_type的查找向量，提高匹配效率
tissue_lookup = setNames(tissue_type_data$tissue_type, tissue_type_data$Sample_ID)

for(i in 1:length(sample_names)) {
  if(sample_names[i] %in% names(tissue_lookup)) {
    tissue_type_matched[i] = tissue_lookup[sample_names[i]]
  } else {
    tissue_type_matched[i] = "Unknown"
  }
}

cat("正在创建CellDataSet对象...\n")
LUAD <- LUAD[rownames(LUAD) != "Stage_Simple", ]
LUAD <- LUAD[rownames(LUAD) != "Cluster", ]
genes = rownames(LUAD)
LUAD = as.data.frame(lapply(LUAD, function(col) {
  if (is.logical(col)) {num_col <- as.numeric(col)} else {num_col <- as.numeric(as.character(col))}
  num_col[is.na(num_col)] <- 0
  num_col
}))
rownames(LUAD) = genes
LUAD = as(as.matrix(LUAD), 'sparseMatrix')

cat("正在进行数据预处理...\n")
cds <- newCellDataSet(
  cellData = LUAD, 
  phenoData = new('AnnotatedDataFrame', data = data.frame(
    orig.ident = rep("RNA", ncol(LUAD)),
    nCount_RNA = colSums(LUAD, na.rm = TRUE),
    seurat_annotations = as.character(stage_simple[1, ]),
    cluster = as.character(cluster[1, ]),
    tissue_type = tissue_type_matched,
    stringsAsFactors = FALSE
  )),
  featureData = new('AnnotatedDataFrame', data = data.frame(
    gene_short_name = row.names(LUAD),
    row.names = row.names(LUAD)
  )),
  lowerDetectionLimit = 0.1,
  expressionFamily = negbinomial.size()
)

cat("正在进行降维...\n")
cds <- estimateSizeFactors(cds)
cds <- estimateDispersions(cds)
cds <- setOrderingFilter(cds, genes)
cds <- reduceDimension(cds, method = 'DDRTree', max_components = 2)

cat("正在排序细胞...\n")
cds <- orderCells(cds)

cat("正在生成图表...\n")
ggsave("trajectory_by_state.png", width=5, height=5, plot = plot_cell_trajectory(cds, color_by="State"))
ggsave("trajectory_by_state_faceted.png",
       width=3*ceiling(length(base::unique(pData(cds)$State))/ceiling(sqrt(length(base::unique(pData(cds)$State))))), height=3*ceiling(sqrt(length(base::unique(pData(cds)$State)))),
       plot = plot_cell_trajectory(cds, color_by="State") + facet_wrap(~State, nrow=ceiling(sqrt(length(base::unique(pData(cds)$State)))), axes="all"))
ggsave("trajectory_by_pseudotime.png", width=5, height=5, plot = plot_cell_trajectory(cds, color_by="Pseudotime"))
ggsave("trajectory_by_stage.png", width=5, height=5, plot = plot_cell_trajectory(cds, color_by="seurat_annotations"))
ggsave("trajectory_by_cluster.png", width=5, height=5, plot = plot_cell_trajectory(cds, color_by="cluster"))

# 生成基于tissue_type的轨迹图
cat("正在生成tissue_type轨迹图...\n")
ggsave("trajectory_by_tissue_type.png", width=5, height=5, plot = plot_cell_trajectory(cds, color_by="tissue_type"))

cat("正在生成pseudotime结果文件...\n")
# 提取pseudotime数据
pseudotime_data <- pData(cds)

# 创建结果数据框，包含tissue_type信息
result_df <- data.frame(
  Sample_ID = gsub("\\.", "-", rownames(pseudotime_data)),  # 将点号替换为横线
  Pseudotime = pseudotime_data$Pseudotime,
  State = pseudotime_data$State,
  Cluster = pseudotime_data$cluster,
  Tissue_Type = pseudotime_data$tissue_type,
  stringsAsFactors = FALSE
)

# 按照pseudotime递增排序
result_df <- result_df[order(result_df$Pseudotime), ]

# 保存为CSV文件
write.csv(result_df, "pseudotime_results.csv", row.names = FALSE)

cat("正在保存CellDataSet对象...\n")
# 保存完整的CellDataSet对象
saveRDS(cds, "monocle_celldataset.rds")

# 也可以保存为RData格式
save(cds, file = "monocle_celldataset.RData")

cat("分析完成！生成的图片文件：\n")
cat("- trajectory_by_state.png\n")
cat("- trajectory_by_pseudotime.png\n")
cat("- trajectory_by_stage.png\n")
cat("- trajectory_by_cluster.png\n")
cat("- trajectory_by_tissue_type.png (新增：基于Tumor/Normal的轨迹图)\n")
cat("- trajectory_by_state_faceted.png\n")
cat("生成的数据文件：\n")
cat("- pseudotime_results.csv (按pseudotime递增排序，包含tissue_type信息)\n")
cat("- monocle_celldataset.rds (完整的CellDataSet对象)\n")
cat("- monocle_celldataset.RData (CellDataSet对象的RData格式)\n")
cat("\n使用方法：\n")
cat("- 加载RDS: cds <- readRDS('monocle_celldataset.rds')\n")
cat("- 加载RData: load('monocle_celldataset.RData')\n")

# 输出tissue_type统计信息
cat("\n样本组织类型统计：\n")
tissue_type_stats = table(pseudotime_data$tissue_type)
print(tissue_type_stats)
