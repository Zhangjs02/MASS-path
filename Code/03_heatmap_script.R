# 载入必要的库
library(readr)
library(dplyr)
library(ComplexHeatmap)
library(circlize)

# 读取数据
pseudotime_data <- read_csv("pseudotime_ordered_table_.csv")
luad_data <- read_csv("LUAD.csv")
pathway_types <- read_tsv("pathway_types.txt")
delete_pathways <- readLines("delete_pathway.txt")
delete_pathways <- delete_pathways[delete_pathways != ""]  # 移除空行

# 查看数据结构
print("Pseudotime data structure:")
print(head(pseudotime_data))
print(paste("Pseudotime data dimensions:", nrow(pseudotime_data), "x", ncol(pseudotime_data)))

print("LUAD data structure:")
print(colnames(luad_data)[1:10])  # 显示前10个列名
print(paste("LUAD data dimensions:", nrow(luad_data), "x", ncol(luad_data)))

print("Pathway types structure:")
print(head(pathway_types))
print(paste("Pathway types dimensions:", nrow(pathway_types), "x", ncol(pathway_types)))

print("Pathways to delete:")
print(delete_pathways)

# 将LUAD数据的第一列设为行名
luad_matrix <- as.data.frame(luad_data)
rownames(luad_matrix) <- luad_matrix[,1]
luad_matrix <- luad_matrix[,-1]  # 移除第一列（样本ID列）

# 转换为数值矩阵
luad_matrix <- as.matrix(luad_matrix)

# 获取pseudotime顺序
sample_order <- pseudotime_data$Sample_ID

# 找到共同的样本ID
common_samples <- intersect(sample_order, rownames(luad_matrix))
print(paste("Common samples found:", length(common_samples)))

# 按照pseudotime顺序重新排列LUAD数据
luad_ordered <- luad_matrix[common_samples, ]

# 删除指定的通路
pathways_to_keep <- setdiff(colnames(luad_ordered), delete_pathways)
luad_ordered <- luad_ordered[, pathways_to_keep]
print(paste("Removed", length(delete_pathways), "pathways. Remaining pathways:", ncol(luad_ordered)))

# 检查数据范围
print(paste("Data range: min =", min(luad_ordered, na.rm = TRUE), 
            "max =", max(luad_ordered, na.rm = TRUE)))

# 创建自定义颜色映射
col_fun <- colorRamp2(c(0, 0.5, 1, 1.5, max(luad_ordered, na.rm = TRUE)), 
                      c("blue", "blue", "white", "red", "red"))

# 按照pathway_types.txt中的顺序重新排列通路（排除已删除的通路）
available_pathways <- colnames(luad_ordered)
pathway_order <- pathway_types$Pathway_ID[pathway_types$Pathway_ID %in% available_pathways]

# 添加任何在数据中但不在pathway_types.txt中的通路
missing_pathways <- setdiff(available_pathways, pathway_order)
if(length(missing_pathways) > 0) {
  print(paste("Warning: Found", length(missing_pathways), "pathways not in pathway_types.txt"))
  pathway_order <- c(pathway_order, missing_pathways)
}

# 按照指定顺序重新排列通路
luad_ordered <- luad_ordered[, pathway_order]

# 创建通路类型的颜色映射
pathway_type_info <- pathway_types[pathway_types$Pathway_ID %in% pathway_order, ]
pathway_type_info <- pathway_type_info[match(pathway_order, pathway_type_info$Pathway_ID), ]

# 为缺失的通路添加"Unknown"类型
if(length(missing_pathways) > 0) {
  missing_df <- data.frame(
    Pathway_ID = missing_pathways,
    Pathway_Type = "Unknown"
  )
  pathway_type_info <- rbind(pathway_type_info, missing_df)
}

# 创建通路类型的颜色向量
unique_types <- unique(pathway_type_info$Pathway_Type)
type_colors <- rainbow(length(unique_types))
names(type_colors) <- unique_types

# 创建左侧行注释（通路类型信息）
left_annotation <- rowAnnotation(
  Pathway_Type = pathway_type_info$Pathway_Type,
  col = list(Pathway_Type = type_colors),
  width = unit(1, "cm")
)

# 创建右侧行注释（通路ID）
right_annotation <- rowAnnotation(
  Pathway_ID = anno_text(pathway_order, 
                        location = 0, 
                        just = "left",
                        gp = gpar(fontsize = 2.5)),
  width = unit(1.2, "cm")
)

# 创建列注释（伪时间信息）
pseudotime_subset <- pseudotime_data[pseudotime_data$Sample_ID %in% common_samples, ]
pseudotime_subset <- pseudotime_subset[match(common_samples, pseudotime_subset$Sample_ID), ]

# 为State和Cluster创建命名颜色向量
state_colors <- rainbow(length(unique(pseudotime_subset$State)))
names(state_colors) <- unique(pseudotime_subset$State)

cluster_colors <- rainbow(length(unique(pseudotime_subset$Cluster)))
names(cluster_colors) <- unique(pseudotime_subset$Cluster)

col_annotation <- HeatmapAnnotation(
  Pseudotime = pseudotime_subset$Pseudotime,
  State = pseudotime_subset$State,
  Cluster = pseudotime_subset$Cluster,
  col = list(
    Pseudotime = colorRamp2(c(min(pseudotime_subset$Pseudotime), 
                             max(pseudotime_subset$Pseudotime)), 
                           c("white", "darkblue")),
    State = state_colors,
    Cluster = cluster_colors
  )
)

# ========== 第一个热图：按pseudotime顺序排列 ==========
print("Generating first heatmap: samples ordered by pseudotime...")

# 创建热图
ht1 <- Heatmap(
  t(luad_ordered),  # 转置矩阵，使样本作为列
  name = "Expression",
  col = col_fun,
  top_annotation = col_annotation,
  left_annotation = left_annotation,
  right_annotation = right_annotation,
  show_column_names = FALSE,  # 不显示列名（样本ID）
  show_row_names = FALSE,     # 不显示行名（基因名）
  cluster_columns = FALSE,    # 不对列进行聚类（保持pseudotime顺序）
  cluster_rows = FALSE,       # 不对行进行聚类（保持pathway_types顺序）
  column_title = "Samples ordered by Pseudotime",
  row_title = "Pathways (ordered by type)",
  heatmap_legend_param = list(
    title = "Expression\nLevel",
    legend_direction = "vertical",
    legend_height = unit(4, "cm")
  )
)

# 保存第一个热图
png("heatmap_pseudotime_ordered.png", width = 13.5, height = 10, units = "in", res = 300)
draw(ht1)
dev.off()

pdf("heatmap_pseudotime_ordered.pdf", width = 13.5, height = 10)
draw(ht1)
dev.off()

print("First heatmap saved as 'heatmap_pseudotime_ordered.png' and 'heatmap_pseudotime_ordered.pdf'")

# ========== 第二个热图：按state分组，state内部按pseudotime排序 ==========
print("Generating second heatmap: samples ordered by state then pseudotime...")

# 按State分组，每个State内部按Pseudotime排序
pseudotime_state_ordered <- pseudotime_subset %>%
  arrange(State, Pseudotime)

# 获取新的样本顺序
state_ordered_samples <- pseudotime_state_ordered$Sample_ID

# 按照新顺序重新排列LUAD数据
luad_state_ordered <- luad_matrix[state_ordered_samples, ]
# 按照通路顺序重新排列
luad_state_ordered <- luad_state_ordered[, pathway_order]

# 创建新的列注释
col_annotation_state <- HeatmapAnnotation(
  Pseudotime = pseudotime_state_ordered$Pseudotime,
  State = pseudotime_state_ordered$State,
  Cluster = pseudotime_state_ordered$Cluster,
  col = list(
    Pseudotime = colorRamp2(c(min(pseudotime_state_ordered$Pseudotime), 
                             max(pseudotime_state_ordered$Pseudotime)), 
                           c("white", "darkblue")),
    State = state_colors,
    Cluster = cluster_colors
  )
)

# 创建第二个热图
ht2 <- Heatmap(
  t(luad_state_ordered),  # 转置矩阵，使样本作为列
  name = "Expression",
  col = col_fun,
  top_annotation = col_annotation_state,
  left_annotation = left_annotation,
  right_annotation = right_annotation,
  show_column_names = FALSE,  # 不显示列名（样本ID）
  show_row_names = FALSE,     # 不显示行名（基因名）
  cluster_columns = FALSE,    # 不对列进行聚类（保持state-pseudotime顺序）
  cluster_rows = FALSE,       # 不对行进行聚类（保持pathway_types顺序）
  column_title = "Samples ordered by State then Pseudotime",
  row_title = "Pathways (ordered by type)",
  heatmap_legend_param = list(
    title = "Expression\nLevel",
    legend_direction = "vertical",
    legend_height = unit(4, "cm")
  )
)

# 保存第二个热图
png("heatmap_state_pseudotime_ordered.png", width = 13.5, height = 10, units = "in", res = 300)
draw(ht2)
dev.off()

pdf("heatmap_state_pseudotime_ordered.pdf", width = 13.5, height = 10)
draw(ht2)
dev.off()

print("Second heatmap saved as 'heatmap_state_pseudotime_ordered.png' and 'heatmap_state_pseudotime_ordered.pdf'")

# 显示摘要信息
print("========== SUMMARY ==========")
print(paste("Final matrix dimensions:", nrow(luad_ordered), "samples x", ncol(luad_ordered), "pathways"))
print("Color mapping:")
print("  <= 0.5: Blue")
print("  = 1.0: White") 
print("  >= 1.5: Red")
print("")
print("Generated files:")
print("1. heatmap_pseudotime_ordered.png/pdf - Samples ordered by pseudotime")
print("2. heatmap_state_pseudotime_ordered.png/pdf - Samples ordered by state then pseudotime")

# 显示每个state的样本数量
state_summary <- pseudotime_state_ordered %>%
  group_by(State) %>%
  summarise(
    Sample_count = n(),
    Min_pseudotime = min(Pseudotime),
    Max_pseudotime = max(Pseudotime)
  )
print("State summary:")
print(state_summary)

# 显示通路类型分布
pathway_type_summary <- pathway_type_info %>%
  group_by(Pathway_Type) %>%
  summarise(Count = n()) %>%
  arrange(desc(Count))
print("Pathway type summary:")
print(pathway_type_summary) 