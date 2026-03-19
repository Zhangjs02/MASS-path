# 基于脚本位置计算路径
args <- commandArgs(trailingOnly = FALSE)
script_path <- sub("--file=", "", args[grep("--file=", args)])
if (length(script_path) == 0) script_path <- "."
SCRIPT_DIR <- dirname(normalizePath(script_path))
PROJECT_DIR <- dirname(SCRIPT_DIR)
DATA_03_DIR <- file.path(PROJECT_DIR, "data", "03")

# 载入必要的库
library(readr)
library(dplyr)
library(ComplexHeatmap)
library(circlize)
## 使用Arial字体（可选，showtext 不可用时自动跳过）
HAS_SHOWTEXT <- suppressWarnings(suppressMessages(require(showtext, quietly = TRUE)))
if (HAS_SHOWTEXT) {
  try({
    arial_paths <- c("/Library/Fonts/Arial.ttf",
                     "/System/Library/Fonts/Supplemental/Arial.ttf",
                     "C:/Windows/Fonts/arial.ttf")
    arial_file <- arial_paths[file.exists(arial_paths)][1]
    if (!is.na(arial_file) && nzchar(arial_file)) {
      showtext::font_add("Arial", arial_file)
      showtext::showtext_auto()
    }
  }, silent = TRUE)
} else {
  message("showtext not available, using default fonts.")
}

# 读取数据
pseudotime_data <- read_csv(file.path(DATA_03_DIR, "pseudotime_ordered_table_.csv"))
luad_data <- read_csv(file.path(DATA_03_DIR, "LUAD.csv"))
pathway_types <- read_tsv(file.path(DATA_03_DIR, "pathway_types.txt"))
delete_pathways <- readLines(file.path(DATA_03_DIR, "delete_pathway.txt"))
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

# 创建自定义颜色映射（仿照图片中的蓝色-白色-红色配色）
col_fun <- colorRamp2(c(0, 0.5, 1, 1.5, max(luad_ordered, na.rm = TRUE)), 
                      c("#0000FF", "#4169E1", "white", "#FF0000", "#DC143C"))

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

# 创建通路类型的颜色向量，并固定图例顺序
unique_types <- unique(pathway_type_info$Pathway_Type)
# 期望顺序
desired_type_order <- c(
  "Metabolism",
  "Genetic Information Processing",
  "Environmental Information Processing",
  "Cellular Processes",
  "Organismal Systems",
  "Human Diseases"
)
# 按期望顺序 + 其余保留
ordered_types <- c(intersect(desired_type_order, unique_types),
                   setdiff(unique_types, desired_type_order))
pathway_type_info$Pathway_Type <- factor(pathway_type_info$Pathway_Type, levels = ordered_types)
type_colors <- setNames(rainbow(length(ordered_types)), ordered_types)

# 创建左侧行注释（通路类型信息，极细）
left_annotation <- rowAnnotation(
  Pathway_Type = pathway_type_info$Pathway_Type,
  col = list(Pathway_Type = type_colors),
  width = unit(0.25, "cm"),
  show_annotation_name = FALSE,
  annotation_legend_param = list(
    Pathway_Type = list(
      at = ordered_types,
      title = "Pathway_Type",
      title_gp = gpar(fontfamily = "sans"),
      labels_gp = gpar(fontfamily = "sans")
    )
  )
)

# 创建列注释（伪时间信息）
pseudotime_subset <- pseudotime_data[pseudotime_data$Sample_ID %in% common_samples, ]
pseudotime_subset <- pseudotime_subset[match(common_samples, pseudotime_subset$Sample_ID), ]

# 为State创建固定颜色映射（与示例图一致）
state_color_map <- c(
  "1" = "#FF6F6F",  # 红
  "2" = "#B58900",  # 赭黄
  "3" = "#73C476",  # 草绿
  "4" = "#66C2A4",  # 青绿
  "5" = "#56B4E9",  # 天蓝
  "6" = "#8C6BB1",  # 紫色
  "7" = "#F17CB0"   # 品红
)
present_states <- as.character(sort(unique(pseudotime_subset$State)))
state_colors <- state_color_map[present_states]
names(state_colors) <- present_states

# ========== 按state分组，state内部按pseudotime排序 ==========

# 按State分组，每个State内部按Pseudotime排序
# 自定义State顺序：按照1726354顺序排列
custom_state_order <- function(state) {
  case_when(
    state == 1 ~ 1,
    state == 7 ~ 2,
    state == 2 ~ 3,
    state == 6 ~ 4,
    state == 3 ~ 5,
    state == 5 ~ 6,
    state == 4 ~ 7,
    TRUE ~ state + 10  # 其他state排在最后
  )
}

pseudotime_state_ordered <- pseudotime_subset %>%
  mutate(State_Order = custom_state_order(State)) %>%
  arrange(State_Order, Pseudotime) %>%
  select(-State_Order)  # 移除辅助列

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
  col = list(
    Pseudotime = colorRamp2(c(0, 80), 
                           c("darkblue", "white")),
    State = state_colors
  ),
  simple_anno_size = unit(0.5, "cm"),
  annotation_height = unit(c(0.5, 0.5), "cm"),
  annotation_legend_param = list(
    Pseudotime = list(title_gp = gpar(fontfamily = "sans"), labels_gp = gpar(fontfamily = "sans")),
    State = list(title_gp = gpar(fontfamily = "sans"), labels_gp = gpar(fontfamily = "sans"))
  ),
  show_annotation_name = FALSE
)

# ========== 生成核心热图（纯热图，无底部曲线） ==========
print("Generating core heatmap...")

ht_core <- Heatmap(
  t(luad_state_ordered),
  name = "Expression",
  col = col_fun,
  top_annotation = col_annotation_state,
  left_annotation = left_annotation,
  show_column_names = FALSE,
  show_row_names = FALSE,
  cluster_columns = FALSE,
  cluster_rows = FALSE,
  column_title = NULL,
  row_title = NULL,
  width = unit(19, "cm"),
  height = unit(12.5, "cm"),
  heatmap_legend_param = list(
    title = "Expression\nLevel",
    legend_direction = "horizontal",
    title_gp = gpar(fontsize = 8, fontfamily = "sans"),
    labels_gp = gpar(fontsize = 7, fontfamily = "sans"),
    legend_width = unit(6, "cm")
  )
)

png(file.path(DATA_03_DIR, "heatmap_state_pseudotime_ordered_core.png"), width = 9, height = 7, units = "in", res = 300)
try(showtext::showtext_begin(), silent = TRUE)
draw(ht_core,
     show_heatmap_legend = FALSE,
     show_annotation_legend = FALSE,
     padding = unit(c(2, 2, 2, 2), "mm"))
try(showtext::showtext_end(), silent = TRUE)
dev.off()

print("Core heatmap saved.")

# 显示摘要信息
print("========== SUMMARY ==========")
print(paste("Final matrix dimensions:", nrow(luad_ordered), "samples x", ncol(luad_ordered), "pathways"))
print("Color mapping:")
print("  <= 0.5: Blue")
print("  = 1.0: White") 
print("  >= 1.5: Red")
print("")
print("Generated file:")
print("1. heatmap_state_pseudotime_ordered_core.png - Core heatmap with mean curve")

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