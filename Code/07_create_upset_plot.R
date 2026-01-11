#!/usr/bin/env Rscript

# 设置CRAN镜像
options(repos = c(CRAN = "https://cran.rstudio.com/"))

# 设置全局字体为Arial
library(grDevices)
if (.Platform$OS.type == "windows") {
    windowsFonts(Arial = windowsFont("Arial"))
} else if (Sys.info()["sysname"] == "Darwin") {
    quartzFonts(Arial = quartzFont(rep("Arial", 4)))
}
par(family = "Arial", font = 2, font.axis = 2, font.lab = 2, font.main = 2, font.sub = 2)

# 安装和加载必要的包
if (!require(UpSetR)) {
    cat("正在安装UpSetR包...\n")
    install.packages("UpSetR")
    library(UpSetR)
}

if (!require(readxl)) {
    cat("正在安装readxl包...\n")
    install.packages("readxl")
    library(readxl)
}

if (!require(dplyr)) {
    cat("正在安装dplyr包...\n")
    install.packages("dplyr")
    library(dplyr)
}

# 读取Excel文件（相对路径，相对于Code目录）
cat("正在读取Excel文件...\n")
data <- read_excel("../Data/feature_indices_comparison.xlsx")

# 查看数据结构
cat("数据维度:", dim(data), "\n")
cat("列名:", colnames(data), "\n")

# 定义指定的集合顺序 (从上到下)
desired_order <- c("State6_VS_State7", "State4_VS_State7", "State4_VS_State6", 
                   "State1_VS_State7", "State1_VS_State6", "State1_VS_State4")

# 重新排列数据列以匹配指定顺序
data <- data[, desired_order]

# 创建UpSet图所需的数据格式
# 首先获取所有唯一的特征索引
all_features <- unique(unlist(data, use.names = FALSE))
all_features <- all_features[!is.na(all_features)]
all_features <- sort(all_features)

cat("总共唯一特征数量:", length(all_features), "\n")

# 创建二进制矩阵
binary_matrix <- matrix(0, nrow = length(all_features), ncol = ncol(data))
colnames(binary_matrix) <- colnames(data)
rownames(binary_matrix) <- all_features

# 填充二进制矩阵
for (i in 1:ncol(data)) {
    col_name <- colnames(data)[i]
    features_in_col <- data[[i]][!is.na(data[[i]])]
    
    # 找到这些特征在all_features中的位置
    feature_positions <- match(features_in_col, all_features)
    binary_matrix[feature_positions, i] <- 1
    
    cat("列", col_name, "包含", length(features_in_col), "个特征\n")
}

# 转换为数据框
upset_data <- as.data.frame(binary_matrix)
upset_data$Feature_Index <- as.numeric(rownames(upset_data))

# 重新排列列，将Feature_Index放在第一列
upset_data <- upset_data[, c("Feature_Index", colnames(data))]

# 创建自定义排序函数
create_custom_order <- function(upset_data, desired_order) {
    # 获取所有可能的交集组合
    set_names <- desired_order
    n_sets <- length(set_names)
    
    # 生成所有可能的交集模式
    all_patterns <- list()
    pattern_info <- list()
    
    # 单个集合 (1个集合的交集)
    for (i in 1:n_sets) {
        pattern <- rep(0, n_sets)
        pattern[i] <- 1
        pattern_key <- paste(pattern, collapse = "")
        all_patterns[[pattern_key]] <- pattern
        pattern_info[[pattern_key]] <- list(size = 1, order = i)
    }
    
    # 多个集合的交集 (2到n个集合)
    for (k in 2:n_sets) {
        combinations <- combn(1:n_sets, k, simplify = FALSE)
        for (combo in combinations) {
            pattern <- rep(0, n_sets)
            pattern[combo] <- 1
            pattern_key <- paste(pattern, collapse = "")
            all_patterns[[pattern_key]] <- pattern
            # 使用最大索引作为排序依据（让下方的集合优先）
            pattern_info[[pattern_key]] <- list(size = k, order = max(combo))
        }
    }
    
    # 按交集大小排序，相同大小按顺序排序（降序让下方集合优先）
    sorted_patterns <- names(all_patterns)[order(
        sapply(pattern_info, function(x) x$size),
        -sapply(pattern_info, function(x) x$order)  # 负号表示降序
    )]
    
    return(sorted_patterns)
}

# 创建UpSet图
cat("正在创建UpSet图...\n")

# 获取自定义排序
custom_order <- create_custom_order(upset_data, desired_order)

# 定义与图片匹配的颜色
set_colors <- c(
    "State6_VS_State7" = "#FF6B6B",    # 红色
    "State4_VS_State6" = "#4ECDC4",    # 青色
    "State1_VS_State6" = "#45B7D1",    # 蓝色
    "State1_VS_State4" = "#2E3A87",    # 深蓝色
    "State1_VS_State7" = "#FF8E53",    # 橙色
    "State4_VS_State7" = "#6C5B7B"     # 紫色
)

# 设置Arial字体和加粗
par(family = "Arial", font = 2)  # font = 2 表示加粗

# 设置图形参数 - 增大上方柱状图高度
png("../Data/output/upset_plot_R.png", width = 1400, height = 1000, res = 150, family = "Arial")

upset(upset_data, 
      sets = desired_order,
      sets.bar.color = set_colors[desired_order],
      main.bar.color = "#4682B4",      # 更深的蓝色柱状图 (SteelBlue)
      matrix.color = "#2E3A87",        # 深蓝色连接点和线
      shade.color = "grey90",
      shade.alpha = 0.5,
      mb.ratio = c(0.75, 0.25),        # 上方柱状图75%，下方矩阵25%
      number.angles = 0,
      point.size = 3.5,
      line.size = 2,
      mainbar.y.label = "Intersection Size",
      sets.x.label = "Set Size",
      text.scale = c(1.2, 1.0, 1.0, 1.0, 1.2, 1.5), # 第4个值控制柱状图顶部数字
      set_size.scale_max = 60,         # 适中的左侧集合柱状图长度
      show.numbers = "yes",            # 确保显示数字
      order.by = "degree",             # 按交集数量排序
      decreasing = FALSE,              # 升序（1个点先显示）
      keep.order = TRUE)

dev.off()

cat("UpSet图已保存为: upset_plot_R.png\n")

# 创建更详细的统计信息
cat("\n正在生成详细统计信息...\n")

# 计算各种交集
intersections <- list()
set_names <- colnames(data)

# 单个集合的大小
for (i in 1:length(set_names)) {
    set_name <- set_names[i]
    features <- data[[set_name]][!is.na(data[[set_name]])]
    intersections[[paste0("Only_", set_name)]] <- length(features)
}

# 计算所有可能的交集
# 生成所有可能的组合
for (k in 2:length(set_names)) {
    combinations <- combn(set_names, k, simplify = FALSE)
    
    for (combo in combinations) {
        # 计算交集
        intersection_features <- all_features
        
        for (set_name in combo) {
            set_features <- data[[set_name]][!is.na(data[[set_name]])]
            intersection_features <- intersect(intersection_features, set_features)
        }
        
        # 从其他集合中排除
        for (other_set in setdiff(set_names, combo)) {
            other_features <- data[[other_set]][!is.na(data[[other_set]])]
            intersection_features <- setdiff(intersection_features, other_features)
        }
        
        if (length(intersection_features) > 0) {
            combo_name <- paste(combo, collapse = " ∩ ")
            intersections[[combo_name]] <- length(intersection_features)
        }
    }
}

# 保存统计结果
stats_df <- data.frame(
    Intersection = names(intersections),
    Size = unlist(intersections),
    stringsAsFactors = FALSE
)

stats_df <- stats_df[order(stats_df$Size, decreasing = TRUE), ]
write.csv(stats_df, "../Data/output/upset_statistics_R.csv", row.names = FALSE)

cat("统计信息已保存为: ../Data/output/upset_statistics_R.csv\n")

# 打印前10个最大的交集
cat("\n前10个最大的交集:\n")
print(head(stats_df, 10))

# 创建简化版的UpSet图（按自定义排序）
cat("\n正在创建简化版UpSet图...\n")

# 设置字体加粗
par(family = "Arial", font = 2)  # font = 2 表示加粗

png("../Data/output/upset_plot_simplified_R.png", width = 1400, height = 1000, res = 150, family = "Arial")

upset(upset_data, 
      sets = desired_order,
      sets.bar.color = set_colors[desired_order],
      main.bar.color = "#4682B4",      # 更深的蓝色柱状图 (SteelBlue)
      matrix.color = "#2E3A87",        # 深蓝色连接点和线
      shade.color = "grey90",
      shade.alpha = 0.5,
      mb.ratio = c(0.75, 0.25),        # 上方柱状图75%，下方矩阵25%
      number.angles = 0,
      point.size = 3.5,
      line.size = 2,
      mainbar.y.label = "Intersection Size",
      sets.x.label = "Set Size",
      text.scale = c(1.2, 1.0, 0.8, 2.5, 1.5, 0.4), # 第4个值控制柱状图顶部数字
      set_size.scale_max = 50,         # 适中的左侧集合柱状图长度
      show.numbers = "yes",            # 确保显示数字
      order.by = "degree",             # 按交集数量排序
      decreasing = FALSE,              # 升序（1个点先显示）
      keep.order = TRUE,
      cutoff = 1,  # 只显示大小>=1的交集
      nintersects = 25)  # 显示更多交集

dev.off()

cat("简化版UpSet图已保存为: ../Data/output/upset_plot_simplified_R.png\n")

cat("\n完成！生成的文件 (保存在 ../Data/output/ 目录下):\n")
cat("1. upset_plot_R.png - 完整UpSet图\n")
cat("2. upset_plot_simplified_R.png - 简化版UpSet图\n")
cat("3. upset_statistics_R.csv - 详细统计信息\n")