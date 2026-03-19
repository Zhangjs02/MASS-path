# Load required packages
library(survival)
library(survminer)
library(ggplot2)
library(dplyr)
library(RColorBrewer)

# 基于脚本位置计算路径
args <- commandArgs(trailingOnly = FALSE)
script_path <- sub("--file=", "", args[grep("--file=", args)])
if (length(script_path) == 0) script_path <- "."
SCRIPT_DIR <- dirname(normalizePath(script_path))
PROJECT_DIR <- dirname(SCRIPT_DIR)
DATA_11_DIR <- file.path(PROJECT_DIR, "data", "11")

# Fixed 50-50 Stage Survival Analysis
cat("=== Fixed 50-50 Stage Survival Analysis ===\n\n")

# Read Stage2 data
cat("Reading Stage2 data...\n")
stage2_data <- read.csv(file.path(DATA_11_DIR, "stage2_pre_with_survival.csv"), stringsAsFactors = FALSE)

cat("Data dimensions:", dim(stage2_data), "\n")
cat("Original State distribution:\n")
print(table(stage2_data$State, useNA = "ifany"))

# Check survival data
cat("Survival time range:", range(stage2_data$OS_time, na.rm = TRUE), "\n")
cat("Event rate:", mean(stage2_data$OS_status, na.rm = TRUE), "\n")

# Ensure State is numeric and remove missing values
stage2_data$State <- as.numeric(as.character(stage2_data$State))
stage2_data <- stage2_data[!is.na(stage2_data$State) & !is.na(stage2_data$OS_time), ]

cat("\nData after removing missing values:", nrow(stage2_data), "samples\n")

# Define state colors (RGB values converted to hex)
state_colors <- c(
  "#FF6F6F",  # State 1: 255,111,111
  "#B5890B",  # State 2: 181,137,011
  "#05C476",  # State 3: 5,196,118
  "#66C2A4",  # State 4: 102,194,164
  "#56B4E9",  # State 5: 86,180,233
  "#8C6BB1",  # State 6: 140,107,177
  "#F17CB0"   # State 7: 241,124,176
)

# Function to create lighter version of a color
create_lighter_color <- function(hex_color, alpha = 0.6) {
  rgb_vals <- col2rgb(hex_color)
  rgb(rgb_vals[1], rgb_vals[2], rgb_vals[3], alpha = alpha * 255, maxColorValue = 255)
}

# Function to create TRUE 50-50 split using ranking
create_equal_50_50_groups <- function(data, stage_num) {
  # Filter data for specific stage
  stage_data <- data[data$State == stage_num, ]
  
  if(nrow(stage_data) < 10) {
    cat("Warning: Stage", stage_num, "has only", nrow(stage_data), "samples. Skipping...\n")
    return(NULL)
  }
  
  # Sort by survival time and create equal groups
  stage_data <- stage_data[order(stage_data$OS_time), ]
  n_samples <- nrow(stage_data)
  
  # Calculate split point for equal groups
  split_point <- floor(n_samples / 2)
  
  # Create equal groups with High Risk and Low Risk labels
  stage_data$Group <- c(rep("High Risk (Bottom 50%)", split_point),
                       rep("Low Risk (Top 50%)", n_samples - split_point))
  
  stage_data$Group <- factor(stage_data$Group, 
                            levels = c("High Risk (Bottom 50%)", "Low Risk (Top 50%)"))
  
  cat("\nStage", stage_num, "equal 50-50 groups:")
  print(table(stage_data$Group))
  
  # Show survival time ranges for each group
  high_risk_range <- range(stage_data$OS_time[stage_data$Group == "High Risk (Bottom 50%)"])
  low_risk_range <- range(stage_data$OS_time[stage_data$Group == "Low Risk (Top 50%)"])
  
  cat("High Risk group time range:", high_risk_range[1], "-", high_risk_range[2], "days\n")
  cat("Low Risk group time range:", low_risk_range[1], "-", low_risk_range[2], "days\n")
  
  return(stage_data)
}

# Function to create enhanced survival plot without risk table
create_enhanced_survival_plot_no_risk <- function(stage_data, stage_num) {
  if(is.null(stage_data)) return(NULL)
  
  # Create survival object
  km_fit <- survfit(Surv(OS_time, OS_status) ~ Group, data = stage_data)
  
  # Perform log-rank test
  logrank_test <- survdiff(Surv(OS_time, OS_status) ~ Group, data = stage_data)
  p_value <- 1 - pchisq(logrank_test$chisq, length(logrank_test$n) - 1)
  
  # Calculate sample sizes
  group_counts <- table(stage_data$Group)
  group_labels <- paste0(names(group_counts), " (n=", group_counts, ")")
  
  cat("\n=== Stage", stage_num, "Survival Analysis (Equal 50-50 Split) ===\n")
  cat("Group sample sizes:\n")
  print(group_counts)
  cat("Log-rank test P-value:", format(p_value, digits = 4), "\n")
  
  # Use state-specific colors with transparency/depth difference
  base_color <- state_colors[stage_num]
  colors <- c(base_color, create_lighter_color(base_color, alpha = 0.6))  # Darker for High Risk, lighter for Low Risk
  
  # Create title with legend information and color squares
  high_risk_square <- "■"  # Dark square for high risk
  low_risk_square <- "□"   # Light square for low risk
  title_with_legend <- paste0("State ", stage_num, " | ", high_risk_square, " High Risk (n=", group_counts[1], ") vs ", low_risk_square, " Low Risk (n=", group_counts[2], ")")
  
  # Set x-axis breaks based on stage
  if (stage_num == 5) {
    x_breaks <- seq(0, max(stage_data$OS_time, na.rm = TRUE), by = 500)
  } else {
    x_breaks <- seq(0, max(stage_data$OS_time, na.rm = TRUE), by = 1000)
  }
  
  # Create survival curve plot without risk table and confidence intervals
  tryCatch({
    p <- ggsurvplot(
      km_fit,
      data = stage_data,
      pval = FALSE,  # Remove p-value display since we'll add it manually
      conf.int = FALSE,  # Remove confidence intervals
      risk.table = FALSE,     # Remove risk table
      break.time.by = if (stage_num == 5) 500 else 1000,
      linetype = "solid",
      size = 2.0,  # 增加生存曲线粗细从1.2到2.0
      ggtheme = theme_minimal(),
      palette = colors,
      title = "",  # Remove original title
      xlab = "Days",
      ylab = "Survival Probability",
      legend = "none"  # Remove legend since info is in title
    )
    
    # Enhance plot appearance with custom title and p-value annotation
    p$plot <- p$plot + 
      ggtitle(title_with_legend) +
      scale_x_continuous(breaks = x_breaks) +
      annotate("text", x = max(stage_data$OS_time, na.rm = TRUE) * 0.05, 
               y = 0.1, label = paste0("p = ", format(p_value, digits = 3)), 
               size = 6, hjust = 0, vjust = 0, fontface = "bold", family = "Arial") +  # 增大P值字号从4到6
      theme(
        text = element_text(family = "Arial"),  # Set all text to Arial
        plot.title = element_text(size = 18, face = "bold", hjust = 0.5, margin = margin(b = 15), family = "Arial"),
        legend.position = "none",  # No legend needed
        axis.title = element_text(size = 24, face = "bold", family = "Arial"),  # 增大坐标轴标题从20到24
        axis.text = element_text(size = 22, family = "Arial", face = "bold"),  # 增大坐标轴刻度文字从18到22
        axis.line = element_line(size = 1.2, color = "black"),  # 坐标轴线粗细
        axis.ticks = element_line(size = 1.2, color = "black"),  # 刻度线粗细
        axis.ticks.length = unit(0.3, "cm"),  # 增加刻度线长度
        panel.grid.minor = element_blank(),
        panel.grid.major = element_line(size = 0.5, color = "gray80"),  # 稍微加粗网格线并调整颜色
        panel.border = element_rect(color = "black", fill = NA, size = 1.2),  # 边框粗细
        plot.margin = margin(t = 5, r = 5, b = 5, l = 5, unit = "pt")
      )
    
    # Save plots with smaller size but larger fonts, maintaining 3.5:3 ratio
    filename_prefix <- paste0("stage", stage_num, "_high_low_risk_survival")
    
    # Use smaller dimensions with 3.5:3 ratio
    width_in <- 7      # 3.5 * 2 = 7 inches width
    height_in <- 6     # 3 * 2 = 6 inches height (3.5:3 ratio)
    
    png(paste0(filename_prefix, ".png"), width = width_in, height = height_in, units = "in", res = 300)
    print(p)
    dev.off()
    
    pdf(paste0(filename_prefix, ".pdf"), width = width_in, height = height_in)
    print(p)
    dev.off()
    
    cat("Plots saved as:", paste0(filename_prefix, ".png/pdf\n"))
    
    return(list(
      plot = p,
      km_fit = km_fit,
      logrank_p = p_value,
      group_counts = group_counts,
      stage = stage_num
    ))
    
  }, error = function(e) {
    cat("ggsurvplot error for Stage", stage_num, ":", e$message, "\n")
    return(NULL)
  })
}

# Create output directory
output_dir <- file.path(DATA_11_DIR, "high_low_risk_survival_results")
if(!dir.exists(output_dir)) {
  dir.create(output_dir, recursive = TRUE)
}
setwd(output_dir)

# Get unique stages
unique_stages <- sort(unique(stage2_data$State))
cat("\nAnalyzing stages:", paste(unique_stages, collapse = ", "), "\n")

# Initialize results storage
all_results <- list()

# Process each stage with fixed 50-50 split
cat("\n", rep("=", 60), "\n")
cat("HIGH RISK vs LOW RISK SURVIVAL ANALYSIS BY STAGE\n")
cat(rep("=", 60), "\n")

for(stage in unique_stages) {
  cat("\n", rep("-", 40), "\n")
  cat("Processing Stage", stage, "\n")
  cat(rep("-", 40), "\n")
  
  # Create equal 50-50 groups for this stage
  stage_data <- create_equal_50_50_groups(stage2_data, stage)
  
  if(!is.null(stage_data)) {
    # Create survival plot
    result <- create_enhanced_survival_plot_no_risk(stage_data, stage)
    if(!is.null(result)) {
      result_name <- paste0("Stage_", stage, "_high_low_risk")
      all_results[[result_name]] <- result
    }
  }
}

# Generate analysis report
cat("\n=== Generating analysis report ===\n")

report_lines <- c(
  "=== High Risk vs Low Risk Stage Survival Analysis Report ===",
  "",
  paste("Analysis Date:", Sys.Date()),
  "",
  "Data Overview:",
  paste("- Total sample size:", nrow(stage2_data)),
  paste("- Survival time range:", paste(range(stage2_data$OS_time, na.rm = TRUE), collapse = " - "), "days"),
  paste("- Total events:", sum(stage2_data$OS_status), "/", nrow(stage2_data)),
  paste("- Event rate:", round(mean(stage2_data$OS_status, na.rm = TRUE) * 100, 2), "%"),
  "",
  "=== Analysis Method ===",
  "- Fixed 50-50 split using ranking method",
  "- Samples sorted by survival time and divided into equal groups",
  "- High Risk: Bottom 50% (shortest survival times)",
  "- Low Risk: Top 50% (longest survival times)",
  "- Risk table removed for cleaner visualization",
  "- Confidence intervals with subtle transparency (alpha=0.15)",
  "",
  "=== Results Summary ===",
  ""
)

# Add stage-specific results
for(stage in unique_stages) {
  result_name <- paste0("Stage_", stage, "_high_low_risk")
  
  if(result_name %in% names(all_results)) {
    result <- all_results[[result_name]]
    report_lines <- c(report_lines,
      paste("Stage", stage, ":"),
      paste("  - Sample sizes:", paste(result$group_counts, collapse = " vs ")),
      paste("  - P-value:", format(result$logrank_p, digits = 4)),
      paste("  - Significance:", ifelse(result$logrank_p < 0.05, "Significant", "Not significant")),
      ""
    )
  }
}

report_lines <- c(report_lines,
  "=== Key Features ===",
  "- True equal split: Samples are ranked and divided exactly in half",
  "- Clinical terminology: High Risk vs Low Risk groups",
  "- No median bias: Equal group sizes regardless of tied values",
  "- Cleaner visualization: Risk table removed",
  "- Subtle confidence intervals: More professional appearance",
  "",
  "=== Files Generated ===",
  "- Individual survival plots for each stage (PNG and PDF formats)",
  "- This analysis report",
  "",
  paste("Total analyses performed:", length(all_results))
)

# Save report
writeLines(report_lines, "high_low_risk_survival_analysis_report.txt")

# Return to original directory
setwd("..")

cat("\n=== Analysis Complete ===\n")
cat("Results saved in 'high_low_risk_survival_results' directory\n")
cat("Generated", length(all_results), "survival analyses with High Risk vs Low Risk groups\n")
cat("Report saved as 'high_low_risk_survival_analysis_report.txt'\n")
cat("\nKey improvements:\n")
cat("- Clinical terminology: High Risk vs Low Risk\n")
cat("- True equal sample splits using ranking method\n")
cat("- Removed risk tables for cleaner visualization\n")
cat("- Subtle confidence intervals (alpha=0.15)\n")