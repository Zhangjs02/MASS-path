# MASS-path

The core step code of MASS-path paper

---

## Code Description

### Code/01_build_strong_gpu.py
**GPU-Accelerated Strongly Connected Network Construction and MCE Calculation**

Implements GPU-accelerated strongly connected network construction using PyTorch and CuPy. Main features:
- Extract pathway subnetworks from sample networks
- Calculate optimal center node and supplement cost
- Complement pathway subnetworks to strongly connected graphs
- Compute pathway MCE (Minimum Communication Entropy) values
- Support automatic GPU/CPU switching with batch processing for multiple samples

---

### Code/02_pathway_mce_calculator.py
**Pathway Network MCE Calculator (Parallel Optimized Version)**

Multi-process parallel pathway MCE calculation program. Main features:
- Load sample π vectors and P transition probability matrices
- Extract pathway-corresponding submatrices
- Vectorized pathway MCE calculation
- Compute MCE ratio (pathway normalized MCE / sample normalized MCE)
- Batch result saving with checkpoint resume support

---

### Code/03_heatmap_script.R
**Pathway Expression Heatmap Visualization**

Generate pathway expression heatmaps using ComplexHeatmap package. Main features:
- Order samples by pseudotime
- Group display by pathway types
- Custom color mapping (blue-white-red)
- Add sample annotations (Pseudotime, State, Cluster)
- Generate two types of heatmaps:
  - Ordered by Pseudotime
  - Grouped by State, then ordered by Pseudotime

---

### Code/04_nsx_code.R
**Monocle Pseudotime Analysis**

Perform sample trajectory analysis using Monocle package. Main features:
- Create CellDataSet object
- DDRTree dimensionality reduction
- Sample ordering and pseudotime inference
- Generate multiple trajectory visualizations:
  - Colored by State
  - Colored by Pseudotime
  - Colored by Stage
  - Colored by Cluster
  - Colored by Tissue Type (Tumor/Normal)
- Output pseudotime results table and CellDataSet object

---

### Code/05_run_manifest.py
**ManiFeSt Feature Selection Algorithm**

Run ManiFeSt GPU-accelerated feature selection algorithm. Main features:
- Load and merge two-class sample data
- Create classification labels (0/1)
- Call ManiFeSt_gpu to compute feature scores
- Rank features by score in descending order
- Save feature ranking results (Rank, Feature_Index, Feature_Name, ManiFeSt_Score)

---

### Code/06_validate_feature_ranking.py
**Feature Ranking Validation and Comparative Analysis**

Validate the effectiveness of feature ranking algorithms. Main features:
- Generate three feature ordering methods:
  - Positive order (algorithm ranking)
  - Negative order (reverse importance)
  - Random order (5-run average)
- Progressive feature evaluation (incrementally adding features)
- Stratified cross-validation using Random Forest classifier
- Compute multiple evaluation metrics (Accuracy, Precision, Recall, F1, AUC)
- Generate comparison curves and statistical reports
- Analyze optimal feature count and feature importance

---

### Code/07_create_upset_plot.R
**UpSet Plot Generation**

Create feature intersection visualizations using UpSetR package. Main features:
- Read feature indices from different state comparisons
- Create binary matrix representing feature membership
- Calculate intersection sizes for all combinations
- Generate UpSet plots showing intersection relationships
- Support custom colors and set ordering
- Output detailed statistics

---

### Code/08_plot_pathway_network.py
**Pathway Network Visualization**

Draw network graphs based on KEGG pathway relationships. Main features:
- Load pathway relationships and type information
- Sector layout grouped by pathway types
- Node size reflects degree (connection count)
- Same-type edges use corresponding colors
- Cross-type edges use gradient colors
- Support Top-K pathway highlighting
- Output PNG and SVG formats

---

### Code/09_create_tripartite_from_mapping.py
**Tripartite Network Diagram (Drug-Gene-Pathway)**

Create three-layer network diagram for drugs, genes, and pathways. Main features:
- Three-layer layout: Pathways (top), Genes (middle), Drugs (bottom)
- Gradient-colored edges connecting adjacent layers
- Edges originate from node circle perimeters
- Automatic grouping of related gene families (ERBB, NTRK, MAP2K, etc.)
- Output high-resolution PNG images

---

## Usage

Each Figure folder contains paired code and data. Navigate to the corresponding directory to run.

---

## Folder Description

### base_data/
**Base Data Files**

| File | Description |
|------|-------------|
| `pathway_363_genes.csv` | Gene list for 363 pathways |
| `expression_485_all_genes.csv` | Full gene expression matrix for 485 samples |
| `methylation450_485.csv` | Methylation data for 485 samples |
| `LUAD_cli.csv` | Sample clinical information |
| `bundle_summary.csv` | Data summary information |

---

### Figure 3_a/
**Monocle Pseudotime Analysis (Figure 3a)**

| File | Description |
|------|-------------|
| `04_nsx_code.R` | Monocle pseudotime analysis code |
| `LUAD_intersection_genes.csv` | LUAD gene expression matrix (input) |
| `LUAD_cli.csv` | LUAD clinical information (input) |

**Output:** Trajectory plots (trajectory_by_*.png), pseudotime results table (pseudotime_results.csv)

---

### Figure 3_b/
**LDA Clustering Plot**

| File | Description |
|------|-------------|
| `pseudotime_ordered_table.csv` | Sample pseudotime ordering table (input) |

**Output:** LDA clustering plot

---

### Figure 3_c/
**Pathway Expression Heatmap (Figure 3c)**

| File | Description |
|------|-------------|
| `03_heatmap_script.R` | Heatmap visualization code |
| `pseudotime_ordered_table.csv` | Sample pseudotime ordering table (input) |
| `LUAD_pathway_expression.csv.csv` | Pathway expression matrix (input) |
| `pathway_types.txt` | Pathway type annotation (input) |

**Output:** Heatmap (heatmap_pseudotime_ordered.png/pdf)

---

### Figure 3_d/
**Survival Analysis (Figure 3d)**

| File | Description |
|------|-------------|
| `stage_survival_50_50_fixed.R` | Survival analysis code |
| `stage2_pre_with_survival.csv` | Sample data with survival information (input) |

**Output:** Kaplan-Meier survival curve

---

### Figure 4/
**ManiFeSt Feature Selection and Validation (Figure 4)**

| File | Description |
|------|-------------|
| `05_run_manifest.py` | ManiFeSt feature selection algorithm |
| `06_validate_feature_ranking.py` | Feature ranking validation code |
| `LUAD_1.csv` | Class 1 sample feature matrix (input) |
| `LUAD_2.csv` | Class 2 sample feature matrix (input) |
| `manifest_results.csv` | ManiFeSt feature ranking results |
| `feature_rank_fig4/` | Validation results output directory |

**Output:** Feature ranking comparison curves, validation reports

---

### Figure 4 upset/
**UpSet Intersection Plot (Figure 4 UpSet)**

| File | Description |
|------|-------------|
| `07_create_upset_plot.R` | UpSet plot generation code |
| `feature_indices_comparison.xlsx` | Feature indices from state comparisons (input) |

**Output:** UpSet plot (upset_plot_R.png), intersection statistics (upset_statistics_R.csv)

---

### Figure 5/
**Pathway Network Graph (Figure 5)**

| File | Description |
|------|-------------|
| `08_plot_pathway_network.py` | Pathway network visualization code |
| `kegg_pathway_relations_summary.csv` | KEGG pathway relationships (input) |
| `pathway_types.txt` | Pathway type annotation (input) |
| `rank.csv` | Pathway ranking (for Top-K highlighting) |
| `feature_rank_fig5/` | ManiFeSt results directory |

**Output:** Pathway network graph (pathway_network.png/svg)

---

### Figure 6/
**Drug-Gene-Pathway Tripartite Network (Figure 6)**

| File | Description |
|------|-------------|
| `09_create_tripartite_from_mapping.py` | Tripartite network generation code |
| `H00014_drug_gene_pathway_mapping_merged.csv` | Drug-gene-pathway mapping (input) |

**Output:** Tripartite network graph (tripartite_pathway_gene_drug.png)

---

### supplement_data/
**Supplementary Data**

| File | Description |
|------|-------------|
| `Supplementary_data_1.xlsx` | Pathway network topology features for each sample (485 samples × 731 columns), including sample network nodes/edges and nodes/edges for 363 pathways |
| `Supplementary_data_2.xlsx` | Pathway MCE ratio matrix for each sample (485 samples × 363 pathways), feature matrix for downstream analysis |
| `Supplementary_data_3.xlsx` | Differential pathway lists for 6 state comparisons, including pathway ID, type, name, and UpSet intersection statistics |

**Supplementary_data_3.xlsx Details:**

| Sheet | Content |
|-------|---------|
| Sheet1 | State1 vs State4 differential pathways (32) |
| Sheet2 | State1 vs State6 differential pathways (38) |
| Sheet3 | State1 vs State7 differential pathways (51) |
| Sheet4 | State4 vs State6 differential pathways (33) |
| Sheet5 | State4 vs State7 differential pathways (57) |
| Sheet6 | State6 vs State7 differential pathways (31) |
| Sheet7 | UpSet intersection statistics across degrees for 6 comparisons |
