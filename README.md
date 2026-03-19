# MASS-path

Core code and data pipeline for the MASS-path paper.

---

## Code Description

All scripts are located in the `Code/` directory. Each script reads inputs from and writes outputs to the corresponding `data/XX/` subdirectory using relative paths.

---

### 01\_build\_strong\_gpu.py
**GPU-Accelerated Strongly Connected Network Construction and MCE Calculation**

Builds strongly connected pathway subnetworks with GPU acceleration via PyTorch and CuPy.
- Extracts pathway subnetworks from sample networks
- Identifies optimal center nodes and computes supplement costs
- Complements pathway subnetworks into strongly connected graphs
- Computes pathway MCE (Minimum Communication Entropy) values
- Supports automatic GPU/CPU fallback and batch processing across samples

**Input**: `data/01/` (pathway_gene_ids.csv, sample network files, etc.)
**Output**: `data/01/` (Sample_MCE_Vectors/, pi vectors, P matrices, etc.)

---

### 02\_pathway\_mce\_calculator.py
**Pathway MCE Calculator (Parallel Optimized)**

Computes pathway-level MCE values using multi-process parallelism.
- Loads sample pi vectors and P transition probability matrices
- Extracts pathway-specific submatrices
- Performs vectorized pathway MCE calculation
- Computes MCE ratio (pathway normalized MCE / sample normalized MCE)
- Supports checkpoint-based resume and batch result saving

**Input**: `data/01/` (outputs from Step 01)
**Output**: `data/01/` (MCE ratio matrix)

---

### 03\_heatmap\_script.R
**Pathway Expression Heatmap Visualization**

Generates pathway MCE ratio heatmaps using the ComplexHeatmap package.
- Samples grouped by Monocle State (1→7→2→6→3→5→4), ordered by Pseudotime within each group
- Pathways grouped by type (Metabolism, Cellular Processes, etc.)
- Blue-white-red color scale (≤0.5 blue, 1.0 white, ≥1.5 red)
- Top annotation bars for Pseudotime and State

**Input**: `data/03/` (pseudotime_ordered_table\_.csv, LUAD.csv, pathway_types.txt)
**Output**: `data/03/heatmap_state_pseudotime_ordered_core.png`

---

### 04\_nsx\_code.R
**Monocle Pseudotime Trajectory Analysis**

Performs sample trajectory inference using Monocle 2.
- Creates CellDataSet object from gene expression and clinical data
- DDRTree dimensionality reduction
- Sample ordering and pseudotime inference
- Generates trajectory plots colored by State, Pseudotime, Stage, Cluster, and Tissue Type
- Outputs pseudotime results table and CDS object

**Input**: `data/04/` (LUAD_intersection_genes.csv, LUAD_cli.csv)
**Output**: `data/04/` (pseudotime_results.csv, trajectory PNG files)

---

### 05\_run\_manifest.py
**ManiFeSt Feature Selection Algorithm**

Runs the GPU-accelerated ManiFeSt feature selection algorithm.
- Loads and merges two-state sample data
- Constructs binary classification labels (0/1)
- Calls ManiFeSt\_gpu to compute feature discriminability scores
- Ranks all features by score in descending order
- Saves ranking results (Rank, Feature\_Index, Feature\_Name, ManiFeSt\_Score)

**Config**: `CANCER_TYPE`, `STATE_A`, `STATE_B`
**Input**: `data/05/feature_selection/` ({CANCER_TYPE}\_state\_{A/B}.csv)
**Output**: `data/05/` ({CANCER_TYPE}\_manifest\_state{A}\_vs\_state{B}.csv)

---

### 06\_validate\_feature\_ranking.py
**Feature Ranking Validation and Comparative Analysis**

Validates the effectiveness of ManiFeSt feature rankings.
- Generates three feature orderings: positive (ManiFeSt ranking), negative (reversed), and random (5-run average)
- Progressive feature evaluation: incrementally adds features and tracks classification performance
- Random Forest classifier with stratified cross-validation
- Computes Accuracy, Precision, Recall, F1, and AUC
- Generates comparison curves and statistical reports
- Identifies the optimal number of features

**Input**: `data/05/` (ManiFeSt ranking files, state data)
**Output**: `data/05/rank_result_{CANCER_TYPE}_state{A}_vs_state{B}/`

---

### 07\_create\_upset\_plot.R
**UpSet Intersection Plot**

Creates feature intersection visualizations using the UpSetR package.
- Reads feature indices from 6 state-pair comparisons (Excel)
- Constructs binary membership matrix
- Computes intersection sizes for all combinations
- Generates UpSet plots showing intersection relationships
- Supports custom colors and set ordering

**Input**: `data/07/` (feature_indices_comparison.xlsx)
**Output**: `data/07/` (upset_plot_R.png, upset_statistics_R.csv)

---

### 08\_plot\_pathway\_network.py
**Pathway Network Visualization**

Draws network graphs based on KEGG pathway relationships.
- Loads pathway relationships and type annotations
- Sector layout grouped by pathway type
- Node size proportional to degree (connection count)
- Intra-type edges colored by type; cross-type edges use gradient colors
- Supports Top-K pathway highlighting

**Input**: `data/08/` (kegg_pathway_relations_summary.csv, pathway_types.txt, rank.csv)
**Output**: `data/08/` (pathway_network.png/svg)

---

### 09\_create\_tripartite\_from\_mapping.py
**Drug-Gene-Pathway Tripartite Network**

Generates a three-layer network diagram from drug-gene-pathway mappings.
- Three-layer layout: Pathways (top) → Genes (middle) → Drugs (bottom)
- Gradient-colored edges connecting adjacent layers
- Edges originate from node circle perimeters
- Automatic grouping of related gene families (ERBB, NTRK, MAP2K, etc.)

**Input**: `data/09/` (H00014_drug_gene_pathway_mapping_merged.csv)
**Output**: `data/09/tripartite_pathway_gene_drug.png`

---

### 10\_supervised\_analysis.py
**Supervised Dimensionality Reduction Analysis**

Evaluates state clustering using multiple dimensionality reduction methods.
- LDA (Linear Discriminant Analysis)
- t-SNE
- PCA
- UMAP
- Visualizes reduction results for each method

**Input**: `data/10/state_result/` (per-state sample CSVs)
**Output**: `data/10/supervised_results/`

---

### 11\_stage\_survival\_50\_50\_fixed.R
**Stage Survival Analysis (50/50 Split)**

Performs survival analysis on stage-specific patient subgroups.
- Reads sample data with survival information
- Splits patients into high-risk and low-risk groups at the median (50/50)
- Draws Kaplan-Meier survival curves using survival + survminer
- Computes Log-rank test p-values

**Input**: `data/11/` (stage2_pre_with_survival.csv)
**Output**: `data/11/high_low_risk_survival_results/`

---

## Data Directory Structure

| Directory | Scripts | Contents |
|-----------|---------|----------|
| `data/01/` | 01, 02 | Sample networks, pi vectors, P matrices, MCE results |
| `data/03/` | 03 | MCE ratio matrix (LUAD.csv), pseudotime table, heatmap output |
| `data/04/` | 04 | Gene expression matrix, clinical info, pseudotime results |
| `data/05/` | 05, 06 | ManiFeSt module, state data, ranking results, validation results |
| `data/07/` | 07 | Feature index Excel, UpSet plot output |
| `data/08/` | 08 | Pathway relationships, type annotations, rankings, network plots |
| `data/09/` | 09 | Drug-gene-pathway mapping, tripartite network output |
| `data/10/` | 10 | State-grouped data, dimensionality reduction results |
| `data/11/` | 11 | Survival data, survival curve output |

---

## Supplementary Data

| File | Contents |
|------|----------|
| `Supplementary_data_1.xlsx` | Pathway network topology features per sample (485 samples × 731 columns) |
| `Supplementary_data_2.xlsx` | Pathway MCE ratio matrix per sample (485 samples × 363 pathways) |
| `Supplementary_data_3.xlsx` | Differential pathway lists for 6 state comparisons with UpSet intersection statistics |
