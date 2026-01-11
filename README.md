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

All code uses relative paths and should be run from the `Code/` directory:

```bash
cd Code
python 09_create_tripartite_from_mapping.py
```

