"""Create tripartite network diagram from H00014_drug_gene_pathway_mapping.csv.

Three layers:
- Layer 1 (top): Pathways
- Layer 2 (middle): Genes
- Layer 3 (bottom): Drugs

Connections: Pathway-Gene, Gene-Drug
"""

from pathlib import Path
import pandas as pd
import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib.patches import FancyBboxPatch
import matplotlib.patches as mpatches


BASE_DIR = Path(__file__).resolve().parent.parent
DATA_DIR = BASE_DIR / "Data"
MAPPING_FILE = DATA_DIR / "H00014_drug_gene_pathway_mapping_merged.csv"
OUTPUT_DIR = DATA_DIR / "output"
OUTPUT_PNG = OUTPUT_DIR / "tripartite_pathway_gene_drug.png"

plt.rcParams["font.family"] = "Arial"


def load_and_prepare_data():
    """Load mapping data and extract nodes and edges."""
    
    df = pd.read_csv(MAPPING_FILE)
    
    # Remove rows without pathway (we'll handle them separately)
    df_with_pathway = df[df["pathway_id"].notna()].copy()
    df_without_pathway = df[df["pathway_id"].isna()].copy()
    
    # Extract unique nodes
    pathways = df_with_pathway["pathway_id"].unique().tolist()
    all_genes = df["gene_id"].unique().tolist()
    all_drugs = df["drug_id"].unique().tolist()
    
    # Get labels
    pathway_labels = {}
    for _, row in df_with_pathway.drop_duplicates("pathway_id").iterrows():
        pathway_labels[row["pathway_id"]] = row["pathway_name"] if pd.notna(row["pathway_name"]) else row["pathway_id"]
    
    gene_labels = {}
    for _, row in df.drop_duplicates("gene_id").iterrows():
        gene_symbol = row["gene_symbol"] if pd.notna(row["gene_symbol"]) else str(row["gene_id"])
        # For merged gene families (e.g., DHFR/DHFR2), only show the first gene
        if "/" in str(gene_symbol):
            gene_symbol = gene_symbol.split("/")[0]
        gene_labels[row["gene_id"]] = gene_symbol
    
    drug_labels = {}
    for _, row in df.drop_duplicates("drug_id").iterrows():
        drug_labels[row["drug_id"]] = row["drug_name"] if pd.notna(row["drug_name"]) else row["drug_id"]
    
    # Extract edges
    pathway_gene_edges = df_with_pathway[["pathway_id", "gene_id"]].drop_duplicates()
    gene_drug_edges = df[["gene_id", "drug_id"]].drop_duplicates()
    
    print(f"Nodes:")
    print(f"  Pathways: {len(pathways)}")
    print(f"  Genes: {len(all_genes)}")
    print(f"  Drugs: {len(all_drugs)}")
    print(f"\nEdges:")
    print(f"  Pathway-Gene: {len(pathway_gene_edges)}")
    print(f"  Gene-Drug: {len(gene_drug_edges)}")
    
    return {
        "pathways": pathways,
        "genes": all_genes,
        "drugs": all_drugs,
        "pathway_labels": pathway_labels,
        "gene_labels": gene_labels,
        "drug_labels": drug_labels,
        "pathway_gene_edges": pathway_gene_edges,
        "gene_drug_edges": gene_drug_edges,
    }


def calculate_positions(data):
    """Calculate positions for all nodes in three layers."""
    
    pathways = data["pathways"]
    genes_raw = data["genes"]
    drugs = data["drugs"]
    
    # Custom sorting for genes: keep ERBB family together
    def gene_sort_key(gene_id):
        """Custom sort key to group gene families together."""
        gene_symbol = data["gene_labels"].get(gene_id, gene_id)
        
        # Group ERBB family
        if gene_symbol.startswith("ERBB"):
            # ERBB2, ERBB3, ERBB4 -> sort as ERBB_2, ERBB_3, ERBB_4
            return f"ERBB_{gene_symbol[4:]}"
        
        # Group NTRK family
        if gene_symbol.startswith("NTRK"):
            return f"NTRK_{gene_symbol[4:]}"
        
        # Group MAP2K family
        if gene_symbol.startswith("MAP2K"):
            return f"MAP2K_{gene_symbol[5:]}"
        
        # Group FLT family
        if gene_symbol.startswith("FLT"):
            return f"FLT_{gene_symbol[3:]}"
        
        # Default: sort by gene symbol
        return gene_symbol
    
    genes = sorted(genes_raw, key=gene_sort_key)
    
    # Y coordinates for each layer (increased spacing x12 = 8 * 1.5)
    y_pathway = 24.0
    y_gene = 12.0
    y_drug = 0.0
    
    # Calculate horizontal spacing
    n_pathway = len(pathways)
    n_gene = len(genes)
    n_drug = len(drugs)
    
    # Use adaptive spacing
    max_width = 50.0
    
    gap_pathway = max_width / max(n_pathway + 1, 1)
    gap_gene = max_width / max(n_gene + 1, 1)
    gap_drug = max_width / max(n_drug + 1, 1)
    
    positions = {}
    
    # Pathway positions
    for i, pathway in enumerate(pathways):
        positions[("pathway", pathway)] = (gap_pathway * (i + 1), y_pathway)
    
    # Gene positions
    for i, gene in enumerate(genes):
        positions[("gene", gene)] = (gap_gene * (i + 1), y_gene)
    
    # Drug positions
    for i, drug in enumerate(drugs):
        positions[("drug", drug)] = (gap_drug * (i + 1), y_drug)
    
    return positions


def draw_gradient_edge(ax, x1, y1, x2, y2, r1, r2, color1, color2, width=0.5, alpha=0.6, n_segments=50):
    """Draw an edge with gradient color, connecting circle edges.
    
    Args:
        ax: matplotlib axes
        x1, y1: center of source node
        x2, y2: center of target node
        r1: radius of source node
        r2: radius of target node
        color1, color2: RGB colors for gradient
        width: line width
        alpha: transparency
        n_segments: number of gradient segments
    """
    
    # Calculate direction vector
    dx = x2 - x1
    dy = y2 - y1
    dist = np.sqrt(dx**2 + dy**2)
    
    if dist < 1e-6:
        return  # Skip if nodes overlap
    
    # Normalize direction
    dx_norm = dx / dist
    dy_norm = dy / dist
    
    # Calculate edge start and end points (on circle perimeters)
    edge_x1 = x1 + dx_norm * r1
    edge_y1 = y1 + dy_norm * r1
    edge_x2 = x2 - dx_norm * r2
    edge_y2 = y2 - dy_norm * r2
    
    # Create gradient by drawing multiple line segments
    for i in range(n_segments):
        t1 = i / n_segments
        t2 = (i + 1) / n_segments
        
        # Interpolate position
        x_start = edge_x1 + t1 * (edge_x2 - edge_x1)
        y_start = edge_y1 + t1 * (edge_y2 - edge_y1)
        x_end = edge_x1 + t2 * (edge_x2 - edge_x1)
        y_end = edge_y1 + t2 * (edge_y2 - edge_y1)
        
        # Interpolate color
        r = color1[0] + t1 * (color2[0] - color1[0])
        g = color1[1] + t1 * (color2[1] - color1[1])
        b = color1[2] + t1 * (color2[2] - color1[2])
        
        # Use gradient color
        ax.plot([x_start, x_end], [y_start, y_end], 
                color=(r, g, b), linewidth=width, alpha=alpha, zorder=1)


def create_tripartite_diagram(data):
    """Create and save the tripartite network diagram."""
    
    positions = calculate_positions(data)
    
    # Create figure
    fig, ax = plt.subplots(figsize=(20, 10))
    
    # Colors (gene and drug swapped)
    pathway_color = np.array([0xD8, 0x76, 0x59]) / 255  # #EF767A - Pink/Red
    gene_color = np.array([0x7D, 0xAE, 0xE0]) / 255     # #48C0AA - Teal
    drug_color = np.array([0x29, 0x9D, 0x8F]) / 255     # #456990 - Blue
    
    # Node sizes (4.5x original = 3x * 1.5)
    pathway_size = 0.54
    gene_size = 0.54
    drug_size = 0.48
    
    # Draw edges first (so they appear behind nodes)
    print("\nDrawing edges...")
    
    # Pathway-Gene edges
    for _, row in data["pathway_gene_edges"].iterrows():
        pathway = row["pathway_id"]
        gene = row["gene_id"]
        
        if ("pathway", pathway) in positions and ("gene", gene) in positions:
            x1, y1 = positions[("pathway", pathway)]
            x2, y2 = positions[("gene", gene)]
            draw_gradient_edge(ax, x1, y1, x2, y2, pathway_size, gene_size,
                             pathway_color, gene_color, width=3.0, alpha=0.6)
    
    # Gene-Drug edges
    for _, row in data["gene_drug_edges"].iterrows():
        gene = row["gene_id"]
        drug = row["drug_id"]
        
        if ("gene", gene) in positions and ("drug", drug) in positions:
            x1, y1 = positions[("gene", gene)]
            x2, y2 = positions[("drug", drug)]
            draw_gradient_edge(ax, x1, y1, x2, y2, gene_size, drug_size,
                             gene_color, drug_color, width=3.0, alpha=0.6)
    
    print("Drawing nodes...")
    
    # Draw pathway nodes (use pathway_id as label)
    for pathway in data["pathways"]:
        x, y = positions[("pathway", pathway)]
        circle = plt.Circle((x, y), pathway_size, color=pathway_color, 
                           ec="white", linewidth=1.5, zorder=3)
        ax.add_patch(circle)
        
        # Use pathway ID instead of name
        label = pathway
        
        # Shift label right to align with rotated text
        label_x_offset = 0.75
        ax.text(x + label_x_offset, y + pathway_size + 0.08, label,
               ha="center", va="bottom", fontsize=16, rotation=45,
               color="black", weight="normal", zorder=4)
    
    # Draw gene nodes (use gene symbol as label)
    for gene in data["genes"]:
        x, y = positions[("gene", gene)]
        circle = plt.Circle((x, y), gene_size, color=gene_color,
                           ec="white", linewidth=1.5, zorder=3)
        ax.add_patch(circle)
        
        # Use gene symbol with gray color (rotated 45 degrees)
        label = data["gene_labels"].get(gene, gene)
        ax.text(x, y, label,
               ha="center", va="center", fontsize=16, rotation=45,
               color="black", weight="normal", zorder=4)
    
    # Draw drug nodes (use drug_id as label)
    for drug in data["drugs"]:
        x, y = positions[("drug", drug)]
        circle = plt.Circle((x, y), drug_size, color=drug_color,
                           ec="white", linewidth=1.5, zorder=3)
        ax.add_patch(circle)
        
        # Use drug ID instead of name
        label = drug
        
        # Shift label left to align with rotated text
        label_x_offset = -0.7
        ax.text(x + label_x_offset, y - drug_size - 0.08, label,
               ha="center", va="top", fontsize=16, rotation=45,
               color="black", weight="normal", zorder=4)
    
    # Set axis limits and style
    ax.set_xlim(-2, 52)
    ax.set_ylim(-2.5, 25.5)
    ax.set_aspect("equal")
    ax.axis("off")
    
    plt.tight_layout()
    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)
    fig.savefig(OUTPUT_PNG, dpi=300, bbox_inches="tight")
    plt.close(fig)
    
    print(f"\nSaved diagram to: {OUTPUT_PNG}")


def main():
    print("\n=== Creating Tripartite Network Diagram ===\n")
    
    data = load_and_prepare_data()
    create_tripartite_diagram(data)
    
    print("\nDone!")


if __name__ == "__main__":
    main()

