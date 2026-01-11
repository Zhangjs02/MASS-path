#!/usr/bin/env python3

import os
import math
import csv
import argparse
from typing import Dict, Set, List, Tuple, Optional

import matplotlib.pyplot as plt
import networkx as nx
import numpy as np
import pandas as pd
from matplotlib.collections import LineCollection

# Typography
plt.rcParams["font.family"] = "Arial"
plt.rcParams["font.sans-serif"] = ["Arial"]

# Color map for pathway types
TYPE_COLORS = {
	"Metabolism": "#FF0000",
	"Genetic Information Processing": "#FFFF00",
	"Environmental Information Processing": "#00FF00",
	"Cellular Processes": "#00FFFF",
	"Organismal Systems": "#0000FF",
	"Human Diseases": "#FF00FF",
	# Note: Drug Development not included as no pathways of this type exist in the network
}


# --- helpers ---

def _hex_to_rgb01(h: str) -> Tuple[float, float, float]:
	h = h.lstrip('#')
	return tuple(int(h[i:i+2], 16) / 255.0 for i in (0, 2, 4))


def load_relations(csv_path: str) -> Dict[str, Set[str]]:
	edges: Dict[str, Set[str]] = {}
	with open(csv_path, newline="", encoding="utf-8") as f:
		rdr = csv.DictReader(f)
		for row in rdr:
			pid = row.get("pathway_id", "").strip()
			rels = row.get("related_pathways", "").strip()
			if not pid:
				continue
			edges.setdefault(pid, set())
			if rels:
				# split by ';' and ','
				cand = [x.strip() for part in rels.split(";") for x in part.split(",")]
				for r in cand:
					if r:
						edges[pid].add(r)
	return edges


def load_types(txt_path: str) -> Dict[str, str]:
	types: Dict[str, str] = {}
	with open(txt_path, newline="", encoding="utf-8") as f:
		# skip header line if present
		first = f.readline()
		for line in f:
			parts = line.strip().split("\t")
			if len(parts) >= 2:
				pid, pt = parts[0].strip(), parts[1].strip()
				if pid:
					types[pid] = pt
	return types


def load_top_pathways(rank_csv: str, top_k: int) -> Set[str]:
	"""Load top-K pathway IDs from rank.csv based on Rank column."""
	df = pd.read_csv(rank_csv)
	if 'Rank' not in df.columns or 'Feature_Name' not in df.columns:
		print(f"Warning: rank.csv missing required columns. Found: {df.columns.tolist()}")
		return set()
	# Sort by Rank and take top-K
	top_df = df.nsmallest(top_k, 'Rank')
	return set(top_df['Feature_Name'].astype(str))


def is_top_edge(u: str, v: str, top_pathways: Set[str]) -> bool:
	"""Check if edge connects two top pathways."""
	return u in top_pathways and v in top_pathways


def build_graph(relations: Dict[str, Set[str]]) -> nx.Graph:
	g = nx.Graph()
	# add all nodes seen
	for a, bs in relations.items():
		g.add_node(a)
		for b in bs:
			g.add_node(b)
	# undirected edges
	for a, bs in relations.items():
		for b in bs:
			if a == b:
				continue
			g.add_edge(a, b)
	return g


def _deterministic_random_sector_points(nodes: List[str], theta0: float, theta1: float, radius: float) -> List[Tuple[float, float]]:
	"""Generate deterministic random positions within angular sector, no overlaps."""
	n = len(nodes)
	if n == 0:
		return []
	
	# Sort nodes by ID for deterministic ordering
	sorted_nodes = sorted(nodes)
	positions = []
	
	# Create deterministic hash function that's stable across runs
	def stable_hash(s: str) -> int:
		"""Stable hash function that produces same result across runs."""
		h = 0
		for char in s:
			h = (h * 31 + ord(char)) % (2**31)
		return h
	
	# Minimum distance to avoid overlaps (based on node size)
	min_dist = 0.35
	
	for i, node in enumerate(sorted_nodes):
		# Create deterministic seed from node ID
		node_seed = stable_hash(node + "fixed_salt_12345") % 1000000
		rng = np.random.default_rng(node_seed)
		
		# Try to find non-overlapping position
		max_attempts = 100
		found_position = False
		
		for attempt in range(max_attempts):
			# Random radius - uniform distribution in disk
			r_uniform = rng.uniform(0.0, 1.0)
			r = radius * math.sqrt(r_uniform)
			
			# Random angle within sector
			sector_span = theta1 - theta0
			angle_uniform = rng.uniform(0.0, 1.0)
			theta = theta0 + sector_span * angle_uniform
			
			x = r * math.cos(theta)
			y = r * math.sin(theta)
			
			# Check for overlaps with existing positions
			overlap = False
			for px, py in positions:
				dist = math.sqrt((x - px)**2 + (y - py)**2)
				if dist < min_dist:
					overlap = True
					break
			
			if not overlap:
				positions.append((x, y))
				found_position = True
				break
		
		# If no non-overlapping position found, use fallback
		if not found_position:
			# Place on a spiral pattern as fallback
			spiral_r = radius * (0.3 + 0.6 * i / n)
			spiral_theta = theta0 + (theta1 - theta0) * (i / n) + 0.5 * i
			x = spiral_r * math.cos(spiral_theta)
			y = spiral_r * math.sin(spiral_theta)
			positions.append((x, y))
	
	return positions


def spherical_with_outer_isolates(g: nx.Graph, types: Dict[str, str], radius_inner: float = 5.3, radius_outer: float = 5.8) -> Dict[str, Tuple[float, float]]:
	"""Generate fully deterministic positions clustered by type."""
	isolates = list(nx.isolates(g))
	connected = [n for n in g.nodes if n not in isolates]
	pos: Dict[str, Tuple[float, float]] = {}
	
	# Handle connected nodes
	if connected:
		# Group nodes by type
		from collections import defaultdict
		groups: Dict[str, List[str]] = defaultdict(list)
		for n in connected:
			groups[types.get(n, "Human Diseases")].append(n)
		
		# Order types deterministically
		type_list = sorted(groups.keys())
		counts = [len(groups[t]) for t in type_list]
		total = sum(counts)
		gap = 0.06  # rad gap between sectors
		angle = 0.0
		
		for t, cnt in zip(type_list, counts):
			if cnt == 0:
				continue
			span = max(0.2, 2 * math.pi * (cnt / total) - gap)
			t0 = angle
			t1 = angle + span
			angle = t1 + gap
			
			# Generate deterministic random positions for this type
			pts = _deterministic_random_sector_points(groups[t], t0, t1, radius_inner)
			
			for n, (x, y) in zip(sorted(groups[t]), pts):
				pos[n] = (x, y)
	
	# Handle isolates - place them closer to main circle
	if isolates:
		n_iso = len(isolates)
		sorted_isolates = sorted(isolates)
		# Place isolates closer to main circle (between inner and outer radius)
		isolate_radius = radius_inner + 0.3 * (radius_outer - radius_inner)  # 30% of the gap
		for i, n in enumerate(sorted_isolates):
			angle = 2 * math.pi * i / max(n_iso, 1)
			x = isolate_radius * math.cos(angle)
			y = isolate_radius * math.sin(angle)
			pos[n] = (x, y)
	
	return pos


def draw_network(g: nx.Graph, types: Dict[str, str], outdir: str, top_pathways: Optional[Set[str]] = None) -> None:
	os.makedirs(outdir, exist_ok=True)
	# Keep all isolates in top-k mode, remove all isolates in normal mode
	g2 = g.copy()
	isolates = list(nx.isolates(g2))
	
	if top_pathways is not None:
		# Keep all isolates in top-k mode to show complete network structure
		top_k_isolates = [node for node in isolates if node in top_pathways]
		print(f"Total isolates: {len(isolates)}, Top-k isolates: {len(top_k_isolates)}")
	else:
		# Remove all isolates in normal mode
		g2.remove_nodes_from(isolates)
	
	if g2.number_of_nodes() == 0:
		return
	pos = spherical_with_outer_isolates(g2, types)
	# Node colors and sizes
	node_colors: List[str] = []
	node_sizes: List[float] = []
	deg_dict = dict(g2.degree())
	if deg_dict:
		deg_vals = np.array(list(deg_dict.values()), dtype=float)
		d_min, d_max = deg_vals.min(), deg_vals.max()
		def scale_size(d: float) -> float:
			if d_max == d_min:
				return 360.0  # 720.0 * 0.5
			norm = max(0.0, (d - d_min) / (d_max - d_min)) ** 1.5
			return 160.0 + norm * (1120.0 - 160.0)  # (320.0 * 0.5) + norm * ((2240.0 * 0.5) - (320.0 * 0.5))
	else:
		def scale_size(d: float) -> float:
			return 360.0  # 720.0 * 0.5
	for n in g2.nodes:
		tp = types.get(n, "Human Diseases")
		# Gray out non-top nodes
		if top_pathways is not None and n not in top_pathways:
			node_colors.append("#888888")
		else:
			node_colors.append(TYPE_COLORS.get(tp, "#999999"))
		node_sizes.append(scale_size(float(deg_dict.get(n, 0))))
	
	# Draw edges: highlight top pathways, gray out others
	plt.figure(figsize=(16, 16), dpi=220)
	ax = plt.gca()
	
	# Separate edges by highlighting status
	both_top_same_segments = []
	both_top_same_colors = []
	both_top_cross_edges = []  # Store cross-type top edges for later
	one_top_segments = []
	gray_segments = []
	
	for u, v in g2.edges():
		tu = types.get(u, "Human Diseases")
		tv = types.get(v, "Human Diseases")
		p0 = pos[u]
		p1 = pos[v]
		
		if top_pathways is None:
			# No filtering mode - use original logic
			if tu == tv:
				both_top_same_segments.append([p0, p1])
				both_top_same_colors.append(TYPE_COLORS.get(tu, "#bbbbbb"))
			else:
				# Store cross-type edge for later drawing
				both_top_cross_edges.append((u, v, p0, p1, tu, tv))
		else:
			# Top-K mode with three edge types
			u_is_top = u in top_pathways
			v_is_top = v in top_pathways
			
			if u_is_top and v_is_top:
				# Both endpoints are top pathways - use original coloring
				if tu == tv:
					both_top_same_segments.append([p0, p1])
					both_top_same_colors.append(TYPE_COLORS.get(tu, "#bbbbbb"))
				else:
					# Store cross-type edge info for later drawing
					both_top_cross_edges.append((u, v, p0, p1, tu, tv))
			elif u_is_top or v_is_top:
				# Only one endpoint is top pathway - use dark gray
				one_top_segments.append([p0, p1])
			else:
				# All other edges - very light gray (background network)
				gray_segments.append([p0, p1])
	
	# Draw in order: background first, important edges last (top layer)
	
	# 1. Draw all background edges (very light gray) - bottom layer
	if gray_segments:
		lc_gray = LineCollection(gray_segments, colors=["#e0e0e0"], linewidths=0.8, alpha=1.0)
		ax.add_collection(lc_gray)
	
	# 2. Draw one-top edges (medium gray) - middle layer  
	if one_top_segments:
		lc_one_top = LineCollection(one_top_segments, colors=["#808080"], linewidths=1.2, alpha=1.0)
		ax.add_collection(lc_one_top)
	
	# 3. Draw both-top same-type edges (full color) - top layer
	if both_top_same_segments:
		lc_same = LineCollection(both_top_same_segments, colors=both_top_same_colors, linewidths=4, alpha=0.9)
		ax.add_collection(lc_same)
	
	# 4. Draw both-top cross-type edges (gradient) - top layer
	for u, v, p0, p1, tu, tv in both_top_cross_edges:
		rgb0 = np.array(_hex_to_rgb01(TYPE_COLORS.get(tu, "#999999")))
		rgb1 = np.array(_hex_to_rgb01(TYPE_COLORS.get(tv, "#999999")))
		steps = 20
		xs = np.linspace(p0[0], p1[0], steps)
		ys = np.linspace(p0[1], p1[1], steps)
		for i in range(steps - 1):
			seg = [(xs[i], ys[i]), (xs[i + 1], ys[i + 1])]
			c = tuple(rgb0 * (1 - i / (steps - 1)) + rgb1 * (i / (steps - 1)))
			lc = LineCollection([seg], colors=[c], linewidths=4, alpha=0.9)
			ax.add_collection(lc)
	
	# Draw nodes with transparency for non-top
	if top_pathways is not None:
		# Split nodes into top and non-top
		top_nodes = [n for n in g2.nodes if n in top_pathways]
		non_top_nodes = [n for n in g2.nodes if n not in top_pathways]
		top_colors = [TYPE_COLORS.get(types.get(n, "Human Diseases"), "#999999") for n in top_nodes]
		top_sizes = [scale_size(float(deg_dict.get(n, 0))) for n in top_nodes]
		# Use type colors for non-top nodes too, to preserve identity information
		non_top_colors = [TYPE_COLORS.get(types.get(n, "Human Diseases"), "#999999") for n in non_top_nodes]
		non_top_sizes = [scale_size(float(deg_dict.get(n, 0))) for n in non_top_nodes]
		
		# Draw non-top nodes with high transparency and colored outline to show identity
		if non_top_nodes:
			nx.draw_networkx_nodes(g2, pos, nodelist=non_top_nodes, node_color=non_top_colors, 
								 node_size=non_top_sizes, linewidths=2.0, edgecolors=non_top_colors, alpha=0.25)
		# Draw top nodes normally with black outline
		if top_nodes:
			nx.draw_networkx_nodes(g2, pos, nodelist=top_nodes, node_color=top_colors, 
								 node_size=top_sizes, linewidths=1.5, edgecolors="#000000")
			
			# Add labels for top nodes
			labels = {n: n for n in top_nodes}
			nx.draw_networkx_labels(g2, pos, labels, font_size=12, font_family='Arial', 
								   font_weight='bold', font_color='black',
								   bbox=dict(boxstyle='round,pad=0.3', facecolor='white', 
											edgecolor='none', alpha=0.7))
	else:
		# Draw all nodes normally
		nx.draw_networkx_nodes(g2, pos, node_color=node_colors, node_size=node_sizes, linewidths=0.6, edgecolors="#1a1a1a")
	plt.axis("off")
	plt.tight_layout()
	plt.savefig(os.path.join(outdir, "pathway_network.png"))
	plt.savefig(os.path.join(outdir, "pathway_network.svg"))
	plt.close()


def main() -> None:
	parser = argparse.ArgumentParser(description="Generate pathway network with optional top-K highlighting")
	parser.add_argument("--top_k", type=int, default=None, help="Highlight top-K pathways from rank.csv")
	parser.add_argument("--rank_csv", type=str, default="../Data/output/manifest_results.csv", help="Path to rank CSV file")
	parser.add_argument("--rel_csv", type=str, default="../Data/kegg_pathway_relations_summary.csv", help="Path to relations CSV")
	parser.add_argument("--type_txt", type=str, default="../Data/pathway_types.txt", help="Path to types file")
	parser.add_argument("--outdir", type=str, default="../Data/output/pathway_network", help="Output directory")
	args = parser.parse_args()
	
	relations = load_relations(args.rel_csv)
	types = load_types(args.type_txt)
	g = build_graph(relations)
	
	top_pathways = None
	if args.top_k is not None:
		top_pathways = load_top_pathways(args.rank_csv, args.top_k)
		print(f"Highlighting top-{args.top_k} pathways: {len(top_pathways)} found")
	
	draw_network(g, types, args.outdir, top_pathways)
	suffix = f"_top{args.top_k}" if args.top_k else ""
	print(f"Saved to {args.outdir}{suffix}")


if __name__ == "__main__":
	main()