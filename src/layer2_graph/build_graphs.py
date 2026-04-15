# =============================================================================
# Layer 2 — Causal Graph Construction
# =============================================================================
#
# Purpose:
#   Take structured event tuples from Layer 1 (extracted_events.json) and build
#   typed heterogeneous graphs. Each report becomes a graph with:
#     - Nodes: ACTOR, SYSTEM, TRIGGER, PHASE, OUTCOME entities
#     - Edges: typed causal/relational edges between entities
#
# Output:
#   - outputs/graphs/graphs.pkl           — list of GraphData objects (one per report)
#   - outputs/graphs/graph_stats.json     — summary statistics
#   - outputs/graphs/graph_visualizations/ — PNG drawings of sample graphs
#
# Graph Schema:
#   ACTOR   --[performed]--> TRIGGER
#   ACTOR   --[operated]-->  SYSTEM
#   SYSTEM  --[involved_in]--> TRIGGER
#   TRIGGER --[led_to]-->    OUTCOME
#   PHASE   --[context_of]--> TRIGGER
#   PHASE   --[context_of]--> OUTCOME
#
# Edge types are assigned by which entity-type pairs are connected.
# If no TRIGGER: ACTOR/SYSTEM connect directly to OUTCOME.
#
# Node features:
#   - entity_type: one-hot (5 types)
#   - text_embedding: simple TF-IDF-like token hash (expandable to BERT later)
#
# =============================================================================

import os
import json
import pickle
import hashlib
import re
from collections import defaultdict
from dataclasses import dataclass, field
from typing import List, Dict, Tuple, Optional

import numpy as np
import networkx as nx
import matplotlib
matplotlib.use("Agg")  # Non-interactive backend for saving PNGs
import matplotlib.pyplot as plt
import matplotlib.patches as mpatches

# ── Paths ─────────────────────────────────────────────────────────────────────
ROOT = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
EXTRACTED_EVENTS_PATH = os.path.join(ROOT, "outputs", "predictions", "extracted_events.json")
OUTPUT_DIR = os.path.join(ROOT, "outputs", "graphs")
VIZ_DIR = os.path.join(OUTPUT_DIR, "graph_visualizations")

os.makedirs(OUTPUT_DIR, exist_ok=True)
os.makedirs(VIZ_DIR, exist_ok=True)

# ── Entity type config ────────────────────────────────────────────────────────
ENTITY_TYPES = ["ACTOR", "SYSTEM", "TRIGGER", "PHASE", "OUTCOME"]
ENTITY_TYPE_IDX = {t: i for i, t in enumerate(ENTITY_TYPES)}
NUM_ENTITY_TYPES = len(ENTITY_TYPES)

# Node colors for visualization
ENTITY_COLORS = {
    "ACTOR":   "#4A90D9",   # blue
    "SYSTEM":  "#7ED321",   # green
    "TRIGGER": "#FF6B6B",   # red
    "PHASE":   "#F5A623",   # orange
    "OUTCOME": "#9B59B6",   # purple
}

# Edge types: (source_type, target_type) -> edge_label
EDGE_SCHEMA = {
    ("ACTOR",   "TRIGGER"): "performed",
    ("ACTOR",   "SYSTEM"):  "operated",
    ("SYSTEM",  "TRIGGER"): "involved_in",
    ("TRIGGER", "OUTCOME"): "led_to",
    ("PHASE",   "TRIGGER"): "context_of",
    ("PHASE",   "OUTCOME"): "context_of",
    ("ACTOR",   "OUTCOME"): "contributed_to",   # fallback: no trigger
    ("SYSTEM",  "OUTCOME"): "contributed_to",   # fallback: no trigger
}

EDGE_TYPE_LIST = list(set(EDGE_SCHEMA.values()))
EDGE_TYPE_IDX = {t: i for i, t in enumerate(EDGE_TYPE_LIST)}
NUM_EDGE_TYPES = len(EDGE_TYPE_LIST)


# =============================================================================
# Data classes
# =============================================================================

@dataclass
class GraphData:
    """Holds a single report's graph in a format ready for PyTorch Geometric."""
    report_idx: int
    narrative_preview: str
    adrep_label: Optional[str]          # ground-truth label (if available)

    # Node-level data
    node_ids: List[str]                 # unique node identifier strings
    node_types: List[str]               # entity type for each node
    node_texts: List[str]               # raw text of each node
    node_features: np.ndarray           # shape (N, NUM_ENTITY_TYPES) one-hot

    # Edge-level data
    edge_index: np.ndarray              # shape (2, E) — [source_indices, target_indices]
    edge_types: List[str]               # edge type label for each edge
    edge_type_ids: np.ndarray           # shape (E,) — integer edge type ids

    # Graph-level metadata
    num_nodes: int
    num_edges: int
    networkx_graph: nx.DiGraph = field(repr=False)


# =============================================================================
# Helper functions
# =============================================================================

def _normalise_entity(text: str) -> str:
    """Clean and normalise entity text for deduplication."""
    return re.sub(r"\s+", " ", text.strip().lower())


def _is_valid_entity(text: str, entity_type: str) -> bool:
    """Filter out placeholder / low-quality extractions."""
    INVALID = {
        "not identified", "unknown", "not explicitly stated",
        "not stated", "n/a", "", "the", "a", "an", "la", "pa", "im",
    }
    norm = _normalise_entity(text)
    if norm in INVALID:
        return False
    if len(norm) <= 1:
        return False
    # For triggers, also drop very generic phrases that are just noise
    if entity_type == "TRIGGER" and norm in {"an", "closed", "evacuation", "lack", "under"}:
        return False
    return True


def _make_node_id(entity_type: str, text: str) -> str:
    """Create a stable unique node ID from type + text."""
    h = hashlib.md5(f"{entity_type}::{text.lower().strip()}".encode()).hexdigest()[:6]
    return f"{entity_type}_{h}"


def _one_hot_entity(entity_type: str) -> np.ndarray:
    """Return a one-hot vector for entity type."""
    vec = np.zeros(NUM_ENTITY_TYPES, dtype=np.float32)
    idx = ENTITY_TYPE_IDX.get(entity_type, -1)
    if idx >= 0:
        vec[idx] = 1.0
    return vec


def _parse_phase(phase_str: str) -> List[str]:
    """Handle multi-phase strings like 'Takeoff / Launch; Initial Climb'."""
    if not phase_str or phase_str.strip().lower() in ("unknown", "", "nan"):
        return []
    # Split on semicolon
    parts = [p.strip() for p in phase_str.split(";")]
    # Keep only the first phase to avoid explosion of nodes
    return [parts[0]] if parts else []


def _parse_outcome(outcome) -> List[str]:
    """Outcomes might be a list or a stringified list."""
    if isinstance(outcome, list):
        return outcome
    if isinstance(outcome, str):
        try:
            parsed = json.loads(outcome)
            if isinstance(parsed, list):
                return parsed
        except Exception:
            pass
        return [outcome]
    return []


# =============================================================================
# Core graph builder
# =============================================================================

def build_graph_for_report(event: Dict, adrep_label: Optional[str] = None) -> Optional[GraphData]:
    """
    Convert a single extracted event dict into a GraphData object.

    Args:
        event: dict from extracted_events.json
        adrep_label: ADREP category label (if known)

    Returns:
        GraphData or None if the graph would be empty
    """
    G = nx.DiGraph()

    # ── Collect valid entity nodes ─────────────────────────────────────────
    entity_buckets: Dict[str, List[str]] = defaultdict(list)

    # ACTOR, SYSTEM, TRIGGER: may be lists
    for etype in ["ACTOR", "SYSTEM", "TRIGGER"]:
        raw = event.get(etype, [])
        if isinstance(raw, str):
            raw = [raw]
        for text in raw:
            if _is_valid_entity(text, etype):
                entity_buckets[etype].append(text)

    # PHASE: parse from string
    for text in _parse_phase(str(event.get("PHASE", ""))):
        if _is_valid_entity(text, "PHASE"):
            entity_buckets["PHASE"].append(text)

    # OUTCOME: parse from list/string
    for text in _parse_outcome(event.get("OUTCOME", [])):
        if _is_valid_entity(text, "OUTCOME"):
            entity_buckets["OUTCOME"].append(text)
            break  # Keep only the primary outcome to control graph size

    # ── Add nodes ─────────────────────────────────────────────────────────
    for etype, texts in entity_buckets.items():
        for text in texts:
            node_id = _make_node_id(etype, text)
            G.add_node(node_id, entity_type=etype, text=text)

    if G.number_of_nodes() == 0:
        return None

    # ── Add typed edges ────────────────────────────────────────────────────
    def _add_edges(src_type, dst_type):
        """Add edges from all src_type nodes to all dst_type nodes."""
        src_nodes = [n for n, d in G.nodes(data=True) if d["entity_type"] == src_type]
        dst_nodes = [n for n, d in G.nodes(data=True) if d["entity_type"] == dst_type]
        edge_label = EDGE_SCHEMA.get((src_type, dst_type), "related_to")
        for s in src_nodes:
            for t in dst_nodes:
                G.add_edge(s, t, edge_type=edge_label)

    has_trigger = len(entity_buckets.get("TRIGGER", [])) > 0

    if has_trigger:
        _add_edges("ACTOR",   "TRIGGER")
        _add_edges("ACTOR",   "SYSTEM")
        _add_edges("SYSTEM",  "TRIGGER")
        _add_edges("TRIGGER", "OUTCOME")
        _add_edges("PHASE",   "TRIGGER")
        _add_edges("PHASE",   "OUTCOME")
    else:
        # No trigger: connect actors/systems directly to outcome
        _add_edges("ACTOR",  "OUTCOME")
        _add_edges("SYSTEM", "OUTCOME")
        _add_edges("ACTOR",  "SYSTEM")
        _add_edges("PHASE",  "OUTCOME")

    # ── Serialise to arrays ────────────────────────────────────────────────
    node_list = list(G.nodes())
    node_idx = {n: i for i, n in enumerate(node_list)}

    node_types = [G.nodes[n]["entity_type"] for n in node_list]
    node_texts = [G.nodes[n]["text"] for n in node_list]
    node_features = np.stack([_one_hot_entity(t) for t in node_types])

    edges = list(G.edges(data=True))
    if edges:
        src_idx = np.array([node_idx[e[0]] for e in edges], dtype=np.int64)
        dst_idx = np.array([node_idx[e[1]] for e in edges], dtype=np.int64)
        edge_index = np.stack([src_idx, dst_idx])
        edge_types_list = [e[2].get("edge_type", "related_to") for e in edges]
        edge_type_ids = np.array(
            [EDGE_TYPE_IDX.get(et, 0) for et in edge_types_list], dtype=np.int64
        )
    else:
        edge_index = np.zeros((2, 0), dtype=np.int64)
        edge_types_list = []
        edge_type_ids = np.zeros(0, dtype=np.int64)

    return GraphData(
        report_idx=event["report_idx"],
        narrative_preview=event.get("narrative_preview", ""),
        adrep_label=adrep_label,
        node_ids=node_list,
        node_types=node_types,
        node_texts=node_texts,
        node_features=node_features,
        edge_index=edge_index,
        edge_types=edge_types_list,
        edge_type_ids=edge_type_ids,
        num_nodes=G.number_of_nodes(),
        num_edges=G.number_of_edges(),
        networkx_graph=G,
    )


# =============================================================================
# Visualisation
# =============================================================================

def visualise_graph(graph_data: GraphData, save_path: str, title: str = ""):
    """Draw a report's causal graph and save to PNG."""
    G = graph_data.networkx_graph
    if G.number_of_nodes() == 0:
        return

    fig, ax = plt.subplots(figsize=(12, 7))
    ax.set_facecolor("#0f0f1a")
    fig.patch.set_facecolor("#0f0f1a")

    # Layout
    try:
        pos = nx.kamada_kawai_layout(G)
    except Exception:
        pos = nx.spring_layout(G, seed=42)

    # Draw edges
    edge_labels = nx.get_edge_attributes(G, "edge_type")
    unique_edge_types = list(set(edge_labels.values()))
    edge_colors = plt.cm.Set2(np.linspace(0, 1, max(len(unique_edge_types), 1)))
    edge_color_map = {et: edge_colors[i] for i, et in enumerate(unique_edge_types)}

    for (u, v), etype in edge_labels.items():
        nx.draw_networkx_edges(
            G, pos, edgelist=[(u, v)],
            edge_color=[edge_color_map[etype]],
            arrows=True, arrowsize=20,
            width=2, alpha=0.8,
            connectionstyle="arc3,rad=0.1",
            ax=ax,
        )

    # Draw nodes
    node_colors = [ENTITY_COLORS.get(G.nodes[n]["entity_type"], "#aaaaaa") for n in G.nodes()]
    nx.draw_networkx_nodes(G, pos, node_color=node_colors, node_size=800, alpha=0.95, ax=ax)

    # Draw labels (truncate long text)
    labels = {n: G.nodes[n]["text"][:18] + "…" if len(G.nodes[n]["text"]) > 18
              else G.nodes[n]["text"]
              for n in G.nodes()}
    nx.draw_networkx_labels(G, pos, labels, font_size=8, font_color="white",
                            font_weight="bold", ax=ax)

    # Draw edge labels
    short_edge_labels = {(u, v): et for (u, v), et in edge_labels.items()}
    nx.draw_networkx_edge_labels(
        G, pos, short_edge_labels,
        font_size=7, font_color="#dddddd", ax=ax,
    )

    # Legend: node types
    node_legend = [
        mpatches.Patch(color=ENTITY_COLORS[t], label=t)
        for t in ENTITY_TYPES if t in [G.nodes[n]["entity_type"] for n in G.nodes()]
    ]
    ax.legend(handles=node_legend, loc="upper left",
              framealpha=0.3, facecolor="#1a1a2e", labelcolor="white",
              fontsize=9)

    # Title
    preview = graph_data.narrative_preview[:80] + "…" if len(
        graph_data.narrative_preview) > 80 else graph_data.narrative_preview
    ax.set_title(
        f"Report #{graph_data.report_idx}  |  {graph_data.num_nodes} nodes  |  "
        f"{graph_data.num_edges} edges\n{preview}",
        color="white", fontsize=10, pad=12
    )
    ax.axis("off")

    plt.tight_layout()
    plt.savefig(save_path, dpi=120, bbox_inches="tight", facecolor=fig.get_facecolor())
    plt.close(fig)


# =============================================================================
# Main pipeline
# =============================================================================

def build_all_graphs(
    events_path: str = EXTRACTED_EVENTS_PATH,
    labels_csv_path: Optional[str] = None,
    visualise_n: int = 5,
) -> List[GraphData]:
    """
    Build graphs for all reports in extracted_events.json.

    Args:
        events_path:     Path to extracted_events.json from Layer 1
        labels_csv_path: Optional path to data_aviation.csv to attach ADREP labels
        visualise_n:     Number of sample graphs to visualise and save as PNGs

    Returns:
        List of GraphData objects
    """
    print("=" * 65)
    print("  Layer 2 — Causal Graph Construction")
    print("=" * 65)

    # ── Load extracted events ──────────────────────────────────────────────
    print(f"\n📂 Loading events from: {events_path}")
    with open(events_path, "r", encoding="utf-8") as f:
        events = json.load(f)
    print(f"   {len(events)} reports loaded")

    # ── Optionally load ADREP labels ───────────────────────────────────────
    label_map: Dict[int, str] = {}
    if labels_csv_path and os.path.exists(labels_csv_path):
        import pandas as pd
        df = pd.read_csv(labels_csv_path)
        if "final_category" in df.columns:
            label_map = dict(enumerate(df["final_category"].fillna("UNKNOWN")))
            print(f"   Labels loaded for {len(label_map)} reports")

    # ── Build graphs ───────────────────────────────────────────────────────
    print("\n🔨 Building graphs...")
    graphs: List[GraphData] = []
    skipped = 0
    node_counts = []
    edge_counts = []

    for event in events:
        label = label_map.get(event["report_idx"])
        g = build_graph_for_report(event, adrep_label=label)
        if g is None:
            skipped += 1
            continue
        graphs.append(g)
        node_counts.append(g.num_nodes)
        edge_counts.append(g.num_edges)

    print(f"   ✅ Built {len(graphs)} graphs  |  ⏭️  Skipped {skipped} empty reports")

    # ── Statistics ─────────────────────────────────────────────────────────
    if graphs:
        stats = {
            "total_graphs": len(graphs),
            "skipped_empty": skipped,
            "nodes": {
                "min": int(np.min(node_counts)),
                "max": int(np.max(node_counts)),
                "mean": float(np.mean(node_counts)),
                "median": float(np.median(node_counts)),
            },
            "edges": {
                "min": int(np.min(edge_counts)),
                "max": int(np.max(edge_counts)),
                "mean": float(np.mean(edge_counts)),
                "median": float(np.median(edge_counts)),
            },
            "entity_type_distribution": defaultdict(int),
            "edge_type_distribution": defaultdict(int),
        }

        for g in graphs:
            for t in g.node_types:
                stats["entity_type_distribution"][t] += 1
            for et in g.edge_types:
                stats["edge_type_distribution"][et] += 1

        # Convert defaultdicts to plain dicts for JSON
        stats["entity_type_distribution"] = dict(stats["entity_type_distribution"])
        stats["edge_type_distribution"] = dict(stats["edge_type_distribution"])

        print("\n📊 Graph Statistics:")
        print(f"   Nodes — min: {stats['nodes']['min']}, max: {stats['nodes']['max']}, "
              f"mean: {stats['nodes']['mean']:.1f}")
        print(f"   Edges — min: {stats['edges']['min']}, max: {stats['edges']['max']}, "
              f"mean: {stats['edges']['mean']:.1f}")
        print(f"\n   Entity type distribution:")
        for etype, count in sorted(stats["entity_type_distribution"].items()):
            print(f"     {etype:<10}: {count}")
        print(f"\n   Edge type distribution:")
        for etype, count in sorted(stats["edge_type_distribution"].items()):
            print(f"     {etype:<20}: {count}")

        # Save stats
        stats_path = os.path.join(OUTPUT_DIR, "graph_stats.json")
        with open(stats_path, "w") as f:
            json.dump(stats, f, indent=2)
        print(f"\n💾 Stats saved to: {stats_path}")

    # ── Save graphs ────────────────────────────────────────────────────────
    graphs_path = os.path.join(OUTPUT_DIR, "graphs.pkl")
    with open(graphs_path, "wb") as f:
        pickle.dump(graphs, f)
    print(f"💾 Graphs saved to: {graphs_path}")

    # ── Visualise sample graphs ────────────────────────────────────────────
    if visualise_n > 0:
        print(f"\n🖼️  Visualising {visualise_n} sample graphs...")
        # Pick graphs with interesting structures (>= 3 nodes and > 0 edges)
        interesting = [g for g in graphs if g.num_nodes >= 3 and g.num_edges > 0]
        sample = interesting[:visualise_n] if len(interesting) >= visualise_n else interesting

        for i, g in enumerate(sample):
            save_path = os.path.join(VIZ_DIR, f"graph_report_{g.report_idx:03d}.png")
            visualise_graph(g, save_path)
            print(f"   Saved: graph_report_{g.report_idx:03d}.png "
                  f"({g.num_nodes} nodes, {g.num_edges} edges)")

    print("\n✅ Layer 2 complete.")
    return graphs


# =============================================================================
# Entry point
# =============================================================================

if __name__ == "__main__":
    import argparse

    parser = argparse.ArgumentParser(description="Layer 2: Causal Graph Construction")
    parser.add_argument(
        "--events", default=EXTRACTED_EVENTS_PATH,
        help="Path to extracted_events.json from Layer 1"
    )
    parser.add_argument(
        "--labels", default=None,
        help="Optional path to data_aviation.csv to attach ADREP labels"
    )
    parser.add_argument(
        "--visualise", type=int, default=8,
        help="Number of sample graphs to visualise (default: 8)"
    )
    args = parser.parse_args()

    graphs = build_all_graphs(
        events_path=args.events,
        labels_csv_path=args.labels,
        visualise_n=args.visualise,
    )
