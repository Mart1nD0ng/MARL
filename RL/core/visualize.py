from __future__ import annotations

from typing import Dict, Tuple, List, Set, Optional, Sequence
import math
import os
import random

try:
    import matplotlib.pyplot as plt
    from matplotlib.lines import Line2D
    from matplotlib.patches import Rectangle, Polygon
except Exception:  # pragma: no cover
    plt = None


def _save_figure(fig, out_path: str, dpi: int = 300, save_vector: bool = True, vector_format: str = 'eps', **kwargs):
    """Save figure in multiple formats (PNG + vector format).
    
    Args:
        fig: matplotlib figure
        out_path: Output path (should end with .png)
        dpi: DPI for raster output
        save_vector: Whether to also save vector format (PDF/EPS)
        vector_format: Vector format to use ('eps' or 'pdf'). EPS is more common for LaTeX.
        **kwargs: Additional kwargs for savefig
    """
    import warnings
    os.makedirs(os.path.dirname(out_path), exist_ok=True)
    
    # Save PNG with high DPI
    fig.savefig(out_path, dpi=dpi, **kwargs)
    
    # Save vector format (EPS for LaTeX compatibility)
    if save_vector:
        vector_path = out_path.rsplit('.', 1)[0] + '.' + vector_format
        # Suppress transparency warning for EPS (transparency will be rendered as opaque)
        with warnings.catch_warnings():
            warnings.filterwarnings('ignore', category=UserWarning)
            fig.savefig(vector_path, format=vector_format, **kwargs)


def _build_grid(bbox: Tuple[float, float, float, float] = (-300, -300, 300, 300), 
                stride: float = 100.0) -> Tuple[List[float], List[float]]:
    """Build grid lines for road visualization.
    
    Args:
        bbox: (min_x, min_y, max_x, max_y) region bounds
        stride: grid line spacing (city block size)
    """
    min_x, min_y, max_x, max_y = bbox
    
    # Generate grid lines within bbox
    xs = []
    curr = math.ceil(min_x / stride) * stride
    while curr <= max_x:
        xs.append(curr)
        curr += stride
    
    ys = []
    curr = math.ceil(min_y / stride) * stride
    while curr <= max_y:
        ys.append(curr)
        curr += stride
    
    return xs, ys


def _draw_roads(ax, xs: Sequence[float], ys: Sequence[float], road_width: float = 14.0) -> None:
    # Extend roads slightly beyond grid
    if not xs or not ys: return
    margin = 60.0
    y_min, y_max = min(ys) - margin, max(ys) + margin
    x_min, x_max = min(xs) - margin, max(xs) + margin
    road_w = max(4.0, float(road_width))
    lane_marker_color = "#f2f2f2"
    road_color = "#5a5a5a"
    for x in xs:
        rect = Rectangle((x - road_w / 2, y_min), road_w, y_max - y_min, color=road_color, alpha=0.92, zorder=0)
        ax.add_patch(rect)
        ax.plot([x, x], [y_min, y_max], color=lane_marker_color, linewidth=1.0, alpha=0.9, linestyle="--", zorder=1)
    for y in ys:
        rect = Rectangle((x_min, y - road_w / 2), x_max - x_min, road_w, color=road_color, alpha=0.92, zorder=0)
        ax.add_patch(rect)
        ax.plot([x_min, x_max], [y, y], color=lane_marker_color, linewidth=1.0, alpha=0.9, linestyle="--", zorder=1)


def _draw_buildings(ax, xs: Sequence[float], ys: Sequence[float], rng: random.Random, stride: float = 100.0) -> None:
    """Draw building blocks between road intersections."""
    spots = set()
    for x in xs:
        for y in ys:
            if rng.random() < 0.4:
                # Place in quadrant
                dx = rng.choice([-0.25 * stride, 0.25 * stride])
                dy = rng.choice([-0.25 * stride, 0.25 * stride])
                spot = (x + dx, y + dy)
                if spot in spots:
                    continue
                spots.add(spot)
    for (bx, by) in spots:
        size = 0.35 * stride
        rect = Rectangle(
            (bx - size/2, by - size/2),
            size,
            size,
            facecolor="#8ac2ff",
            edgecolor="#0b4f9c",
            linewidth=1.2,
            alpha=0.8,
            zorder=0.5
        )
        ax.add_patch(rect)


def _add_scale_bar(
    ax,
    bbox: Tuple[float, float, float, float],
    length_m: Optional[float] = None,
    label: Optional[str] = None,
) -> None:
    """Add a simple scale bar to the lower-left corner."""
    xmin, ymin, xmax, ymax = bbox
    w = xmax - xmin
    h = ymax - ymin
    if w <= 0 or h <= 0:
        return

    # Default scale bar length ~20% of width, rounded to nice values.
    if length_m is None or length_m <= 0:
        target = max(10.0, 0.2 * w)
        # Round to nearest 10m.
        length_m = max(10.0, round(target / 10.0) * 10.0)

    pad_x = 0.06 * w
    pad_y = 0.06 * h
    x0 = xmin + pad_x
    y0 = ymin + pad_y
    x1 = x0 + length_m

    ax.plot([x0, x1], [y0, y0], color="#111111", linewidth=3.0, solid_capstyle="butt", zorder=10)
    ax.plot([x0, x0], [y0 - 0.01 * h, y0 + 0.01 * h], color="#111111", linewidth=2.0, zorder=10)
    ax.plot([x1, x1], [y0 - 0.01 * h, y0 + 0.01 * h], color="#111111", linewidth=2.0, zorder=10)

    text = label or f"{int(length_m)} m"
    ax.text(
        (x0 + x1) / 2.0,
        y0 - 0.03 * h,
        text,
        ha="center",
        va="top",
        fontsize=10,
        color="#111111",
        zorder=10,
        bbox=dict(boxstyle="round,pad=0.2", facecolor="white", alpha=0.7),
    )


def _scatter_vehicle(
    ax,
    center: Tuple[float, float],
    heading: float,
    color: str,
) -> None:
    """Draw a vehicle as a single dot (node)."""
    cx, cy = center
    ax.scatter([cx], [cy], s=60, color=color, edgecolors="white", linewidths=1.0, zorder=4, alpha=0.9)


def plot_topology(graph: Dict, partitions: Dict[int, int], out_path: str, dpi: int = 160, 
                  max_nodes: Optional[int] = None, metrics: Optional[Dict] = None,
                  show_legend: bool = True, show_scale_bar: bool = True, show_stats: bool = True):
    """Plot physical topology with nodes positioned by their actual coordinates.
    
    Args:
        graph: Graph data dict with nodes, edges, node_pos, etc.
        partitions: Dict mapping node_id -> agent_id
        out_path: Output file path
        dpi: Figure DPI
        max_nodes: Max nodes to display (for large graphs)
        metrics: Optional metrics dict to overlay
        show_legend: Whether to show agent legend (default True)
        show_scale_bar: Whether to show scale bar and area info (default True)
        show_stats: Whether to show environment stats text (default True)
    """
    if plt is None:
        return
    nodes: List[int] = list(graph.get('nodes', []))
    node_pos: Dict[int, Tuple[float, float]] = graph.get('node_pos', {}) or {}
    edges_raw: Set[Tuple[int, int]] = set(graph.get('edges', set()))
    
    # Map Data
    buildings = list(graph.get('buildings', []))
    grid_stride = float(graph.get('grid_stride', 50.0))
    bbox = tuple(graph.get('bbox', (-120, -120, 120, 120)))
    
    if max_nodes is not None and max_nodes > 0 and len(nodes) > max_nodes:
        rng = random.Random(0)
        nodes = sorted(rng.sample(nodes, max_nodes))
        
    allowed = set(nodes)
    edges = {tuple(sorted((u, v))) for (u, v) in edges_raw if u in allowed and v in allowed}
    agents = sorted(set(partitions.values())) if partitions else [0]
    
    # Use standard palette
    colors_list = [
        "#E74C3C", "#F39C12", "#F1C40F", "#2ECC71", "#3498DB", "#9B59B6",
        "#1ABC9C", "#34495E", "#E67E22", "#95A5A6", "#D35400", "#7F8C8D"
    ]
    
    fig, ax = plt.subplots(figsize=(12, 10), dpi=dpi) # Slightly larger
    
    # Background: Reconstruct grid lines from bbox + stride
    # We need to compute xs/ys to match the logic in Environment
    xs = []
    curr = math.ceil(bbox[0] / grid_stride) * grid_stride
    while curr <= bbox[2]:
        xs.append(curr)
        curr += grid_stride
    ys = []
    curr = math.ceil(bbox[1] / grid_stride) * grid_stride
    while curr <= bbox[3]:
        ys.append(curr)
        curr += grid_stride
        
    road_width = float(graph.get('road_width', 14.0))
    _draw_roads(ax, xs, ys, road_width=road_width)
    
    # Draw actual buildings from env
    for (bx, by, w, h) in buildings:
        rect = Rectangle(
            (bx, by), # In env we stored bottom-left
            w,
            h,
            facecolor="#8ac2ff",
            edgecolor="#0b4f9c",
            linewidth=1.2,
            alpha=0.8,
            zorder=0.5
        )
        ax.add_patch(rect)

    # Connections
    for (u, v) in edges:
        x1, y1 = node_pos.get(u, (0.0, 0.0))
        x2, y2 = node_pos.get(v, (0.0, 0.0))
        # Determine if internal or cross
        au, av = partitions.get(u), partitions.get(v)
        color = '#cccccc'
        width = 1.0
        zorder = 1.5
        if au is not None and av is not None and au != av:
            color = '#7f8c8d' # darker grey for cross
            width = 1.5
            zorder = 2.0
            
        ax.plot([x1, x2], [y1, y2], color=color, linewidth=width, alpha=0.6, zorder=zorder)

    # Nodes
    for n in allowed:
        aid = partitions.get(n, 0)
        c = colors_list[aid % len(colors_list)]
        pos = node_pos.get(n, (0.0, 0.0))
        _scatter_vehicle(ax, pos, 0.0, c)
    
    # Draw cluster boundaries (convex hull) to show non-overlapping geographic territories
    # Also validate for overlaps
    overlap_detected = False
    try:
        from scipy.spatial import ConvexHull
        import numpy as np
        
        agent_hulls = {}  # Store hulls for overlap detection
        
        for a in agents:
            nodes_a = [n for n in allowed if partitions.get(n) == a]
            if len(nodes_a) >= 3:  # Need at least 3 points for convex hull
                points = np.array([node_pos[n] for n in nodes_a])
                try:
                    hull = ConvexHull(points)
                    color = colors_list[a % len(colors_list)]
                    
                    # Draw hull boundary lines
                    for simplex in hull.simplices:
                        ax.plot(points[simplex, 0], points[simplex, 1], 
                               color=color, linewidth=2.5, alpha=0.9, linestyle='-', zorder=3)
                    
                    # Fill region with semi-transparent color
                    vertices = points[hull.vertices]
                    vertices = np.vstack([vertices, vertices[0]])  # Close polygon
                    ax.fill(vertices[:, 0], vertices[:, 1], 
                           color=color, alpha=0.12, zorder=0.3)
                    
                    # Store hull for overlap detection
                    agent_hulls[a] = vertices[:-1]  # Remove duplicate closing vertex
                    
                except Exception:
                    pass  # Skip if hull computation fails
        
        # Check for overlaps between agent territories
        if len(agent_hulls) > 1:
            from matplotlib.path import Path
            agent_ids = sorted(agent_hulls.keys())
            for i, a1 in enumerate(agent_ids):
                for a2 in agent_ids[i+1:]:
                    hull1 = agent_hulls[a1]
                    hull2 = agent_hulls[a2]
                    
                    # Check if any vertex of hull1 is inside hull2 or vice versa
                    path1 = Path(hull1)
                    path2 = Path(hull2)
                    
                    if any(path1.contains_points(hull2)) or any(path2.contains_points(hull1)):
                        overlap_detected = True
                        print(f"[WARNING] Territory overlap detected between Agent {a1} and Agent {a2}")
                        
        if not overlap_detected and len(agent_hulls) > 1:
            print(f"[✓] No territory overlaps detected - {len(agent_hulls)} non-overlapping regions validated")
            
    except ImportError:
        pass  # scipy not available

    # Metrics Overlay
    if metrics:
        # Filter important metrics
        display_keys = {'episode', 'avg_return', 'success_proxy', 'delay_proxy', 'energy_proxy', 'm', 'rho_bar'}
        lines = []
        for k, v in metrics.items():
            if k in display_keys:
                val_str = f"{v:.4f}" if isinstance(v, float) else str(v)
                lines.append(f"{k}: {val_str}")
        if lines:
            text_str = '\n'.join(lines)
            props = dict(boxstyle='round', facecolor='white', alpha=0.85)
            ax.text(0.02, 0.98, text_str, transform=ax.transAxes, fontsize=11, fontfamily='monospace',
                    verticalalignment='top', bbox=props, zorder=10)

    # Legend
    if show_legend:
        legend_items = [
            Line2D([0], [0], marker="o", color="none", markerfacecolor=colors_list[i % len(colors_list)], 
                   markeredgecolor="white", markersize=10, label=f"Agent {a}")
            for i, a in enumerate(agents[:16]) # Limit legend size
        ]
        if legend_items:
            ax.legend(handles=legend_items, loc="upper right", ncol=2, fontsize=9, framealpha=0.9)
    
    ax.set_aspect("equal", "box")
    ax.axis('off') # Hide axes for cleaner look
    
    # Set bounds
    all_x = [p[0] for p in node_pos.values()]
    all_y = [p[1] for p in node_pos.values()]
    if all_x and all_y:
        margin = 30.0
        ax.set_xlim(min(all_x) - margin, max(all_x) + margin)
        ax.set_ylim(min(all_y) - margin, max(all_y) + margin)

    # Add scale bar and environment summary text
    if show_scale_bar:
        env_w = float(bbox[2] - bbox[0])
        env_h = float(bbox[3] - bbox[1])
        scale_len = float(grid_stride)
        _add_scale_bar(ax, bbox, length_m=scale_len, label=f"{int(scale_len)} m")
        
        if show_stats:
            summary = f"Area: {int(env_w)}m x {int(env_h)}m   Block: {int(grid_stride)}m"
            ax.text(
                0.02,
                0.02,
                summary,
                transform=ax.transAxes,
                fontsize=10,
                fontfamily="monospace",
                verticalalignment="bottom",
                bbox=dict(boxstyle="round", facecolor="white", alpha=0.8),
                zorder=10,
            )

    _save_figure(fig, out_path, dpi=dpi, save_vector=True, bbox_inches='tight', facecolor='white')
    plt.close(fig)


def plot_agent_coverage(partitions: Dict[int, int], node_pos: Dict[int, Tuple[float, float]], out_path: str, 
                        dpi: int = 160, max_nodes: Optional[int] = None,
                        bbox: Tuple[float, float, float, float] = (-300, -300, 300, 300),
                        grid_stride: float = 100.0, road_width: float = 25.0):
    if plt is None:
        return
        
    agents = sorted(set(partitions.values())) if partitions else [0]
    colors_list = [
        "#E74C3C", "#F39C12", "#F1C40F", "#2ECC71", "#3498DB", "#9B59B6",
        "#1ABC9C", "#34495E", "#E67E22", "#95A5A6"
    ]
    
    fig, ax = plt.subplots(figsize=(12, 10), dpi=dpi)
    rng_bg = random.Random(789)
    xs, ys = _build_grid(bbox=bbox, stride=grid_stride)
    _draw_roads(ax, xs, ys, road_width=road_width)
    
    # Draw Hulls
    for idx, a in enumerate(agents):
        nodes = [n for n, p in partitions.items() if p == a]
        if not nodes: 
            continue
        points = [node_pos[n] for n in nodes if n in node_pos]
        if len(points) >= 3:
            # Simple convex hull
            def _convex_hull(points: List[Tuple[float, float]]) -> List[Tuple[float, float]]:
                if len(points) < 3: return points
                start = min(points, key=lambda p: (p[1], p[0]))
                def polar_angle(p):
                    return math.atan2(p[1] - start[1], p[0] - start[0])
                sorted_points = sorted([p for p in points if p != start], key=polar_angle)
                hull = [start, sorted_points[0]]
                for p in sorted_points[1:]:
                    while len(hull) > 1:
                        o, a = hull[-2], hull[-1]
                        cross = (a[0]-o[0])*(p[1]-o[1]) - (a[1]-o[1])*(p[0]-o[0])
                        if cross > 0: break
                        hull.pop()
                    hull.append(p)
                return hull

            hull = _convex_hull(points)
            if len(hull) >= 3:
                c = colors_list[idx % len(colors_list)]
                poly = Polygon(hull, facecolor=c, alpha=0.15, edgecolor=c, linewidth=2.0, linestyle='--')
                ax.add_patch(poly)
                
        # Draw Nodes
        c = colors_list[idx % len(colors_list)]
        xs_n = [p[0] for p in points]
        ys_n = [p[1] for p in points]
        ax.scatter(xs_n, ys_n, s=20, color=c, alpha=0.8)
        if points:
            cx = sum(p[0] for p in points)/len(points)
            cy = sum(p[1] for p in points)/len(points)
            ax.scatter([cx], [cy], s=80, marker='x', color='black', zorder=5, linewidth=2.0)

    ax.set_aspect("equal", "box")
    ax.axis('off')
    
    # Save in multiple formats (PNG + EPS)
    _save_figure(fig, out_path, dpi=dpi, save_vector=True, bbox_inches='tight', facecolor='white')
    plt.close(fig)


def plot_logical_topology(nodes: List[int], edges: Set[Tuple[int, int]], partitions: Dict[int, int], 
                          out_path: str, dpi: int = 160, layout: str = 'spring',
                          show_legend: bool = True, show_title: bool = True, show_stats: bool = True,
                          figsize: Tuple[float, float] = (10, 10)):
    """Plot a logical network topology graph using graph layout algorithms.
    
    Unlike plot_topology which uses physical positions, this function uses 
    force-directed or other graph layout algorithms to position nodes,
    making the network structure clearer.
    
    Args:
        nodes: List of node IDs
        edges: Set of (u, v) edge tuples
        partitions: Dict mapping node_id -> agent_id
        out_path: Output file path
        dpi: Figure DPI
        layout: Layout algorithm ('spring', 'kamada_kawai', 'circular', 'shell')
        show_legend: Whether to show agent legend (default True)
        show_title: Whether to show title (default True)
        show_stats: Whether to show edge statistics (default True)
        figsize: Figure size in inches (width, height), default (10, 10) for consistent sizing
    """
    if plt is None:
        return
    
    try:
        import networkx as nx
    except ImportError:
        print("[Warning] networkx not installed, skipping logical topology plot")
        return
    
    # Build networkx graph
    G = nx.Graph()
    G.add_nodes_from(nodes)
    G.add_edges_from(edges)
    
    # 固定的布局参数，确保一致的坐标范围
    layout_scale = 1.0  # 使用较小的scale，留出边距
    
    # Choose layout algorithm
    if layout == 'kamada_kawai':
        pos = nx.kamada_kawai_layout(G, scale=layout_scale)
    elif layout == 'circular':
        pos = nx.circular_layout(G, scale=layout_scale)
    elif layout == 'shell':
        # Group nodes by agent for shell layout
        agents = sorted(set(partitions.values()))
        shells = [[n for n in nodes if partitions.get(n) == a] for a in agents]
        shells = [s for s in shells if s]  # Remove empty shells
        pos = nx.shell_layout(G, nlist=shells, scale=layout_scale)
    else:  # spring (default)
        pos = nx.spring_layout(G, k=1.5/math.sqrt(len(nodes)+1), iterations=100, seed=42, scale=layout_scale)
    
    # Agent colors
    colors_list = [
        "#E74C3C", "#F39C12", "#F1C40F", "#2ECC71", "#3498DB", 
        "#9B59B6", "#1ABC9C", "#34495E", "#E67E22", "#95A5A6",
        "#C0392B", "#27AE60", "#2980B9", "#8E44AD", "#D35400"
    ]
    agents = sorted(set(partitions.values()))
    agent_color = {a: colors_list[i % len(colors_list)] for i, a in enumerate(agents)}
    
    # Create figure with fixed size (square for consistent aspect ratio)
    fig, ax = plt.subplots(figsize=figsize, dpi=dpi)
    ax.set_facecolor('#f8f9fa')
    
    # Draw edges first (background)
    for (u, v) in edges:
        if u not in pos or v not in pos:
            continue
        x1, y1 = pos[u]
        x2, y2 = pos[v]
        
        # Color by internal (same agent) vs cross (different agents)
        if partitions.get(u) == partitions.get(v):
            # Internal edge - use agent color with higher alpha
            agent = partitions.get(u, 0)
            edge_color = agent_color.get(agent, '#888888')
            alpha = 0.6
            lw = 1.5
        else:
            # Cross-partition edge - gray
            edge_color = '#888888'
            alpha = 0.3
            lw = 1.0
        
        ax.plot([x1, x2], [y1, y2], color=edge_color, alpha=alpha, linewidth=lw, zorder=1)
    
    # Draw nodes by agent
    for agent in agents:
        agent_nodes = [n for n in nodes if partitions.get(n) == agent and n in pos]
        if not agent_nodes:
            continue
        
        xs = [pos[n][0] for n in agent_nodes]
        ys = [pos[n][1] for n in agent_nodes]
        color = agent_color[agent]
        
        # Draw nodes - larger size for better visibility in papers
        ax.scatter(xs, ys, s=350, c=color, edgecolors='white', linewidths=2.0, 
                   zorder=3, alpha=0.9, label=f'Agent {agent}')
        
        # Draw node labels - larger font for better readability
        for n in agent_nodes:
            x, y = pos[n]
            ax.annotate(str(n), (x, y), fontsize=9, ha='center', va='center', 
                        color='white', fontweight='bold', zorder=4)
    
    # Legend
    if show_legend:
        legend_handles = [Line2D([0], [0], marker='o', color='w', markerfacecolor=agent_color[a], 
                                 markersize=10, label=f'Agent {a}') for a in agents]
        ax.legend(handles=legend_handles, loc='upper right', fontsize=9, framealpha=0.9,
                  ncol=2 if len(agents) > 5 else 1)
    
    # Statistics text
    if show_stats:
        n_internal = sum(1 for (u, v) in edges if partitions.get(u) == partitions.get(v))
        n_cross = len(edges) - n_internal
        stats_text = f"Nodes: {len(nodes)} | Edges: {len(edges)} (Internal: {n_internal}, Cross: {n_cross})"
        ax.text(0.5, -0.02, stats_text, transform=ax.transAxes, ha='center', fontsize=10,
                bbox=dict(boxstyle='round', facecolor='white', alpha=0.8))
    
    if show_title:
        ax.set_title('Logical Network Topology', fontsize=14, fontweight='bold', pad=10)
    
    # 设置固定的坐标范围，确保图尺寸一致
    # 根据实际节点位置计算范围，但保证最小范围
    if pos:
        all_x = [p[0] for p in pos.values()]
        all_y = [p[1] for p in pos.values()]
        max_extent = max(abs(min(all_x)), abs(max(all_x)), abs(min(all_y)), abs(max(all_y)), 1.0)
        axis_limit = max_extent * 1.15  # 留出15%边距
    else:
        axis_limit = 1.3
    
    ax.set_xlim(-axis_limit, axis_limit)
    ax.set_ylim(-axis_limit, axis_limit)
    
    ax.axis('off')
    ax.set_aspect('equal')
    
    # Save in multiple formats (PNG + EPS)
    _save_figure(fig, out_path, dpi=dpi, save_vector=True, facecolor='white', pad_inches=0.1)
    plt.close(fig)
