"""
Data Structure Visualizer
Supports: Binary Trees and Linked Lists
All visualizations are 768x768 pixels
"""

import matplotlib.pyplot as plt
import matplotlib.patches as patches
from matplotlib.patches import Circle, Rectangle, FancyBboxPatch, Polygon, FancyArrowPatch
import numpy as np
from PIL import Image
from typing import List, Tuple, Dict
import copy

"""
Data Structure Visualizer
Supports: Binary Trees and Linked Lists
All visualizations are 768x768 pixels
"""

import matplotlib.pyplot as plt
import matplotlib.patches as patches
from matplotlib.patches import Circle, Rectangle, FancyBboxPatch, Polygon, FancyArrowPatch
import numpy as np

import copy

# pip install pillow
from PIL import Image, ImageDraw, ImageFont

def save_text_image(
    text: str,
    out_path: str,
    *,
    font_path: str | None = None,   # e.g., "/usr/share/fonts/truetype/dejavu/DejaVuSans.ttf"
    font_size: int = 42,
    max_width: int = 1024,          # total image width (pixels)
    padding: int = 32,
    text_color: tuple[int, int, int] = (0, 0, 0),
):
    """
    Render `text` onto a white PNG and save to `out_path`.

    - Automatically wraps lines to fit within `max_width - 2*padding`.
    - If `font_path` is None, tries DejaVuSans, then falls back to a default bitmap font.
    """
    # Load font
    if font_path is None:
        try:
            font = ImageFont.truetype("DejaVuSans.ttf", font_size)
        except Exception:
            font = ImageFont.load_default()
    else:
        font = ImageFont.truetype(font_path, font_size)

    # Dummy canvas for measuring
    measure_img = Image.new("RGB", (10, 10))
    measure = ImageDraw.Draw(measure_img)

    # Wrap text to the usable width
    usable_width = max_width - 2 * padding

    def wrap_text(s: str) -> list[str]:
        lines: list[str] = []
        for para in s.splitlines():
            if not para:
                lines.append("")
                continue
            words = para.split(" ")
            line = ""
            for w in words:
                trial = (line + " " + w).strip()
                if measure.textlength(trial, font=font) <= usable_width:
                    line = trial
                else:
                    if line:
                        lines.append(line)
                    # If a single word is too long, hard-wrap by characters
                    if measure.textlength(w, font=font) <= usable_width:
                        line = w
                    else:
                        chunk = ""
                        for ch in w:
                            trial2 = chunk + ch
                            if measure.textlength(trial2, font=font) <= usable_width:
                                chunk = trial2
                            else:
                                if chunk:
                                    lines.append(chunk)
                                chunk = ch
                        line = chunk
            lines.append(line)
        return lines

    lines = wrap_text(text)
    wrapped_text = "\n".join(lines)

    # Compute text block size
    try:
        bbox = measure.multiline_textbbox((0, 0), wrapped_text, font=font, align="left", spacing=int(font.size * 0.2))
        text_w = bbox[2] - bbox[0]
        text_h = bbox[3] - bbox[1]
    except AttributeError:
        # Pillow < 8.0 fallback
        ascent, descent = font.getmetrics()
        line_height = ascent + descent
        text_w = max((int(measure.textlength(l, font=font)) for l in lines), default=0)
        text_h = len(lines) * line_height + max(0, len(lines) - 1) * int(font.size * 0.2)

    img_w = min(max_width, text_w + 2 * padding)
    img_h = text_h + 2 * padding

    # Create white background and draw text
    img = Image.new("RGB", (int(img_w), int(img_h)), color=(255, 255, 255))
    draw = ImageDraw.Draw(img)
    draw.multiline_text(
        (padding, padding),
        wrapped_text,
        font=font,
        fill=text_color,
        align="left",
        spacing=int(font.size * 0.2),
    )

    img.save(out_path, format="PNG")
    return out_path, (int(img_w), int(img_h))

from PIL import Image

def merge_png_vertically(image1_path: str, image2_path: str, output_path: str) -> None:
    """
    Merge two PNG images vertically and save the result.

    Parameters
    ----------
    image1_path : str
        Path to the first PNG image (appears on top).
    image2_path : str
        Path to the second PNG image (appears below the first).
    output_path : str
        Path where the merged PNG will be saved.

    The resulting image width equals the max of both widths, and heights are added.
    The images are centered horizontally if their widths differ.
    """
    # Load images (RGBA keeps transparency)
    img1 = Image.open(image1_path).convert("RGBA")
    img2 = Image.open(image2_path).convert("RGBA")

    # Compute output size
    new_width = max(img1.width, img2.width)
    new_height = img1.height + img2.height

    # Create transparent background
    merged_img = Image.new("RGBA", (new_width, new_height), (255, 255, 255))

    # Compute horizontal offsets for centering
    offset_x1 = (new_width - img1.width) // 2
    offset_x2 = (new_width - img2.width) // 2

    # Paste both images
    merged_img.paste(img1, (offset_x1, 0), img1)
    merged_img.paste(img2, (offset_x2, img1.height), img2)

    # Save result
    merged_img.save(output_path, format="PNG")

    print(f"Merged image saved to: {output_path}")


def generate_llist(length, vary_type="numbers", base_number=1, base_shape="circle", base_color="blue"):
    tree = []
    
    shapes = ["circle", "square", "diamond", "triangle", "rectangle"]
    colors = ["red", "blue", "green", "yellow", "purple", "orange", "cyan", "magenta"]
    
    for i in range(length):
        if vary_type == "numbers":
            number = base_number + i
            shape = base_shape
            color = base_color
        elif vary_type == "shapes":
            number = -1  # Hide numbers
            shape = shapes[i % len(shapes)]
            color = base_color
        elif vary_type == "colors":
            number = -1  # Hide numbers
            shape = base_shape
            color = colors[i % len(colors)]
        else:
            number = base_number
            shape = base_shape
            color = base_color
        
        tree.append({"value": (number, shape, color), "next": [i+1]})
    
    return tree


# ============================================================================
# FUNCTION 1: REVERSE TREE (Mirror the tree)
# ============================================================================

def reverse_tree(tree_list, swap_index=[1,2]):
    """
    Reverse (mirror) a binary tree by swapping left and right children at each level.
    
    Args:
        tree_list: Binary tree in list format [(number, shape, color), ...]
        swap_index: Additional swap to apply after reversal
    
    Returns:
        Reversed tree in list format
    
    Example:
        Original:       1
                       / \
                      2   3
                     / \
                    4   5
        
        Reversed:       1
                       / \
                      3   2
                         / \
                        5   4
    """
    if not tree_list:
        return tree_list
    
    n = len(tree_list)
    reversed_tree = [None] * n
    
    def reverse_helper(src_idx, dst_idx):
        if src_idx >= n or tree_list[src_idx] is None:
            if dst_idx < n:
                reversed_tree[dst_idx] = None
            return
        
        # Copy current node
        reversed_tree[dst_idx] = tree_list[src_idx]
        
        # Get child indices
        src_left = 2 * src_idx + 1
        src_right = 2 * src_idx + 2
        dst_left = 2 * dst_idx + 1
        dst_right = 2 * dst_idx + 2
        
        # Reverse children: source's right goes to dest's left, source's left goes to dest's right
        if src_right < n:
            reverse_helper(src_right, dst_left)
        elif dst_left < n:
            reversed_tree[dst_left] = None
            
        if src_left < n:
            reverse_helper(src_left, dst_right)
        elif dst_right < n:
            reversed_tree[dst_right] = None
    
    reverse_helper(0, 0)
    
    # Apply the swap_index swap
    i, j = swap_index
    if i < len(reversed_tree) and j < len(reversed_tree):
        reversed_tree[i], reversed_tree[j] = reversed_tree[j], reversed_tree[i]
    
    return reversed_tree


# ============================================================================
# FUNCTION 2: GET HEIGHT 2 SUBTREE
# ============================================================================

def get_height2_subtree(tree_list, root_index=0, swap_index = [1,2]):
    """
    Extract a height 2 subtree from a binary tree starting at root_index.
    Height 2 means: root + children + grandchildren (up to 7 nodes total).
    
    Args:
        tree_list: Binary tree in list format [(number, shape, color), ...]
        root_index: Index of the root node for the subtree (default: 0)
    
    Returns:
        Subtree in list format with up to 7 nodes
    
    Example:
        Original tree:      1
                          /   \\
                         2     3
                        / \\   / \\
                       4   5 6   7
                      / \\
                     8   9
        
        get_height2_subtree(tree, 0) returns:
                            1
                          /   \\
                         2     3
                        / \\   / \\
                       4   5 6   7
        
        get_height2_subtree(tree, 1) returns:
                            2
                           / \\
                          4   5
                         / \\
                        8   9
    """
    if not tree_list or root_index >= len(tree_list) or tree_list[root_index] is None:
        return []
    
    subtree = [None] * 7  # Max 7 nodes for height 2 (complete binary tree)
    
    def extract_node(src_idx, dest_idx, current_height):
        """Recursively extract nodes up to height 2"""
        if src_idx >= len(tree_list) or tree_list[src_idx] is None or current_height > 2:
            return
        
        # Copy the node
        subtree[dest_idx] = tree_list[src_idx]
        
        if current_height < 2:
            # Process left child
            left_src = 2 * src_idx + 1
            left_dest = 2 * dest_idx + 1
            extract_node(left_src, left_dest, current_height + 1)
            
            # Process right child
            right_src = 2 * src_idx + 2
            right_dest = 2 * dest_idx + 2
            extract_node(right_src, right_dest, current_height + 1)
    
    extract_node(root_index, 0, 0)
    
    # Remove trailing None values
    while subtree and subtree[-1] is None:
        subtree.pop()
    
    i,j = swap_index
    subtree[i], subtree[j] = subtree[j], subtree[i]
    return subtree



# ============================================================================
# BINARY TREE VISUALIZER
# ============================================================================

def draw_binary_tree(tree_list, output_path="tree.png", scale_by_height=False, show_root_line=False):
    """
    Generate an image of a binary tree from a leetcode-style list format.

    Args:
        tree_list: List of tuples (number, shape, color) or None for null nodes
                  number: integer to display on node, -1 to hide number
                  shape can be: "circle", "square", "rectangle", "diamond", "triangle"
                  color can be any matplotlib color (e.g., "red", "blue", "#FF5733")
        output_path: Path to save the output image (default: "tree.png")
        scale_by_height: If True, scale node size based on depth (smaller at lower levels)
        show_root_line: If True, draw a vertical line down from root to indicate binary tree

    Example:
        tree_list = [(1, "circle", "orange"), (2, "circle", "red"), (3, "circle", "red")]
        draw_binary_tree(tree_list, "my_tree.png", scale_by_height=True, show_root_line=True)
    """

    # Filter out None values but keep track of structure
    if not tree_list or tree_list[0] is None:
        print("Empty tree!")
        return

    # Create figure with 768x768 pixels (assuming 100 dpi)
    fig, ax = plt.subplots(figsize=(7.68, 7.68), dpi=100)
    ax.set_xlim(0, 10)
    ax.set_ylim(0, 10)
    ax.axis('off')

    # Calculate tree depth and number of actual nodes
    n = len(tree_list)

    # Count actual non-None nodes at each level
    def get_tree_depth():
        max_depth = 0
        for i in range(len(tree_list)):
            if tree_list[i] is not None:
                depth = int(np.log2(i + 1)) if i > 0 else 0
                max_depth = max(max_depth, depth)
        return max_depth + 1

    depth = get_tree_depth()

    # Node positions dictionary
    positions = {}
    node_levels = {}  # Track level of each node for size scaling

    def count_leaves(index):
        """Count the number of leaf nodes in the subtree"""
        if index >= len(tree_list) or tree_list[index] is None:
            return 0

        left_child = 2 * index + 1
        right_child = 2 * index + 2

        # If no children, this is a leaf
        if (left_child >= len(tree_list) or tree_list[left_child] is None) and \
           (right_child >= len(tree_list) or tree_list[right_child] is None):
            return 1

        # Otherwise, sum the leaves of children
        left_leaves = count_leaves(left_child)
        right_leaves = count_leaves(right_child)
        return left_leaves + right_leaves

    def calculate_positions(index, left_bound, right_bound, y, level):
        """Recursively calculate positions with better spacing"""
        if index >= len(tree_list) or tree_list[index] is None:
            return

        left_child = 2 * index + 1
        right_child = 2 * index + 2

        # Count leaves in left and right subtrees
        left_leaves = count_leaves(left_child) if left_child < len(tree_list) else 0
        right_leaves = count_leaves(right_child) if right_child < len(tree_list) else 0
        total_leaves = left_leaves + right_leaves

        if total_leaves == 0:
            # This is a leaf node, center it in the available space
            x = (left_bound + right_bound) / 2
        else:
            # Position based on the proportion of leaves in each subtree
            if left_leaves == 0:
                x = right_bound - (right_bound - left_bound) * 0.25
            elif right_leaves == 0:
                x = left_bound + (right_bound - left_bound) * 0.25
            else:
                # Weighted position based on leaf distribution
                x = left_bound + (right_bound - left_bound) * (left_leaves / total_leaves)

        positions[index] = (x, y)
        node_levels[index] = level  # Track the level for size scaling

        # Calculate positions for children
        new_y = y - 1.8

        if left_child < len(tree_list) and tree_list[left_child] is not None:
            # Left child gets space proportional to its leaves
            if total_leaves > 0:
                mid_point = left_bound + (right_bound - left_bound) * (left_leaves / total_leaves)
                calculate_positions(left_child, left_bound, mid_point, new_y, level + 1)
            else:
                mid_point = (left_bound + x) * 0.75 + x * 0.25
                calculate_positions(left_child, left_bound, mid_point, new_y, level + 1)

        if right_child < len(tree_list) and tree_list[right_child] is not None:
            # Right child gets space proportional to its leaves
            if total_leaves > 0:
                mid_point = left_bound + (right_bound - left_bound) * (left_leaves / total_leaves)
                calculate_positions(right_child, mid_point, right_bound, new_y, level + 1)
            else:
                mid_point = x * 0.25 + (x + right_bound) * 0.75
                calculate_positions(right_child, mid_point, right_bound, new_y, level + 1)

    # Start calculating positions from root with full width
    calculate_positions(0, 0.5, 9.5, 9, 0)

    # Draw vertical line from root if requested
    if show_root_line and 0 in positions:
        root_x, root_y = positions[0]
        # Draw a dashed vertical line extending down from the root
        ax.plot([root_x, root_x], [root_y - 0.5, 0.5], 'k--', linewidth=2, alpha=0.5, zorder=0)

    # Draw edges first (so they appear behind nodes)
    for index in positions:
        left_child = 2 * index + 1
        right_child = 2 * index + 2

        if left_child in positions:
            x1, y1 = positions[index]
            x2, y2 = positions[left_child]
            ax.plot([x1, x2], [y1, y2], 'k-', linewidth=2, zorder=1)

        if right_child in positions:
            x1, y1 = positions[index]
            x2, y2 = positions[right_child]
            ax.plot([x1, x2], [y1, y2], 'k-', linewidth=2, zorder=1)

    # Draw nodes
    base_node_size = 0.4

    for index, (x, y) in positions.items():
        number, shape, color = tree_list[index]

        # Scale node size based on level if requested
        if scale_by_height:
            level = node_levels[index]
            # Reduce size by 15% for each level down
            node_size = base_node_size * (0.85 ** level)
        else:
            node_size = base_node_size

        if shape.lower() == "circle":
            circle = Circle((x, y), node_size, facecolor=color, edgecolor='black',
                          linewidth=2, zorder=2)
            ax.add_patch(circle)

        elif shape.lower() == "square":
            square = Rectangle((x - node_size, y - node_size),
                              node_size * 2, node_size * 2,
                              facecolor=color, edgecolor='black',
                              linewidth=2, zorder=2)
            ax.add_patch(square)

        elif shape.lower() == "rectangle":
            rect = Rectangle((x - node_size * 1.2, y - node_size * 0.7),
                           node_size * 2.4, node_size * 1.4,
                           facecolor=color, edgecolor='black',
                           linewidth=2, zorder=2)
            ax.add_patch(rect)

        elif shape.lower() == "diamond":
            diamond = Polygon([
                (x, y + node_size),           # top
                (x + node_size, y),           # right
                (x, y - node_size),           # bottom
                (x - node_size, y)            # left
            ], facecolor=color, edgecolor='black', linewidth=2, zorder=2)
            ax.add_patch(diamond)

        elif shape.lower() == "triangle":
            triangle = Polygon([
                (x, y + node_size),                    # top
                (x - node_size, y - node_size),        # bottom left
                (x + node_size, y - node_size)         # bottom right
            ], facecolor=color, edgecolor='black', linewidth=2, zorder=2)
            ax.add_patch(triangle)

        else:
            # Default to circle if shape not recognized
            circle = Circle((x, y), node_size, facecolor=color, edgecolor='black',
                          linewidth=2, zorder=2)
            ax.add_patch(circle)
        
        # Add number label if not -1
        if number != -1:
            ax.text(x, y, str(number), fontsize=10, ha='center', va='center',
                   fontweight='bold', color='white', zorder=3)

    plt.tight_layout()
    plt.savefig(output_path, dpi=100, bbox_inches='tight', pad_inches=0.1)
    plt.close()

    print(f"Tree image saved to: {output_path}")


# ============================================================================
# LINKED LIST VISUALIZER
# ============================================================================

def draw_linked_list(nodes_list, output_path="linked_list.png", layout="horizontal",
                     scale_by_position=False, show_indices=False):
    """
    Generate an image of a linked list/graph structure from a list format.

    Args:
        nodes_list: List of dicts with format:
                   {"value": (number, shape, color), "next": [list of indices or None]}
                   Example: [
                       {"value": (1, "circle", "red"), "next": [1]},
                       {"value": (2, "square", "blue"), "next": [2]},
                       {"value": (-1, "circle", "green"), "next": [None]}  # -1 means no number display
                   ]
        output_path: Path to save the output image (default: "linked_list.png")
        layout: "horizontal" (left-to-right) or "vertical" (top-to-bottom)
        scale_by_position: If True, scale node sizes based on position in list
        show_indices: If True, display index numbers on nodes (overrides number from value)

    Example:
        # Simple linked list with numbers
        nodes = [
            {"value": (1, "circle", "red"), "next": [1]},
            {"value": (2, "circle", "blue"), "next": [2]},
            {"value": (3, "circle", "green"), "next": [None]}
        ]
        draw_linked_list(nodes, "my_list.png")

        # Circular linked list with hidden numbers
        nodes = [
            {"value": (-1, "circle", "red"), "next": [1]},
            {"value": (-1, "circle", "blue"), "next": [0]}  # Points back to 0
        ]
        draw_linked_list(nodes, "circular.png")
    """

    if not nodes_list:
        print("Empty list!")
        return

    # Create figure with 768x768 pixels (assuming 100 dpi)
    fig, ax = plt.subplots(figsize=(7.68, 7.68), dpi=100)
    ax.set_xlim(0, 10)
    ax.set_ylim(0, 10)
    ax.axis('off')

    n = len(nodes_list)
    positions = {}
    base_node_size = 0.35

    # Calculate positions based on layout
    if layout == "horizontal":
        # Arrange nodes horizontally
        if n <= 5:
            spacing = 8.0 / max(n - 1, 1)
            start_x = 1.0
        else:
            spacing = 8.0 / (n - 1)
            start_x = 1.0

        for i in range(n):
            x = start_x + i * spacing
            y = 5.0
            positions[i] = (x, y)

    elif layout == "vertical":
        # Arrange nodes vertically (top to bottom)
        if n <= 5:
            spacing = 8.0 / max(n - 1, 1)
            start_y = 9.0
        else:
            spacing = 8.0 / (n - 1)
            start_y = 9.0

        for i in range(n):
            x = 5.0
            y = start_y - i * spacing
            positions[i] = (x, y)

    # Draw arrows/edges first (so they appear behind nodes)
    drawn_arrows = set()  # Track drawn arrows to avoid duplicates

    for i, node in enumerate(nodes_list):
        if "next" not in node or node["next"] is None:
            continue

        next_indices = node["next"]
        if next_indices is None or next_indices == [None]:
            continue

        x1, y1 = positions[i]

        for next_idx in next_indices:
            if next_idx is None or next_idx >= len(nodes_list):
                continue

            # Avoid drawing duplicate arrows
            arrow_key = (i, next_idx)
            if arrow_key in drawn_arrows:
                continue
            drawn_arrows.add(arrow_key)

            # Special handling for self-loops (node points to itself)
            if next_idx == i:
                # Draw a circular arrow around the node
                if layout == "horizontal":
                    # Draw loop above the node
                    loop = patches.FancyBboxPatch(
                        (x1 - base_node_size * 0.3, y1 + base_node_size * 0.5),
                        base_node_size * 0.6,
                        base_node_size * 1.2,
                        boxstyle="round,pad=0.05",
                        linewidth=2.5,
                        edgecolor='black',
                        facecolor='none',
                        zorder=1
                    )
                    ax.add_patch(loop)
                    # Add arrowhead
                    arrow = FancyArrowPatch(
                        (x1 + base_node_size * 0.3, y1 + base_node_size * 1.5),
                        (x1 + base_node_size * 0.4, y1 + base_node_size * 0.6),
                        arrowstyle='->,head_width=5,head_length=5',
                        color='black',
                        linewidth=2.5,
                        zorder=1
                    )
                    ax.add_patch(arrow)
                else:  # vertical
                    # Draw loop to the right of the node
                    loop = patches.FancyBboxPatch(
                        (x1 + base_node_size * 0.5, y1 - base_node_size * 0.3),
                        base_node_size * 1.2,
                        base_node_size * 0.6,
                        boxstyle="round,pad=0.05",
                        linewidth=2.5,
                        edgecolor='black',
                        facecolor='none',
                        zorder=1
                    )
                    ax.add_patch(loop)
                    # Add arrowhead
                    arrow = FancyArrowPatch(
                        (x1 + base_node_size * 1.5, y1 - base_node_size * 0.3),
                        (x1 + base_node_size * 0.6, y1 - base_node_size * 0.4),
                        arrowstyle='->,head_width=5,head_length=5',
                        color='black',
                        linewidth=2.5,
                        zorder=1
                    )
                    ax.add_patch(arrow)
                continue

            x2, y2 = positions[next_idx]

            # Check if this is a circular/back reference
            is_back_reference = next_idx <= i

            # Calculate arrow style based on direction
            if layout == "horizontal":
                if is_back_reference:
                    # Draw curved arrow for back references with prominent arrowhead
                    arrow = FancyArrowPatch(
                        (x1 + base_node_size, y1),
                        (x2 + base_node_size, y2),
                        arrowstyle='->,head_width=5,head_length=5',
                        connectionstyle="arc3,rad=0.3",
                        color='black',
                        linewidth=2.5,
                        zorder=1,
                        alpha=0.8
                    )
                else:
                    # Straight arrow for forward references with prominent arrowhead
                    arrow = FancyArrowPatch(
                        (x1 + base_node_size, y1),
                        (x2 - base_node_size, y2),
                        arrowstyle='->,head_width=5,head_length=5',
                        color='black',
                        linewidth=2.5,
                        zorder=1
                    )
            else:  # vertical
                if is_back_reference:
                    # Draw curved arrow for back references with prominent arrowhead
                    arrow = FancyArrowPatch(
                        (x1, y1 - base_node_size),
                        (x2, y2 - base_node_size),
                        arrowstyle='->,head_width=0.5,head_length=0.5',
                        connectionstyle="arc3,rad=-0.3",
                        color='black',
                        linewidth=2.5,
                        zorder=1,
                        alpha=0.8
                    )
                else:
                    # Straight arrow for forward references with prominent arrowhead
                    arrow = FancyArrowPatch(
                        (x1, y1 - base_node_size),
                        (x2, y2 + base_node_size),
                        arrowstyle='->,head_width=0.5,head_length=0.5',
                        color='black',
                        linewidth=2.5,
                        zorder=1
                    )

            ax.add_patch(arrow)

    # Draw nodes
    for i, node in enumerate(nodes_list):
        x, y = positions[i]

        if "value" not in node:
            continue

        # Extract number, shape, and color from triplet
        number, shape, color = node["value"]

        # Scale node size based on position if requested
        if scale_by_position:
            node_size = base_node_size * (0.9 ** i)
        else:
            node_size = base_node_size

        # Draw the shape
        if shape.lower() == "circle":
            circle = Circle((x, y), node_size, facecolor=color, edgecolor='black',
                          linewidth=2, zorder=2)
            ax.add_patch(circle)

        elif shape.lower() == "square":
            square = Rectangle((x - node_size, y - node_size),
                              node_size * 2, node_size * 2,
                              facecolor=color, edgecolor='black',
                              linewidth=2, zorder=2)
            ax.add_patch(square)

        elif shape.lower() == "rectangle":
            rect = Rectangle((x - node_size * 1.2, y - node_size * 0.7),
                           node_size * 2.4, node_size * 1.4,
                           facecolor=color, edgecolor='black',
                           linewidth=2, zorder=2)
            ax.add_patch(rect)

        elif shape.lower() == "diamond":
            diamond = Polygon([
                (x, y + node_size),           # top
                (x + node_size, y),           # right
                (x, y - node_size),           # bottom
                (x - node_size, y)            # left
            ], facecolor=color, edgecolor='black', linewidth=2, zorder=2)
            ax.add_patch(diamond)

        elif shape.lower() == "triangle":
            triangle = Polygon([
                (x, y + node_size),                    # top
                (x - node_size, y - node_size),        # bottom left
                (x + node_size, y - node_size)         # bottom right
            ], facecolor=color, edgecolor='black', linewidth=2, zorder=2)
            ax.add_patch(triangle)

        else:
            # Default to circle if shape not recognized
            circle = Circle((x, y), node_size, facecolor=color, edgecolor='black',
                          linewidth=2, zorder=2)
            ax.add_patch(circle)

        # Add label (number or index)
        if show_indices:
            # Show index if requested (overrides number)
            ax.text(x, y, str(i), fontsize=10, ha='center', va='center',
                   fontweight='bold', color='white', zorder=3)
        elif number != -1:
            # Show number if it's not -1
            ax.text(x, y, str(number), fontsize=10, ha='center', va='center',
                   fontweight='bold', color='white', zorder=3)

    plt.tight_layout()
    plt.savefig(output_path, dpi=100, bbox_inches='tight', pad_inches=0.1)
    plt.close()

    print(f"Linked list image saved to: {output_path}")
    

def merge_png_horizontally(image1_path, image2_path, output_path):
    """
    Merges two PNG images horizontally.

    Args:
        image1_path (str): Path to the first PNG image.
        image2_path (str): Path to the second PNG image.
        output_path (str): Path to save the merged image.
    """
    try:
        img1 = Image.open(image1_path)
        img2 = Image.open(image2_path)

        # Ensure both images are in RGBA mode for transparency handling
        img1 = img1.convert("RGBA")
        img2 = img2.convert("RGBA")

        # Determine the dimensions of the new merged image
        total_width = img1.width + img2.width
        max_height = max(img1.height, img2.height)

        # Create a new blank image with the calculated dimensions
        merged_image = Image.new("RGBA", (total_width, max_height), (0, 0, 0, 0)) # Transparent background

        # Paste the first image
        merged_image.paste(img1, (0, 0))

        # Paste the second image next to the first
        merged_image.paste(img2, (img1.width, 0))

        # Save the merged image
        merged_image.save(output_path, format="PNG")
        print(f"Images merged successfully and saved to {output_path}")

    except FileNotFoundError:
        print("Error: One or both image files not found.")
    except Exception as e:
        print(f"An error occurred: {e}")



def reverse_linked_list(nodes_list: List[Dict], mode: str = "arrow", swap: List[Tuple[int, int]] = None):
    """
    Reverse a linked list in two different ways.
    
    Args:
        nodes_list: Original linked list as list of dicts with "value" and "next"
        mode: Either "node" or "arrow"
              - "node": Reverse the physical order of nodes, keep arrow logic same
              - "arrow": Keep nodes in same positions, reverse arrow directions
        swap: List of (index1, index2) pairs to swap after reversing (for testing)
              Example: [(0, 2)] swaps nodes at index 0 and 2
    
    Returns:
        Reversed linked list in the same format
    
    Examples:
        Original: Red→Blue→Green
        
        Mode "node": Green→Blue→Red (nodes physically reversed, arrows updated)
        Mode "arrow": Red←Blue←Green (arrows point backwards, so Green→Blue→Red)
    """
    
    if not nodes_list:
        return []
    
    # Deep copy to avoid modifying original
    result = copy.deepcopy(nodes_list)
    n = len(result)
    
    if mode == "node":
        # MODE 1: REVERSE THE NODES (physical positions), update arrows accordingly
        # The node values/shapes reverse positions
        # Arrows are updated to maintain the reversed order
        
        # Reverse the node values
        reversed_values = [result[i]["value"] for i in range(n-1, -1, -1)]
        
        # Assign reversed values to nodes
        for i in range(n):
            result[i]["value"] = reversed_values[i]
        
        # Update the next pointers to create a forward chain in the new order
        for i in range(n):
            if i < n - 1:
                result[i]["next"] = [i + 1]
            else:
                result[i]["next"] = [None]
    
    elif mode == "arrow":
        # MODE 2: REVERSE THE ARROWS (pointer directions), keep nodes in place
        # Nodes stay in their original positions
        # But the next pointers are reversed
        
        # Build reverse mapping: for each node, find what points to it
        incoming = {i: [] for i in range(n)}
        
        for i, node in enumerate(result):
            next_indices = node.get("next", [None])
            if next_indices and next_indices != [None]:
                for next_idx in next_indices:
                    if next_idx is not None and next_idx < n:
                        incoming[next_idx].append(i)
        
        # Find the tail (node that points to None)
        tail = None
        for i, node in enumerate(result):
            next_indices = node.get("next", [None])
            if not next_indices or next_indices == [None]:
                tail = i
                break
        
        # Reverse the pointers
        for i in range(n):
            if i == tail:
                # Old tail becomes new head, points to what pointed to it
                if incoming[i]:
                    result[i]["next"] = incoming[i]
                else:
                    result[i]["next"] = [None]
            else:
                # Each node now points to what previously pointed to it
                if incoming[i]:
                    result[i]["next"] = incoming[i]
                else:
                    # This was the head, now becomes tail
                    result[i]["next"] = [None]
    
    else:
        raise ValueError(f"Invalid mode: {mode}. Must be 'node' or 'arrow'")
    
    # Apply swaps if provided
    if swap:
        for idx1, idx2 in swap:
            if 0 <= idx1 < n and 0 <= idx2 < n:
                # Swap the node values (not the structure)
                result[idx1]["value"], result[idx2]["value"] = result[idx2]["value"], result[idx1]["value"]
    
    return result
