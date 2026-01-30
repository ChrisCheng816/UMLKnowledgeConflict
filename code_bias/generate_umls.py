import inspect
import matplotlib.pyplot as plt
from matplotlib.patches import Rectangle, Polygon
from matplotlib.lines import Line2D

# ---------- helpers to normalize inputs ----------

def _class_members(cls):
    """Return (attributes, methods) defined on the class (skip dunders)."""
    attrs, methods = [], []
    for name, value in cls.__dict__.items():
        if name.startswith("_"):
            continue
        if isinstance(value, property):
            attrs.append(f"{name}: property")
        elif inspect.isfunction(value) or inspect.ismethoddescriptor(value):
            methods.append(f"{name}()")
        elif not inspect.isroutine(value):
            attrs.append(f"{name}: {type(value).__name__}")
    # de-dupe preserve order
    seen = set(); attrs = [a for a in attrs if not (a in seen or seen.add(a))]
    seen = set(); methods = [m for m in methods if not (m in seen or seen.add(m))]
    return attrs, methods

def _to_spec(obj):
    """
    Accept:
      - dict spec: {'name': str, 'attributes': [...], 'methods': [...]}
      - Python class: will introspect and convert to the same spec
    """
    if isinstance(obj, dict):
        name = obj.get('name', 'Unnamed')
        attributes = list(obj.get('attributes', []))
        methods = list(obj.get('methods', []))
        # coerce to strings
        attributes = [str(a) for a in attributes]
        methods = [str(m) for m in methods]
        return {'name': name, 'attributes': attributes, 'methods': methods}

    if inspect.isclass(obj):
        attrs, meths = _class_members(obj)
        return {'name': obj.__name__, 'attributes': attrs, 'methods': meths}

    raise TypeError("Each input must be either a class object or a dict spec.")

# ---------- drawing primitives ----------

def _draw_class_box(ax, x, y, title, attrs, methods, fontsize=10):
    """
    Draw a UML-style class box whose top-left corner is (x,y) in axes coords.
    Returns (box_width, box_height) in the same coordinates.
    """
    # prepare text
    attrs_text = "\n".join(attrs) if attrs else "«no attributes»"
    methods_text = "\n".join(methods) if methods else "«no methods»"

    def longest_line(s): return max((len(l) for l in s.splitlines()), default=0)

    max_chars = max(longest_line(title), longest_line(attrs_text), longest_line(methods_text))
    n_attrs = attrs_text.count("\n") + 1
    n_methods = methods_text.count("\n") + 1

    # coarse layout heuristics in axes fraction
    w_ax = max(0.28, min(0.9, 0.015 * max_chars + 0.18))
    title_h = 0.05
    attrs_h = max(0.06, 0.02 * n_attrs + 0.02)
    methods_h = max(0.06, 0.02 * n_methods + 0.02)
    h_ax = title_h + attrs_h + methods_h

    # outer box
    rect = Rectangle((x, y - h_ax), w_ax, h_ax, color='lightblue', linewidth=1.5)
    ax.add_patch(rect)

    # separators
    y_title_sep = y - title_h
    y_attr_sep = y - title_h - attrs_h
    ax.add_line(Line2D([x, x + w_ax], [y_title_sep, y_title_sep], linewidth=0.75, color='black'))
    ax.add_line(Line2D([x, x + w_ax], [y_attr_sep, y_attr_sep], linewidth=0.75, color='black'))

    # text
    ax.text(x + 0.01, y - title_h/2, title, va="center", ha="left",
            fontsize=fontsize+2, fontweight="bold", family="monospace")
    ax.text(x + 0.01, y_title_sep - attrs_h/2, attrs_text, va="center", ha="left",
            fontsize=fontsize, family="monospace")
    ax.text(x + 0.01, y_attr_sep - methods_h/2, methods_text, va="center", ha="left",
            fontsize=fontsize, family="monospace")

    return w_ax, h_ax

def _draw_inheritance(ax, child_center, parent_center, head_size=0.02, relationship='isa'):
    """Draw UML generalization (hollow triangle) from child → parent."""
    cx, cy = child_center
    px, py = parent_center
    vx, vy = px - cx, py - cy
    L = (vx**2 + vy**2) ** 0.5
    if L == 0: return
    ux, uy = vx / L, vy / L

    tip = (px, py)
    base_c = (px - ux * head_size*1.5, py - uy * head_size*1.5)
    perp = (-uy, ux)
    bw = head_size
    p1 = (base_c[0] + perp[0]*bw/2, base_c[1] + perp[1]*bw/2)
    p2 = (base_c[0] - perp[0]*bw/2, base_c[1] - perp[1]*bw/2)

    ax.add_line(Line2D([cx, base_c[0]], [cy, base_c[1]], linewidth=1.2, color='black'))
    if relationship == 'isa':
        patch = Polygon([tip, p1, p2], closed=True, fill=False, linewidth=1.5)
    else:
        back = (base_c[0] - ux * head_size*1.5, base_c[1] - uy * head_size*1.5)
        patch = Polygon([tip, p1, back, p2], closed=True, facecolor='black', edgecolor='black', linewidth=1.5)
        
    ax.add_patch(patch)

# ---------- public API ----------

def draw_UML(parent, child, save_path, relationship='isa'):
    """
    Draw a minimal UML diagram for a parent and child using matplotlib.
    Accepts either dict specs or class objects.

    Dict spec format:
        {'name': str, 'attributes': [str, ...], 'methods': [str, ...]}
    """
    p_spec = _to_spec(parent)
    c_spec = _to_spec(child)

    fig, ax = plt.subplots(figsize=(8, 6))
    ax.set_xlim(0, 1); ax.set_ylim(0, 1); ax.axis("off")

    # layout: parent top-center, child bottom-center
    parent_x, parent_y = 0.35, 0.85
    child_x, child_y = 0.35, 0.45

    pw, ph = _draw_class_box(ax, parent_x, parent_y,
                             p_spec['name'], p_spec['attributes'], p_spec['methods'])
    cw, ch = _draw_class_box(ax, child_x, child_y,
                             c_spec['name'], c_spec['attributes'], c_spec['methods'])

    parent_center_bottom = (parent_x + pw/2, parent_y - ph)
    child_center_top = (child_x + cw/2, child_y)

    _draw_inheritance(ax, child_center_top, parent_center_bottom, relationship=relationship)
    plt.tight_layout();
    # plt.show()
    plt.savefig(save_path)
