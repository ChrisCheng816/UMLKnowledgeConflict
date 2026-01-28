import os


def read_inheritance_edges(wsd_path):
    """
    作用
    - 读取一个 .wsd 文件
    - 提取所有形如:
      Parent <|-- Child
      的继承边

    返回
    - edges: 列表，每个元素是 (parent, child)

    重要约定
    - 在 PlantUML 里: Parent <|-- Child 表示 Child 是 Parent 的子类
    - 也就是 “右边是左边的子节点”
    """
    edges = []

    with open(wsd_path, "r", encoding="utf-8", errors="ignore") as f:
        for raw in f:
            line = raw.strip()
            if "<|--" not in line:
                continue

            left, right = line.split("<|--", 1)
            parent = left.strip()
            child = right.strip()

            if parent and child:
                edges.append((parent, child))

    return edges


def build_chain_from_edges(edges, wsd_path):
    """
    作用
    - 给定一组 (parent, child) 边
    - 尝试把它们还原成一条链:
      node0 node1 node2 ...

    理想情况
    - 每个节点的出度最多为一
    - 每个节点的入度最多为一
    - 只有一个 root (只出不入)
    - 这样可以唯一还原链

    非理想情况处理
    - 如果出现分叉或多个 root:
      会选择“最长的一条链”作为主链
      其余节点会在末尾按确定性顺序追加
      同时在终端打印警告，方便你后续清理数据
    """
    if not edges:
        return []

    # parent_to_children: parent -> [child1, child2, ...]
    parent_to_children = {}
    in_degree = {}
    out_degree = {}
    nodes = set()

    for parent, child in edges:
        nodes.add(parent)
        nodes.add(child)

        parent_to_children.setdefault(parent, []).append(child)

        in_degree[child] = in_degree.get(child, 0) + 1
        in_degree.setdefault(parent, in_degree.get(parent, 0))

        out_degree[parent] = out_degree.get(parent, 0) + 1
        out_degree.setdefault(child, out_degree.get(child, 0))

    # roots: nodes with in_degree == 0
    roots = [n for n in nodes if in_degree.get(n, 0) == 0]
    roots.sort()

    def follow_from(start):
        """
        作用
        - 从 start 开始沿着 parent_to_children 一直往下走
        - 如果某个 parent 有多个 child，只取字典序最小的那个作为主链分支
        """
        chain = [start]
        visited = set([start])

        cur = start
        while True:
            children = parent_to_children.get(cur, [])
            if not children:
                break

            # 若有分叉，按字典序选一个，保证稳定
            next_child = sorted(children)[0]
            if next_child in visited:
                # 出现环，停止
                break

            chain.append(next_child)
            visited.add(next_child)
            cur = next_child

        return chain

    # 如果没有 root，说明可能有环，随便选一个最小节点开始
    if not roots:
        start = sorted(nodes)[0]
        chain = follow_from(start)
        print(f"[WARN] No root found (cycle suspected): {wsd_path}")
    else:
        # 多个 root 时，选择能产生最长链的 root
        best_chain = []
        for r in roots:
            c = follow_from(r)
            if len(c) > len(best_chain):
                best_chain = c
        chain = best_chain

        if len(roots) > 1:
            print(f"[WARN] Multiple roots found in {wsd_path}: {roots}")

    # 检查是否是严格单链
    branching_parents = [p for p, ch in parent_to_children.items() if len(ch) > 1]
    multi_in_nodes = [n for n in nodes if in_degree.get(n, 0) > 1]

    if branching_parents:
        print(f"[WARN] Branching detected in {wsd_path}, parents: {sorted(branching_parents)}")
    if multi_in_nodes:
        print(f"[WARN] Multiple inheritance detected in {wsd_path}, nodes: {sorted(multi_in_nodes)}")

    # 把没覆盖到的节点追加到末尾，保证“一个文件一行”不会丢信息
    chain_set = set(chain)
    leftover = sorted([n for n in nodes if n not in chain_set])
    if leftover:
        chain = chain + leftover
        print(f"[WARN] Unused nodes appended in {wsd_path}: {leftover}")

    return chain


def wsd_to_data_line(wsd_path):
    """
    作用
    - 把一个 .wsd 文件转换成 data.txt 的一行

    输出格式
    - 节点之间用一个空格分隔
    - 例如:
      ToyotaCamry Automobile Car
    """
    edges = read_inheritance_edges(wsd_path)
    chain = build_chain_from_edges(edges, wsd_path)

    # 如果文件里没有继承边，就输出空行会很怪
    # 这里选择跳过，避免污染 data.txt
    if not chain:
        return None

    return " ".join(chain)


def build_data_txt_for_folder(folder_path):
    """
    作用
    - 针对一个文件夹
      找到该文件夹内所有 .wsd 文件
      每个 .wsd 生成一行
      写到该文件夹的 data.txt

    关键点
    - 一个 .wsd 文件就是一个 instance
    - 所以 data.txt 里一行对应一个 .wsd
    """
    wsd_files = []
    for name in os.listdir(folder_path):
        full = os.path.join(folder_path, name)
        if os.path.isfile(full) and name.lower().endswith(".wsd"):
            wsd_files.append(full)

    if not wsd_files:
        return

    wsd_files.sort()

    lines = []
    for wsd_path in wsd_files:
        line = wsd_to_data_line(wsd_path)
        if line is not None:
            lines.append(line)

    data_path = os.path.join(folder_path, "data.txt")
    with open(data_path, "w", encoding="utf-8") as out:
        out.write("\n".join(lines))
        if lines:
            out.write("\n")


def traverse_all_folders(root_dir):
    """
    作用
    - 从 root_dir 开始递归遍历所有子文件夹
    - 每个文件夹各自生成自己的 data.txt
    """
    for current_dir, subdirs, files in os.walk(root_dir):
        build_data_txt_for_folder(current_dir)


if __name__ == "__main__":
    """
    用法
    - 把脚本放在任意位置
    - root_dir 改成你的根目录
    - 运行后，每个包含 .wsd 的文件夹都会生成 data.txt
    """
    root_dir = "./Nagetive"  # 改成你的实际根目录路径
    traverse_all_folders(root_dir)
