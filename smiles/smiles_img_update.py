import os
import csv
import re
import math
import argparse
from collections import deque
from typing import Optional, Tuple, List, Dict, Set

from rdkit import Chem
from rdkit.Chem import AllChem
from rdkit.Chem.Draw import rdMolDraw2D

try:
    from rdkit.Chem import rdDepictor
except Exception:
    rdDepictor = None


# =========================
# 1) 字符串清洗与规范化
# =========================
def clean_smiles_text(s: str) -> str:
    """只保留 SMILES 主体；如果含 CXSMILES 的 |...|，仅取 '|' 前面。"""
    if not s:
        return ""
    s = s.strip()
    if not s:
        return ""
    s = s.split()[0]
    if "|" in s:
        s = s.split("|", 1)[0].strip()
    return s


def normalize_nonstandard_tokens(s: str) -> str:
    """
    处理一些非标准标记：
    - 将 {-} 替换成 -
    - 删除除 {+n} 以外的 {...}（避免 RDKit 解析失败）
    """
    if not s:
        return ""
    s = s.replace("{-}", "-")
    s = re.sub(r"\{(?!\+n\})[^}]*\}", "", s)
    return s


# =========================
# 2) 聚合物锚点：用 {+n} -> [*:999] 标记
# =========================
POLY_ANCHOR_MAP = 999  # 用于原子映射号，尽量选个不常用的大号


def contains_polymer(smiles: str) -> bool:
    return bool(smiles) and ("{+n}" in smiles)


def polymer_to_anchored_smiles(smiles: str) -> str:
    """
    把 {+n} 替换成 [*:999]（哑原子 + 原子映射号）
    这样 RDKit 解析后我们能精确定位重复连接点。
    """
    return smiles.replace("{+n}", f"[*:{POLY_ANCHOR_MAP}]")


def find_poly_anchor_atom(mol: Chem.Mol) -> Optional[int]:
    """在分子里找到原子映射号为 999 的哑原子索引。"""
    for a in mol.GetAtoms():
        if a.GetAtomMapNum() == POLY_ANCHOR_MAP:
            return a.GetIdx()
    return None


# =========================
# 3) 清理会导致奇怪标记的小属性
# =========================
def strip_atom_tags_and_notes(mol: Chem.Mol) -> Chem.Mol:
    """
    清理绘图时可能显示出来的原子属性，避免出现小方块/奇怪字符。
    注意：聚合物锚点使用 atom map，需要保留 999，所以这里不清掉 999。
    """
    if mol is None:
        return mol
    for a in mol.GetAtoms():
        # 保留 999，其它 map 清掉
        if a.GetAtomMapNum() and a.GetAtomMapNum() != POLY_ANCHOR_MAP:
            a.SetAtomMapNum(0)
        for p in ("_MolFileRLabel", "_MolFileAlias", "molFileAlias", "atomNote", "_atomNote"):
            if a.HasProp(p):
                a.ClearProp(p)
    return mol


# =========================
# 4) 盐的处理（默认保留盐；可选删除）
# =========================
def keep_largest_fragment(mol: Chem.Mol) -> Chem.Mol:
    """去盐：只保留最大片段（重原子数最多）。"""
    frags = Chem.GetMolFrags(mol, asMols=True, sanitizeFrags=False)
    if len(frags) <= 1:
        return mol
    frags = sorted(frags, key=lambda m: m.GetNumHeavyAtoms(), reverse=True)
    main = frags[0]
    try:
        Chem.SanitizeMol(main)
    except Exception:
        pass
    return main


# =========================
# 5) 金属识别（用于禁用 kekulize 等）
# =========================
METALS = {
    "Li", "Na", "K", "Rb", "Cs", "Mg", "Ca", "Sr", "Ba",
    "Ti", "V", "Cr", "Mn", "Fe", "Co", "Ni", "Cu", "Zn",
    "Zr", "Mo", "W", "Ru", "Rh", "Pd", "Ag", "Cd", "Pt", "Au", "Hg",
    "Al", "Ga", "In", "Sn", "Pb", "Bi"
}


def has_metal(mol: Chem.Mol) -> bool:
    return any(a.GetSymbol() in METALS for a in mol.GetAtoms())


# =========================
# 6) 稳定 2D 坐标：多次尝试选最不重叠（慢一点但稳）
# =========================
def overlap_score(mol: Chem.Mol) -> float:
    """非成键原子太近会惩罚，分数越低越不挤。"""
    if mol.GetNumConformers() == 0:
        return 1e18
    conf = mol.GetConformer()
    n = mol.GetNumAtoms()

    bonded = set()
    for b in mol.GetBonds():
        i, j = b.GetBeginAtomIdx(), b.GetEndAtomIdx()
        if i > j:
            i, j = j, i
        bonded.add((i, j))

    score = 0.0
    for i in range(n):
        pi = conf.GetAtomPosition(i)
        for j in range(i + 1, n):
            if (i, j) in bonded:
                continue
            pj = conf.GetAtomPosition(j)
            dx = pi.x - pj.x
            dy = pi.y - pj.y
            d2 = dx * dx + dy * dy
            if d2 < 0.25:
                score += 50.0
            elif d2 < 0.64:
                score += 10.0
            elif d2 < 1.0:
                score += 2.0
    return score


def compute_2d_coords_robust(mol: Chem.Mol, tries: int = 12) -> Chem.Mol:
    """多次生成 2D 坐标，选 overlap_score 最小的那个。"""
    best = None
    best_sc = 1e18

    for t in range(max(1, tries)):
        m = Chem.Mol(mol)

        if rdDepictor is not None and hasattr(rdDepictor, "SetPreferCoordGen"):
            rdDepictor.SetPreferCoordGen((t % 2) == 0)

        try:
            if rdDepictor is not None and hasattr(rdDepictor, "Compute2DCoords"):
                rdDepictor.Compute2DCoords(m, canonOrient=((t % 3) == 0))
            else:
                AllChem.Compute2DCoords(m)
        except Exception:
            try:
                AllChem.Compute2DCoords(m)
            except Exception:
                continue

        sc = overlap_score(m)
        if sc < best_sc:
            best_sc = sc
            best = m

    if best is None:
        best = Chem.Mol(mol)
        AllChem.Compute2DCoords(best)

    return best


# =========================
# 7) 旋转：用聚合物锚点两侧邻接原子连线 -> 水平（极稳）
# =========================
def rotate_mol_2d_inplace(mol: Chem.Mol, angle_rad: float, center: Tuple[float, float]):
    """将分子 2D 坐标绕 center 旋转 angle_rad（弧度）。"""
    if mol.GetNumConformers() == 0:
        return
    conf = mol.GetConformer()
    cx, cy = center
    ca = math.cos(angle_rad)
    sa = math.sin(angle_rad)

    for i in range(mol.GetNumAtoms()):
        p = conf.GetAtomPosition(i)
        x, y = float(p.x), float(p.y)
        x -= cx
        y -= cy
        xr = x * ca - y * sa
        yr = x * sa + y * ca
        conf.SetAtomPosition(i, Chem.rdGeometry.Point3D(xr + cx, yr + cy, float(p.z)))


def orient_by_poly_anchor_horizontal(mol2d: Chem.Mol, anchor_idx: int) -> bool:
    """
    让“聚合物连接点”水平：
    - anchor 是 [*:999]，它通常有 2 个邻居（左/右）
    - 用这两个邻居的连线作为主链方向，旋转到水平
    """
    if mol2d.GetNumConformers() == 0:
        return False
    a = mol2d.GetAtomWithIdx(anchor_idx)
    nbs = [nb.GetIdx() for nb in a.GetNeighbors()]
    if len(nbs) < 2:
        return False

    conf = mol2d.GetConformer()
    p1 = conf.GetAtomPosition(nbs[0])
    p2 = conf.GetAtomPosition(nbs[1])

    dx = float(p2.x - p1.x)
    dy = float(p2.y - p1.y)
    if abs(dx) < 1e-9 and abs(dy) < 1e-9:
        return False

    angle = math.atan2(dy, dx)
    cx = (float(p1.x) + float(p2.x)) / 2.0
    cy = (float(p1.y) + float(p2.y)) / 2.0
    rotate_mol_2d_inplace(mol2d, -angle, (cx, cy))
    return True


# =========================
# 8) 选择 repeat_smarts 的“正确 match”（靠近锚点优先）
# =========================
def min_graph_distance_to_set(mol: Chem.Mol, src_atoms: Set[int], tgt_atoms: Set[int]) -> int:
    """返回 src 集合到 tgt 集合的最小图距离（BFS）。"""
    if not src_atoms or not tgt_atoms:
        return 10**9

    q = deque()
    dist = {a: 0 for a in src_atoms}
    for a in src_atoms:
        q.append(a)

    while q:
        u = q.popleft()
        if u in tgt_atoms:
            return dist[u]
        for nb in mol.GetAtomWithIdx(u).GetNeighbors():
            v = nb.GetIdx()
            if v not in dist:
                dist[v] = dist[u] + 1
                q.append(v)
    return 10**9


def choose_best_match_near_anchor(mol: Chem.Mol, matches: List[Tuple[int, ...]], anchor_idx: int) -> Tuple[int, ...]:
    """
    在多个 matches 里选择“最接近锚点”的那个：
    - 优先：match 包含锚点邻居（锚点两侧那两个原子）
    - 其次：match 到锚点邻居集合的最小图距离最小
    """
    anchor_atom = mol.GetAtomWithIdx(anchor_idx)
    anchor_nbs = {nb.GetIdx() for nb in anchor_atom.GetNeighbors()}

    # 1) 直接包含锚点邻居的优先
    candidates = []
    for m in matches:
        ms = set(int(i) for i in m)
        hit = len(ms & anchor_nbs)
        candidates.append((hit, m))

    candidates.sort(key=lambda x: (-x[0], -len(x[1])))
    if candidates and candidates[0][0] > 0:
        return candidates[0][1]

    # 2) 用图距离选最接近的
    best = matches[0]
    best_d = 10**9
    for m in matches:
        ms = set(int(i) for i in m)
        d = min_graph_distance_to_set(mol, anchor_nbs, ms)
        if d < best_d:
            best_d = d
            best = m
    return best


# =========================
# 9) BFS 邻域：repeat_smarts 缺失/失败时，用锚点邻域做重复单元近似
# =========================
def bfs_neighborhood_atoms(mol: Chem.Mol, seed_atoms: Set[int], max_depth: int) -> Set[int]:
    """从 seed_atoms 出发 BFS 扩展 max_depth，返回覆盖到的原子集合。"""
    visited = set(seed_atoms)
    q = deque([(a, 0) for a in seed_atoms])
    while q:
        u, d = q.popleft()
        if d >= max_depth:
            continue
        for nb in mol.GetAtomWithIdx(u).GetNeighbors():
            v = nb.GetIdx()
            if v not in visited:
                visited.add(v)
                q.append((v, d + 1))
    return visited


# =========================
# 10) SVG 工具：读取 viewBox / 插入 overlay / bbox
# =========================
def ensure_svg_has_size(svg: str, width: int, height: int) -> str:
    """强制给 <svg> 注入 width/height/viewBox。"""
    if not svg:
        return svg
    m = re.search(r"<svg\b[^>]*>", svg, flags=re.IGNORECASE)
    if not m:
        return svg

    tag = m.group(0)
    has_w = re.search(r"\bwidth\s*=", tag, flags=re.IGNORECASE) is not None
    has_h = re.search(r"\bheight\s*=", tag, flags=re.IGNORECASE) is not None
    has_vb = re.search(r"\bviewBox\s*=", tag, flags=re.IGNORECASE) is not None

    inject = ""
    if not has_w:
        inject += f" width='{width}px'"
    if not has_h:
        inject += f" height='{height}px'"
    if not has_vb:
        inject += f" viewBox='0 0 {width} {height}'"

    if inject:
        new_tag = tag[:-1] + inject + ">"
        svg = svg[:m.start()] + new_tag + svg[m.end():]
    return svg


def get_viewbox(svg: str) -> Optional[Tuple[float, float, float, float]]:
    m = re.search(r"viewBox\s*=\s*(['\"])(.*?)\1", svg, flags=re.IGNORECASE)
    if not m:
        return None
    parts = m.group(2).strip().split()
    if len(parts) != 4:
        return None
    return float(parts[0]), float(parts[1]), float(parts[2]), float(parts[3])


def insert_before_svg_end(svg: str, overlay: str) -> str:
    new_svg, n = re.subn(r"</\s*svg\s*>", overlay + r"\n</svg>", svg, count=1, flags=re.IGNORECASE)
    return new_svg if n else svg


def svg_content_bbox(svg: str) -> Optional[Tuple[float, float, float, float]]:
    """粗略解析所有 path 坐标估 bbox（退化用）。"""
    ds = re.findall(r"\bd\s*=\s*(['\"])(.*?)\1", svg, flags=re.IGNORECASE)
    if not ds:
        return None
    xs, ys = [], []
    num_re = re.compile(r"-?\d+(?:\.\d+)?")
    for _, d in ds:
        nums = num_re.findall(d)
        for i in range(0, len(nums) - 1, 2):
            xs.append(float(nums[i]))
            ys.append(float(nums[i + 1]))
    if not xs or not ys:
        return None
    return min(xs), min(ys), max(xs), max(ys)


def svg_bbox_for_atom_indices(svg: str, atom_indices: Set[int]) -> Optional[Tuple[float, float, float, float]]:
    """
    用 atom-<idx> 相关 path + text 共同求 bbox（包含字母位置），用于避免括号压字。
    """
    if not svg or not atom_indices:
        return None

    idx_set = set(int(i) for i in atom_indices)
    xs, ys = [], []

    path_re = re.compile(r"<path\b[^>]*>", re.IGNORECASE)
    text_re = re.compile(r"<text\b[^>]*>", re.IGNORECASE)

    class_re = re.compile(r"class\s*=\s*(['\"])(.*?)\1", re.IGNORECASE)
    d_re = re.compile(r"\bd\s*=\s*(['\"])(.*?)\1", re.IGNORECASE)
    x_re = re.compile(r"\bx\s*=\s*(['\"])(.*?)\1", re.IGNORECASE)
    y_re = re.compile(r"\by\s*=\s*(['\"])(.*?)\1", re.IGNORECASE)
    num_re = re.compile(r"-?\d+(?:\.\d+)?")

    for tag in path_re.findall(svg):
        mcls = class_re.search(tag)
        md = d_re.search(tag)
        if not mcls or not md:
            continue
        cls = mcls.group(2)
        if not any(f"atom-{i}" in cls for i in idx_set):
            continue
        nums = num_re.findall(md.group(2))
        for k in range(0, len(nums) - 1, 2):
            xs.append(float(nums[k]))
            ys.append(float(nums[k + 1]))

    for tag in text_re.findall(svg):
        mcls = class_re.search(tag)
        mx = x_re.search(tag)
        my = y_re.search(tag)
        if not mcls or not mx or not my:
            continue
        cls = mcls.group(2)
        if not any(f"atom-{i}" in cls for i in idx_set):
            continue
        try:
            xs.append(float(mx.group(2)))
            ys.append(float(my.group(2)))
        except Exception:
            pass

    if not xs or not ys:
        return None
    return min(xs), min(ys), max(xs), max(ys)


# =========================
# 11) 聚合物括号：去掉“-”，尽量不压字；方向正确：[  ]
# =========================
def add_polymer_brackets_canvas(svg: str, label: str = "n") -> str:
    """A 模式：画布固定位置括号。"""
    vb = get_viewbox(svg)
    if vb is None:
        return svg
    minx, miny, w, h = vb

    y1 = miny + h * 0.35
    y2 = miny + h * 0.65
    xL = minx + w * 0.10
    xR = minx + w * 0.90
    hook = max(8.0, min(w, h) * 0.015)

    overlay = f"""
<g id='polymer_brackets' stroke='#000' stroke-width='2' fill='none'>
  <!-- 左括号 [ ：横线向右 -->
  <path d='M {xL} {y1} L {xL+hook} {y1} M {xL} {y1} L {xL} {y2} M {xL} {y2} L {xL+hook} {y2}' />
  <!-- 右括号 ] ：横线向左 -->
  <path d='M {xR} {y1} L {xR-hook} {y1} M {xR} {y1} L {xR} {y2} M {xR} {y2} L {xR-hook} {y2}' />
  <text x='{xR+12}' y='{y2+18}' font-family='Helvetica, Arial, sans-serif' font-size='18' fill='#000'>{label}</text>
</g>
"""
    return insert_before_svg_end(svg, overlay)


def add_polymer_brackets_repeat_unit(svg: str,
                                    atom_set_for_unit: Set[int],
                                    label: str = "n") -> str:
    """
    B 模式：括号包住 atom_set_for_unit 的 bbox（含文字），并加大外边距，避免压字。
    不画括号中间的“-”短横线。
    """
    vb = get_viewbox(svg)
    if vb is None:
        return svg
    minx, miny, w, h = vb

    bbox = svg_bbox_for_atom_indices(svg, atom_set_for_unit)
    if bbox is None:
        bbox = svg_content_bbox(svg)
        if bbox is None:
            return add_polymer_brackets_canvas(svg, label=label)

    x0, y0, x1, y1 = bbox

    # 外边距：尽量让括号在字母外侧
    pad_x = max(26.0, min(w, h) * 0.055)
    pad_y = max(18.0, min(w, h) * 0.04)

    x0 -= pad_x
    x1 += pad_x
    y0 -= pad_y
    y1 += pad_y

    # 约束到画布范围
    x0 = max(minx + 2, x0)
    x1 = min(minx + w - 2, x1)
    y0 = max(miny + 2, y0)
    y1 = min(miny + h - 2, y1)

    hook = max(12.0, min(w, h) * 0.03)

    n_x = min(minx + w - 10, x1 + 12)
    n_y = min(miny + h - 6, y1 + 18)

    overlay = f"""
<g id='polymer_brackets' stroke='#000' stroke-width='2' fill='none'>
  <!-- 左括号 [ ：横线向右 -->
  <path d='M {x0} {y0} L {x0+hook} {y0} M {x0} {y0} L {x0} {y1} M {x0} {y1} L {x0+hook} {y1}' />
  <!-- 右括号 ] ：横线向左 -->
  <path d='M {x1} {y0} L {x1-hook} {y0} M {x1} {y0} L {x1} {y1} M {x1} {y1} L {x1-hook} {y1}' />
  <text x='{n_x}' y='{n_y}' font-family='Helvetica, Arial, sans-serif' font-size='18' fill='#000'>{label}</text>
</g>
"""
    return insert_before_svg_end(svg, overlay)


# =========================
# 12) 绘图（黑白、去(R/S)、隐藏锚点“*”）
# =========================
def draw_svg(mol: Chem.Mol,
             width: int,
             height: int,
             metal_mode: bool,
             bond_length: int,
             padding: float,
             anchor_idx: Optional[int]) -> str:
    drawer = rdMolDraw2D.MolDraw2DSVG(width, height, -1, -1)
    opt = drawer.drawOptions()

    # 黑白
    opt.useBWAtomPalette()

    # 线条 / 键长 / 边距
    opt.bondLineWidth = 1.5
    opt.fixedBondLength = bond_length
    opt.padding = padding
    opt.clearBackground = True

    # 不显示 (R)/(S)
    opt.addStereoAnnotation = False
    opt.explicitMethyl = False

    # 金属体系：不 kekulize 更稳，字号略小
    if hasattr(opt, "fontSize"):
        opt.fontSize = 0.7 if metal_mode else 0.8
    if hasattr(opt, "minFontSize"):
        opt.minFontSize = 10 if metal_mode else 12
    if hasattr(opt, "maxFontSize"):
        opt.maxFontSize = 14 if metal_mode else 18

    # 隐藏锚点 [*:999] 的“*”标签，避免图上出现星号
    if anchor_idx is not None and hasattr(opt, "atomLabels"):
        try:
            opt.atomLabels[anchor_idx] = ""
        except Exception:
            pass

    kek = False if metal_mode else True
    rdMolDraw2D.PrepareAndDrawMolecule(drawer, mol, kekulize=kek)
    drawer.FinishDrawing()
    return drawer.GetDrawingText()


# =========================
# 13) 单条生成入口（核心逻辑）
# =========================
def generate_image(raw_smiles: str,
                   output_path: str,
                   width: int = 700,
                   height: int = 520,
                   fmt: str = "svg",
                   polymer_style: str = "bracket",
                   repeat_smarts: str = "",
                   robust: bool = True,
                   layout_tries: int = 20,
                   keep_salts: bool = True,
                   repeat_radius: int = 4) -> bool:
    """
    repeat_radius：
    - 当 repeat_smarts 缺失/匹配失败时，用锚点邻域 BFS 近似重复单元
    - 数值越大，括号包得越“宽”
    """
    try:
        # 1) 清洗 SMILES
        s = clean_smiles_text(raw_smiles)
        if not s:
            return False
        s = normalize_nonstandard_tokens(s)

        # 2) 聚合物：把 {+n} 替换成 [*:999] 作为锚点
        is_poly = contains_polymer(s)
        s_draw = polymer_to_anchored_smiles(s) if is_poly else s

        mol = Chem.MolFromSmiles(s_draw)
        if not mol:
            return False

        mol = strip_atom_tags_and_notes(mol)

        # 3) 是否去盐
        if not keep_salts:
            mol = keep_largest_fragment(mol)

        metal_mode = has_metal(mol)

        # 4) 2D 坐标：稳（慢）或普通（快）
        if robust:
            mol2d = compute_2d_coords_robust(mol, tries=layout_tries)
        else:
            mol2d = Chem.Mol(mol)
            AllChem.Compute2DCoords(mol2d)

        # 5) 找锚点
        anchor_idx = find_poly_anchor_atom(mol2d) if is_poly else None

        # 6) B 模式：用锚点两侧连线强制水平（不依赖 repeat_smarts，极稳）
        if is_poly and polymer_style == "repeat_unit" and anchor_idx is not None:
            orient_by_poly_anchor_horizontal(mol2d, anchor_idx)

        # 7) 根据挤压程度动态调 bondLength/padding（更稳更不挤）
        sc = overlap_score(mol2d)
        if metal_mode:
            if sc > 30:
                bond_length, padding = 38, 0.13
            elif sc > 10:
                bond_length, padding = 36, 0.12
            else:
                bond_length, padding = 34, 0.11
        else:
            if sc > 30:
                bond_length, padding = 34, 0.10
            elif sc > 10:
                bond_length, padding = 32, 0.09
            else:
                bond_length, padding = 30, 0.08

        # 8) 输出
        if fmt == "svg":
            svg = draw_svg(mol2d, width, height, metal_mode, bond_length, padding, anchor_idx)
            svg = ensure_svg_has_size(svg, width, height)

            # 9) 聚合物括号
            if is_poly and polymer_style == "bracket":
                svg = add_polymer_brackets_canvas(svg, label="n")

            elif is_poly and polymer_style == "repeat_unit":
                # 9.1 先确定“重复单元原子集合”
                unit_atoms: Set[int] = set()

                if anchor_idx is not None:
                    # 锚点邻居作为 seed（通常 2 个）
                    anchor_nbs = {nb.GetIdx() for nb in mol2d.GetAtomWithIdx(anchor_idx).GetNeighbors()}
                else:
                    anchor_nbs = set()

                # 9.2 优先用 repeat_smarts（更精准），并用“靠近锚点”规则选 match
                if repeat_smarts:
                    patt = Chem.MolFromSmarts(repeat_smarts)
                    if patt is not None:
                        matches = mol2d.GetSubstructMatches(patt)
                        if matches:
                            if anchor_idx is not None:
                                best = choose_best_match_near_anchor(mol2d, list(matches), anchor_idx)
                            else:
                                best = max(matches, key=lambda m: len(m))
                            unit_atoms = set(int(i) for i in best)

                # 9.3 若 repeat_smarts 缺失/失败：用 BFS 邻域近似（更稳）
                if not unit_atoms:
                    if anchor_nbs:
                        unit_atoms = bfs_neighborhood_atoms(mol2d, anchor_nbs, max_depth=repeat_radius)
                    else:
                        # 最退化：包住整图
                        unit_atoms = set(range(mol2d.GetNumAtoms()))

                svg = add_polymer_brackets_repeat_unit(svg, unit_atoms, label="n")

            with open(output_path, "w", encoding="utf-8") as f:
                f.write(svg)
            return True

        elif fmt == "png":
            # PNG：Cairo 输出（如果环境支持）
            try:
                drawer = rdMolDraw2D.MolDraw2DCairo(width, height)
            except AttributeError:
                print("错误：当前 RDKit 环境不支持 MolDraw2DCairo，无法输出 PNG。")
                return False

            opt = drawer.drawOptions()
            opt.useBWAtomPalette()
            opt.bondLineWidth = 1.5
            opt.fixedBondLength = bond_length
            opt.padding = padding
            opt.clearBackground = True
            opt.addStereoAnnotation = False
            opt.explicitMethyl = False

            # 隐藏锚点星号
            if anchor_idx is not None and hasattr(opt, "atomLabels"):
                try:
                    opt.atomLabels[anchor_idx] = ""
                except Exception:
                    pass

            kek = False if metal_mode else True
            rdMolDraw2D.PrepareAndDrawMolecule(drawer, mol2d, kekulize=kek)
            drawer.FinishDrawing()
            content = drawer.GetDrawingText()
            with open(output_path, "wb") as f:
                f.write(content)
            return True

        return False

    except Exception as e:
        print(f"生成失败：{raw_smiles}，原因：{e}")
        return False


# =========================
# 14) 批量处理
# =========================
def process_batch(input_csv: str,
                  output_dir: str,
                  width: int,
                  height: int,
                  fmt: str,
                  polymer_style: str,
                  repeat_smarts: str,
                  robust: bool,
                  layout_tries: int,
                  keep_salts: bool,
                  repeat_radius: int):
    if not os.path.exists(output_dir):
        os.makedirs(output_dir)
        print(f"已创建输出目录：{output_dir}")

    count = 0
    success = 0
    print(f"开始批量处理：{input_csv}")
    smiles_path_list = []

    with open(input_csv, "r", encoding="utf-8") as f:
        reader = csv.reader(f)
        for row in reader:
            if not row:
                continue
            raw = row[0].strip() if row[0] else ""
            filename = os.path.join(output_dir, f"{count}.{fmt}")

            if generate_image(raw, filename, width, height, fmt,
                              polymer_style=polymer_style,
                              repeat_smarts=repeat_smarts,
                              robust=robust,
                              layout_tries=layout_tries,
                              keep_salts=keep_salts,
                              repeat_radius=repeat_radius):
                success += 1
                smiles_path_list.append({"smiles": raw, "path": filename})

            count += 1
            if count % 100 == 0:
                print(f"已处理 {count} 条（成功 {success} 条）", end="\r")
            if count % 1000 == 0:
                break

    with open(os.path.join(output_dir, "result.json"), 'w', encoding='utf-8', newline='') as f:
        import json
        json.dump(smiles_path_list, f, ensure_ascii=False, indent=2)

    print(f"\n批量完成：成功 {success}/{count}，输出目录：{output_dir}")

# =========================
# 15) 命令行入口
# =========================
if __name__ == "__main__":
    parser = argparse.ArgumentParser(
        description="将 SMILES 转换为黑白 SVG/PNG（稳定版）。B 模式用 {+n} 锚点强制水平，并用靠近锚点的 repeat_smarts 或 BFS 邻域定位括号；去掉括号中的“-”，并尽量避免压字。"
    )
    parser.add_argument("--smiles", help="单条 SMILES")
    parser.add_argument("-i", "--input", help="批量模式：输入 CSV（第一列为 SMILES）")
    parser.add_argument("-o", "--output", help="单条模式：输出文件名；批量模式：输出目录")
    parser.add_argument("--format", choices=["svg", "png"], default="png", help="输出格式")
    parser.add_argument("--width", type=int, default=700, help="图片宽度")
    parser.add_argument("--height", type=int, default=520, help="图片高度")

    parser.add_argument(
        "--polymer_style",
        choices=["bracket", "repeat_unit", "off"],
        default="bracket",
        help="聚合物（含 {+n}）显示方式：bracket=画布固定括号(A)；repeat_unit=锚点水平+重复单元括号(B)；off=不加括号"
    )
    parser.add_argument(
        "--repeat_smarts",
        default="",
        help="B模式：重复单元 SMARTS（建议给能唯一定位的，例如 OCCO；太短会多匹配）。"
    )
    parser.add_argument(
        "--repeat_radius",
        type=int,
        default=4,
        help="当 repeat_smarts 缺失/失败时，使用锚点邻域 BFS 的深度（越大括号包得越宽）。默认 4"
    )

    parser.add_argument("--robust", action="store_true", help="启用更稳的 2D 布局（慢一点但更稳）")
    parser.add_argument("--layout_tries", type=int, default=20, help="稳布局尝试次数（越大越稳但更慢）")
    parser.add_argument("--remove_salts", action="store_true", help="去盐：只保留最大片段（默认保留盐/对离子）")

    args = parser.parse_args()
    keep_salts = not args.remove_salts

    if args.input:
        out_dir = args.output if args.output else "output_images"
        process_batch(args.input, out_dir, args.width, args.height, args.format,
                      args.polymer_style, args.repeat_smarts, args.robust, args.layout_tries,
                      keep_salts, args.repeat_radius)
    elif args.smiles:
        out_file = args.output if args.output else f"mol.{args.format}"
        ok = generate_image(args.smiles, out_file, args.width, args.height, args.format,
                            polymer_style=args.polymer_style,
                            repeat_smarts=args.repeat_smarts,
                            robust=args.robust,
                            layout_tries=args.layout_tries,
                            keep_salts=keep_salts,
                            repeat_radius=args.repeat_radius)
        print(f"已保存：{out_file}" if ok else "失败：无法解析/绘制该 SMILES")
    else:
        print("错误：请提供 --smiles 或 --input")
