# -*- coding: utf-8 -*-
import os
import csv
import re
import math
from collections import deque
from typing import Optional, Tuple, Set

from rdkit import Chem
from rdkit.Chem import AllChem
from rdkit.Chem.Draw import rdMolDraw2D

try:
    from rdkit.Chem import rdDepictor
except Exception:
    rdDepictor = None
from PIL import Image, ImageChops

import sys
sys.path.append("/root/binghao/smiles/app/img2smles/utils")
from smiles_selfies_converter import smiles_to_selfies, selfies_to_smiles
# =========================
# 🔧 配置区（用户可修改）
# =========================

DEFAULT_CONFIG = {
    "mode": "batch",  # 'single' 或 'batch'
    "input_csv": "/root/binghao/smiles/smiles_export.csv",  # 批量输入路径
    "output_dir": "/root/binghao/smiles/app/img2smles/dataset/images",  # 批量输出目录
    "single_smiles": "CC{+n}",  # 单条模式输入
    "single_output": "mol.svg",
    "format": "png",  # "svg" 或 "png"
    "width": 384,
    "height": 384,
    "polymer_style": "repeat_unit",  # "bracket", "repeat_unit", "off"
    "repeat_smarts": "",  # 如 "COCO"，用于精确定位重复单元（可选）
    "repeat_radius": 4,   # BFS邻域深度（fallback用）
    "robust_layout": True,
    "layout_tries": 20,
    "keep_salts": True,
}


# =========================
# 1) 工具函数：SMILES 清洗
# =========================
def clean_smiles_text(s: str) -> str:
    s = s.strip().split()[0] if s else ""
    return s.split("|", 1)[0].strip() if "|" in s else s


def normalize_nonstandard_tokens(s: str) -> str:
    s = s.replace("{-}", "-")
    s = re.sub(r"\{(?!\+n\})[^}]*\}", "", s)
    return s


# =========================
# 2) 聚合物锚点处理
# =========================
POLY_ANCHOR_MAP = 999


def contains_polymer(smiles: str) -> bool:
    return "{+n}" in smiles


def polymer_to_anchored_smiles(smiles: str) -> str:
    return smiles.replace("{+n}", f"[*:{POLY_ANCHOR_MAP}]")


def find_poly_anchor_atom(mol: Chem.Mol) -> Optional[int]:
    for a in mol.GetAtoms():
        if a.GetAtomMapNum() == POLY_ANCHOR_MAP:
            return a.GetIdx()
    return None


def clear_all_atom_maps_except_anchor(mol: Chem.Mol):
    """只保留 POLY_ANCHOR_MAP=999，其它 atom map 全部清除（防红框）"""
    for a in mol.GetAtoms():
        if a.GetAtomMapNum() != POLY_ANCHOR_MAP:
            a.SetAtomMapNum(0)


# =========================
# 3) 分子预处理
# =========================
def strip_atom_notes(mol: Chem.Mol):
    props_to_clear = ["_MolFileRLabel", "_MolFileAlias", "atomNote", "_atomNote"]
    for a in mol.GetAtoms():
        for p in props_to_clear:
            if a.HasProp(p):
                a.ClearProp(p)


def keep_largest_fragment(mol: Chem.Mol) -> Chem.Mol:
    frags = Chem.GetMolFrags(mol, asMols=True, sanitizeFrags=False)
    if len(frags) <= 1:
        return mol
    main = max(frags, key=lambda m: m.GetNumHeavyAtoms())
    try:
        Chem.SanitizeMol(main)
    except Exception:
        pass
    return main


def has_metal(mol: Chem.Mol) -> bool:
    METALS = {"Li", "Na", "K", "Mg", "Ca", "Fe", "Co", "Ni", "Cu", "Zn", "Pt", "Au", "Hg", "Al", "Pb", "Bi"}
    return any(a.GetSymbol() in METALS for a in mol.GetAtoms())


# =========================
# 4) 稳定 2D 坐标生成
# =========================
def overlap_score(mol: Chem.Mol) -> float:
    if not mol.GetNumConformers():
        return 1e18
    conf = mol.GetConformer()
    bonded = {(min(b.GetBeginAtomIdx(), b.GetEndAtomIdx()),
               max(b.GetBeginAtomIdx(), b.GetEndAtomIdx())) for b in mol.GetBonds()}
    score = 0.0
    n = mol.GetNumAtoms()
    for i in range(n):
        pi = conf.GetAtomPosition(i)
        for j in range(i + 1, n):
            if (i, j) in bonded:
                continue
            pj = conf.GetAtomPosition(j)
            d2 = (pi.x - pj.x)**2 + (pi.y - pj.y)**2
            if d2 < 0.25:
                score += 50.0
            elif d2 < 0.64:
                score += 10.0
            elif d2 < 1.0:
                score += 2.0
    return score


def compute_2d_coords_robust(mol: Chem.Mol, tries: int = 12) -> Chem.Mol:
    best, best_sc = None, 1e18
    for _ in range(max(1, tries)):
        m = Chem.Mol(mol)
        use_coordgen = (_ % 2) == 0
        canon_orient = (_ % 3) == 0

        if rdDepictor and hasattr(rdDepictor, "SetPreferCoordGen"):
            rdDepictor.SetPreferCoordGen(use_coordgen)

        success = False
        try:
            if rdDepictor and hasattr(rdDepictor, "Compute2DCoords"):
                rdDepictor.Compute2DCoords(m, canonOrient=canon_orient)
            else:
                AllChem.Compute2DCoords(m)
            success = True
        except Exception:
            try:
                AllChem.Compute2DCoords(m)
                success = True
            except Exception:
                pass

        if success:
            sc = overlap_score(m)
            if sc < best_sc:
                best_sc, best = sc, m

    return best or Chem.Mol(mol)  # fallback


# =========================
# 5) 旋转对齐锚点连线至水平
# =========================
def rotate_mol_2d_inplace(mol: Chem.Mol, angle_rad: float, center: Tuple[float, float]):
    cx, cy = center
    ca, sa = math.cos(angle_rad), math.sin(angle_rad)
    conf = mol.GetConformer()
    for i in range(mol.GetNumAtoms()):
        p = conf.GetAtomPosition(i)
        x, y = p.x - cx, p.y - cy
        xr = x * ca - y * sa
        yr = x * sa + y * ca
        conf.SetAtomPosition(i, Chem.rdGeometry.Point3D(xr + cx, yr + cy, p.z))


def orient_by_poly_anchor_horizontal(mol: Chem.Mol, anchor_idx: int) -> bool:
    nbs = [nb.GetIdx() for nb in mol.GetAtomWithIdx(anchor_idx).GetNeighbors()]
    if len(nbs) < 2:
        return False
    conf = mol.GetConformer()
    p1 = conf.GetAtomPosition(nbs[0])
    p2 = conf.GetAtomPosition(nbs[1])
    dx, dy = p2.x - p1.x, p2.y - p1.y
    if abs(dx) < 1e-9 and abs(dy) < 1e-9:
        return False
    angle = math.atan2(dy, dx)
    cx = (p1.x + p2.x) / 2.0
    cy = (p1.y + p2.y) / 2.0
    rotate_mol_2d_inplace(mol, -angle, (cx, cy))
    return True


# =========================
# 6) 定位重复单元原子集合
# =========================
def min_graph_distance_to_set(mol: Chem.Mol, src: Set[int], tgt: Set[int]) -> int:
    visited = dict.fromkeys(src, 0)
    q = deque(src)
    while q:
        u = q.popleft()
        if u in tgt:
            return visited[u]
        for nb in mol.GetAtomWithIdx(u).GetNeighbors():
            v = nb.GetIdx()
            if v not in visited:
                visited[v] = visited[u] + 1
                q.append(v)
    return 10**9


def choose_best_match_near_anchor(mol: Chem.Mol, matches: list, anchor_idx: int) -> tuple:
    anchor_nbs = {nb.GetIdx() for nb in mol.GetAtomWithIdx(anchor_idx).GetNeighbors()}
    scored = [(len(set(m) & anchor_nbs), -len(m), m) for m in matches]
    scored.sort()
    best = scored[-1][-1]
    return best if scored[-1][0] > 0 else min(
        matches,
        key=lambda m: min_graph_distance_to_set(mol, anchor_nbs, set(m))
    )


def bfs_neighborhood_atoms(mol: Chem.Mol, seeds: Set[int], max_depth: int) -> Set[int]:
    visited = set(seeds)
    q = deque((a, 0) for a in seeds)
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
# 7) SVG 工具函数
# =========================
def ensure_svg_has_size(svg: str, w: int, h: int) -> str:
    tag_match = re.search(r"<svg\b[^>]*>", svg, re.IGNORECASE)
    if not tag_match:
        return svg
    tag = tag_match.group(0)
    inject = ""
    if not re.search(r"\bwidth=", tag, re.IGNORECASE):
        inject += f" width='{w}px'"
    if not re.search(r"\bheight=", tag, re.IGNORECASE):
        inject += f" height='{h}px'"
    if not re.search(r"\bviewBox=", tag, re.IGNORECASE):
        inject += f" viewBox='0 0 {w} {h}'"
    if inject:
        new_tag = tag[:-1] + inject + ">"
        svg = svg[:tag_match.start()] + new_tag + svg[tag_match.end():]
    return svg


def get_viewbox(svg: str) -> Optional[Tuple[float, float, float, float]]:
    m = re.search(r"viewBox\s*=\s*(['\"])(.*?)\1", svg, re.IGNORECASE)
    if not m:
        return None
    parts = m.group(2).strip().split()
    if len(parts) != 4:
        return None
    return tuple(map(float, parts))


def insert_before_svg_end(svg: str, overlay: str) -> str:
    return re.sub(r"</svg>", f"{overlay}\n</svg>", svg, count=1, flags=re.IGNORECASE)


def svg_bbox_for_atom_indices(svg: str, atom_indices: Set[int]) -> Optional[Tuple[float, float, float, float]]:
    xs, ys = [], []
    idx_set = set(atom_indices)
    num_re = re.compile(r"-?\d+(?:\.\d+)?")

    # 提取含 atom-X 的 path 和 text 坐标
    for tag in re.finditer(r"<path\b[^>]*>", svg, re.IGNORECASE):
        cls = re.search(r"class=['\"](.*?)['\"]", tag.group(), re.IGNORECASE)
        d = re.search(r"d=['\"](.*?)['\"]", tag.group(), re.IGNORECASE)
        if not cls or not d:
            continue
        if not any(f"atom-{i}" in cls.group(1) for i in idx_set):
            continue
        nums = num_re.findall(d.group(1))
        for k in range(0, len(nums) - 1, 2):
            xs.append(float(nums[k]))
            ys.append(float(nums[k + 1]))

    for tag in re.finditer(r"<text\b[^>]*>", svg, re.IGNORECASE):
        cls = re.search(r"class=['\"](.*?)['\"]", tag.group(), re.IGNORECASE)
        x = re.search(r"x=['\"](.*?)['\"]", tag.group(), re.IGNORECASE)
        y = re.search(r"y=['\"](.*?)['\"]", tag.group(), re.IGNORECASE)
        if not cls or not x or not y:
            continue
        if not any(f"atom-{i}" in cls.group(1) for i in idx_set):
            continue
        try:
            xs.append(float(x.group(1)))
            ys.append(float(y.group(1)))
        except Exception:
            pass

    return (min(xs), min(ys), max(xs), max(ys)) if xs and ys else None


# =========================
# 8) 添加聚合物括号
# =========================
def add_polymer_brackets_canvas(svg: str, label: str = "n") -> str:
    vb = get_viewbox(svg)
    if not vb:
        return svg
    minx, miny, w, h = vb
    y1, y2 = miny + h * 0.35, miny + h * 0.65
    xL, xR = minx + w * 0.10, minx + w * 0.90
    hook = max(8.0, min(w, h) * 0.015)
    overlay = f"""
<g stroke='#000' stroke-width='2' fill='none'>
  <path d='M{xL},{y1}L{xL+hook},{y1}M{xL},{y1}L{xL},{y2}M{xL},{y2}L{xL+hook},{y2}'/>
  <path d='M{xR},{y1}L{xR-hook},{y1}M{xR},{y1}L{xR},{y2}M{xR},{y2}L{xR-hook},{y2}'/>
  <text x='{xR+12}' y='{y2+18}' font-family='sans-serif' font-size='18'>{label}</text>
</g>"""
    return insert_before_svg_end(svg, overlay)


def add_polymer_brackets_repeat_unit(svg: str, unit_atoms: Set[int], label: str = "n") -> str:
    vb = get_viewbox(svg)
    if not vb:
        return svg
    minx, miny, w, h = vb

    bbox = svg_bbox_for_atom_indices(svg, unit_atoms) or (minx, miny, minx + w, miny + h)
    x0, y0, x1, y1 = bbox

    pad_x = max(26.0, min(w, h) * 0.055)
    pad_y = max(18.0, min(w, h) * 0.04)
    x0 -= pad_x; x1 += pad_x; y0 -= pad_y; y1 += pad_y

    x0 = max(minx + 2, x0)
    x1 = min(minx + w - 2, x1)
    y0 = max(miny + 2, y0)
    y1 = min(miny + h - 2, y1)

    hook = max(12.0, min(w, h) * 0.03)
    n_x = min(minx + w - 10, x1 + 12)
    n_y = min(miny + h - 6, y1 + 18)

    overlay = f"""
<g stroke='#000' stroke-width='2' fill='none'>
  <path d='M{x0},{y0}L{x0+hook},{y0}M{x0},{y0}L{x0},{y1}M{x0},{y1}L{x0+hook},{y1}'/>
  <path d='M{x1},{y0}L{x1-hook},{y0}M{x1},{y0}L{x1},{y1}M{x1},{y1}L{x1-hook},{y1}'/>
  <text x='{n_x}' y='{n_y}' font-family='sans-serif' font-size='18'>{label}</text>
</g>"""
    return insert_before_svg_end(svg, overlay)


# =========================
# 9) 绘图主函数
# =========================
def draw_molecule_image(mol: Chem.Mol,
                        width: int,
                        height: int,
                        fmt: str,
                        is_polymer: bool,
                        polymer_style: str,
                        repeat_smarts: str,
                        anchor_idx: Optional[int],
                        repeat_radius: int) -> Optional[str]:
    metal_mode = has_metal(mol)
    sc = overlap_score(mol)

    # 动态调整键长和边距
    bond_length = 34 if sc > 30 else 32 if sc > 10 else 30
    padding = 0.10 if sc > 30 else 0.09 if sc > 10 else 0.08
    if metal_mode:
        bond_length += 2
        padding += 0.01

    # SVG 输出
    if fmt == "svg":
        drawer = rdMolDraw2D.MolDraw2DSVG(width, height)
        opt = drawer.drawOptions()
        opt.useBWAtomPalette()
        opt.bondLineWidth = 1.5
        opt.fixedBondLength = bond_length
        opt.padding = padding
        opt.clearBackground = True
        opt.addStereoAnnotation = False
        opt.explicitMethyl = False
        if hasattr(opt, "fontSize"):
            opt.fontSize = 0.7 if metal_mode else 0.8

        # 隐藏锚点星号
        if anchor_idx is not None and hasattr(opt, "atomLabels"):
            opt.atomLabels[anchor_idx] = ""

        kekulize = not metal_mode
        rdMolDraw2D.PrepareAndDrawMolecule(drawer, mol, kekulize=kekulize)
        drawer.FinishDrawing()
        svg = drawer.GetDrawingText()
        svg = ensure_svg_has_size(svg, width, height)

        # 添加括号
        if is_polymer and polymer_style == "bracket":
            svg = add_polymer_brackets_canvas(svg)
        elif is_polymer and polymer_style == "repeat_unit":
            unit_atoms = set()

            # 优先 SMARTS 匹配
            if repeat_smarts and anchor_idx is not None:
                patt = Chem.MolFromSmarts(repeat_smarts)
                if patt is not None:
                    matches = mol.GetSubstructMatches(patt)
                    if matches:
                        best = choose_best_match_near_anchor(mol, matches, anchor_idx)
                        unit_atoms = set(best)

            # fallback：BFS 邻域
            if not unit_atoms and anchor_idx is not None:
                anchor_nbs = {nb.GetIdx() for nb in mol.GetAtomWithIdx(anchor_idx).GetNeighbors()}
                if anchor_nbs:
                    unit_atoms = bfs_neighborhood_atoms(mol, anchor_nbs, repeat_radius)
            if not unit_atoms:
                unit_atoms = set(range(mol.GetNumAtoms()))

            svg = add_polymer_brackets_repeat_unit(svg, unit_atoms)

        return svg

    # PNG 输出
    elif fmt == "png":
        try:
            drawer = rdMolDraw2D.MolDraw2DCairo(width, height)
        except AttributeError:
            print("❌ 不支持 Cairo，无法生成 PNG")
            return None

        opt = drawer.drawOptions()
        opt.useBWAtomPalette()
        opt.bondLineWidth = 1.5
        opt.fixedBondLength = bond_length
        opt.padding = padding
        opt.clearBackground = True
        opt.addStereoAnnotation = False
        if hasattr(opt, "atomLabels") and anchor_idx is not None:
            opt.atomLabels[anchor_idx] = ""

        kekulize = not metal_mode
        rdMolDraw2D.PrepareAndDrawMolecule(drawer, mol, kekulize=kekulize)
        drawer.FinishDrawing()
        return drawer.GetDrawingText()

    return None

def trim_png_whitespace(png_path: str, output_path: str, bg_color=(255, 255, 255)):
    """
    裁剪 PNG 图像的白色背景边框
    :param png_path: 输入路径
    :param output_path: 输出路径
    :param bg_color: 背景色，默认白色
    """
    img = Image.open(png_path).convert("RGB")
    bg = Image.new(img.mode, img.size, bg_color)
    diff = ImageChops.difference(img, bg)
    bbox = diff.getbbox()  # 获取非背景区域的包围盒
    if bbox:
        cropped = img.crop(bbox)
        # 可选：加一点边距
        padded = Image.new(img.mode, (
            cropped.width + 30,
            cropped.height + 30
        ), bg_color)
        padded.paste(cropped, (10, 10))
        padded.save(output_path, "PNG")
    else:
        img.save(output_path, "PNG")

# =========================
# 10) 主流程封装
# =========================

def generate_image(raw_smiles: str, output_path: str, config=DEFAULT_CONFIG) -> bool:
    try:
        s = clean_smiles_text(raw_smiles)
        if not s:
            return False
        s = normalize_nonstandard_tokens(s)

        # 聚合物锚点
        is_poly = contains_polymer(s)
        s_draw = polymer_to_anchored_smiles(s) if is_poly else s

        mol = Chem.MolFromSmiles(s_draw)
        cano_smi = Chem.MolToSmiles(mol, canonical=True)
        mol = Chem.MolFromSmiles(cano_smi)
        if not mol:
            return False

        strip_atom_notes(mol)
        if not config["keep_salts"]:
            mol = keep_largest_fragment(mol)

        # 坐标生成
        if config["robust_layout"]:
            mol = compute_2d_coords_robust(mol, tries=config["layout_tries"])
        else:
            AllChem.Compute2DCoords(mol)

        # 锚点处理
        anchor_idx = find_poly_anchor_atom(mol) if is_poly else None
        if is_poly and config["polymer_style"] == "repeat_unit" and anchor_idx is not None:
            orient_by_poly_anchor_horizontal(mol, anchor_idx)

        # 清除非锚点 map（关键！防止红框）
        clear_all_atom_maps_except_anchor(mol)

        # 绘图
        img_data = draw_molecule_image(
            mol=mol,
            width=config["width"],
            height=config["height"],
            fmt=config["format"],
            is_polymer=is_poly,
            polymer_style=config["polymer_style"],
            repeat_smarts=config["repeat_smarts"],
            anchor_idx=anchor_idx,
            repeat_radius=config["repeat_radius"]
        )

        if img_data is None:
            return False

        # 保存
        mode = 'wb' if config["format"] == "png" else 'w'
        with open(output_path, mode, encoding='utf-8' if mode == 'w' else None) as f:
            f.write(img_data)
        return True

    except Exception as e:
        print(f"❌ 失败：{raw_smiles} | {e}")
        return False

def process_batch(config: dict):
    """
    批量处理 SMILES，仅在：
    - SMILES 合法（可被解析）
    - 可转换为标准 SELFIES
    - 成功生成图像
    三个条件同时满足时，才保存图片和记录。
    """
    input_file = config["input_csv"]
    output_dir = config["output_dir"]
    os.makedirs(output_dir, exist_ok=True)

    # 存储最终有效的 smiles
    valid_smiles_list = []
    count = 0  # 实际保存的数量

    print(f"🔍 开始批量处理: {input_file}")

    with open(input_file, newline='', encoding='utf-8') as f:
        reader = csv.reader(f)
        
        for i, row in enumerate(reader):
            if not row:
                continue
            
            raw_smiles = row[0].strip()
            if not raw_smiles:
                continue

            # --- 步骤 3: 生成图像文件 ---
            filename = os.path.join(output_dir, f"{count}.{config['format']}")
            success = generate_image(raw_smiles, filename, config)

            if not success:
                print(f"❌ 图像生成失败: {raw_smiles}")
                continue

            # --- 步骤 4: PNG 裁剪白边（可选）---
            if config['format'] == 'png':
                trim_png_whitespace(filename, filename, bg_color=(255, 255, 255))

            # --- 全部通过！保存到结果列表 ---
            valid_smiles_list.append(raw_smiles)
            count += 1

            if count % 100 == 0:
                print(f"✅ 已成功处理并保存 {count} 条")

    # --- 最终保存有效 SMILES 列表 ---
    smiles_file = os.path.join(output_dir, "smiles.txt")
    with open(smiles_file, "w", encoding="utf-8") as f:
        for smi in valid_smiles_list:
            f.write(smi + "\n")

    # --- 输出统计结果 ---
    total_input = i + 1
    success_count = len(valid_smiles_list)
    print(f"\n🎉 批量处理完成！")
    print(f"📊 输入总数: {total_input}")
    print(f"✅ 成功生成: {success_count} 条")
    print(f"📁 输出目录: {output_dir}")
    print(f"📝 有效 SMILES 列表已保存至: {smiles_file}")
    


def run_single(config: dict):
    success = generate_image(config["single_smiles"], config["single_output"], config)
    print(f"🎯 单条生成 {'成功' if success else '失败'}：{config['single_output']}")


# =========================
# ✅ 主入口（无需命令行）
# =========================
if __name__ == "__main__":
    cfg = DEFAULT_CONFIG

    print(f"🚀 启动图像生成器 | 模式: {cfg['mode']} | 格式: {cfg['format']}")

    if cfg["mode"] == "batch":
        if not os.path.exists(cfg["input_csv"]):
            print(f"❌ 输入文件不存在: {cfg['input_csv']}")
        else:
            process_batch(cfg)
    elif cfg["mode"] == "single":
        run_single(cfg)
    else:
        print("⚠️ 未知模式，请设置 mode='single' 或 'batch'")
