# -*- coding: utf-8 -*-
import os
import json
import math
import csv
from typing import Optional, Tuple, List, Dict
from rdkit import Chem
from rdkit.Chem import AllChem, rdDepictor
from rdkit.Chem.Draw import rdMolDraw2D
from PIL import Image, ImageChops

# =========================
# 🔧 配置区
# =========================
CONFIG = {
    "width": 384,
    "height": 384,
    "format": "png",
    "tries": 15,          # 鲁棒布局尝试次数
    "output_dir": "./dataset_output",
    "trim_padding": 10    # 裁剪后的留白
}

class MoleculeDataGenerator:
    def __init__(self, config: Dict):
        self.cfg = config
        os.makedirs(self.cfg["output_dir"], exist_ok=True)

    def _compute_robust_coords(self, mol: Chem.Mol) -> bool:
        """多算法尝试，寻找重叠分数最低的布局"""
        best_sc = 1e18
        best_conf = None
        
        for i in range(self.cfg["tries"]):
            temp_mol = Chem.Mol(mol)
            use_coordgen = (i % 2 == 0)
            if hasattr(rdDepictor, "SetPreferCoordGen"):
                rdDepictor.SetPreferCoordGen(use_coordgen)
            
            try:
                rdDepictor.Compute2DCoords(temp_mol, canonOrient=(i % 3 == 0))
                # 简单的重叠评分（原子间距越小分数越高）
                sc = self._get_overlap_score(temp_mol)
                if sc < best_sc:
                    best_sc = sc
                    best_conf = Chem.Conformer(temp_mol.GetConformer())
            except:
                continue
        
        if best_conf:
            mol.RemoveAllConformers()
            mol.AddConformer(best_conf)
            return True
        return False

    def _get_overlap_score(self, mol: Chem.Mol) -> float:
        conf = mol.GetConformer()
        score = 0.0
        pts = [conf.GetAtomPosition(i) for i in range(mol.GetNumAtoms())]
        for i in range(len(pts)):
            for j in range(i + 1, len(pts)):
                d2 = (pts[i].x - pts[j].x)**2 + (pts[i].y - pts[j].y)**2
                if d2 < 0.5: score += 10.0
        return score

    def process_single(self, smiles: str, idx: int):
        """处理单条 SMILES 并保存图像与坐标数据"""
        mol = Chem.MolFromSmiles(smiles)
        if not mol: return False

        # 1. 坐标布局优化
        self._compute_robust_coords(mol)
        
        # 2. 初始化绘图器
        drawer = rdMolDraw2D.MolDraw2DCairo(self.cfg["width"], self.cfg["height"])
        opt = drawer.drawOptions()
        opt.useBWAtomPalette()
        opt.bondLineWidth = 2.0
        opt.padding = 0.12
        
        # 3. 绘制分子
        rdMolDraw2D.PrepareAndDrawMolecule(drawer, mol)
        drawer.FinishDrawing()
        
        # 4. 提取像素坐标
        atom_coords = []
        for atom in mol.GetAtoms():
            a_idx = atom.GetIdx()
            # 关键：使用修复后的 idx 传参，获取像素坐标
            p_canvas = drawer.GetDrawCoords(a_idx)
            atom_coords.append({
                "aid": a_idx,
                "symbol": atom.GetSymbol(),
                "x": round(p_canvas.x, 2),
                "y": round(p_canvas.y, 2)
            })

        bond_coords = []
        for bond in mol.GetBonds():
            b_p1 = drawer.GetDrawCoords(bond.GetBeginAtomIdx())
            b_p2 = drawer.GetDrawCoords(bond.GetEndAtomIdx())
            bond_coords.append({
                "ba": bond.GetBeginAtomIdx(),
                "ea": bond.GetEndAtomIdx(),
                "type": str(bond.GetBondType()),
                "center": {"x": round((b_p1.x + b_p2.x)/2, 2), "y": round((b_p1.y + b_p2.y)/2, 2)}
            })

        # 5. 保存图像
        img_path = os.path.join(self.cfg["output_dir"], f"{idx}.png")
        with open(img_path, 'wb') as f:
            f.write(drawer.GetDrawingText())

        # 6. 裁剪图像并获取边界框 (Optional but helpful for training)
        bbox = self._trim_and_save(img_path)

        # 7. 最终数据汇总
        data = {
            "id": idx,
            "smiles": smiles,
            "width": self.cfg["width"],
            "height": self.cfg["height"],
            "atoms": atom_coords,
            "bonds": bond_coords,
            "crop_bbox": bbox # [xmin, ymin, xmax, ymax]
        }
        
        json_path = os.path.join(self.cfg["output_dir"], f"{idx}.json")
        with open(json_path, 'w', encoding='utf-8') as f:
            json.dump(data, f, indent=4, ensure_ascii=False)

        return True

    def _trim_and_save(self, path: str):
        """裁剪白边并返回裁剪前后的偏移量"""
        img = Image.open(path).convert("RGB")
        bg = Image.new(img.mode, img.size, (255, 255, 255))
        diff = ImageChops.difference(img, bg)
        bbox = diff.getbbox()
        if bbox:
            # 适当留白
            left, top, right, bottom = bbox
            p = self.cfg["trim_padding"]
            cropped = img.crop((max(0, left-p), max(0, top-p), min(img.width, right+p), min(img.height, bottom+p)))
            cropped.save(path)
            return [left, top, right, bottom]
        return None

# =========================
# 🚀 运行示例
# =========================
if __name__ == "__main__":
    # 模拟输入数据
    smiles_list = [
        "CC(=O)OC1=CC=CC=C1C(=O)O", # 阿司匹林
        "CN1C=NC2=C1C(=O)N(C(=O)N2C)C", # 咖啡因
        "C1=CC=C(C=C1)C(=O)O" # 苯甲酸
    ]
    
    generator = MoleculeDataGenerator(CONFIG)
    
    print("开始生成数据...")
    for i, smi in enumerate(smiles_list):
        success = generator.process_single(smi, i)
        if success:
            print(f"✅ 已处理: {i} | {smi[:20]}...")
    
    print(f"\n🎉 处理完成。请在 {CONFIG['output_dir']} 目录下查看 PNG 和 JSON 文件。")