#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
SMILES <-> SELFIES 转换工具
支持：
- 单个字符串转换
- 批量 CSV/TSV 文件转换
- 生成标准（canonical）形式
- 错误自动过滤

使用前请安装：
pip install selfies rdkit
"""

import argparse
import csv
import sys
from pathlib import Path
import deepsmiles

try:
    from rdkit import Chem
except ImportError:
    print("❌ 未安装 rdkit，请运行：pip install rdkit")
    exit(1)

try:
    import selfies as sf
except ImportError:
    print("❌ 未安装 selfies，请运行：pip install selfies")
    exit(1)


def smiles_to_selfies(smi):
    try:
        # Step 1: 标准化 SMILES（RDKit）
        mol = Chem.MolFromSmiles(smi)
        if mol is None:
            return None
        canonical_smi = Chem.MolToSmiles(mol, canonical=True)

        # Step 2: 转为 SELFIES（此时输入已经是标准形式）
        selfi = sf.encoder(canonical_smi)  # 不需要 canonize 参数
        return selfi
    except Exception:
        return None

def selfies_to_smiles(selfies_str: str) -> str:
    """
    将 SELFIES 转为 SMILES
    :param selfies_str: 输入 SELFIES 字符串
    :return: SMILES 字符串 或 None（失败时）
    """
    try:
        smi = sf.decoder(selfies_str)
        # 再次标准化输出
        mol = Chem.MolFromSmiles(smi)
        if mol is None:
            return None
        return Chem.MolToSmiles(mol, canonical=True)
    except Exception:
        return None

def all_smiles_to_selfies():
    """
    批量将 SMILES 列表转换为 SELFIES 列表
    :param smiles_list: SMILES 字符串列表
    :return: SELFIES 字符串列表（转换失败的为 None）
    """
    smile_txt_file = "/root/binghao/smiles/app/img2smles/dataset/smiles.txt"
    selfies_txt_file = "/root/binghao/smiles/app/img2smles/dataset/selfies.txt"
    selfie_list = []
    with open(smile_txt_file, "r", encoding="utf-8") as f:
        for line in f:
            smi = line.strip()
            selfi = smiles_to_selfies(smi)
            print(f"SMILES: {smi} -> SELFIES: {selfi}")
            if selfi is None:
                selfi = ""
                print(f"Failed to convert SMILES to SELFIES: {smi}")
            selfie_list.append(selfi if selfi is not None else "")
    with open(selfies_txt_file, "w", encoding="utf-8") as f:
        for selfi in selfie_list:
            f.write(selfi + "\n")

def all_selfies_to_smiles():
    """
    批量将 SELFIES 列表转换为 SMILES 列表
    :param selfies_list: SELFIES 字符串列表
    :return: SMILES 字符串列表（转换失败的为 None）
    """
    smile_txt_file = "/root/binghao/smiles/app/img2smles/dataset/smiles_from_selfies.txt"
    selfies_txt_file = "/root/binghao/smiles/app/img2smles/dataset/smiles.txt"
    smile_list = []
    with open(selfies_txt_file, "r", encoding="utf-8") as f:
        for line in f:
            selfi = line.strip()
            smi = selfies_to_smiles(selfi)
            print(f"SELFIES: {selfi} -> SMILES: {smi}")
            if smi is None:
                raise ValueError(f"Failed to convert SELFIES to SMILES: {selfi}")
            smile_list.append(smi if smi is not None else "")
    with open(smile_txt_file, "w", encoding="utf-8") as f:
        for smi in smile_list:
            f.write(smi + "\n")

def all_smiles_to_deepsmiles():
    """
    批量将 SMILES 列表转换为 DeepSMILES 列表
    :param smiles_list: SMILES 字符串列表
    :return: DeepSMILES 字符串列表（转换失败的为 None）
    """
    from deepsmiles import Converter

    smile_txt_file = "/root/binghao/smiles/app/img2smles/dataset/smiles.txt"
    deepsmiles_txt_file = "/root/binghao/smiles/app/img2smles/dataset/deepsmiles.txt"
    converter = deepsmiles.Converter(rings=True, branches=True)
    deepsmiles_list = []
    with open(smile_txt_file, "r", encoding="utf-8") as f:
        for line in f:
            smi = line.strip()
            try:
                dsmi = converter.encode(smi)
                print(f"SMILES: {smi} -> DeepSMILES: {dsmi}")
                deepsmiles_list.append(dsmi)
            except Exception:
                print(f"Failed to convert SMILES to DeepSMILES: {smi}")
                raise ValueError(f"Failed to convert SMILES to DeepSMILES: {smi}")
                # deepsmiles_list.append("")
    with open(deepsmiles_txt_file, "w", encoding="utf-8") as f:
        for dsmi in deepsmiles_list:
            f.write(dsmi + "\n")

if __name__ == "__main__":
    # all_smiles_to_selfies()
    print(sf.decoder("[Cl][P][Branch1][C][Cl][O][C][C][=C][C][=C][C][=C][Ring1][=Branch1]"))