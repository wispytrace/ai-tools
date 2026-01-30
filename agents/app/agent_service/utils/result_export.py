from dataclasses import dataclass, field, asdict
import json
from tabulate import tabulate
from rxnmapper import RXNMapper, BatchedMapper



def extract_fusion_result(fusion_result):
    
    extracted_results = []
    for extracted in fusion_result:
        reactions = extracted.get('reactions', [])
        bbox = extracted.get('bbox', [])
        page_idx = extracted.get('page_idx', 0)
        for reaction in reactions:
            reaction_condition = {}
            # Extract reactants (rxn_smiles)
            reaction_condition['reactants'] = reaction.get('rxn_smiles', "")

            # Extract targets from products
            products = reaction.get('products', [])
            targets = [product.get('smiles', '') for product in products]
            reaction_condition['target'] = ','.join(targets)

            # Extract materials from reactions
            reactants = reaction.get('reactants')
            materials = [reactant.get('smiles', '') for reactant in reactants]
            reaction_condition['materials'] = ','.join(materials)

            # Solvents and experiments
            reaction_condition['solvents'] = reaction.get('solvent', '')
            reaction_condition['experiments'] = reaction.get('experiments', '')

            # Location from evidence
            evidence = reaction.get('evidence', {})

            reaction_condition['page'] = page_idx
            reaction_condition['bbox'] = bbox
            # reaction_condition['location'] = {"page": page, "bbox": bbox}

            yields = reaction.get('yields', [])
            yiled_result = []
            for yiled_detail in yields:
                yiled_result.append(f"{yiled_detail['substrate_label']}: {yiled_detail['value']}{yiled_detail['unit']}")

            reaction_condition['yields'] = ','.join(yiled_result)

            extracted_results.append(reaction_condition)

    return extracted_results


def extract_from_fusion_file(file_path='/mnt/binghao/output/results.json'):
    try:
        with open(file_path, 'r', encoding='utf-8') as f:
            results = json.load(f)

        fusion_result_list = results.get('fusion_result', [])
        see = json.loads(results['fusion_result']['content'])
        print(see)
        if not fusion_result_list:
            print("⚠️ 警告：JSON 文件中未找到 'fusion_result' 数据。")
            return

        # 提取所有反应条件
        extracted = extract_fusion_result(fusion_result_list)
        print(extracted)
        reaction_smiles = []
        for i in range(len(extracted)):
            reaction_smiles.append(extracted[i]['reactants'])
        rxn_mapper =  BatchedMapper(batch_size=32)
        rxn_results = rxn_mapper.map_reactions_with_info(reaction_smiles)
        for i in range(len(extracted)):
            if rxn_results[i] != {}:
                extracted[i]['mapped_rxn'] = rxn_results[i]['mapped_rxn']
                extracted[i]['confidence'] = rxn_results[i]['confidence']
            else:
                extracted[i]['mapped_rxn'] = ''
                extracted[i]['confidence'] = 0.0

        if not extracted:
            print("📭 未提取到任何反应信息。")
            return

        # 逐条打印，每个 key-value 占一行
        print("\n📋 开始输出提取的反应信息：\n")
        for idx, item in enumerate(extracted, start=1):
            print(f"✅ 反应 {idx}:")
            for key, value in item.items():
                # 格式化输出，key 左对齐固定宽度
                print(f"  {key:<12} : {value}")
            print("-" * 50)  # 分隔线

        # （可选）仍然保存为 CSV
        # save_to_csv(extracted, "extracted_reactions.csv")

    except FileNotFoundError:
        print(f"❌ 错误：文件未找到，请检查路径：{file_path}")
    except json.JSONDecodeError as e:
        print(f"❌ JSON 解析失败：{e}")
    except Exception as e:
        import traceback
        traceback.print_exc()
        print(f"❌ 发生未知错误：{e}")


def save_to_csv(data, csv_file):
    import csv
    if not data:
        return
    keys = data[0].keys()
    with open(csv_file, 'w', newline='', encoding='utf-8') as f:
        writer = csv.DictWriter(f, fieldnames=keys)
        writer.writeheader()
        writer.writerows(data)
    print(f"\n💾 已保存结果到：{csv_file}")


if __name__ == "__main__":
    extract_from_fusion_file()

