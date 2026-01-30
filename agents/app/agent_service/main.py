# chemistry_extraction/main.py
import os
import json
import argparse
from typing import Dict, Any
from .state import ChemistryExtractionState
from .workflow import ChemistryWorkflow
from .config import Config
from datetime import datetime

Agents = ['compound_extraction', 'reaction_extraction']

def main():
    """主执行函数"""
    
    # 配置

    Config.validate()
        # 初始化状态
    initial_state = ChemistryExtractionState(
        pdf_path = "/mnt/binghao/papers/CN110551144B.pdf",
        current_stage=["initialized"],
    )
    
    agent = "compound_extraction"  # or "compound_extraction"
    # 创建并运行工作流
    print("🔄 Setting up workflow...")
    workflow = ChemistryWorkflow()
    workflow.set_workflow(agent)
    app = workflow.compile()
    
    print("🚀 Starting extraction process...")
    print(type(initial_state))
    final_state = app.invoke(initial_state)
    
    # 保存结果
    timestamp = datetime.now().strftime("%Y-%m-%d_%H-%M-%S")
    output_with_timestamp = f"./output/{timestamp}_results.json"
    print(f"💾 Saving results to: {output_with_timestamp}")
    with open(output_with_timestamp, 'w', encoding='utf-8') as f:
        json.dump(dict(final_state), f, ensure_ascii=False, indent=2)

    # 打印摘要
    metadata = final_state.get("metadata", {})
    print("\n✅ Extraction completed successfully!")
    print(f"  - Text sections processed: {metadata.get('text_sections_count', 0)}")
    print(f"  - Images analyzed: {metadata.get('image_extractions_count', 0)}")
    print(f"  - Fusion success: {metadata.get('fusion_success', False)}")
    
    if final_state["errors"]:
        print(f"\n⚠️  Encountered {len(final_state["errors"])} errors:")
        for i, err in enumerate(final_state["errors"][:3]):  # 只显示前3个错误
            print(err)
        if len(final_state["errors"]) > 3:
            print(f"  ... and {len(final_state["errors"]) - 3} more errors")

if __name__ == "__main__":
    import time
    main()
