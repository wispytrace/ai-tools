import os
import shutil
import multiprocessing
import concurrent.futures
from tqdm import tqdm
from rdkit import Chem, RDLogger
from rdkit.Chem import Draw
from rdkit.Chem import AllChem
try:
    import selfies as sf
except ImportError:
    print("❌ 缺失 selfies 库")
    exit(1)

# 导入您的自定义生成函数
from smiles2img import generate_image as custom_generate_fn

# 抑制 RDKit 警告
RDLogger.DisableLog('rdApp.*')

def rdkit_generate_image(smiles, save_path, size=(384, 384)):
    """
    最通用的 RDKit SMILES 转图片函数
    :param smiles: 输入的 SMILES 字符串
    :param save_path: 图片保存路径 (支持 .png, .jpg, .svg)
    :param size: 图片尺寸 (像素宽度, 像素高度)
    """
    try:
        # 1. 解析 SMILES
        mol = Chem.MolFromSmiles(smiles)
        if mol is None:
            print(f"❌ 无法解析 SMILES: {smiles}")
            return False

        # 2. 生成 2D 坐标（确保分子展开整齐，而不是挤在一起）
        # 这是最关键的一步，能显著提升图片质量
        AllChem.Compute2DCoords(mol)

        # 3. 渲染并保存
        # useSVG=False 生成像素图(PNG)，useSVG=True 则生成矢量图
        Draw.MolToFile(mol, save_path, size=size, kekulize=True, imageType="png")
        
        return True
    except Exception as e:
        print(f"❌ 生成过程中出错: {e}")
        return False

# --- 1. 配置中心 ---
class GenConfig:
    # 数据输入输出路径
    INPUT_FILE = "/root/binghao/smiles/app/selfies_top30.txt"
    OUTPUT_DIR = "/root/binghao/smiles/app/img2smles/dataset/images_24_color"
    
    # 运行参数
    NUM_WORKERS = 8       # 并行进程数
    TIMEOUT = 5           # 单个任务超时时间（秒）
    MAX_COUNT = 2000000   # 最大生成数量
    
    # 指定生成函数句柄
    # 你可以随时切换为其他函数，只要函数签名接受 (smiles, path) 即可
    GENERATE_ENGINE = rdkit_generate_image 

# --- 2. 工作进程逻辑 ---
def _worker_process(args):
    """
    单个图像生成任务
    args: (line, save_path, engine_func)
    """
    line, save_path, engine_func = args
    try:
        # SELFIES 解码为 SMILES
        smiles = sf.decoder(line)
        
        # 验证分子合法性
        mol = Chem.MolFromSmiles(smiles)
        if mol is None:
            return False
            
        # 调用传入的生成引擎
        engine_func(smiles, save_path)
        
        return os.path.exists(save_path)
    except Exception:
        # 发生异常时尝试清理残余文件
        if os.path.exists(save_path):
            try: os.remove(save_path)
            except: pass
        return False

# --- 3. 核心批处理函数 ---
def batch_generate_images(cfg: GenConfig):
    """
    基于配置类进行并行生成
    """
    if not os.path.exists(cfg.OUTPUT_DIR):
        os.makedirs(cfg.OUTPUT_DIR)
        print(f"📁 已创建目录: {cfg.OUTPUT_DIR}")
    
    # 同步标签文件
    label_dest = os.path.join(cfg.OUTPUT_DIR, "label.txt")
    shutil.copy(cfg.INPUT_FILE, label_dest)
    
    # 读取数据
    with open(cfg.INPUT_FILE, "r") as f:
        # 过滤空行并限制数量
        lines = [line.strip() for line in f if line.strip()][:cfg.MAX_COUNT]

    # 构建任务列表（跳过已存在的文件）
    tasks = []
    for i, line in enumerate(lines, 1):
        save_path = os.path.join(cfg.OUTPUT_DIR, f"{i}.png")
        if not os.path.exists(save_path):
            tasks.append((line, save_path, cfg.GENERATE_ENGINE))
    
    if not tasks:
        print("✨ 所有图像已存在，无需生成。")
        return

    print(f"🚀 准备生成 {len(tasks)} 张图像 (使用 {cfg.NUM_WORKERS} 进程)...")
    
    success = 0
    # 使用 spawn 模式在 Linux 上更稳定，避免 CUDA/RDKit 锁死
    ctx = multiprocessing.get_context('spawn')
    
    with concurrent.futures.ProcessPoolExecutor(max_workers=cfg.NUM_WORKERS, mp_context=ctx) as executor:
        # 提交任务
        future_to_path = {executor.submit(_worker_process, t): t[1] for t in tasks}
        
        # 进度条显示
        pbar = tqdm(total=len(tasks), desc="🖼️ 生成进度")
        
        for future in concurrent.futures.as_completed(future_to_path):
            try:
                if future.result(timeout=cfg.TIMEOUT):
                    success += 1
            except Exception as e:
                path = future_to_path[future]
                # print(f"❌ 任务超时或失败: {path}") 
            finally:
                pbar.update(1)
        pbar.close()

    print(f"\n✅ 处理完成!")
    print(f"📊 成功: {success}")
    print(f"📊 失败/跳过: {len(tasks) - success}")
    print(f"📍 存储位置: {cfg.OUTPUT_DIR}")

# --- 4. 入口 ---
if __name__ == "__main__":
    # 使用方式：直接传入配置类
    batch_generate_images(GenConfig)