import os

class Config:
    # --- 1. 路径与断点配置 ---
    CHECKPOINT_DIR = "checkpoints"
    TEMP_MODEL_DIR = "/root/binghao/smiles/app/img2smles/temp_models"
    PRETRAINED_PATH = "/root/binghao/smiles/app/img2smles/pretrained_models/swinv2_large_patch4_window12to24_192to384_22kto1k_ft.pth"
    TOKENIZER_PATH = "tokenizer-selfies-ultimate.json"
    FREQUENCT_PARH = "token_frequencies.txt"
    RESUME_PATH = os.path.join(TEMP_MODEL_DIR, "checkpoint_latest.pth")
    
    RESUME = True       # 是否加载断点
    IS_AUG = True       # 是否数据增强
    # --- 2. 训练步数与频率 ---
    BATCH_SIZE = 48
    EPOCHS = 40
    STEPS_PER_EPOCH = 4000     # 每个 epoch 训练的最大步数
    SAVE_STEP_INTERVAL = 1500   # 每隔多少步保存一次临时断点
    LOG_STEP_INTERVAL = 20     # 打印日志的步数频率
    ACCUMULATION_STEPS = 4
    
    # --- 3. 优化器与学习率 ---
    ENCODER_LR = 5e-6
    DECODER_LR = 5e-5
    MAX_LR = [1e-5, 1e-4]
    WEIGHT_DECAY = 1e-2
    PCT_START = 0.05
    
    # --- 4. 损失函数配置 ---
    EOS_WEIGHT = 5.0            # 给 <eos> 标记分配的权重
    LABEL_SMOOTHING = 0.002     # 标签平滑系数
    RARE_WEIGHT = 5.0
    
    # --- 5. 数据源配置 ---
    DATA_SOURCES = [
        ("dataset/images_24", "dataset/images_24/label.txt", 3000000),
        ("dataset/images_24_except", "dataset/images_24_except/label.txt", 1500000),
        ("dataset/images_24_color", "dataset/images_24_color/label.txt", 3000000)
    ]
    TRAIN_RATIO = 0.92          # 训练集占比