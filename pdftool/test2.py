# simple_translate_with_maps.py

from pdf2zh.high_level import translate
from pdf2zh.translator import (
    GoogleTranslator,
    BingTranslator,
    DeepLTranslator,
    DeepLXTranslator,
    OllamaTranslator,
    XinferenceTranslator,
    AzureOpenAITranslator,
    OpenAITranslator,
    ZhipuTranslator,
    ModelScopeTranslator,
    SiliconTranslator,
    GeminiTranslator,
    AzureTranslator,
    TencentTranslator,
    DifyTranslator,
    AnythingLLMTranslator,
    ArgosTranslator,
    GrokTranslator,
    GroqTranslator,
    DeepseekTranslator,
    OpenAIlikedTranslator,
    QwenMtTranslator,
)
from pdf2zh.doclayout import ModelInstance
import os
from pathlib import Path
import shutil

# ======================
# 保留原始映射表（map）
# ======================

# 服务映射：字符串 → Translator 类
service_map = {
    "Google": GoogleTranslator,
    "Bing": BingTranslator,
    "DeepL": DeepLTranslator,
    "DeepLX": DeepLXTranslator,
    "Ollama": OllamaTranslator,
    "Xinference": XinferenceTranslator,
    "AzureOpenAI": AzureOpenAITranslator,
    "OpenAI": OpenAITranslator,
    "Zhipu": ZhipuTranslator,
    "ModelScope": ModelScopeTranslator,
    "Silicon": SiliconTranslator,
    "Gemini": GeminiTranslator,
    "Azure": AzureTranslator,
    "Tencent": TencentTranslator,
    "Dify": DifyTranslator,
    "AnythingLLM": AnythingLLMTranslator,
    "Argos Translate": ArgosTranslator,
    "Grok": GrokTranslator,
    "Groq": GroqTranslator,
    "DeepSeek": DeepseekTranslator,
    "OpenAI-liked": OpenAIlikedTranslator,
    "Ali Qwen-Translation": QwenMtTranslator,
}

# 语言映射：中文名 → ISO 代码
lang_map = {
    "Simplified Chinese": "zh",
    "Traditional Chinese": "zh-TW",
    "English": "en",
    "French": "fr",
    "German": "de",
    "Japanese": "ja",
    "Korean": "ko",
    "Russian": "ru",
    "Spanish": "es",
    "Italian": "it",
}

# 页面范围映射（可选）
page_map = {
    "All": None,
    "First": [0],
    "First 5 pages": list(range(0, 5)),
    "Others": None,
}

# 默认环境变量（模拟配置管理器）
# 实际使用时你可以从 ConfigManager 加载
DEFAULT_ENVS = {
    # 示例：
    # "OPENAI_API_KEY": "***",
    # "GOOGLE_API_KEY": "your_key_here",
}


def translate_pdf(
    file_path: str,
    output_dir: str = "./pdf2zh_files",
    service: str = "Google",           # 对应 service_map 键
    lang_from: str = "English",        # 对应 lang_map 键
    lang_to: str = "Simplified Chinese",
    page_range: str = "All",           # 对应 page_map 键，或传具体列表
    custom_pages: list = None,         # 当 page_range="Others" 时使用
    threads: int = 4,
    ignore_cache: bool = False,
    skip_subset_fonts: bool = False,
    prompt: str = None,
    vfont: str = "",
    envs: dict = None,                 # 外部传入密钥等环境变量
):
    """
    使用 pdf2zh 高阶接口翻译 PDF，完全基于原始 map 结构。

    Args:
        file_path: 输入 PDF 文件路径
        output_dir: 输出目录
        service: 翻译服务名称（必须是 service_map 中的 key）
        lang_from: 源语言显示名（lang_map 的 key）
        lang_to: 目标语言显示名
        page_range: 页面范围名称（如 "All", "First 5 pages"），或设为 "Others"
        custom_pages: 若 page_range="Others"，则使用此页码列表（0-indexed）
        threads: 并行线程数
        ignore_cache: 是否忽略缓存
        skip_subset_fonts: 是否跳过字体子集化
        prompt: 自定义 LLM 提示词模板（可选）
        vfont: 公式字体正则表达式
        envs: API keys 等环境变量字典，如 {"GOOGLE_API_KEY": "xxx"}

    Returns:
        (mono_pdf_path, dual_pdf_path)
    """
    # 解析参数
    if service not in service_map:
        raise ValueError(f"Unsupported service: {service}. Choose from {list(service_map.keys())}")

    if lang_from not in lang_map or lang_to not in lang_map:
        raise ValueError(f"Language not supported. Use from: {list(lang_map.keys())}")

    selected_pages = page_map.get(page_range)
    if page_range == "Others" and custom_pages is not None:
        selected_pages = custom_pages

    translator_cls = service_map[service]
    src_lang = lang_map[lang_from]
    tgt_lang = lang_map[lang_to]

    # 创建输出目录
    output_dir = Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    # 复制文件
    filename = Path(file_path).stem
    file_raw = output_dir / f"{filename}.pdf"
    shutil.copy(file_path, file_raw)

    # 合并 envs（优先使用传入的，否则留空）
    final_envs = {}
    if envs:
        final_envs.update(envs)

    print(f"🚀 Starting translation: {src_lang} → {tgt_lang} using {service}")
    print(f"📄 Input:  {file_raw}")
    print(f"📁 Output: {output_dir}/")

    # 构造参数
    param = {
        "files": [str(file_raw)],
        "pages": selected_pages,
        "lang_in": src_lang,
        "lang_out": tgt_lang,
        "service": translator_cls.name,
        "output": output_dir,
        "thread": int(threads),
        "callback": None,
        "cancellation_event": None,
        "envs": final_envs,
        "prompt": prompt,
        "skip_subset_fonts": skip_subset_fonts,
        "ignore_cache": ignore_cache,
        "vfont": vfont,
        "model": ModelInstance.value,  # 布局模型（可选）
    }

    # 执行翻译
    try:
        translate(**param)
    except Exception as e:
        print(f"❌ Translation failed: {e}")
        raise

    # 返回结果路径
    mono_pdf = output_dir / f"{filename}-mono.pdf"
    dual_pdf = output_dir / f"{filename}-dual.pdf"

    if not mono_pdf.exists():
        raise FileNotFoundError("Mono PDF was not generated.")
    if not dual_pdf.exists():
        raise FileNotFoundError("Dual PDF was not generated.")

    print(f"✅ Success! Files saved:")
    print(f"   - Mono:  {mono_pdf}")
    print(f"   - Dual:  {dual_pdf}")

    return str(mono_pdf), str(dual_pdf)


# ======================
# 使用示例
# ======================

if __name__ == "__main__":
    # 示例参数（与你要求一致）
    params = {
        "file_path": "/app/pdf2zh_files/bai2009.pdf",             # ← 替换为你自己的 PDF 路径
        "output_dir": "./pdf2zh_files",
        "service": "Bing",
        "lang_from": "English",
        "lang_to": "Simplified Chinese",
        "page_range": "All",                    # 或 "First 5 pages", "First"
        # "custom_pages": [0, 1, 2],            # 如果使用 "Others"
        "threads": 4,
        "ignore_cache": False,
        "skip_subset_fonts": False,
        "vfont": "",                            # 可选：自定义公式字体规则
        "envs": {
            # "GOOGLE_API_KEY": "your-key-here", # 如果需要
        },
    }

    translate_pdf(**params)
