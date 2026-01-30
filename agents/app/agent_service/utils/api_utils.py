import requests
import mimetypes
from pathlib import Path


def convert_image_to_smiles(image_path: str):
    """
    将化学结构图像转换为SMILES表示
    
    Args:
        image_path: 本地图像路径
        
    Returns:
        {"smiles": "..."} 或 None（失败时）
    """
    url = "http://192.168.1.239:30869/ocr_api/img_to_smiles"
    headers = {"accept": "application/json"}
    
    # 推测 MIME 类型
    mime_type, _ = mimetypes.guess_type(image_path)
    ext = Path(image_path).suffix.lower()
    if ext in ['.jpg', '.jpeg']:
        mime_type = 'image/jpeg'
    elif ext == '.png':
        mime_type = 'image/png'
    else:
        print(f"[SMILES] Unsupported image type: {ext}")
        return None

    file_name = Path(image_path).name
    try:
        with open(image_path, 'rb') as f:
            files = {'file': (file_name, f, mime_type)}
            response = requests.post(url, headers=headers, files=files, timeout=30)
        
        if response.status_code == 200:
            result = response.json()
            # 假设返回格式: {"smiles": "C1=CC=..."} 或带 confidence 的对象
            if isinstance(result, dict) and "smiles" in result:
                return result
            else:
                print(f"[SMILES] Invalid response format: {result}")
                return ''
        else:
            try:
                error = response.json().get("detail", response.text)
            except:
                error = response.text
            print(f"[SMILES] API Error {response.status_code}: {error}")
            return None
    except Exception as e:
        print(f"[SMILES] Request failed: {str(e)}")
        return None


def convert_image_to_bboxs(image_path: str):
    """
    将化学结构图像转换化合物以及名称位置的bbox表示
    
    Args:
        image_path: 本地图像路径
        
    Returns:
        {"compound": "...", "text": "...."} 或 '空字符串（失败时）'
    """
    url = "http://192.168.1.239:6789/detect"
    headers = {"accept": "application/json"}
    
    # 推测 MIME 类型
    mime_type, _ = mimetypes.guess_type(image_path)
    ext = Path(image_path).suffix.lower()
    if ext in ['.jpg', '.jpeg']:
        mime_type = 'image/jpeg'
    elif ext == '.png':
        mime_type = 'image/png'
    else:
        print(f"[Yolo] Unsupported image type: {ext}")
        return None

    file_name = Path(image_path).name
    try:
        with open(image_path, 'rb') as f:
            files = {'file': (file_name, f, mime_type)}
            response = requests.post(url, headers=headers, files=files, timeout=30)
        
        if response.status_code == 200:
            result = response.json()
            # 假设返回格式: {"smiles": "C1=CC=..."} 或带 confidence 的对象
            if isinstance(result, dict) and "detections" in result:
                return result["detections"]
            else:
                print(f"[Yolo] Invalid response format: {result}")
                return []
        else:
            try:
                error = response.json().get("detail", response.text)
            except:
                error = response.text
            print(f"[Yolo] API Error {response.status_code}: {error}")
            return None
    except Exception as e:
        print(f"[SMILES] Request failed: {str(e)}")
        return None

def convert_image_to_text(image_path: str):
    """
    将图像转换为文本表示（通过调用 /recognize API）
    
    Args:
        image_path: 本地图像路径
        
    Returns:
        提取的文本字符串 或 空字符串（失败或无结果时）
    """
    url = "http://192.168.1.239:6788/recognize"
    headers = {"accept": "application/json"}
    
    # 标准化路径
    path = Path(image_path)
    if not path.exists():
        print(f"[OCR] Image file not found: {image_path}")
        return ''
    
    # 推测 MIME 类型
    ext = path.suffix.lower()
    if ext in ['.jpg', '.jpeg']:
        mime_type = 'image/jpeg'
    elif ext == '.png':
        mime_type = 'image/png'
    else:
        print(f"[OCR] Unsupported image type: {ext}")
        return ''

    try:
        with open(path, 'rb') as f:
            # ✅ 关键修复：字段名必须是 'images'，且格式要支持多图
            files = [('images', (path.name, f, mime_type))]  # 注意：使用 list of tuple 支持同名字段
            response = requests.post(url, headers=headers, files=files, timeout=30)
        
        if response.status_code == 200:
            result = response.json()
            
            # ✅ 安全解析返回结构
            if isinstance(result, dict):
                if result.get("success") and "results" in result:
                    results_list = result["results"]
                    if len(results_list) > 0:
                        text = results_list[0].get("text", "").strip()
                        return text or ''  # 避免返回 None
            
            print(f"[OCR] No valid text extracted from response: {result}")
            return ''
        
        else:
            try:
                error_detail = response.json().get("detail", response.text)
            except:
                error_detail = response.text
            print(f"[OCR] API Error {response.status_code}: {error_detail}")
            return ''
            
    except requests.exceptions.Timeout:
        print("[OCR] Request timed out after 30s")
        return ''
    except Exception as e:
        print(f"[OCR] Request failed: {str(e)}")
        return ''



def translate_text_with_ollama(text: str,
                            model: str = "huihui_ai/hunyuan-mt-abliterated:latest",
                            ollama_host: str = "http://192.168.1.239:11434") -> str:
    """
    如果输入是中文，直接返回；
    否则调用 Ollama 模型将其翻译为中文并返回。
    
    Args:
        text (str): 输入文本
        model (str): Ollama 翻译模型名称
        ollama_host (str): Ollama 服务地址
    
    Returns:
        str: 中文文本（原样或翻译后）；失败时返回 None
    """
    def is_chinese(text: str, threshold=0.5) -> bool:
        """
        判断文本是否主要是中文。
        
        Args:
            text (str): 输入文本
            threshold (float): 中文字符占比阈值（默认 0.5）
        
        Returns:
            bool: True 表示主要是中文
        """
        if not text.strip():
            return False

        chinese_count = 0
        letter_count = 0  # 所有“类字母”字符（中/英文字母等）

        for char in text:
            # 判断是否为中文字符：基本汉字范围 U+4E00 ~ U+9FFF
            if '\u4e00' <= char <= '\u9fff':
                chinese_count += 1
                letter_count += 1
            elif char.isalpha():  # 英文字母也算“字母”
                letter_count += 1

        # 如果没有发现任何字母类字符，认为不是有效语言文本
        if letter_count == 0:
            return False

        return chinese_count / letter_count >= threshold


    # === 主逻辑 ===
    if not text or not text.strip():
        print("[Translate] Empty input.")
        return text  # 原样返回空串或空白

    text = text.strip()

    if is_chinese(text):
        return text  # 是中文，直接返回

    # 非中文，调用 Ollama 翻译成中文
    url = f"{ollama_host}/api/generate"
    headers = {"Content-Type": "application/json"}
    data = {
        "model": model,
        "prompt": (
            "You are a precise scientific translator. "
            "Translate the following text into formal, accurate Chinese. "
            "Preserve formatting such as lists and new lines. "
            "Preserve technical terms and sentence structure as much as possible.\n\n"
            f"{text}"
        ),
        "stream": False
    }

    try:
        response = requests.post(url, json=data, headers=headers, timeout=60)
        if response.status_code == 200:
            result = response.json()
            if "response" in result:
                translated = result["response"].strip()
                return translated
            else:
                print(f"[Ollama] Invalid response format: {result}")
                return None
        else:
            try:
                error_msg = response.json().get("error", response.text)
            except:
                error_msg = response.text
            print(f"[Ollama] API Error {response.status_code}: {error_msg}")
            return None

    except requests.exceptions.RequestException as e:
        print(f"[Ollama] Request failed: {str(e)}")
        return None
    except Exception as e:
        print(f"[Ollama] Unknown error: {str(e)}")
        return None