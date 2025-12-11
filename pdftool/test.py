import requests
import json
from typing import Dict, Optional

# 配置服务基础地址
BASE_URL = "http://192.168.1.239:11008/v1/translate"


def submit_translate_task(
    file_path: str,
    lang_in: str = "en",
    lang_out: str = "zh",
    service: str = "google",
    thread: int = 4,
) -> Optional[str]:
    """
    提交翻译任务
    :param file_path: 要上传的PDF文件路径
    :param lang_in: 输入语言
    :param lang_out: 输出语言
    :param service: 翻译服务（如 google）
    :param thread: 线程数
    :return: 返回任务ID，失败返回None
    """
    url = BASE_URL
    data = {
        "lang_in": lang_in,
        "lang_out": lang_out,
        "service": service,
        "thread": thread,
    }

    with open(file_path, "rb") as f:
        files = {
            "file": f,
            "data": (None, json.dumps(data), "application/json"),
        }
        response = requests.post(url, files=files)

    if response.status_code == 200:
        try:
            result = response.json()
            task_id = result.get("id")
            print(f"✅ 任务提交成功，任务ID: {task_id}")
            return task_id
        except json.JSONDecodeError:
            print("❌ 响应不是有效的JSON")
            return None
    else:
        print(f"❌ 任务提交失败，状态码: {response.status_code}, 响应: {response.text}")
        return None


def check_progress(task_id: str) -> Optional[Dict]:
    """
    查询翻译任务进度
    :param task_id: 任务ID
    :return: 返回状态信息字典，失败返回None
    """
    url = f"{BASE_URL}/{task_id}"
    response = requests.get(url)

    if response.status_code == 200:
        try:
            result = response.json()
            state = result.get("state")
            if state == "PROGRESS":
                info = result.get("info", {})
                print(f"📊 进度: {info.get('n', 0)}/{info.get('total', 0)}")
            elif state == "SUCCESS":
                print("✅ 任务已完成")
            return result
        except json.JSONDecodeError:
            print("❌ 响应不是有效的JSON")
            return None
    else:
        print(f"❌ 查询失败，状态码: {response.status_code}")
        return None


def save_monolingual_file(task_id: str, output_path: str) -> bool:
    """
    下载单语翻译结果文件（仅目标语言）
    :param task_id: 任务ID
    :param output_path: 保存文件路径
    :return: 是否成功
    """
    url = f"{BASE_URL}/{task_id}/mono"
    response = requests.get(url, stream=True)

    if response.status_code == 200:
        with open(output_path, "wb") as f:
            for chunk in response.iter_content(chunk_size=8192):
                f.write(chunk)
        print(f"📄 单语文件已保存至: {output_path}")
        return True
    else:
        print(f"❌ 下载单语文件失败，状态码: {response.status_code}")
        return False


def save_bilingual_file(task_id: str, output_path: str) -> bool:
    """
    下载双语对照文件
    :param task_id: 任务ID
    :param output_path: 保存文件路径
    :return: 是否成功
    """
    url = f"{BASE_URL}/{task_id}/dual"
    response = requests.get(url, stream=True)

    if response.status_code == 200:
        with open(output_path, "wb") as f:
            for chunk in response.iter_content(chunk_size=8192):
                f.write(chunk)
        print(f"📄 双语文件已保存至: {output_path}")
        return True
    else:
        print(f"❌ 下载双语文件失败，状态码: {response.status_code}")
        return False


def interrupt_and_delete_task(task_id: str) -> bool:
    """
    中断并删除正在运行的任务
    :param task_id: 任务ID
    :return: 是否成功
    """
    url = f"{BASE_URL}/{task_id}"
    response = requests.delete(url)

    if response.status_code == 200:
        print(f"🗑️ 任务 {task_id} 已被中断并删除")
        return True
    else:
        print(f"❌ 删除任务失败，状态码: {response.status_code}")
        return False


# ===========================
# 示例：组合调用流程
# ===========================
if __name__ == "__main__":
    # 1. 提交任务
    task_id = submit_translate_task("/root/binghao/pdftool/bai2009.pdf")
    if not task_id:
        exit(1)

    # 2. 轮询进度直到完成
    import time

    while True:
        status = check_progress(task_id)
        if not status:
            break
        if status.get("state") == "SUCCESS":
            break
        time.sleep(2)  # 每2秒查一次

    # 3. 下载双语和单语文件
    save_bilingual_file(task_id, "example-dual.pdf")
    save_monolingual_file(task_id, "example-mono.pdf")

    # （可选）如果想中途取消任务：
    # interrupt_and_delete_task(task_id)
