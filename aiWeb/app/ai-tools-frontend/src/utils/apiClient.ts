// src/utils/apiClient.ts
import axios from 'axios';
import type { InputType } from '../api/endpoints';
import type { AxiosRequestConfig } from 'axios';
// 可以根据你的后端路径设置 baseURL，比如：'http://localhost:8000'
const api = axios.create({
  baseURL: '',
  timeout: 60000
});

export interface ApiRequestPayload {
  endpoint: string;
  method: 'GET' | 'POST';
  inputTypes: InputType[];
  textInput?: string;
  jsonInput?: string;    // JSON 文本
  files?: File[];        // 上传的文件
}

// 响应统一结构（前端解析）
export type ParsedResponseType = 'json' | 'text' | 'image' | 'file';

export interface ParsedResponse {
  type: ParsedResponseType;
  data: any;          // JSON 对象 / 字符串 / blob URL / 下载信息等
  raw?: any;
  filename?: string;  // 对于下载文件
  mimeType?: string;
}

export async function callApi(payload: ApiRequestPayload): Promise<ParsedResponse> {
  const { endpoint, method, inputTypes, textInput, jsonInput, files } = payload;

  const config: AxiosRequestConfig = {
    url: endpoint,
    method
  };

  // 根据 inputTypes 判断要用 JSON 还是 multipart/form-data
  const useMultipart = inputTypes.includes('file');

  if (method === 'GET') {
    // 简单 GET 示例：可根据需要扩展 query
    config.params = { text: textInput };
  } else {
    if (useMultipart) {
      const formData = new FormData();
      if (textInput) {
        formData.append('text', textInput);
      }
      if (jsonInput) {
        // 通常后端把这个字段再做 json.loads
        formData.append('json', jsonInput);
      }
      if (files && files.length > 0) {
        // 注意字段名应和 FastAPI 接收参数名对应，如 files: List[UploadFile]
        files.forEach((file) => formData.append('files', file));
      }
      config.data = formData;
      // axios 会自动加上 Content-Type: multipart/form-data
    } else {
      // 使用 JSON
      let jsonBody: Record<string, any> = {};
      if (textInput) jsonBody.text = textInput;
      if (jsonInput) {
        try {
          jsonBody.json = JSON.parse(jsonInput);
        } catch {
          // 若 JSON 解析失败，仍然以字符串形式发送，或者抛错
          jsonBody.json = jsonInput;
        }
      }
      config.headers = {
        'Content-Type': 'application/json'
      };
      config.data = jsonBody;
    }
  }

  // 为了能拿到二进制等，统一先用 'arraybuffer'，再根据 content-type 自己解析
  config.responseType = 'arraybuffer';

  const response = await api.request<ArrayBuffer>(config);

  // 解析响应类型
  const contentType = response.headers['content-type'] as string | undefined;
  const contentDisposition = response.headers['content-disposition'] as string | undefined;

  // 从 content-disposition 提取文件名（如果有）
  let filename: string | undefined;
  if (contentDisposition) {
    const match = /filename\*?=(?:UTF-8''|")?([^\";]+)/i.exec(contentDisposition);
    if (match && match[1]) {
      filename = decodeURIComponent(match[1].replace(/\"/g, ''));
    }
  }

  // 封装 ArrayBuffer 为 Blob
  const mimeType = contentType || 'application/octet-stream';
  const blob = new Blob([response.data], { type: mimeType });

  // 根据 content-type 判断如何渲染
  if (mimeType.includes('application/json')) {
    const text = await blob.text();
    try {
      const json = JSON.parse(text);
      return {
        type: 'json',
        data: json,
        raw: text,
        mimeType
      };
    } catch {
      // JSON 解析失败，当作纯文本
      return {
        type: 'text',
        data: text,
        raw: text,
        mimeType
      };
    }
  }

  if (mimeType.startsWith('text/')) {
    const text = await blob.text();
    return {
      type: 'text',
      data: text,
      raw: text,
      mimeType
    };
  }

  if (mimeType.startsWith('image/')) {
    // 将 Blob 转为可以直接用于 <img> 的 URL
    const imageUrl = URL.createObjectURL(blob);
    return {
      type: 'image',
      data: imageUrl,
      raw: blob,
      filename,
      mimeType
    };
  }

  // 其他文件类型：PDF、ZIP 等，当作可下载文件
  const fileUrl = URL.createObjectURL(blob);
  return {
    type: 'file',
    data: fileUrl,
    raw: blob,
    filename,
    mimeType
  };
}