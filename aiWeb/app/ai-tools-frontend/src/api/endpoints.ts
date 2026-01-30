// src/api/endpoints.ts

type InputType = 'text' | 'json' | 'file';

interface ApiData {
  id: string;
  name: string;
  desc: string;
  intro: string; // 简短介绍（支持简单 HTML）
  path: string;
  method: 'GET' | 'POST';
  inputs: InputType[];
  req?: Record<string, any>;
  res?: Array<{ title: string; data: any }>;
  errors?: Array<{ code: number; msg: string }>;
}

const fmt = (obj: any) => JSON.stringify(obj, null, 2).replace(/[<>]/g, m => m === '<' ? '&lt;' : '&gt;');

const render = ({ path, method, inputs, intro, req, res, errors }: ApiData): string => {
  let h = intro;

  // 请求信息
  h += `<h4>请求</h4><p><strong>URL:</strong> <code>${path}</code><br><strong>Method:</strong> ${method}`;
  if (inputs.includes('file')) h += '<br><strong>Content-Type:</strong> multipart/form-data';
  else if (req) h += '<br><strong>Content-Type:</strong> application/json';
  h += '</p>';

  // 输入字段
  if (inputs.includes('text') || inputs.includes('file')) {
    h += '<h4>输入</h4><ul>';
    if (inputs.includes('text')) h += '<li><code>text</code>: 文本</li>';
    if (inputs.includes('file')) h += '<li><code>file(s)</code>: 文件</li>';
    h += '</ul>';
  }

  // 示例
  if (req) h += `<h4>请求示例</h4><pre><code>${fmt(req)}</code></pre>`;
  if (res) res.forEach(e => h += `<h4>${e.title}</h4><pre><code>${fmt(e.data)}</code></pre>`);
  if (errors?.length) {
    h += '<h4>错误</h4><ul>';
    errors.forEach(e => h += `<li><code>${e.code}</code>: ${e.msg}</li>`);
    h += '</ul>';
  }

  return h;
};

const data: ApiData[] = [
  {
    id: 'generate_text',
    name: 'AI 文本生成',
    desc: '根据提示词生成文本',
    intro: '<p>支持中英文，建议 &lt;500 字。</p>',
    path: '/api/generate-text',
    method: 'POST',
    inputs: ['text'],
    req: { prompt: "写一首春天的诗", max_length: 200 },
    res: [
      { title: '成功 (200)', data: { success: true, result: "春风拂面花自开..." } },
      { title: '失败 (400)', data: { success: false, error: "prompt 不能为空" } }
    ],
    errors: [{ code: 400, msg: "参数无效" }, { code: 500, msg: "服务器错误" }]
  },
  {
    id: 'generate_image',
    name: 'AI 图像生成',
    desc: '根据文本生成图像',
    intro: '<p>仅支持英文 prompt（≤100字符）</p>',
    path: '/api/generate-image',
    method: 'POST',
    inputs: ['json'],
    req: { prompt: "a cute cat", num_images: 1, size: "512x512" },
    res: [{ title: '成功 (200)', data: { success: true, images: ["data:image/png;base64,..."] } }]
  },
  {
    id: 'analyze_document',
    name: '文档分析',
    desc: '上传文档进行分析',
    intro: '<p>支持 PDF/DOCX/TXT</p>',
    path: '/api/analyze-document',
    method: 'POST',
    inputs: ['file'],
    res: [{ title: '成功 (200)', data: { success: true, summary: "本文介绍...", tags: ["climate"] } }],
    errors: [{ code: 400, msg: "文件无效" }, { code: 413, msg: "文件过大" }]
  },
  {
    id: 'multi_modal',
    name: '多模态分析',
    desc: '图文联合分析',
    intro: '<p>需同时提供文本和图片</p>',
    path: '/api/multi-modal',
    method: 'POST',
    inputs: ['text', 'file'],
    res: [{ title: '成功 (200)', data: { success: true, analysis: "场景匹配", confidence: 0.92 } }],
    errors: [{ code: 400, msg: "缺少文本或图片" }]
  }
];

export interface ApiEndpointConfig {
  id: string;
  name: string;
  description: string;
  detail: string;
  endpoint: string;
  method: 'GET' | 'POST';
  inputTypes: InputType[];
}

export const apiEndpoints: ApiEndpointConfig[] = data.map(d => ({
  id: d.id,
  name: d.name,
  description: d.desc,
  detail: render(d),
  endpoint: d.path,
  method: d.method,
  inputTypes: d.inputs
}));