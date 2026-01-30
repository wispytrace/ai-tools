<template>
  <div class="api-card">
    <!-- 卡片头部：标题 + 简短描述 + 展开按钮 -->
    <div class="api-card-header" @click="toggleExpand">
      <div>
        <h3 class="api-card-title">{{ config.name }}</h3>
        <p class="api-card-description">{{ config.description }}</p>
      </div>
      <button class="api-card-toggle-btn" type="button">
        {{ expanded ? '收起' : '展开' }}
      </button>
    </div>

    <!-- 展开内容 -->
    <div v-if="expanded" class="api-card-body">
      <!-- 详细说明（支持 HTML） -->
      <div class="api-card-detail" v-html="config.detail"></div>

      <!-- 输入区域 -->
      <div class="api-card-section">
        <h4 class="section-title">输入</h4>

        <div v-if="config.inputTypes.includes('text')" class="input-group">
          <label class="input-label">文本输入</label>
          <textarea
            v-model="textInput"
            class="text-input"
            rows="3"
            placeholder="请输入文本，例如提示词、说明等"
          ></textarea>
        </div>

        <div v-if="config.inputTypes.includes('json')" class="input-group">
          <label class="input-label">
            JSON 输入
            <span class="input-helper">（请填写合法 JSON）</span>
          </label>
          <textarea
            v-model="jsonInput"
            class="json-input"
            rows="6"
            placeholder='例如：{"size": "512x512", "num_images": 1}'
          ></textarea>
        </div>

        <div v-if="config.inputTypes.includes('file')" class="input-group">
          <label class="input-label">文件上传</label>
          <input
            type="file"
            multiple
            @change="onFileChange"
          />
          <ul v-if="selectedFiles.length" class="file-list">
            <li
              v-for="(file, index) in selectedFiles"
              :key="index"
              class="file-item"
            >
              {{ file.name }} ({{ formatFileSize(file.size) }})
            </li>
          </ul>
        </div>

        <div class="action-row">
          <button
            class="execute-btn"
            type="button"
            :disabled="loading"
            @click.stop="execute"
          >
            {{ loading ? '执行中...' : '执行' }}
          </button>
          <span v-if="error" class="error-text">{{ error }}</span>
        </div>
      </div>

      <!-- 输出区域 -->
      <div class="api-card-section">
        <h4 class="section-title">输出</h4>

        <div v-if="loading" class="loading-text">正在请求接口，请稍候...</div>

        <div v-else-if="response">
          <!-- JSON -->
          <div v-if="response.type === 'json'" class="output-json">
            <pre class="json-pretty">{{ prettyJson }}</pre>
          </div>

          <!-- 文本 -->
          <div v-else-if="response.type === 'text'" class="output-text">
            <pre class="text-pre">{{ response.data }}</pre>
          </div>

          <!-- 图片 -->
          <div v-else-if="response.type === 'image'" class="output-image">
            <img :src="response.data" alt="接口返回图片" class="output-img" />
            <button
              v-if="response.data"
              class="download-btn"
              type="button"
              @click="downloadFile"
            >
              下载图片
            </button>
          </div>

          <!-- 文件 -->
          <div v-else-if="response.type === 'file'" class="output-file">
            <p>接口返回文件：</p>
            <button
              class="download-btn"
              type="button"
              @click="downloadFile"
            >
              下载{{ response.filename || '文件' }}
            </button>
          </div>
        </div>

        <div v-else class="placeholder-text">
          尚无输出，请先输入参数并点击“执行”。
        </div>
      </div>
    </div>
  </div>
</template>

<script setup lang="ts">
import { computed, ref } from 'vue';
import type { ApiEndpointConfig } from '../api/endpoints';
import { callApi, type ParsedResponse } from '../utils/apiClient';

interface Props {
  config: ApiEndpointConfig;
}

const props = defineProps<Props>();

const expanded = ref(false);
const textInput = ref('');
const jsonInput = ref('');
const selectedFiles = ref<File[]>([]);
const loading = ref(false);
const error = ref<string | null>(null);
const response = ref<ParsedResponse | null>(null);

const prettyJson = computed(() => {
  if (!response.value || response.value.type !== 'json') return '';
  try {
    return JSON.stringify(response.value.data, null, 2);
  } catch {
    return String(response.value.data);
  }
});

function toggleExpand() {
  expanded.value = !expanded.value;
}

function onFileChange(event: Event) {
  const target = event.target as HTMLInputElement;
  selectedFiles.value = [];
  if (target.files) {
    selectedFiles.value = Array.from(target.files);
  }
}

function formatFileSize(bytes: number): string {
  if (bytes < 1024) return `${bytes} B`;
  if (bytes < 1024 * 1024) return `${(bytes / 1024).toFixed(1)} KB`;
  return `${(bytes / (1024 * 1024)).toFixed(1)} MB`;
}

async function execute() {
  loading.value = true;
  error.value = null;
  response.value = null;

  try {
    const res = await callApi({
      endpoint: props.config.endpoint,
      method: props.config.method,
      inputTypes: props.config.inputTypes,
      textInput: textInput.value || undefined,
      jsonInput: jsonInput.value || undefined,
      files: selectedFiles.value.length ? selectedFiles.value : undefined
    });

    response.value = res;
  } catch (e: any) {
    console.error(e);
    error.value = e?.message || '请求失败，请稍后重试。';
  } finally {
    loading.value = false;
  }
}

function downloadFile() {
  if (!response.value || !response.value.data) return;

  const url = response.value.data as string;
  const link = document.createElement('a');
  link.href = url;
  link.download = response.value.filename || 'download';
  document.body.appendChild(link);
  link.click();
  document.body.removeChild(link);
}
</script>