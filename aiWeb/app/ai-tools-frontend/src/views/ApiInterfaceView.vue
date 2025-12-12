<template>
  <div class="api-interface-view">
    <h2 class="page-title">AI 工具接口</h2>
    <p class="page-subtitle">点击卡片展开，输入参数并执行。</p>

    <div class="api-card-list">
      <ApiCard
        v-for="endpoint in endpoints"
        :key="endpoint.id"
        :config="endpoint"
      />
    </div>
  </div>
</template>

<script setup lang="ts">
import { apiEndpoints } from '../api/endpoints';
import ApiCard from '../components/ApiCard.vue';

const endpoints = apiEndpoints;
</script>