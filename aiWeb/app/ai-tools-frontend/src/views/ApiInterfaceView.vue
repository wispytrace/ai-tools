<template>
  <div class="api-interface-view">
    <h2 class="page-title">Api接口</h2>
    <p class="page-subtitle">点击卡片展开，输入参数并执行。</p>

    <!-- 新增筛选栏 -->
    <ApiFilterBar v-model="filter" @search="onSearch" />

    <div class="api-card-list">
      <ApiCard
        v-for="endpoint in filteredEndpoints"
        :key="endpoint.id"
        :config="endpoint"
      />
    </div>

    <div v-if="filteredEndpoints.length === 0" class="no-results">
      没有找到匹配的接口。
    </div>
  </div>
</template>

<script setup lang="ts">
import { apiEndpoints } from '../api/endpoints';
import ApiCard from '../components/ApiCard.vue';
import { ref, computed } from 'vue';
import ApiFilterBar from '../components/ApiFilterBar.vue';

// const endpoints = apiEndpoints;
const endpointsWithCategory = apiEndpoints.map(ep => {
  let category1 = '', category2 = '', category3 = '';
  
  if (ep.id.includes('generate')) {
    category1 = '生成类';
    if (ep.id.includes('text')) {
      category2 = '文本生成';
      category3 = '摘要';
    } else if (ep.id.includes('image')) {
      category2 = '图像生成';
      category3 = '写实';
    }
  } else if (ep.id.includes('analyze')) {
    category1 = '分析类';
    category2 = '文档分析';
    category3 = 'PDF';
  } else if (ep.id.includes('multi')) {
    category1 = '多模态';
    category2 = '图文理解';
    category3 = 'OCR';
  }

  return {
    ...ep,
    category1,
    category2,
    category3
  };
});

// 筛选状态
const filter = ref({
  category1: '',
  category2: '',
  category3: '',
  keyword: ''
});

// 过滤逻辑
const filteredEndpoints = computed(() => {
  const { category1, category2, category3, keyword } = filter.value;
  return endpointsWithCategory.filter(ep => {
    // 分类匹配
    const matchCat1 = !category1 || ep.category1 === category1;
    const matchCat2 = !category2 || ep.category2 === category2;
    const matchCat3 = !category3 || ep.category3 === category3;

    // 关键字匹配
    const matchKeyword = 
      !keyword ||
      ep.name.toLowerCase().includes(keyword.toLowerCase()) ||
      ep.description.toLowerCase().includes(keyword.toLowerCase());

    return matchCat1 && matchCat2 && matchCat3 && matchKeyword;
  });
});

// 搜索触发（可用于埋点或加载状态）
const onSearch = () => {
  console.log('执行筛选:', filter.value);
};
</script>
