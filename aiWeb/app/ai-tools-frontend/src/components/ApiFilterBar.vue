<!-- src/components/ApiFilterBar.vue -->
<template>
  <div class="api-filter-bar">
    <!-- 第一行：分类筛选 -->
    <div class="filter-row filter-row--top">
      <div class="filter-item">
        <label>筛选类型：</label>
        <select v-model="localFilter.category1" @change="handleCategoryChange('category1')">
          <option value="">全部</option>
          <option v-for="cat in categoryOptions.level1" :key="cat" :value="cat">
            {{ cat }}
          </option>
        </select>
      </div>

      <div class="filter-item">
        <label>二级分类：</label>
        <select v-model="localFilter.category2" @change="handleCategoryChange('category2')">
          <option value="">全部</option>
          <option
            v-for="cat in availableLevel2"
            :key="cat"
            :value="cat"
            :disabled="!localFilter.category1"
          >
            {{ cat }}
          </option>
        </select>
      </div>

      <div class="filter-item">
        <label>三级分类：</label>
        <select v-model="localFilter.category3">
          <option value="">全部</option>
          <option
            v-for="cat in availableLevel3"
            :key="cat"
            :value="cat"
            :disabled="!localFilter.category2"
          >
            {{ cat }}
          </option>
        </select>
      </div>
    </div>

    <!-- 第二行：关键词 + 操作按钮 -->
    <div class="filter-row filter-row--bottom">
      <div class="filter-item keyword-item">
        <label>关键字：</label>
        <input
          v-model="localFilter.keyword"
          type="text"
          placeholder="名称或描述中搜索..."
          @keyup.enter="applyFilter"
        />
      </div>

      <div class="button-group">
        <button class="btn btn--reset" @click="resetFilter">重置</button>
        <button class="btn btn--search" @click="applyFilter">查找</button>
      </div>
    </div>
  </div>
</template>

<script setup lang="ts">
import { ref, watch, computed} from 'vue';
// 定义分类选项（可根据实际业务扩展）
const categoryOptions = {
  level1: ['生成类', '分析类', '多模态'],
  level2: {
    '生成类': ['文本生成', '图像生成'],
    '分析类': ['文档分析', '数据解析'],
    '多模态': ['图文理解', '视频分析']
  },
  level3: {
    '文本生成': ['诗歌', '摘要', '翻译'],
    '图像生成': ['写实', '卡通', '艺术'],
    '文档分析': ['PDF', 'Word', 'Excel'],
    '图文理解': ['OCR', 'VQA']
  }
};

// Props / Emits
const props = defineProps<{
  modelValue: {
    category1: string;
    category2: string;
    category3: string;
    keyword: string;
  };
}>();

const emit = defineEmits<{
  (e: 'update:modelValue', value: typeof props.modelValue): void;
  (e: 'search'): void;
}>();

// 本地状态
const localFilter = ref({ ...props.modelValue });

// 计算可用的二级、三级选项
const availableLevel2 = computed(() => {
  if (!localFilter.value.category1) return [];
  return categoryOptions.level2[localFilter.value.category1 as keyof typeof categoryOptions.level2] || [];
});

const availableLevel3 = computed(() => {
  if (!localFilter.value.category2) return [];
  return categoryOptions.level3[localFilter.value.category2 as keyof typeof categoryOptions.level3] || [];
});

// 处理一级/二级变更时清空下级
const handleCategoryChange = (level: 'category1' | 'category2') => {
  if (level === 'category1') {
    localFilter.value.category2 = '';
    localFilter.value.category3 = '';
  } else if (level === 'category2') {
    localFilter.value.category3 = '';
  }
};

// 应用筛选
const applyFilter = () => {
  emit('update:modelValue', { ...localFilter.value });
  emit('search');
};

// 重置
const resetFilter = () => {
  localFilter.value = {
    category1: '',
    category2: '',
    category3: '',
    keyword: ''
  };
  emit('update:modelValue', { ...localFilter.value });
  emit('search');
};

// 监听外部 props 变化（如父组件重置）
watch(
  () => props.modelValue,
  (newVal) => {
    localFilter.value = { ...newVal };
  },
  { deep: true }
);
</script>
