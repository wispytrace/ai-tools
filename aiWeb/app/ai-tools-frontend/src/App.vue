<template>
  <div class="app-container">
    <aside class="sidebar">
      <div class="sidebar-header">
        <h1 class="logo">Api 工具面板</h1>
      </div>
      <nav class="sidebar-nav">
        <RouterLink
          to="/api"
          class="nav-link"
          :class="{ active: $route.path.startsWith('/api') }"
        >
          API 接口
        </RouterLink>
        <RouterLink
          to="/about"
          class="nav-link"
          :class="{ active: $route.path.startsWith('/about') }"
        >
          关于
        </RouterLink>
      </nav>
    </aside>

    <main class="main-content">
      <RouterView />
    </main>
  </div>
</template>

<script setup lang="ts">
import { RouterLink, RouterView, useRoute } from 'vue-router';
const route = useRoute(); // 若需要可以使用
</script>