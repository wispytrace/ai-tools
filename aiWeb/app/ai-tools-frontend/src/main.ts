// src/main.ts
import { createApp } from 'vue';
import App from './App.vue';
import { router } from './router';

// 全局样式
import './styles/common.css';
import './styles/layout.css';
import './styles/card.css';
import './styles/api-card.css';

const app = createApp(App);
app.use(router);
app.mount('#app');