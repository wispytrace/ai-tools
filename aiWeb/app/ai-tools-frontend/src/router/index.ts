// src/router/index.ts
import { createRouter, createWebHistory} from 'vue-router';
import ApiInterfaceView from '../views/ApiInterfaceView.vue';
import AboutView from '../views/AboutView.vue';
import type { RouteRecordRaw } from 'vue-router'; // ✅ 正确：只用于类型注解

const routes: RouteRecordRaw[] = [
  {
    path: '/',
    redirect: '/api'
  },
  {
    path: '/api',
    name: 'ApiInterface',
    component: ApiInterfaceView
  },
  {
    path: '/about',
    name: 'About',
    component: AboutView
  }
];

export const router = createRouter({
  history: createWebHistory(),
  routes
});