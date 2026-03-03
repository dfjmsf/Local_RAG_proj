<!-- frontend/src/App.vue -->
<script setup lang="ts">
import { NConfigProvider, NMessageProvider, NGlobalStyle } from 'naive-ui'
// 引入我们刚才新建的组件
import ChatLayout from './components/ChatLayout.vue'
</script>

<template>
  <n-config-provider>
    <n-global-style />
    <n-message-provider>
      <!-- 渲染主界面 -->
      <ChatLayout />
    </n-message-provider>
  </n-config-provider>
</template>

<style>
/* 全局样式覆盖 */
body {
  font-family: 'Inter', -apple-system, BlinkMacSystemFont, "Segoe UI", Roboto, sans-serif;
  margin: 0;
  padding: 0;
}

/* 自定义滚动条 */
::-webkit-scrollbar {
  width: 5px;
  height: 5px;
}
::-webkit-scrollbar-track {
  background: transparent;
}
::-webkit-scrollbar-thumb {
  background: #e2e8f0;
  border-radius: 3px;
}
::-webkit-scrollbar-thumb:hover {
  background: #cbd5e1;
}

/* Markdown 代码块样式 */
.prose pre {
  background-color: #1e293b !important;
  color: #e2e8f0 !important;
  padding: 1rem;
  border-radius: 0.5rem;
  overflow-x: auto;
  font-size: 0.85em;
  margin: 0.5rem 0;
}
.prose code {
  background-color: #f1f5f9;
  color: #ef4444;
  padding: 0.2em 0.4em;
  border-radius: 0.25rem;
  font-size: 0.9em;
  font-weight: 500;
}
.prose pre code {
  background-color: transparent;
  color: inherit;
  padding: 0;
  font-size: 1em;
}
.prose ul {
  list-style-type: disc;
  padding-left: 1.25rem;
  margin: 0.5rem 0;
}
.prose ol {
  list-style-type: decimal;
  padding-left: 1.25rem;
  margin: 0.5rem 0;
}
.prose p {
  margin: 0.5rem 0;
  line-height: 1.6;
}
</style>