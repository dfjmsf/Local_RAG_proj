<script setup lang="ts">
import { ref, nextTick, watch, onMounted, computed } from 'vue'
import {
  NLayout, NLayoutSider, NLayoutContent,
  NInput, NButton, NSpace, NTag, NCollapse, NCollapseItem, NSpin,
  NUpload, NUploadDragger, NList, NListItem, NScrollbar, NDivider, NNumberAnimation
} from 'naive-ui' // 移除了未使用的 NStatistic
import {
  PaperPlane, TrashBin, Refresh, CloudUpload, DocumentText, Flash, Rocket,
  CheckmarkCircle, Time, ChatboxEllipses, Add, StatsChart
} from '@vicons/ionicons5'
import MarkdownIt from 'markdown-it'
import hljs from 'highlight.js'
import 'highlight.js/styles/github-dark.css'
import { useChat } from '../composables/useChat'
import { useKnowledgeBase } from '../composables/useKnowledgeBase'
import { useChatSessions } from '../composables/useChatSessions'

// --- Markdown 配置 ---
const md = new MarkdownIt({
  html: true,
  linkify: true,
  highlight: (str, lang) => {
    if (lang && hljs.getLanguage(lang)) {
      try { return '<pre class="hljs"><code>' + hljs.highlight(str, { language: lang, ignoreIllegals: true }).value + '</code></pre>'; } catch (__) {}
    }
    return '';
  }
})

// --- 逻辑引入 ---
const { messages, isLoading, sendMessage, clearChat, loadSessionHistory } = useChat()
const {
  indexedFiles, pendingFiles, isRebuilding,
  fetchFiles, uploadFiles, rebuildDb, resetDb
} = useKnowledgeBase()
const {
  sessions, currentSessionId, fetchSessions, createSession, deleteSession, selectSession
} = useChatSessions()

// --- UI 状态 ---
const inputVal = ref("")
const chatContainer = ref<HTMLElement | null>(null)
const mode = ref("pro")

// --- 计算属性：全局 Token 估算 ---
// 增加 ?. 防止 TS 报错 Object is possibly 'undefined'
const totalSessionTokens = computed(() => {
  return messages.value.reduce((acc, msg) => {
    return acc + (msg?.content?.length || 0) + (msg?.thought?.length || 0)
  }, 0)
})

// --- 初始化与交互 ---
onMounted(async () => {
  await fetchFiles()
  await fetchSessions()
  // 增加安全的空值检查，防止 TS 报错
  if (sessions.value && sessions.value.length > 0 && sessions.value[0]) {
    handleSelectSession(sessions.value[0].id)
  } else {
    await handleNewChat()
  }
})

const handleSelectSession = async (id: string) => {
  if (currentSessionId.value === id) return
  selectSession(id)
  await loadSessionHistory(id)
}

const handleNewChat = async () => {
  const newId = await createSession()
  if (newId) {
    clearChat()
  }
}

const handleDeleteSession = async (id: string, e: Event) => {
  e.stopPropagation()
  await deleteSession(id)
  if (currentSessionId.value === null) await handleNewChat()
}

const handleSend = async () => {
  if (!inputVal.value.trim() || isLoading.value) return
  if (!currentSessionId.value) await handleNewChat()

  const q = inputVal.value
  inputVal.value = ""
  // 确保 ID 存在
  if (currentSessionId.value) {
    await sendMessage(q, mode.value, currentSessionId.value)
    fetchSessions()
  }
}

// Enter 发送，Shift+Enter 换行
const handleKeydown = (e: KeyboardEvent) => {
  if (e.key === 'Enter' && !e.shiftKey) {
    e.preventDefault()
    handleSend()
  }
}

watch(messages, () => {
  nextTick(() => {
    if (chatContainer.value) {
      chatContainer.value.scrollTop = chatContainer.value.scrollHeight
    }
  })
}, { deep: true })

const customRequest = async ({ file, onFinish, onError }: any) => {
  try {
    // 增加边界检查
    if (file && file.file) {
      await uploadFiles([file.file])
      onFinish()
    } else {
      onError()
    }
  } catch (e) {
    onError()
  }
}
</script>

<template>
  <n-layout has-sider class="h-screen bg-[#f8f9fa]">

    <n-layout-sider
      bordered
      width="240"
      content-style="background-color: #f9fafb; display: flex; flex-direction: column;"
    >
      <div class="p-4 border-b border-gray-200 bg-white/50 backdrop-blur">
        <n-button type="primary" dashed block @click="handleNewChat" class="shadow-sm">
          <template #icon><Add /></template>
          新建对话
        </n-button>
      </div>

      <n-scrollbar class="flex-1">
        <n-list hoverable clickable class="bg-transparent">
          <n-list-item
            v-for="session in sessions"
            :key="session.id"
            :class="{'!bg-white !border-r-4 !border-r-blue-500 shadow-sm': currentSessionId === session.id}"
            @click="handleSelectSession(session.id)"
          >
            <div class="px-4 py-3 flex justify-between items-center group">
              <div class="flex items-center gap-3 overflow-hidden">
                <ChatboxEllipses class="w-4 h-4 text-gray-400" :class="{'text-blue-500': currentSessionId === session.id}"/>
                <div class="flex flex-col">
                  <span class="text-sm text-gray-700 truncate w-32 font-medium" :title="session.title">
                    {{ session.title }}
                  </span>
                  <span class="text-[10px] text-gray-400">{{ session.created_at.substring(5, 16) }}</span>
                </div>
              </div>

              <n-button
                size="tiny" quaternary circle type="error"
                class="opacity-0 group-hover:opacity-100 transition-opacity"
                @click.stop="(e) => handleDeleteSession(session.id, e)"
              >
                <template #icon><TrashBin /></template>
              </n-button>
            </div>
          </n-list-item>
        </n-list>
      </n-scrollbar>
    </n-layout-sider>

    <n-layout has-sider sider-placement="right">

      <n-layout-content content-style="display: flex; flex-direction: column; height: 100vh;">

        <div class="h-16 border-b bg-white flex items-center px-6 justify-between shadow-sm z-10">
          <div class="flex items-center gap-3">
            <div class="bg-blue-50 p-2 rounded-lg text-xl">🤖</div>
            <div>
              <div class="font-bold text-gray-800 leading-tight">Local RAG Assistant</div>
              <div class="text-[10px] text-gray-400 font-medium tracking-wide">POWERED BY DEEPSEEK R1</div>
            </div>
          </div>

          <div class="flex items-center gap-4">
            <div class="hidden sm:flex flex-col items-end mr-4">
              <span class="text-[10px] text-gray-400 uppercase tracking-wider font-bold">Total Tokens</span>
              <div class="text-sm font-mono font-bold text-blue-600 flex items-center gap-1">
                <StatsChart class="w-3 h-3"/>
                <n-number-animation :from="0" :to="totalSessionTokens" />
              </div>
            </div>

            <n-button size="small" quaternary type="error" @click="clearChat">
              <template #icon><TrashBin /></template>
              清屏
            </n-button>
          </div>
        </div>

        <div ref="chatContainer" class="flex-1 overflow-y-auto p-4 sm:p-8 space-y-8 scroll-smooth bg-[#f8f9fa]">

          <div v-if="messages.length === 0" class="flex flex-col items-center justify-center h-full text-gray-300 select-none pb-20">
            <div class="bg-white p-6 rounded-full shadow-sm mb-4">
              <div class="text-4xl text-blue-500">💬</div>
            </div>
            <div class="text-sm font-medium text-gray-400">选择模式，上传文档，开始对话</div>
          </div>

          <div v-for="(msg, index) in messages" :key="index" class="flex flex-col group animate-fade-in">

            <div v-if="msg.role === 'user'" class="self-end max-w-[85%] sm:max-w-[70%]">
              <div class="bg-blue-600 text-white px-5 py-3 rounded-2xl rounded-tr-sm shadow-md text-base leading-relaxed">
                {{ msg.content }}
              </div>
            </div>

            <div v-else class="self-start max-w-[95%] sm:max-w-[85%] w-full">
              <div class="flex gap-4">
                <div class="w-8 h-8 rounded-full bg-white border border-gray-100 flex items-center justify-center shadow-sm shrink-0 mt-1 text-sm">
                  🤖
                </div>
                <div class="flex-1 min-w-0 space-y-3">

                  <div v-if="msg.intent" class="flex items-center gap-2">
                    <n-tag size="tiny" :type="msg.intent === 'SEARCH' ? 'info' : 'success'" :bordered="false" round class="font-bold bg-opacity-10 px-2">
                      {{ msg.intent === 'SEARCH' ? '🔍 知识库检索' : '💬 闲聊模式' }}
                    </n-tag>
                  </div>

                  <div v-if="msg.thought" class="max-w-3xl">
                    <n-collapse arrow-placement="right">
                      <n-collapse-item name="1">
                        <template #header>
                          <div class="flex items-center gap-2">
                            <span class="text-xs text-gray-500 font-bold">深度思考链</span>
                            <n-tag size="tiny" round :bordered="false" class="bg-gray-200 text-gray-500 font-mono">
                              {{ msg.thought.length }} tokens
                            </n-tag>
                          </div>
                        </template>
                        <div class="text-xs text-gray-600 bg-white p-4 rounded-lg border border-gray-200 font-mono leading-relaxed whitespace-pre-wrap shadow-sm">
                          {{ msg.thought }}
                        </div>
                      </n-collapse-item>
                    </n-collapse>
                  </div>

                  <div v-if="msg.content" class="relative group/content">
                    <div class="prose prose-slate max-w-none bg-white px-6 py-5 rounded-2xl rounded-tl-sm border border-gray-100 shadow-sm"
                         v-html="md.render(msg.content)">
                    </div>
                    <div class="absolute -bottom-5 right-2 opacity-0 group-hover/content:opacity-100 transition-opacity text-[10px] text-gray-400 font-mono">
                      Response: {{ msg.content.length }} chars
                    </div>
                  </div>

                  <div v-if="msg.isStreaming && !msg.content && !msg.thought">
                    <div class="flex items-center gap-2 text-gray-400 text-sm bg-white px-4 py-2 rounded-full shadow-sm inline-flex">
                      <n-spin size="small" />
                      <span>DeepSeek 正在思考...</span>
                    </div>
                  </div>

                  <div v-if="msg.sources && msg.sources.length > 0" class="max-w-3xl pt-1">
                    <n-collapse>
                      <n-collapse-item name="source">
                        <template #header>
                          <span class="text-xs text-blue-600 font-bold flex items-center gap-1">
                            <DocumentText class="w-3 h-3"/> 引用 {{ msg.sources.length }} 个文档
                          </span>
                        </template>
                        <div class="grid gap-3 sm:grid-cols-2 mt-2">
                          <div v-for="(source, idx) in msg.sources" :key="idx"
                               class="bg-blue-50/30 p-3 rounded-lg border border-blue-100/50 hover:border-blue-200 transition-colors cursor-default">
                            <div class="flex items-start justify-between mb-1">
                              <div class="font-bold text-xs text-blue-700 truncate w-3/4" :title="source.source">
                                {{ idx+1 }}. {{ source.source }}
                              </div>
                              <div class="text-[10px] text-blue-400 bg-white px-1.5 py-0.5 rounded border border-blue-100">P{{ source.page }}</div>
                            </div>
                            <div class="text-xs text-gray-500 line-clamp-2 leading-relaxed">
                              {{ source.content }}
                            </div>
                          </div>
                        </div>
                      </n-collapse-item>
                    </n-collapse>
                  </div>

                </div>
              </div>
            </div>
          </div>
        </div>

        <div class="p-6 bg-white border-t border-gray-100 shrink-0">
          <div class="max-w-4xl mx-auto relative">
            <n-input
              v-model:value="inputVal"
              type="textarea"
              :autosize="{ minRows: 1, maxRows: 6 }"
              placeholder="请输入问题 (Enter 发送, Shift+Enter 换行)..."
              @keydown="handleKeydown"
              class="shadow-sm !rounded-2xl !py-3 !pl-4 !pr-14 bg-gray-50 focus-within:!bg-white focus-within:!shadow-md transition-all text-base"
              size="large"
            />
            <n-button
              type="primary"
              circle
              class="absolute right-2 bottom-2 shadow-lg"
              @click="handleSend"
              :disabled="isLoading || !inputVal.trim()"
              color="#2563eb"
            >
              <template #icon><PaperPlane /></template>
            </n-button>
          </div>
        </div>
      </n-layout-content>

      <n-layout-sider
          bordered
          width="320"
          content-style="background-color: #fff;"
          collapse-mode="transform"
          :collapsed-width="0"
          show-trigger="arrow-circle"
          class="shadow-xl z-20"
      >
        <div class="p-6 h-full flex flex-col bg-white">
          <div class="font-bold text-base mb-6 text-gray-800 flex items-center gap-2 border-b pb-4">
            <span class="text-xl">⚙️</span> 控制台 (Control)
          </div>

          <n-scrollbar class="flex-1 pr-4 -mr-4">
            <n-space vertical size="large">

              <div>
                <div class="text-xs font-bold text-gray-400 mb-3 uppercase tracking-wider pl-1">Retrieval Mode</div>
                <div class="flex flex-col gap-3">
                  <div
                    class="relative p-4 rounded-xl border-2 cursor-pointer transition-all duration-200 group"
                    :class="mode === 'flash' ? 'border-orange-400 bg-orange-50' : 'border-gray-100 hover:border-orange-200 bg-white'"
                    @click="mode = 'flash'"
                  >
                    <div class="flex items-center justify-between mb-1">
                      <div class="flex items-center gap-2">
                        <div class="p-1.5 rounded-lg" :class="mode === 'flash' ? 'bg-orange-200 text-orange-700' : 'bg-gray-100 text-gray-400'">
                          <Flash class="w-5 h-5"/>
                        </div>
                        <span class="font-bold text-gray-800">Flash (极速)</span>
                      </div>
                      <div v-if="mode === 'flash'" class="w-3 h-3 rounded-full bg-orange-500 shadow-[0_0_8px_rgba(249,115,22,0.6)]"></div>
                    </div>
                    <p class="text-xs text-gray-500 pl-10 leading-relaxed">
                      仅使用向量检索 (Top-3)。<br>速度最快，适合简单查询。
                    </p>
                  </div>

                  <div
                    class="relative p-4 rounded-xl border-2 cursor-pointer transition-all duration-200 group"
                    :class="mode === 'pro' ? 'border-blue-500 bg-blue-50' : 'border-gray-100 hover:border-blue-200 bg-white'"
                    @click="mode = 'pro'"
                  >
                    <div class="flex items-center justify-between mb-1">
                      <div class="flex items-center gap-2">
                        <div class="p-1.5 rounded-lg" :class="mode === 'pro' ? 'bg-blue-200 text-blue-700' : 'bg-gray-100 text-gray-400'">
                          <Rocket class="w-5 h-5"/>
                        </div>
                        <span class="font-bold text-gray-800">Pro (深度)</span>
                      </div>
                      <div v-if="mode === 'pro'" class="w-3 h-3 rounded-full bg-blue-600 shadow-[0_0_8px_rgba(37,99,235,0.6)]"></div>
                    </div>
                    <p class="text-xs text-gray-500 pl-10 leading-relaxed">
                      检索更多 (Top-20) 并使用 Rerank 重排序。<br>精准度高，但稍慢。
                    </p>
                  </div>
                </div>
              </div>

              <n-divider />

              <div>
                <div class="text-xs font-bold text-gray-400 mb-2 uppercase tracking-wider pl-1">Knowledge Base</div>
                <n-upload
                  multiple
                  directory-dnd
                  :custom-request="customRequest"
                  :show-file-list="false"
                >
                  <n-upload-dragger class="!p-6 !rounded-xl border-dashed border-2 border-blue-100 hover:border-blue-400 hover:bg-blue-50 transition-all group cursor-pointer bg-slate-50">
                    <div class="flex flex-col items-center gap-3 text-gray-400 group-hover:text-blue-500">
                      <CloudUpload class="w-10 h-10 opacity-50 group-hover:opacity-100 transition-opacity"/>
                      <span class="text-xs font-medium">点击或拖拽上传文档</span>
                    </div>
                  </n-upload-dragger>
                </n-upload>
              </div>

              <div class="flex flex-col gap-4">
                <div v-if="pendingFiles.length > 0" class="animate-pulse">
                  <div class="flex justify-between items-center mb-2 px-1">
                    <span class="text-[10px] font-bold text-orange-500 uppercase tracking-wider flex items-center gap-1">
                      <Time class="w-3 h-3"/> 待入库 ({{ pendingFiles.length }})
                    </span>
                  </div>
                  <div class="border border-orange-200 bg-orange-50 rounded-lg overflow-hidden">
                    <n-list hoverable clickable size="small" class="bg-transparent">
                      <n-list-item v-for="file in pendingFiles" :key="file">
                        <div class="flex items-center gap-2 text-xs text-orange-700 px-2 py-1">
                          <span class="truncate block w-full" :title="file">{{ file }}</span>
                        </div>
                      </n-list-item>
                    </n-list>
                  </div>
                </div>

                <div>
                  <div class="flex justify-between items-center mb-2 px-1">
                    <span class="text-[10px] font-bold text-green-600 uppercase tracking-wider flex items-center gap-1">
                      <CheckmarkCircle class="w-3 h-3"/> 已生效 ({{ indexedFiles.length }})
                    </span>
                    <n-button text size="tiny" @click="fetchFiles" class="text-gray-400 hover:text-blue-500">
                      <template #icon><Refresh /></template>
                    </n-button>
                  </div>
                  <div class="border border-gray-200 bg-white rounded-lg overflow-hidden shadow-sm min-h-[80px]">
                    <n-scrollbar style="max-height: 200px;">
                      <n-list hoverable clickable size="small" class="bg-transparent">
                        <template v-if="indexedFiles.length > 0">
                          <n-list-item v-for="file in indexedFiles" :key="file">
                            <div class="flex items-center gap-2 text-xs text-gray-700 px-2 py-1">
                              <DocumentText class="w-3 h-3 text-green-500 shrink-0"/>
                              <span class="truncate block w-full" :title="file">{{ file }}</span>
                            </div>
                          </n-list-item>
                        </template>
                        <template v-else>
                          <div class="p-4 text-center text-[10px] text-gray-400 italic flex flex-col items-center gap-1 mt-2">
                            <div class="w-4 h-4 border-2 border-gray-200 rounded-full"></div>
                            <span>知识库为空</span>
                          </div>
                        </template>
                      </n-list>
                    </n-scrollbar>
                  </div>
                </div>
              </div>

              <div class="grid gap-3 pt-6 mt-auto">
                <n-button type="primary" block secondary @click="rebuildDb" :loading="isRebuilding" size="medium" class="!font-bold shadow-sm">
                  <template #icon><Refresh /></template>
                  构建/更新索引
                </n-button>
                <n-divider dashed class="!my-1 text-gray-300">Danger Zone</n-divider>
                <n-button secondary type="error" block @click="resetDb" :disabled="isRebuilding" size="medium">
                  <template #icon><TrashBin /></template>
                  清空并重置
                </n-button>
              </div>

            </n-space>
          </n-scrollbar>
        </div>
      </n-layout-sider>

    </n-layout> </n-layout>
</template>

<style>
/* CSS 动画：让消息平滑淡入 */
@keyframes fadeIn {
  from { opacity: 0; transform: translateY(10px); }
  to { opacity: 1; transform: translateY(0); }
}
.animate-fade-in {
  animation: fadeIn 0.3s ease-out forwards;
}
</style>