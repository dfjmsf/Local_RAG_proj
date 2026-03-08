import { ref } from 'vue'
import type { ChatMessage } from '../types/chat'

export function useChat() {
    const messages = ref<ChatMessage[]>([])
    const isLoading = ref(false)

    // [新增] 加载指定会话的历史记录
    const loadSessionHistory = async (sessionId: string) => {
        messages.value = [] // 先清空当前屏幕
        isLoading.value = true
        try {
            const res = await fetch(`/api/sessions/${sessionId}/messages`)
            if (res.ok) {
                const historyData = await res.json()
                // 映射后端数据到前端格式
                messages.value = historyData.map((msg: any) => ({
                    role: msg.role,
                    content: msg.content,
                    thought: msg.thought || '', // 恢复思考过程
                    sources: msg.sources || []  // 恢复参考来源
                }))
            }
        } catch (e) {
            console.error("加载历史记录失败", e)
        } finally {
            isLoading.value = false
        }
    }

    // [修改] 发送消息增加 sessionId 参数
    const sendMessage = async (question: string, mode: string, sessionId: string | null) => {

        // 1. 准备历史 payload (给 AI 看的，只包含 role 和 content)
        const historyPayload = messages.value.map(msg => ({
            role: msg.role,
            content: msg.content || ""
        }))

        // 2. UI 添加用户消息
        messages.value.push({ role: 'user', content: question })

        // 3. UI 添加 AI 占位
        const aiMessage = ref<ChatMessage>({
            role: 'assistant',
            content: '',
            thought: '',
            isStreaming: true
        })
        messages.value.push(aiMessage.value)
        isLoading.value = true

        try {
            const response = await fetch('/api/chat', {
                method: 'POST',
                headers: { 'Content-Type': 'application/json' },
                body: JSON.stringify({
                    question: question,
                    history: historyPayload,
                    mode: mode,
                    session_id: sessionId // <--- [关键] 告诉后端存到哪个会话里
                })
            })

            if (!response.ok) throw new Error('Network error')
            if (!response.body) throw new Error('No readable stream')

            const reader = response.body.getReader()
            const decoder = new TextDecoder()

            let rawBuffer = ''  // [修复] 累积全部原始输出，避免标签跨 chunk
            let buffer = ''     // NDJSON 行拼接缓冲

            while (true) {
                const { done, value } = await reader.read()
                if (done) break

                const chunk = decoder.decode(value, { stream: true })
                buffer += chunk
                const lines = buffer.split('\n')
                buffer = lines.pop() || ''

                for (const line of lines) {
                    if (!line.trim()) continue
                    try {
                        const json = JSON.parse(line)

                        if (json.type === 'intent') aiMessage.value.intent = json.data
                        else if (json.type === 'sources') aiMessage.value.sources = json.data
                        else if (json.type === 'content') {
                            const text = json.data
                            // [修复] 累积到 rawBuffer，通过全局状态判断标签
                            rawBuffer += text

                            if (rawBuffer.includes('</think>')) {
                                // 标签已闭合：提取思考和正文
                                const parts = rawBuffer.split('</think>')
                                aiMessage.value.thought = parts[0].replace('<think>', '')
                                aiMessage.value.content = parts.slice(1).join('')
                            } else if (rawBuffer.includes('<think>')) {
                                // 标签未闭合：全部是思考内容
                                aiMessage.value.thought = rawBuffer.replace('<think>', '')
                                aiMessage.value.content = ''
                            } else {
                                // 无标签：全部是正文
                                aiMessage.value.content = rawBuffer
                            }
                        }
                        else if (json.type === 'error') aiMessage.value.content += `\n[Error: ${json.data}]`
                    } catch (e) { }
                }
            }
        } catch (e) {
            aiMessage.value.content += "\n[请求失败]"
        } finally {
            isLoading.value = false
            aiMessage.value.isStreaming = false
        }
    }

    const clearChat = () => {
        messages.value = []
    }

    return {
        messages,
        isLoading,
        sendMessage,
        clearChat,
        loadSessionHistory // 导出新方法
    }
}