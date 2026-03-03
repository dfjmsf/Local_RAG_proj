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

            let isThinking = false
            let buffer = ''

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
                            if (text.includes('<think>')) { isThinking = true; continue }
                            if (text.includes('</think>')) { isThinking = false; continue }

                            if (isThinking) aiMessage.value.thought = (aiMessage.value.thought || "") + text
                            else aiMessage.value.content += text
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