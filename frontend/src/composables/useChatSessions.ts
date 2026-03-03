import { ref } from 'vue'
import { useMessage } from 'naive-ui'

export interface ChatSession {
    id: string;
    title: string;
    created_at: string;
}

export function useChatSessions() {
    const message = useMessage()
    const sessions = ref<ChatSession[]>([])
    const currentSessionId = ref<string | null>(null)
    const isLoadingSessions = ref(false)

    // 1. 获取会话列表
    const fetchSessions = async () => {
        isLoadingSessions.value = true
        try {
            const res = await fetch('/api/sessions')
            if (res.ok) {
                sessions.value = await res.json()
            }
        } catch (e) {
            console.error("获取会话列表失败", e)
        } finally {
            isLoadingSessions.value = false
        }
    }

    // 2. 创建新会话
    const createSession = async () => {
        try {
            const res = await fetch('/api/sessions', { method: 'POST' })
            if (res.ok) {
                const newSession = await res.json()
                sessions.value.unshift(newSession) // 加到开头
                currentSessionId.value = newSession.id // 自动选中
                return newSession.id
            }
        } catch (e) {
            message.error("创建会话失败")
        }
        return null
    }

    // 3. 删除会话
    const deleteSession = async (id: string) => {
        try {
            await fetch(`/api/sessions/${id}`, { method: 'DELETE' })
            // 从列表中移除
            sessions.value = sessions.value.filter(s => s.id !== id)
            // 如果删的是当前选中的，重置选中状态
            if (currentSessionId.value === id) {
                currentSessionId.value = null
            }
            message.success("会话已删除")
        } catch (e) {
            message.error("删除失败")
        }
    }

    // 4. 切换会话 (只改变 ID，具体加载消息由 useChat 负责)
    const selectSession = (id: string) => {
        currentSessionId.value = id
    }

    return {
        sessions,
        currentSessionId,
        isLoadingSessions,
        fetchSessions,
        createSession,
        deleteSession,
        selectSession
    }
}