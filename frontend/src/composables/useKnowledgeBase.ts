import { ref } from 'vue'
import { useMessage } from 'naive-ui'

export function useKnowledgeBase() {
    const message = useMessage()

    // [修改] 拆分为两个列表
    const indexedFiles = ref<string[]>([])
    const pendingFiles = ref<string[]>([])

    const isUploading = ref(false)
    const isRebuilding = ref(false)

    // [修改] 获取文件列表并分类
    const fetchFiles = async () => {
        try {
            const res = await fetch('/api/files')
            if (res.ok) {
                const data = await res.json()
                indexedFiles.value = data.indexed || []
                pendingFiles.value = data.pending || []
            }
        } catch (e) {
            console.error("获取文件列表失败", e)
        }
    }

    const uploadFiles = async (files: File[]) => {
        if (files.length === 0) return

        isUploading.value = true
        const formData = new FormData()
        files.forEach(file => {
            formData.append('files', file)
        })

        try {
            const res = await fetch('/api/upload', {
                method: 'POST',
                body: formData
            })
            if (res.ok) {
                message.success(`成功上传 ${files.length} 个文件`)
                await fetchFiles() // 刷新列表 (此时新文件应该在 pending 里)
            } else {
                const err = await res.json()
                message.error("上传失败: " + err.message)
            }
        } catch (e) {
            message.error("网络错误")
        } finally {
            isUploading.value = false
        }
    }

    const rebuildDb = async () => {
        isRebuilding.value = true
        try {
            const res = await fetch('/api/rebuild', { method: 'POST' })
            const data = await res.json()
            if (res.ok && data.status === 'success') {
                message.success(data.message)
                await fetchFiles() // [关键] 重建成功后刷新，文件应该从 pending 跑到 indexed
            } else {
                message.error("重建失败: " + data.message)
            }
        } catch (e) {
            message.error("请求超时或网络错误")
        } finally {
            isRebuilding.value = false
        }
    }

    const resetDb = async () => {
        // 二次确认通常在 UI 层做，这里直接执行逻辑
        isRebuilding.value = true
        try {
            const res = await fetch('/api/reset', { method: 'POST' })
            if (res.ok) {
                message.success("已恢复出厂设置")
                await fetchFiles() // 刷新，两个列表都应该变空
            } else {
                message.error("重置失败")
            }
        } catch (e) {
            message.error("网络错误")
        } finally {
            isRebuilding.value = false
        }
    }

    return {
        indexedFiles, // 导出
        pendingFiles, // 导出
        isUploading,
        isRebuilding,
        fetchFiles,
        uploadFiles,
        rebuildDb,
        resetDb
    }
}