export interface SourceDoc {
    source: string;
    page: number;
    content: string;
}

export interface ChatMessage {
    role: 'user' | 'assistant';
    content: string;        // 正文内容
    thought?: string;       // 思考过程 (若有)
    sources?: SourceDoc[];  // 参考来源
    isStreaming?: boolean;  // 是否正在接收流
    intent?: string;        // 意图 (SEARCH/CHAT)
}