import sqlite3
import json
import os
import uuid
from datetime import datetime

# --- 配置 ---
# 数据库文件存放在 data 目录下
CURRENT_DIR = os.path.dirname(os.path.abspath(__file__))
DB_PATH = os.path.join(CURRENT_DIR, "../data/chat_history.db")


def get_db_connection():
    """获取数据库连接"""
    conn = sqlite3.connect(DB_PATH)
    # 这行代码让查询结果像字典一样可以通过列名访问 (row['id'])
    conn.row_factory = sqlite3.Row
    return conn


def init_db():
    """初始化数据库表结构"""
    conn = get_db_connection()
    c = conn.cursor()

    # 1. 会话表 (Sessions)
    c.execute('''
        CREATE TABLE IF NOT EXISTS sessions (
            id TEXT PRIMARY KEY,
            title TEXT NOT NULL,
            created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP
        )
    ''')

    # 2. 消息表 (Messages)
    # thought: 存储 DeepSeek 的思考过程
    # sources: 存储参考来源 (JSON 字符串)
    c.execute('''
        CREATE TABLE IF NOT EXISTS messages (
            id INTEGER PRIMARY KEY AUTOINCREMENT,
            session_id TEXT NOT NULL,
            role TEXT NOT NULL,
            content TEXT NOT NULL,
            thought TEXT,
            sources TEXT,
            created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP,
            FOREIGN KEY (session_id) REFERENCES sessions (id) ON DELETE CASCADE
        )
    ''')

    conn.commit()
    conn.close()
    print(f"✅ 数据库初始化完成: {DB_PATH}")


# ===========================
# 会话管理 (Sessions)
# ===========================

def create_session(title="新对话"):
    """创建一个新会话，返回 session_id"""
    conn = get_db_connection()
    session_id = str(uuid.uuid4())
    conn.execute('INSERT INTO sessions (id, title) VALUES (?, ?)', (session_id, title))
    conn.commit()
    conn.close()
    return session_id


def get_all_sessions():
    """获取所有会话列表 (按时间倒序)"""
    conn = get_db_connection()
    sessions = conn.execute('SELECT * FROM sessions ORDER BY created_at DESC').fetchall()
    conn.close()
    # 转为字典列表返回
    return [dict(s) for s in sessions]


def delete_session(session_id):
    """删除指定会话及其所有消息"""
    conn = get_db_connection()
    # 因为有外键约束(ON DELETE CASCADE)可能不生效(取决于SQLite版本配置)，手动删两张表最稳
    conn.execute('DELETE FROM messages WHERE session_id = ?', (session_id,))
    conn.execute('DELETE FROM sessions WHERE id = ?', (session_id,))
    conn.commit()
    conn.close()


def update_session_title(session_id, new_title):
    """更新会话标题"""
    conn = get_db_connection()
    conn.execute('UPDATE sessions SET title = ? WHERE id = ?', (new_title, session_id))
    conn.commit()
    conn.close()


# ===========================
# 消息管理 (Messages)
# ===========================

def add_message(session_id, role, content, thought=None, sources=None):
    """添加一条消息"""
    conn = get_db_connection()

    # 如果 sources 是对象/列表，转为 JSON 字符串存储
    if sources and not isinstance(sources, str):
        sources = json.dumps(sources, ensure_ascii=False)

    conn.execute('''
        INSERT INTO messages (session_id, role, content, thought, sources)
        VALUES (?, ?, ?, ?, ?)
    ''', (session_id, role, content, thought, sources))
    conn.commit()
    conn.close()


def get_session_messages(session_id):
    """获取指定会话的所有消息"""
    conn = get_db_connection()
    messages = conn.execute('''
        SELECT * FROM messages 
        WHERE session_id = ? 
        ORDER BY id ASC
    ''', (session_id,)).fetchall()
    conn.close()

    result = []
    for msg in messages:
        m = dict(msg)
        # 将 sources 从 JSON 字符串转回对象
        if m['sources']:
            try:
                m['sources'] = json.loads(m['sources'])
            except:
                m['sources'] = []
        result.append(m)
    return result


# 模块被导入时自动检查初始化
if not os.path.exists(DB_PATH):
    # 确保父目录存在
    os.makedirs(os.path.dirname(DB_PATH), exist_ok=True)
    init_db()