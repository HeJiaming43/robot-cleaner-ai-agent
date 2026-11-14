"""
智净灵境 — 扫地/拖地机器人智能客服系统
Streamlit Web 服务，双页面：智能客服（用户端）+ 知识库管理（商家端）

关于定位：原「高德 IP 自动定位」已移除。该能力通过服务端公网 IP 反查，
获取的是服务器所在城市而非访客所在城市，属伪个性化；且失败时回退固定常量，
会把虚构位置讲给用户。城市信息现由用户在对话中自然语言提供。
"""
import os
import threading
import uuid
from datetime import datetime

import streamlit as st

from utils.config_handler import chroma_conf
from utils.logger_handler import logger
from utils.path_tool import get_abs_path


_preload_done = threading.Event()
_preloaded = {"vector_service": None, "agent": None}


# ============================================================
# 页面配置
# ============================================================
st.set_page_config(
    page_title="智净灵境 - 扫地/拖地机器人智能客服系统",
    page_icon="🤖",
    layout="wide",
    initial_sidebar_state="expanded",
)


# ============================================================
# CSS 样式
# ============================================================
def inject_css():
    st.markdown("""
    <style>
    .main-header {
        text-align: center;
        padding: 1.8rem 0;
        background: linear-gradient(135deg, #667eea 0%, #764ba2 100%);
        border-radius: 14px;
        margin-bottom: 0.8rem;
    }
    .main-header h1 {
        color: #fff;
        font-size: 2.2rem;
        margin: 0;
        letter-spacing: 2px;
    }
    .main-header .subtitle {
        color: rgba(255,255,255,0.88);
        font-size: 1rem;
        margin: 0.4rem 0 0 0;
    }
    .location-badge {
        text-align: center;
        padding: 0.7rem 1rem;
        background: rgba(102, 126, 234, 0.12);
        border: 1px solid rgba(102, 126, 234, 0.35);
        border-left: 4px solid #667eea;
        border-radius: 10px;
        margin: 0.8rem 0 1.2rem 0;
        font-size: 1rem;
        color: #e8e8f0;
    }
    .footer {
        text-align: center;
        padding: 1.2rem;
        color: #aaa;
        font-size: 0.82rem;
        margin-top: 2rem;
    }
    .kb-file-list li {
        margin: 0.3rem 0;
    }
    .sidebar-section-title {
        font-size: 1rem;
        font-weight: 600;
        margin-top: 0.6rem;
    }
    </style>
    """, unsafe_allow_html=True)


# ============================================================
# 全局缓存资源（重量级对象跨 rerun 复用）
# ============================================================
@st.cache_resource(show_spinner=False)
def get_vector_service():
    # 等待后台预热完成，避免与预热线程竞争创建重复实例
    _preload_done.wait(timeout=95)
    if _preloaded["vector_service"] is not None:
        return _preloaded["vector_service"]
    from rag.vector_store import get_shared_vector_store
    return get_shared_vector_store()


@st.cache_resource(show_spinner=False)
def get_agent():
    # 等待后台预热完成，避免与预热线程竞争创建重复实例
    _preload_done.wait(timeout=95)
    if _preloaded["agent"] is not None:
        return _preloaded["agent"]
    from agent.react_agent import ReactAgent
    return ReactAgent()


def _load_vector_service():
    """后台线程专用：纯导入 + 初始化，不触碰 Streamlit API。"""
    if _preloaded["vector_service"] is not None:
        return
    logger.info("[预热] 开始加载向量库服务")
    from rag.vector_store import get_shared_vector_store
    _preloaded["vector_service"] = get_shared_vector_store()
    logger.info("[预热] 向量库服务加载完成")


def _load_agent():
    """后台线程专用：纯导入 + 初始化，不触碰 Streamlit API。"""
    if _preloaded["agent"] is not None:
        return
    logger.info("[预热] 开始加载智能客服引擎")
    from agent.react_agent import ReactAgent
    _preloaded["agent"] = ReactAgent()
    logger.info("[预热] 智能客服引擎加载完成")


def _preload_heavy_services():
    """
    后台预热重量级服务（Agent 引擎 + 向量库），
    让首次页面渲染不被 15~20 秒的依赖导入阻塞。
    预热结果写入模块级 _preloaded，缓存函数优先取用。
    """
    threads = [
        threading.Thread(target=_load_vector_service, daemon=True),
        threading.Thread(target=_load_agent, daemon=True),
    ]
    try:
        # 两个重依赖并行导入；join 设置上限，避免异常环境导致永久阻塞
        for thread in threads:
            thread.start()
        for thread in threads:
            thread.join(timeout=90)
    except Exception as e:
        logger.error(f"[预热] 后台初始化失败: {e}", exc_info=True)
    finally:
        logger.info("[预热] 后台初始化流程结束")
        _preload_done.set()


# ============================================================
# 会话状态初始化
# ============================================================
def init_session_state():
    defaults = {
        "messages": [],            # [{"role", "content", "timestamp", "question"}]
        "pending_question": "",
        "kg_authed": False,        # 知识库管理鉴权状态
    }
    for key, value in defaults.items():
        if key not in st.session_state:
            st.session_state[key] = value

    # 会话身份：仅生成一次并贯穿整个会话，不再使用进程级随机用户ID
    if not st.session_state.get("session_id"):
        st.session_state.session_id = uuid.uuid4().hex[:12]


# ============================================================
# Markdown 导出
# ============================================================
def generate_chat_markdown(messages: list) -> str:
    """将整个对话历史导出为 Markdown 文件。"""
    lines = [
        "# 智净灵境 · 对话记录",
        "",
        f"**导出时间：** {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}",
        "",
        "---",
        "",
    ]
    for msg in messages:
        role_icon = "🧑" if msg["role"] == "user" else "🤖"
        role_name = "用户" if msg["role"] == "user" else "智净灵境 · 智能客服"
        ts = msg.get("timestamp", "")
        question = msg.get("question", "")
        lines.append(f"### {role_icon} {role_name}")
        if ts:
            lines.append(f"*{ts}*")
        lines.append("")
        if msg["role"] == "assistant" and question:
            lines.append(f"> **关联问题：** {question}")
            lines.append("")
        lines.append(msg["content"])
        lines.append("")
        lines.append("---")
        lines.append("")
    return "\n".join(lines)


def generate_answer_markdown(question: str, answer: str, ts: str) -> str:
    """将单条问答导出为 Markdown 文件。"""
    return "\n".join([
        "# 智净灵境 · 问答记录",
        "",
        f"**时间：** {ts}",
        "",
        "---",
        "",
        "## 🧑 用户问题",
        "",
        question,
        "",
        "## 🤖 智能回复",
        "",
        answer,
        "",
        "---",
        "",
        "*由智净灵境扫地/拖地机器人智能客服系统生成*",
    ])


# ============================================================
# 页面一：智能客服（用户端）
# ============================================================
def render_session_sidebar():
    st.markdown('<div class="sidebar-section-title">⚙️ 会话管理</div>', unsafe_allow_html=True)

    msg_count = len(st.session_state.messages)
    st.caption(f"当前会话共 {msg_count} 条消息")

    col_clear, col_export = st.columns(2)
    with col_clear:
        if st.button("🗑️ 清空对话", use_container_width=True):
            st.session_state.messages = []
            st.rerun()
    with col_export:
        if st.session_state.messages:
            md_data = generate_chat_markdown(st.session_state.messages)
            st.download_button(
                label="📥 导出全部",
                data=md_data,
                file_name=f"chat_{datetime.now().strftime('%Y%m%d_%H%M%S')}.md",
                mime="text/markdown",
                use_container_width=True,
            )

    st.divider()
    st.markdown('<div class="sidebar-section-title">⚡ 快捷提问</div>', unsafe_allow_html=True)
    quick_questions = [
        "扫地机器人日常如何保养？",
        "机器人报E13错误代码怎么办？",
        "如何选购适合宠物家庭的扫地机？",
        "扫拖一体机和单独扫地机哪个好？",
        "机器人吸力变弱如何排查？",
    ]
    for q in quick_questions:
        if st.button(q, key=f"qq_{q[:12]}", use_container_width=True):
            st.session_state.pending_question = q
            st.rerun()


def page_customer_service():
    # ---- 顶部横幅 ----
    st.markdown("""
    <div class="main-header">
        <h1>🤖 智净灵境</h1>
        <p class="subtitle">扫地 / 拖地机器人 · AI 智能客服系统</p>
    </div>
    """, unsafe_allow_html=True)

    # ---- 欢迎语（城市信息由用户在对话中提供，不再做 IP 自动定位） ----
    st.markdown(f"""
    <div class="location-badge">
        🌐 欢迎您咨询智净灵境扫地 / 拖地机器人智能客服 · 会话 {st.session_state.session_id}
    </div>
    """, unsafe_allow_html=True)

    # ---- 侧边栏：会话管理 + 快捷提问 ----
    with st.sidebar:
        render_session_sidebar()

    # ---- 历史对话 ----
    for index, msg in enumerate(st.session_state.messages):
        with st.chat_message(msg["role"], avatar="🧑" if msg["role"] == "user" else "🤖"):
            st.markdown(msg["content"])
            if msg["role"] == "assistant":
                ts = msg.get("timestamp", "")
                question = msg.get("question", "")
                md_data = generate_answer_markdown(question, msg["content"], ts)
                st.download_button(
                    label="📥 下载此回答",
                    data=md_data,
                    file_name=f"answer_{index}_{datetime.now().strftime('%H%M%S')}.md",
                    mime="text/markdown",
                    key=f"dl_{index}",
                )

    # ---- 输入区 ----
    question = st.chat_input("请输入您关于扫地/拖地机器人的问题…")
    if not question and st.session_state.get("pending_question"):
        question = st.session_state.pending_question
        st.session_state.pending_question = ""

    if question:
        ts = datetime.now().strftime("%Y-%m-%d %H:%M:%S")
        st.session_state.messages.append({
            "role": "user",
            "content": question,
            "timestamp": ts,
        })

        with st.chat_message("user", avatar="🧑"):
            st.markdown(question)

        with st.chat_message("assistant", avatar="🤖"):
            # 思考/检索阶段提示：在首个文字出现前一直显示
            thinking = st.empty()
            thinking.markdown("🤔 **客服思考中…**")
            stream_started = False

            def collect(generator):
                nonlocal stream_started
                for delta in generator:
                    # 首个文字出现，移除"思考中"提示，切换为流式渲染
                    if not stream_started:
                        thinking.empty()
                        stream_started = True
                    yield delta

            try:
                # 已把当前问题追加进 messages，故上下文取其之前的部分
                history = st.session_state.messages[:-1]
                answer = st.write_stream(
                    collect(get_agent().execute_stream(
                        question,
                        history=history,
                        session_id=st.session_state.session_id,
                    ))
                ).strip()
                if not answer:
                    thinking.empty()
                    answer = "抱歉，我暂时无法回答这个问题。"
                    st.markdown(answer)
            except Exception as e:
                thinking.empty()
                answer = f"抱歉，系统出错了：{e}"
                st.error(answer)

            ans_ts = datetime.now().strftime("%Y-%m-%d %H:%M:%S")
            st.session_state.messages.append({
                "role": "assistant",
                "content": answer,
                "timestamp": ans_ts,
                "question": question,
            })

        # 刷新页面，让侧边栏消息计数与历史循环同步
        st.rerun()


# ============================================================
# 页面二：知识库管理（商家端）
# ============================================================
@st.fragment(run_every=3)
def render_kb_status():
    """知识库状态卡片：引擎未就绪时先显示预热状态，就绪后自动刷新片段总数。"""
    if not _preload_done.is_set():
        st.metric("向量片段总数", "预热中…")
        st.caption("后台引擎加载中，预计几秒后自动刷新…")
        return

    try:
        vector_service = get_vector_service()
        count = vector_service.vector_store._collection.count()
        st.metric("向量片段总数", count)
    except Exception as e:
        st.metric("向量片段总数", "加载中…")
        st.caption(f"*ChromaDB 未就绪: {e}*")


def render_kb_sidebar():
    st.markdown('<div class="sidebar-section-title">📋 支持格式</div>', unsafe_allow_html=True)
    st.info(
        "**PDF** (.pdf) · **文本文件** (.txt)\n\n"
        "上传的文件将自动分割、去重并加入向量知识库。"
    )

    st.divider()
    st.markdown('<div class="sidebar-section-title">📊 知识库状态</div>', unsafe_allow_html=True)
    render_kb_status()

    st.divider()
    st.markdown('<div class="sidebar-section-title">🔄 重建知识库</div>', unsafe_allow_html=True)
    if st.button("从 data 目录全量重建", use_container_width=True):
        with st.spinner("重建中，请稍候…"):
            vector_service = get_vector_service()
            count = vector_service.rebuild_knowledge_base()
        st.success(f"重建完成！共入库 {count} 个文档片段")
        st.rerun()


def page_knowledge_management():
    st.markdown("""
    <div class="main-header">
        <h1>📚 知识库管理</h1>
        <p class="subtitle">上传文档 · 扩充知识 · 提升服务质量</p>
    </div>
    """, unsafe_allow_html=True)

    with st.sidebar:
        render_kb_sidebar()

    col_left, col_right = st.columns([3, 2], gap="large")

    # ---- 左栏：上传新文档 ----
    with col_left:
        st.markdown("### 📤 上传新文档")
        uploaded_files = st.file_uploader(
            "选择 PDF 或 TXT 文件（可多选）",
            type=["pdf", "txt"],
            accept_multiple_files=True,
        )
        st.caption("单个文件不超过 2 MB")

        valid_files = []
        if uploaded_files:
            for file in uploaded_files:
                if file.size > 2 * 1024 * 1024:
                    st.warning(f"⛔ `{file.name}` 超过 2 MB，已跳过")
                else:
                    valid_files.append(file)

            if st.button("🚀 导入知识库", type="primary", use_container_width=True):
                if not valid_files:
                    st.warning("没有符合大小限制的文件可导入。")
                else:
                    data_dir = get_abs_path(chroma_conf["data_path"])
                    os.makedirs(data_dir, exist_ok=True)

                    progress = st.progress(0)
                    for idx, file in enumerate(valid_files):
                        # 清洗文件名，防止路径穿越（如 ../etc/passwd）
                        safe_name = os.path.basename(file.name)
                        with open(os.path.join(data_dir, safe_name), "wb") as f:
                            f.write(file.getbuffer())
                        progress.progress((idx + 1) / len(valid_files))

                    with st.spinner("正在分割、去重并写入向量知识库…"):
                        vector_service = get_vector_service()
                        processed = vector_service.load_document()

                    st.success(f"✅ 导入完成，本次新增入库 {processed} 个文件")
                    st.rerun()

    # ---- 右栏：已入库文件 ----
    with col_right:
        st.markdown("### 📁 已入库文件")
        data_dir = get_abs_path(chroma_conf["data_path"])
        if os.path.isdir(data_dir):
            files = sorted(
                [f for f in os.listdir(data_dir) if f.endswith((".pdf", ".txt", ".text"))],
                key=lambda f: os.path.getmtime(os.path.join(data_dir, f)),
                reverse=True,
            )
            if files:
                st.markdown('<ul class="kb-file-list">', unsafe_allow_html=True)
                for fname in files:
                    fpath = os.path.join(data_dir, fname)
                    size_kb = os.path.getsize(fpath) / 1024
                    mtime = datetime.fromtimestamp(os.path.getmtime(fpath))
                    st.markdown(
                        f"- **{fname}** "
                        f"*({size_kb:.1f} KB, {mtime.strftime('%m-%d %H:%M')})*"
                    )
                st.markdown("</ul>", unsafe_allow_html=True)
            else:
                st.info("暂无文件，请上传")
        else:
            st.warning("data 目录不存在")


# ============================================================
# 主入口
# ============================================================
def main():
    inject_css()
    init_session_state()

    # 后台预热重量级服务（Agent 引擎 + 向量库），
    # 让首屏渲染不被 15~20 秒的依赖导入阻塞
    if not st.session_state.get("_preload_started"):
        st.session_state._preload_started = True
        threading.Thread(target=_preload_heavy_services, daemon=True).start()

    # 首次加载不再做位置探测：原逻辑需阻塞等待最长 10 秒并额外触发一次全页 rerun
    # ---- 侧边导航 ----
    with st.sidebar:
        st.markdown("## 🧭 导航")
        page = st.radio(
            "切换页面",
            ["💬 智能客服", "📚 知识库管理"],
            label_visibility="collapsed",
        )
        st.divider()
        st.caption("智净灵境 v1.0")
        st.caption("Powered by LangChain + RAG + MCP")
        st.divider()

    if "智能客服" in page:
        page_customer_service()
    else:
        page_knowledge_management()

    st.markdown("""
    <div class="footer">
        智净灵境 &copy; 2026 · 扫地/拖地机器人智能客服系统
    </div>
    """, unsafe_allow_html=True)


if __name__ == "__main__":
    main()
