from langchain.agents import create_agent
from model.factory import chat_model
from utils.prompt_loader import load_system_prompts
from utils.logger_handler import logger
from rag.retrieval_gate import get_retrieval_gate, format_grounding_block
from agent.identity import session_scope
from agent.tools.agent_tools import (rag_summarize,get_weather,get_user_id,
                                     get_current_month,fetch_external_data,fill_context_for_report)
from agent.tools.middleware import monitor_tool,log_before_model,report_prompt_switch


# 检索前置之后的回答约束：把「以资料为依据」写成每一轮都会出现的硬性要求
ANSWER_RULES = (
    "【本轮回答要求】\n"
    "1. 系统已在回答前完成知识库检索，上述资料可直接使用；无需再调用 rag_summarize，除非需要更换检索词补充检索。\n"
    "2. 资料足以回答时，必须依据资料组织内容，并在相应结论后标注 [参考资料N]。\n"
    "3. 资料不足时，明确说明「知识库中暂未查到该信息」，不得编造参数、型号、故障码或处理步骤。\n"
    "4. 工具返回不可用（如实时天气数据不可用）时如实告知，不得用推测数值替代。"
)


def _build_turn_input(query: str, gate_result) -> str:
    """把检索结果 + 回答约束 + 用户问题合成为本轮输入。"""
    return "\n".join([
        format_grounding_block(gate_result),
        "",
        ANSWER_RULES,
        "",
        f"【用户问题】{query}",
    ])


# 多轮上下文限制：避免长会话把上下文撑爆，也控制单次请求的 token 成本
MAX_HISTORY_MESSAGES = 12       # 最多回带的轮次消息数（user / assistant 各计一条）
MAX_HISTORY_CHARS = 4000        # 历史内容总字符预算


def _build_history(history: list | None) -> list[dict]:
    """
    把前端会话历史裁剪为模型可接受的消息列表。

    裁剪策略：先按条数截断，再按字符预算从最近往前累计，
    保证「最近的上下文优先保留」，避免长回答把早期关键信息挤出窗口。
    非 user / assistant 角色、空内容、以及带「抱歉，系统出错」之类失败回答
    不进入上下文，防止错误自我强化。
    """
    if not history:
        return []

    cleaned: list[dict] = []
    for msg in history:
        if not isinstance(msg, dict):
            continue
        role = msg.get("role")
        content = (msg.get("content") or "").strip()
        if role not in ("user", "assistant") or not content:
            continue
        cleaned.append({"role": role, "content": content})

    if not cleaned:
        return []

    cleaned = cleaned[-MAX_HISTORY_MESSAGES:]

    kept: list[dict] = []
    used = 0
    for msg in reversed(cleaned):
        used += len(msg["content"])
        if used > MAX_HISTORY_CHARS and kept:
            break
        kept.append(msg)

    return list(reversed(kept))


class ReactAgent:
    def __init__(self):
        self.agent = create_agent(
            model = chat_model,
            system_prompt = load_system_prompts(),
            tools = [rag_summarize,get_weather,get_user_id,
                     get_current_month,fetch_external_data,fill_context_for_report],
            middleware = [monitor_tool,report_prompt_switch,log_before_model]
        )


    def execute_stream(self, query: str, history: list | None = None, session_id: str | None = None):
        """
        按增量流式输出模型的回答（stream_mode='messages'）。
        每个 chunk 都是新生成的文本增量，前端可直接逐字渲染；
        工具调用、工具返回等非回答消息会被过滤掉。

        history：此前轮次的消息（[{"role","content"}, ...]），用于多轮指代解析。
        调用方须传入「不含当前问题」的历史，当前问题通过 query 单独提交。
        session_id：会话标识。工具据此解析当前用户身份，
                    模型无法通过参数访问其他用户的数据。
        """
        messages = _build_history(history)

        # 检索前置：无论模型如何判断，本轮都先检索并注入资料（问题 P0-2）
        gate_result = get_retrieval_gate().retrieve(query)
        messages.append({'role': 'user', 'content': _build_turn_input(query, gate_result)})

        if len(messages) > 1:
            logger.info(f"[多轮] 本轮携带 {len(messages) - 1} 条历史消息进入模型")

        input_dict = {'messages': messages}

        # 绑定会话身份：工具内通过 ContextVar 读取，退出即恢复，不跨请求泄漏
        with session_scope(session_id or "anonymous"):
            # context 是上下文 runtime 中的信息，做提示词切换的标记
            for message_chunk, _metadata in self.agent.stream(
                input_dict,
                stream_mode = 'messages',
                context = {'report': False},
            ):
                # 只取模型生成的文本增量，跳过工具调用与工具返回消息
                if getattr(message_chunk, 'type', '') != 'AIMessageChunk':
                    continue

                content = message_chunk.content
                if not content:
                    continue

                if not isinstance(content, str):
                    # 兼容 content blocks 场景，仅拼接文本块
                    parts = []
                    for block in content:
                        if isinstance(block, dict) and block.get('type') == 'text':
                            parts.append(block.get('text', ''))
                        elif isinstance(block, str):
                            parts.append(block)
                    content = ''.join(parts)

                if content:
                    yield content

if __name__ == '__main__':
    agent = ReactAgent()

    for chunk in agent.execute_stream('扫地机器人日常如何保养', session_id='cli-demo'):
        print(chunk, end='', flush=True)
