from typing import Callable
from utils.prompt_loader import load_system_prompts, load_report_prompts
from langchain.agents import AgentState
from langchain.agents.middleware import wrap_tool_call, before_model, dynamic_prompt, ModelRequest
from langchain_core.messages import ToolMessage
from langgraph.prebuilt.tool_node import ToolCallRequest
from langgraph.runtime import Runtime
from langgraph.types import Command
from utils.logger_handler import logger


@wrap_tool_call
def monitor_tool(
        request: ToolCallRequest,
        handler: Callable[[ToolCallRequest], ToolMessage | Command]
) -> ToolMessage | Command:
    tool_name = request.tool_call["name"]
    tool_args = request.tool_call["args"]
    logger.info(f'[tool monitor]执行工具: {tool_name}')
    logger.info(f'[tool monitor]传入参数: {tool_args}')

    try:
        result = handler(request)
        logger.info(f'[tool monitor]工具{tool_name}调用成功')

        if tool_name == 'fill_context_for_report':
            request.runtime.context['report'] = True

        return result
    except Exception as e:
        # 单个工具失败不应击穿整轮对话（问题 P0-6）：
        # 转为结构化错误回传给模型，由模型决定换工具或如实告知用户。
        logger.error(f'工具{tool_name}调用失败,原因:{str(e)}', exc_info=True)
        return ToolMessage(
            content=(
                f"工具 {tool_name} 执行失败（{type(e).__name__}）。"
                "请勿编造该工具本应返回的数据；可改用其他工具，"
                "或如实告知用户该项信息暂时无法获取。"
            ),
            tool_call_id=request.tool_call.get("id", ""),
            name=tool_name,
            status="error",
        )


@before_model
def log_before_model(
        state: AgentState,
        runtime: Runtime,
):
    msg_count = len(state["messages"])
    logger.info(f'[log_before_model]即将调用模型，带有{msg_count}条消息')
    last_msg = state["messages"][-1]
    logger.debug(f'[log_before_model]{type(last_msg).__name__} | {last_msg.content.strip()}')

    return None


@dynamic_prompt
def report_prompt_switch(request: ModelRequest):
    is_report = request.runtime.context.get('report', False)
    if is_report:
        logger.info('[report_prompt_switch] 检测到报告生成场景，切换为报告提示词')
        return load_report_prompts()

    return load_system_prompts()
