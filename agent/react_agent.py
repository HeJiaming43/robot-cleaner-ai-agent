from langchain.agents import create_agent
from model.factory import chat_model
from utils.prompt_loader import load_system_prompts
from agent.tools.agent_tools import (rag_summarize,get_weather,get_user_id,
                                     get_current_month,fetch_external_data,fill_context_for_report)
from agent.tools.middleware import monitor_tool,log_before_model,report_prompt_switch



class ReactAgent:
    def __init__(self):
        self.agent = create_agent(
            model = chat_model,
            system_prompt = load_system_prompts(),
            tools = [rag_summarize,get_weather,get_user_id,
                     get_current_month,fetch_external_data,fill_context_for_report],
            middleware = [monitor_tool,report_prompt_switch,log_before_model]
        )


    def execute_stream(self, query: str):
        """
        按增量流式输出模型的回答（stream_mode='messages'）。
        每个 chunk 都是新生成的文本增量，前端可直接逐字渲染；
        工具调用、工具返回等非回答消息会被过滤掉。
        """
        input_dict = {
            'messages':[
                {'role': 'user', 'content': query},
            ]
        }

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

    for chunk in agent.execute_stream('扫地机器人日常如何保养'):
        print(chunk, end='', flush=True)
