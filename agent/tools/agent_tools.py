import os
import csv
import json
import functools
from datetime import datetime
from utils.logger_handler import logger
from langchain_core.tools import tool
import random
from utils.config_handler import agent_conf
from utils.path_tool import get_abs_path
from utils.location_service import get_amap_weather, get_location_text


_rag = None


def _get_rag():
    """
    懒加载 RAG 服务单例。
    避免模块导入时立即初始化向量库（Chroma + 嵌入模型），
    把启动阶段的重活推迟到第一次真正检索资料时。
    """
    global _rag
    if _rag is None:
        from rag.rag_service import RagSummarizeService
        _rag = RagSummarizeService()
    return _rag

user_ids = ["1001", "1002", "1003", "1004", "1005", "1006", "1007", "1008", "1009", "1010"]
_session_user_id = None  # 单次会话内保持一致的用户ID

external_data = {}


@tool(description="从向量知识库检索与用户问题相关的参考资料原文，返回带[参考资料N]标记的文本，最终回答必须基于这些资料生成")
def rag_summarize(query: str) -> str:
    """
    检索知识库并返回参考资料原文。
    不再额外调用模型做"总结"——检索结果直接交给主模型，由最终回答一次性生成，
    从而把一次问答的串行模型调用从 3 次降到 2 次。
    """
    docs = _get_rag().retrieve_docs(query)

    if not docs:
        return "未检索到相关参考资料"

    context_lines = []
    for idx, doc in enumerate(docs, 1):
        context_lines.append(
            f"[参考资料{idx}] {doc.page_content} | 元数据: {doc.metadata}"
        )
    return "\n".join(context_lines)

@tool(description="获取指定城市的天气，以消息字符串的形式返回")
def get_weather(city: str) -> str:
    weather = get_amap_weather(city)
    if weather:
        return weather
    # 未配置高德 Key 或查询失败时的演示兜底
    return f'城市{city}天气为晴天，气温26摄氏度，空气湿度50%，南风1级，AQI21，最近6小时降雨概率极低'

_cached_location = None

def _get_cached_location() -> str:
    """带缓存的位置获取，避免重复网络请求。"""
    global _cached_location
    if _cached_location is None:
        _cached_location = get_location_text()
    return _cached_location


@tool(description="获取用户所在城市的名称，以纯字符串形式返回")
def get_user_location() -> str:
    # 高德 IP 定位返回"省份+城市"（如：福建省厦门市），这里转换为纯城市名
    location = _get_cached_location()
    return _parse_city_from_location(location)

@tool(description="获取用户的ID，以纯字符串形式返回")
def get_user_id() -> str:
    global _session_user_id
    if _session_user_id is None:
        _session_user_id = random.choice(user_ids)
    return _session_user_id

@tool(description="获取当前月份，以纯字符串形式返回")
def get_current_month() -> str:
    return datetime.now().strftime('%Y-%m')


def _parse_city_from_location(location: str) -> str:
    """将"福建省厦门市"这类定位文本解析为纯城市名（直辖市直接返回）。"""
    for suffix in ('省', '自治区', '特别行政区'):
        idx = location.find(suffix)
        if idx != -1:
            city = location[idx + len(suffix):]
            return city or location
    return location

def generate_external_data():
    if not external_data:
        external_data_path = get_abs_path(agent_conf['external_data_path'])

        if not os.path.exists(external_data_path):
            raise FileNotFoundError(f'外部数据文件{external_data_path}不存在')

        with open(external_data_path, 'r', encoding='utf-8') as f:
            reader = csv.reader(f)
            next(reader, None)  # 跳过表头行
            for row_num, row in enumerate(reader, start=2):
                if len(row) < 6:
                    logger.warning(f'[generate_external_data] 第{row_num}行列数不足6列（实际{len(row)}列），已跳过: {row}')
                    continue

                user_id = row[0].strip().strip('"')
                feature = row[1].strip().strip('"')
                efficiency = row[2].strip().strip('"')
                consumables = row[3].strip().strip('"')
                comparison = row[4].strip().strip('"')
                time = row[5].strip().strip('"')

                if not user_id:
                    logger.warning(f'[generate_external_data] 第{row_num}行user_id为空，已跳过')
                    continue

                if user_id not in external_data:
                    external_data[user_id] = {}

                external_data[user_id][time] = {
                    'feature': feature,
                    'efficiency': efficiency,
                    'consumables': consumables,
                    'comparison': comparison,
                }

@tool(description="从外部系统中获取指定用户在指定月份的使用记录，以纯字符串形式返回， 如果未检索到返回空字符串")
def fetch_external_data(user_id: str, month: str) -> str:
    generate_external_data()

    try:
        return json.dumps(external_data[user_id][month], ensure_ascii=False)
    except KeyError:
        logger.warning(f'[fetch_external_data]未能检索到用户：{user_id}在{month}的使用记录数据')

        # 数据兜底：外部数据只有历史月份时，返回该用户最近一个可查月份的数据
        user_records = external_data.get(user_id, {})
        if user_records:
            latest_month = max(user_records.keys())
            logger.info(f'[fetch_external_data]数据兜底：{user_id}最近可查月份为{latest_month}')
            return (
                f'未找到{user_id}在{month}的使用记录，'
                f'已返回最近可查月份{latest_month}的数据：\n{user_records[latest_month]}'
            )
        return ''

@tool(description="无入参，调用后触发中间件拦截，自动为报告生成场景切换提示词。返回值无实际意义，仅用于触发中间件")
def fill_context_for_report():
    return 'fill_context_for_report已调用'
