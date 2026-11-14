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
        from rag.vector_store import get_shared_vector_store
        # 复用进程内共享实例：与检索前置、UI 状态读取共用同一 Chroma 客户端
        _rag = RagSummarizeService(vector_store=get_shared_vector_store())
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

_GEO_URL = "https://geocoding-api.open-meteo.com/v1/search"
_FORECAST_URL = "https://api.open-meteo.com/v1/forecast"


def _weather_unavailable(city: str) -> str:
    """天气数据不可用时的统一回应：明确告知，禁止模型编造。"""
    return (
        f"实时天气数据不可用，无法获取{city}当前天气。"
        "请勿编造任何天气数值；应改为基于知识库中潮湿、雨季、低温等环境相关的保养建议作答，"
        "并向用户明确说明当前无法查询实时天气。"
    )


def _fetch_open_meteo(city: str) -> str:
    """
    调用免密钥的 Open-Meteo（地理编码 + 实时天气）。
    仅在 weather_provider=open-meteo 时启用；import requests 延迟到此处，
    使默认不联网的部署完全不依赖 requests。
    返回空字符串表示未取到数据，由调用方降级。
    """
    import requests

    geo_resp = requests.get(
        _GEO_URL, params={"name": city, "count": 1, "language": "zh"}, timeout=3
    )
    results = (geo_resp.json() or {}).get("results") or []
    if not results:
        logger.info(f"[get_weather] 城市地理编码未命中: {city}")
        return ""

    latitude = results[0]["latitude"]
    longitude = results[0]["longitude"]

    resp = requests.get(
        _FORECAST_URL,
        params={
            "latitude": latitude,
            "longitude": longitude,
            "current": "temperature_2m,relative_humidity_2m,weather_code",
            "timezone": "auto",
        },
        timeout=3,
    )
    current = (resp.json() or {}).get("current") or {}
    if not current:
        return ""

    return (
        f"{city}实时天气：天气代码{current.get('weather_code', '未知')}，"
        f"气温{current.get('temperature_2m', '未知')}摄氏度，"
        f"相对湿度{current.get('relative_humidity_2m', '未知')}%，"
        f"观测时间{current.get('time', '未知')}（数据来源：Open-Meteo）"
    )


@tool(
    description=(
        "获取指定城市的实时天气。city 为用户所在城市名称。"
        "返回内容可能为『实时天气数据不可用』，此时你必须如实告知用户无法查询，"
        "并改为基于知识库给出环境适配建议，严禁编造天气数值"
    )
)
def get_weather(city: str) -> str:
    """
    天气工具：数据源由 config/agent.yml 的 weather_provider 决定。
      - none（默认）：不发起任何外网请求，返回明确的不可用提示
      - open-meteo：调用免密钥的 Open-Meteo，失败时同样明确降级
    任何分支都不会返回伪造的天气数据。
    """
    provider = str(agent_conf.get("weather_provider", "none") or "none").strip().lower()
    city = (city or "").strip()

    if not city:
        return "未提供城市名称，无法查询天气。请先向用户确认其所在城市。"

    if provider == "none":
        return _weather_unavailable(city)

    try:
        return _fetch_open_meteo(city) or _weather_unavailable(city)
    except Exception as e:
        logger.warning(f"[get_weather] 天气数据源异常（provider={provider}）: {e}")
        return _weather_unavailable(city)

@tool(description="获取用户的ID，以纯字符串形式返回")
def get_user_id() -> str:
    global _session_user_id
    if _session_user_id is None:
        _session_user_id = random.choice(user_ids)
    return _session_user_id

@tool(description="获取当前月份，以纯字符串形式返回")
def get_current_month() -> str:
    return datetime.now().strftime('%Y-%m')


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
