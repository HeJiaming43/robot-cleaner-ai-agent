"""
定位服务：通过高德地图 Web 服务 API 实现 IP 定位与实时天气查询。

能力：
1. 自动探测公网 IPv4（多个服务容灾）
2. 调用高德「IP 定位」API，返回"省份+城市"（如：福建省厦门市）
3. 调用高德「天气查询」API，返回指定城市的实时天气文本

密钥优先级：环境变量 AMAP_API_KEY > config/agent.yml 中的 amap_api_key。
未配置密钥或请求失败时，自动回退到配置的兜底位置（默认：福建省厦门市）。
"""
import os
import re
from concurrent.futures import ThreadPoolExecutor

import requests

from utils.config_handler import agent_conf
from utils.logger_handler import logger


IP_PROBE_SERVICES = [
    "https://api.ipify.org",
    "https://ifconfig.me/ip",
    "https://myip.ipip.net",
    "https://cip.cc",
]

AMAP_IP_URL = "https://restapi.amap.com/v3/ip"
AMAP_WEATHER_URL = "https://restapi.amap.com/v3/weather/weatherInfo"

_IP_PATTERN = re.compile(r"\d{1,3}\.\d{1,3}\.\d{1,3}\.\d{1,3}")


def _is_private_ip(ip: str) -> bool:
    """判断是否为内网/私有 IP 地址（10.x, 172.16-31.x, 192.168.x）。"""
    try:
        parts = [int(p) for p in ip.split(".")]
        if len(parts) != 4:
            return True
        # 10.0.0.0/8
        if parts[0] == 10:
            return True
        # 172.16.0.0/12
        if parts[0] == 172 and 16 <= parts[1] <= 31:
            return True
        # 192.168.0.0/16
        if parts[0] == 192 and parts[1] == 168:
            return True
        # 127.0.0.0/8 (loopback)
        if parts[0] == 127:
            return True
        return False
    except (ValueError, IndexError):
        return True


def get_amap_key() -> str:
    """获取高德 Web 服务 Key，优先读环境变量，其次读配置文件。"""
    key = os.environ.get("AMAP_API_KEY", "").strip()
    if not key:
        key = str(agent_conf.get("amap_api_key", "") or "").strip()
    return key


def get_fallback_location() -> str:
    """未配置 Key / 定位失败时的兜底位置。"""
    fallback = agent_conf.get("fallback_location", "福建省厦门市")
    return str(fallback or "福建省厦门市")


def get_public_ip(timeout: int = 3) -> str:
    """
    并发探测公网 IPv4 地址，多服务容灾。
    全部失败时返回空字符串（由调用方决定是否降级）。
    """
    def _probe(url: str) -> str:
        try:
            resp = requests.get(url, timeout=timeout)
            resp.raise_for_status()
            match = _IP_PATTERN.search(resp.text)
            if match:
                ip = match.group(0)
                if _is_private_ip(ip):
                    logger.debug(f"[定位服务] 跳过内网 IP: {url} -> {ip}")
                    return ""
                logger.info(f"[定位服务] 公网 IP 探测成功: {url} -> {ip}")
                return ip
        except Exception as e:
            logger.debug(f"[定位服务] IP 探测失败 {url}: {e}")
        return ""

    with ThreadPoolExecutor(max_workers=len(IP_PROBE_SERVICES)) as pool:
        for ip in pool.map(_probe, IP_PROBE_SERVICES):
            if ip:
                return ip
    return ""


def get_amap_location(ip: str | None = None, timeout: int = 3) -> str:
    """
    调用高德「IP 定位」API，返回"省份+城市"文本。
    未配置 Key 或请求失败时返回兜底位置。
    """
    key = get_amap_key()
    if not key:
        logger.info("[定位服务] 未配置 AMAP_API_KEY，使用兜底位置")
        return get_fallback_location()

    params = {"key": key}
    if ip:
        params["ip"] = ip

    try:
        resp = requests.get(AMAP_IP_URL, params=params, timeout=timeout)
        data = resp.json()

        if data.get("status") == "1":
            province = data.get("province", "")
            city = data.get("city", "")

            # 直辖市场景下 city 可能返回空列表 []
            if isinstance(city, list):
                city = city[0] if city else ""

            if province or city:
                location = f"{province}{city}"
                logger.info(f"[定位服务] 高德定位成功: {location}")
                return location

        logger.warning(f"[定位服务] 高德返回异常: infocode={data.get('infocode')}, info={data.get('info')}")
    except Exception as e:
        logger.error(f"[定位服务] 获取位置失败: {e}")

    return get_fallback_location()


def get_amap_weather(city: str, timeout: int = 3) -> str:
    """
    调用高德「天气查询」API，返回实时天气文本。
    未配置 Key / 查询失败时返回空字符串（由调用方自行兜底）。
    """
    key = get_amap_key()
    if not key:
        return ""

    try:
        resp = requests.get(
            AMAP_WEATHER_URL,
            params={"key": key, "city": city, "extensions": "base"},
            timeout=timeout,
        )
        data = resp.json()

        if data.get("status") == "1" and data.get("lives"):
            live = data["lives"][0]
            weather = live.get("weather", "")
            temperature = live.get("temperature", "")
            humidity = live.get("humidity", "")
            wind_power = live.get("windpower", "")
            wind_dir = live.get("winddirection", "")
            report = (
                f"{live.get('city', city)}天气{weather}，"
                f"气温{temperature}摄氏度，空气湿度{humidity}%，"
                f"{wind_dir}风{wind_power}级"
            )
            logger.info(f"[定位服务] 高德天气查询成功: {report}")
            return report

        logger.warning(f"[定位服务] 天气查询失败: info={data.get('info')}")
    except Exception as e:
        logger.error(f"[定位服务] 获取天气失败: {e}")

    return ""


def get_location_text() -> str:
    """对外统一入口：探测公网 IP 并返回定位文本（未配置 Key 时直接返回兜底位置）。"""
    if not get_amap_key():
        return get_fallback_location()

    ip = get_public_ip()
    return get_amap_location(ip)


if __name__ == "__main__":
    print(f"定位结果: {get_location_text()}")
    print(f"天气结果: {get_amap_weather('厦门市') or '未配置 Key / 查询失败'}")
