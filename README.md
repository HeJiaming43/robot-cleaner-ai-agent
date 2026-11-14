# 智净灵境 — 扫地/拖地机器人智能客服系统

面向扫地/拖地机器人品牌的 AI 智能客服系统，基于 **LangChain ReAct 智能体 + RAG 检索增强生成** 构建，覆盖「文档解析与清洗 → 知识库构建 → 向量检索 → 大模型生成 → 引用溯源」完整链路，支持用户端实时咨询与商家端知识库管理双场景。

## 功能特性

- **ReAct 多工具智能体**：天气查询、用户 ID、月份、外部使用记录拉取等工具联动
- **多轮对话**：会话历史按条数与字符预算裁剪后进入模型，支持指代追问
- **检索前置**：每轮强制检索并按相似度阈值判定可信度，资料以「不可信数据」边界注入，回答标注 `[参考资料N]` 来源
- **会话级身份**：用户身份由会话 ID 稳定解析并作用域传递，取数工具不接受 user_id 参数，无法查询他人数据
- **个性化使用报告**：中间件按意图切换提示词，拉取本人使用数据生成 Markdown 报告；数据缺失时显式标注兜底月份
- **RAG 检索增强**：ChromaDB 向量库 + 中文分片优化；错误码/型号走关键词通道补召回，候选按词法覆盖率重排
- **知识库管理**：口令鉴权 + 上传内容注入检测 + MD5 去重；全量重建采用蓝绿切换，重建期间线上查询不中断
- **分层兜底**：异常 / 空回答 / 知识库未覆盖 / 用户要求人工时自动给出转人工入口并生成工单
- **反馈闭环**：每条回答可赞踩，事件落 JSONL，用于定位知识缺口
- **可观测性**：request_id 贯穿检索与生成，记录命中数、相似度、检索耗时与首字延迟（p50/p95 可算）
- **诚实降级**：天气/检索等外部依赖不可用时明确告知，全链路不返回伪造数据
- **流式输出**：token 级增量渲染；带静默超时看门狗，模型卡住会明确报错而不是无限等待
- **后台预热**：多线程预加载重量级服务，状态托管在被导入模块中，跨 rerun 不会重复初始化
- **内置评测**：独立评测基线（`eval/eval_suite.py`）+ 52 项离线行为契约用例（`tests/`）

## 技术栈

| 模块 | 技术 |
|------|------|
| Web 界面 | Streamlit |
| Agent 框架 | LangChain / LangGraph ReAct |
| 大模型 | 通义千问 qwen3-max（DashScope） |
| 向量检索 | ChromaDB + text-embedding-v4 |
| 天气数据（可选） | Open-Meteo（免密钥，`weather_provider: open-meteo` 时启用；默认关闭） |
| 文档解析 | PyPDFLoader / TextLoader |

## 快速开始

```bash
# 1. 安装依赖
pip install -r requirements.txt

# 2. 配置环境变量（或写入 config/agent.yml）
#    DASHSCOPE_API_KEY=你的DashScope密钥
#    KB_ADMIN_PASSWORD=知识库管理页口令（不配置则该页仅本地可写，见 app.py）

# 3. 导入知识库（data/ 下的文档 → 向量库）
python -c "from rag.vector_store import VectorStoreService; VectorStoreService().load_document()"

# 4. 启动应用
streamlit run app.py
```

## 知识库评测

**主评测基线**（独立题目、生产链路、含越界拒答与 p95 延迟）：

```bash
python eval/eval_suite.py --no-llm      # 只测检索，不消耗模型额度
python eval/eval_suite.py               # 完整评测：引用率 / 越界拒答率 / p50 / p95
```

评测集 `eval/gold_qa.json` 的题目独立编写，不取自知识库既有问答对；
命中判定用人工标注的关键词，只看「资料是否召回到」，不替代人工评分。

历史脚本 `eval/rag_eval.py` 仍在，但它**与知识库同源、且测的是 `rag_summarize` 单链**，
其 96.7% 只能当上限参考，不可作为泛化能力证据（脚本头部已标注该局限）。

## 验收用例

```bash
python tests/test_p0_acceptance.py      # 29 项：多轮、检索前置、身份隔离、鉴权、异常收敛
python tests/test_p1p2_acceptance.py    # 23 项：脱敏埋点、检索增强、蓝绿指针、预热幂等、超时、转人工
```

全部离线运行，不发起模型请求，可在无 API 额度或 CI 环境回归。

## 项目结构

```
├── app.py                  # Streamlit 主入口（智能客服 + 知识库管理）
├── agent/                  # ReAct 智能体、工具、中间件、会话身份、兜底判定
├── rag/                    # 向量库服务（蓝绿重建）、检索前置、候选重排
├── model/                  # 模型工厂（ChatTongyi / DashScopeEmbeddings）
├── utils/                  # 配置、日志、预热、鉴权、脱敏、埋点、工单
├── prompts/                # 系统提示词 / 报告提示词 / RAG 提示词
├── config/                 # YAML 配置
├── data/                   # 知识库源文档（txt/pdf）+ 运行时工单
├── eval/                   # 评测基线与评测集
├── tests/                  # 离线行为契约用例（无需模型额度）
├── scripts/                # 健康检查
├── Dockerfile              # 容器镜像（含 HEALTHCHECK）
└── chroma_db/              # 向量库（运行时生成，不入库）
```

## 运维与可观测性

- **事件日志**：`logs/events_YYYYMMDD.jsonl`，每条含 `request_id`；事件类型
  `turn_retrieval`（命中数/最高相似度/检索耗时）、`turn_answer`（耗时/是否转人工）、
  `turn_error`、`feedback`、`ticket_created`、`kb_import`。字段中的用户输入已脱敏截断。
- **工单**：`data/tickets/tickets_YYYYMMDD.jsonl`，转人工时生成，可直接接续处理。
- **延迟**：用 `utils.telemetry.percentile` 对事件流统计 p50/p95。
- **健康检查**：`python scripts/healthcheck.py`（容器 `HEALTHCHECK` 已接入，不触发模型调用）。

## 容器化

```bash
docker build -t smart-cs .
docker run -p 8501:8501 -e DASHSCOPE_API_KEY=... -e KB_ADMIN_PASSWORD=... smart-cs
```

## 免责声明

`data/external/records.csv` 为演示用模拟数据，请勿在生产环境存放真实用户信息。
