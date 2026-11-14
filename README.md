# 智净灵境 — 扫地/拖地机器人智能客服系统

面向扫地/拖地机器人品牌的 AI 智能客服系统，基于 **LangChain ReAct 智能体 + RAG 检索增强生成** 构建，覆盖「文档解析与清洗 → 知识库构建 → 向量检索 → 大模型生成 → 引用溯源」完整链路，支持用户端实时咨询与商家端知识库管理双场景。

## 功能特性

- **ReAct 多工具智能体**：天气查询、用户 ID、月份、外部使用记录拉取等工具联动
- **多轮对话**：会话历史按条数与字符预算裁剪后进入模型，支持指代追问
- **检索前置**：每轮强制检索并按相似度阈值判定可信度，资料以「不可信数据」边界注入，回答标注 `[参考资料N]` 来源
- **会话级身份**：用户身份由会话 ID 稳定解析并作用域传递，取数工具不接受 user_id 参数，无法查询他人数据
- **个性化使用报告**：中间件按意图切换提示词，拉取本人使用数据生成 Markdown 报告；数据缺失时显式标注兜底月份
- **RAG 检索增强**：ChromaDB 向量库 + 中文分片优化，回答严格基于检索资料生成
- **知识库管理**：口令鉴权 + 上传内容注入检测 + MD5 去重、增量加载与全量重建
- **诚实降级**：天气/检索等外部依赖不可用时明确告知，全链路不返回伪造数据
- **流式输出**：token 级增量渲染，问答体验流畅
- **后台预热**：多线程预加载重量级服务，首屏秒级响应
- **内置评测**：200 条标准问答对的 RAG 评测脚本 + P0 行为契约的离线验收用例（`tests/`）

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

基于 `data/` 下 200 条标准问答对进行抽样评测（LLM 判分）：

```bash
python eval/rag_eval.py --all --sample 30 --judge
```

参考结果（30 条样本）：检索命中率 96.7%，回答准确率 96.7%，问题解决率 100%，平均总响应约 3.9s。
> 注：该评测集与知识库同源，且跑的是 `rag_summarize` 单链而非生产 ReAct 链路，结果应视为上限参考，不可作为泛化能力证据。

## 验收用例

P0 改造的行为契约可在无模型额度环境下回归：

```bash
python tests/test_p0_acceptance.py
```

覆盖：多轮历史裁剪、检索强制注入与阈值判定、身份会话隔离与越权拒绝、外部数据兜底标记、鉴权正反用例、注入特征检测、错误文案不外泄、工具失败不击穿对话。

## 项目结构

```
├── app.py                  # Streamlit 主入口（智能客服 + 知识库管理）
├── agent/                  # ReAct 智能体、工具函数、中间件
├── rag/                    # 向量库服务、RAG 总结服务
├── model/                  # 模型工厂（ChatTongyi / DashScopeEmbeddings）
├── utils/                  # 配置、日志、文件、提示词加载
├── prompts/                # 系统提示词 / 报告提示词 / RAG 提示词
├── config/                 # YAML 配置
├── data/                   # 知识库源文档（txt/pdf）
├── eval/                   # RAG 评测脚本
├── tests/                  # P0 行为契约离线验收用例（无需模型额度）
└── chroma_db/              # 向量库（运行时生成，不入库）
```

## 免责声明

`data/external/records.csv` 为演示用模拟数据，请勿在生产环境存放真实用户信息。
