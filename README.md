# 🧠 学科知识整合智能体

> 2026 AI 全栈黑客松 — 多教材知识图谱构建 | 跨教材去重提纯 | RAG 精准问答

## 📋 项目简介

开发一个 AI 智能体，对多本教材进行知识整合：构建可视化知识图谱、跨教材去重提纯、RAG 精准问答。

## 🛠️ 技术栈

| 层级 | 技术 |
|------|------|
| 前端 | Gradio |
| 后端 | Python |
| 文件解析 | PyMuPDF (PDF) / python-docx / markdown |
| LLM | OpenAI API / 通义千问 / DeepSeek |
| 向量嵌入 | sentence-transformers |
| 向量检索 | FAISS (CPU) |
| 知识图谱可视化 | ECharts |

## 📦 安装

```bash
# 1. 克隆仓库
git clone <repo-url>
cd <project-dir>

# 2. 安装依赖
pip install -r requirements.txt

# 3. 配置 API Key
cp .env.example .env
# 编辑 .env，填入你的 LLM_API_KEY
```

## ⚙️ 配置

编辑 `.env` 文件：

```env
LLM_API_KEY=sk-your-key-here          # 必填
LLM_BASE_URL=https://api.openai.com/v1
LLM_MODEL=gpt-3.5-turbo
EMBED_MODEL=paraphrase-multilingual-MiniLM-L12-v2
PORT=7860
```

## 🚀 运行

```bash
python app.py
```

打开浏览器访问 `http://localhost:7860`

## 📂 项目结构

```
├── app.py                    # Gradio 主入口
├── requirements.txt          # Python 依赖
├── .env.example              # 配置模板
├── README.md                 # 本文件
├── src/
│   └── backend/
│       ├── parser/           # 文件解析 (PDF/MD/TXT/DOCX)
│       ├── knowledge/        # 知识提取 & 图谱构建
│       ├── integration/      # 跨教材整合 & 去重
│       ├── rag/              # RAG Pipeline
│       └── dialogue/         # 多轮对话
├── docs/                     # 设计文档
├── report/                   # 整合报告
└── data/
    ├── textbooks/            # 教材文件 (不提交 Git)
    └── uploads/              # 用户上传目录
```

## 📝 功能

1. 📚 **多格式教材上传** — PDF / Markdown / TXT / DOCX
2. 🗺️ **知识图谱构建** — LLM 提取知识点 + 关系 → ECharts 可视化
3. 🔀 **跨教材整合** — Embedding 语义对齐 → 去重合并 → ≤30% 压缩
4. 💬 **RAG 精准问答** — Chunk → Embed → FAISS → 检索 → 生成 (带引用)
5. 👩‍🏫 **教师对话** — 多轮对话调整整合方案
6. 📊 **整合报告** — 自动生成 Markdown 报告

## 🔧 API 兼容

支持 OpenAI / 通义千问 / DeepSeek 等兼容 API。

修改 `.env` 中的 `LLM_BASE_URL` 即可切换：

```env
# 通义千问
LLM_BASE_URL=https://dashscope.aliyuncs.com/compatible-mode/v1
LLM_MODEL=qwen-plus

# DeepSeek
LLM_BASE_URL=https://api.deepseek.com/v1
LLM_MODEL=deepseek-chat
```
