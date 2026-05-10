"""
学科知识整合智能体 — Gradio 主入口
功能: 教材上传 → 解析 → 知识图谱 → 跨教材整合 → RAG问答 → 对话

运行: python app.py
"""

import os, sys, uuid, json, tempfile, shutil
from pathlib import Path
from datetime import datetime
import gradio as gr
from dotenv import load_dotenv

# 加载 .env 文件（本地开发用，服务器上会优先用平台环境变量）
load_dotenv()

# 确保 src 在路径中
sys.path.insert(0, str(Path(__file__).parent / "src"))
sys.path.insert(0, str(Path(__file__).parent / "src" / "backend"))

from backend.parser.parser import parse_textbook, TextbookInfo, PARSERS
from backend.knowledge.extractor import build_knowledge_graph, graph_to_echarts, merge_graphs
from backend.integration.integrator import integrate_graphs, IntegrationResult
from backend.rag.rag_pipeline import TextChunker, VectorIndex, rag_query, RAGResponse
from backend.dialogue.manager import DialogueManager

# ── 启动自检 ──────────────────────────────────
print("=" * 50)
print("[自检] 模块导入完成")

# 网络连通性测试
try:
    import requests
    r = requests.get("https://ms-ens-f8274faf-bcde.api-inference.modelscope.cn/v1/models", 
                     headers={"Authorization": "Bearer ms-b992cd79-197b-42f7-9c1b-d14c0ed0f9b2"},
                     timeout=10)
    print(f"[自检] API 连通性: HTTP {r.status_code}")
except Exception as e:
    print(f"[自检] ⚠️ 网络不通: {e}")

# LLM 快速测试
try:
    from llm_client import call_llm
    test_reply = call_llm("回复OK", "你只回复OK两个字母", temperature=0.1)
    print(f"[自检] LLM 测试: {test_reply[:80]}")
except Exception as e:
    print(f"[自检] ⚠️ LLM 调用失败: {e}")

print("=" * 50)

# ── 全局状态 ──────────────────────────────────
UPLOAD_DIR = Path("data/uploads")
UPLOAD_DIR.mkdir(parents=True, exist_ok=True)
OUTPUT_DIR = Path("output")
OUTPUT_DIR.mkdir(parents=True, exist_ok=True)

_books: dict = {}           # textbook_id → TextbookInfo
_graphs: dict = {}          # textbook_id → graph dict
_vector_idx = VectorIndex()
_dialogue = DialogueManager()
_integration: IntegrationResult = None

# ── 辅助函数 ──────────────────────────────────

def _book_list_md():
    """生成教材列表 Markdown"""
    if not _books:
        return "📭 暂无教材，请上传\n\n> 支持 PDF / Markdown / TXT"
    rows = ["| # | 教材名 | 格式 | 页数 | 字数 | 章节数 | 状态 |",
            "|---|--------|------|------|------|--------|------|"]
    for bid, b in _books.items():
        rows.append(f"| {bid} | {b.title} | {b.format} | {b.total_pages} | {b.total_chars:,} | {len(b.chapters)} | ✅ {b.status} |")
    return "\n".join(rows)

# ── Tab 1: 教材上传 ───────────────────────────

def handle_upload(files):
    """处理文件上传"""
    if not files:
        return _book_list_md(), "⚠️ 请先选择文件", gr.update(choices=list(_books.keys()))
    results = []
    for f in files:
        try:
            fpath = Path(f.name)
            ext = fpath.suffix.lower()
            if ext not in ['.pdf', '.md', '.txt', '.docx']:
                results.append(f"⚠️ {fpath.name}: 不支持的格式")
                continue
            bid = f"book_{uuid.uuid4().hex[:6]}"
            dest = UPLOAD_DIR / f"{bid}{ext}"
            shutil.copy(f.name, str(dest))
            book = parse_textbook(str(dest), bid)
            _books[bid] = book
            results.append(f"✅ {book.title} ({book.format}) — {book.total_pages}页 {book.total_chars:,}字")
        except Exception as e:
            results.append(f"❌ {Path(f.name).name}: {e}")
    return _book_list_md(), "\n".join(results), gr.update(choices=list(_books.keys()))


def load_sample_books():
    """加载 data/textbooks/ 下的示例教材"""
    sample_dir = Path("data/textbooks")
    if not sample_dir.exists():
        return _book_list_md(), "📭 data/textbooks/ 目录不存在，请手动上传教材", gr.update(choices=list(_books.keys()))
    pdfs = list(sample_dir.glob("*.pdf")) + list(sample_dir.glob("*.md")) + list(sample_dir.glob("*.txt"))
    if not pdfs:
        return _book_list_md(), "📭 data/textbooks/ 下没有教材文件", gr.update(choices=list(_books.keys()))
    results = []
    for fpath in pdfs:
        try:
            bid = f"book_{uuid.uuid4().hex[:6]}"
            book = parse_textbook(str(fpath), bid)
            _books[bid] = book
            results.append(f"✅ {book.title} ({book.format}) — {book.total_pages}页")
        except Exception as e:
            results.append(f"❌ {fpath.name}: {e}")
    return _book_list_md(), "\n".join(results), gr.update(choices=list(_books.keys()))

# ── Tab 2: 知识图谱 ───────────────────────────

def build_graph_for_book(book_id: str):
    """为指定教材构建知识图谱"""
    if not book_id or book_id not in _books:
        return None, "⚠️ 请先选择一个教材"
    book = _books[book_id]
    graph = build_knowledge_graph(book.to_dict())
    _graphs[book_id] = graph
    # 生成 ECharts 可用的数据
    echarts_data = graph_to_echarts(graph)
    stats = f"### {book.title}\n- 知识点: {graph['stats']['total_nodes']}\n- 关系: {graph['stats']['total_edges']}"
    return json.dumps(echarts_data, ensure_ascii=False), stats


def get_graph_for_frontend(book_id: str):
    """返回图谱 JSON + stats"""
    if book_id and book_id in _graphs:
        g = _graphs[book_id]
        return json.dumps(graph_to_echarts(g), ensure_ascii=False), f"节点: {g['stats']['total_nodes']} | 关系: {g['stats']['total_edges']}"
    return "{}", "请先构建图谱"


def get_book_choices():
    return list(_books.keys())

# ── Tab 3: 跨教材整合 ─────────────────────────

def run_integration():
    """执行跨教材整合"""
    global _integration
    if len(_books) < 1:
        return "⚠️ 请先上传至少 1 本教材", "{}", ""
    # 确保每本书都有图谱
    total_chars = 0
    graphs = []
    for bid, book in _books.items():
        total_chars += book.total_chars
        if bid not in _graphs:
            _graphs[bid] = build_knowledge_graph(book.to_dict())
        graphs.append(_graphs[bid])

    result = integrate_graphs(graphs, total_chars, target_ratio=0.30)
    _integration = result

    merged_graph_data = json.dumps(graph_to_echarts(merge_graphs(graphs)), ensure_ascii=False)

    md = f"""## 📊 整合报告

| 指标 | 数值 |
|------|------|
| 原始教材数 | {len(_books)} |
| 原始总字数 | {total_chars:,} |
| 原始知识点 | {result.original_node_count} |
| 整合后知识点 | {result.merged_node_count} |
| 整合决策数 | {len(result.decisions)} |
| **压缩比** | **{result.compression_ratio:.1%}** |
| 目标压缩比 | ≤ 30% |

### 整合决策摘要
- 合并 (merge): {sum(1 for d in result.decisions if str(d.action) == 'merge')} 项
- 保留 (keep): {sum(1 for d in result.decisions if str(d.action) == 'keep')} 项
- 删除 (remove): {sum(1 for d in result.decisions if str(d.action) == 'remove')} 项
"""
    if result.compression_ratio <= 0.30:
        md += "\n✅ **压缩比达标！**"
    else:
        md += f"\n⚠️ 压缩比 {result.compression_ratio:.1%} 超过 30% 目标，可能需要调整阈值"
    return md, merged_graph_data, "✅ 整合完成"


def get_decisions_text():
    """获取整合决策列表文本"""
    if not _integration:
        return "暂无整合结果"
    lines = [f"共 {len(_integration.decisions)} 项决策:\n"]
    for i, d in enumerate(_integration.decisions[:20]):
        lines.append(f"{i+1}. [{d.action}] {d.reason} (置信度: {d.confidence})")
    return "\n".join(lines)

# ── Tab 4: RAG 问答 ───────────────────────────

def build_rag_index():
    """为所有教材建立 RAG 索引"""
    chunker = TextChunker(chunk_size=600, overlap=80)
    all_chunks = []
    for bid, book in _books.items():
        all_chunks.extend(chunker.chunk_textbook(book.to_dict()))
    if not all_chunks:
        return "⚠️ 没有可索引的内容，请先上传教材"
    msg = _vector_idx.index(all_chunks)
    return f"✅ {msg}\n共 {len(_books)} 本教材"


def ask_question(question: str, history: list):
    """RAG 问答"""
    if not question.strip():
        return history, history
    if _vector_idx.embeddings is None or len(_vector_idx.chunks) == 0:
        history.append({"role": "user", "content": question})
        history.append({"role": "assistant", "content": "⚠️ 请先构建 RAG 索引（点击「建立索引」按钮）"})
        return history, history

    resp: RAGResponse = rag_query(question, _vector_idx, top_k=5)
    # 构建回答
    answer = resp.answer
    if resp.citations:
        answer += "\n\n**引用来源：**\n"
        for i, c in enumerate(resp.citations[:3], 1):
            answer += f"\n{i}. 《{c.textbook}》{c.chapter} p{c.page}（相关度: {c.relevance_score:.2f}）"

    history.append({"role": "user", "content": question})
    history.append({"role": "assistant", "content": answer})
    return history, history

# ── Tab 5: 多轮对话 ───────────────────────────

def chat_with_teacher(message: str, history: list):
    """教师对话"""
    if not message.strip():
        return "", history
    sid = "default_session"
    decisions = _integration.decisions if _integration else None
    reply = _dialogue.chat(sid, message, decisions)
    history.append({"role": "user", "content": message})
    history.append({"role": "assistant", "content": reply})
    return "", history

# ── 全局样式 ──────────────────────────────────
CSS = """
.container { max-width: 1400px; margin: auto; }
.status-ok { color: #2e7d32; }
footer { visibility: hidden; }
"""

# ── 构建 Gradio UI ────────────────────────────

def create_ui():
    with gr.Blocks(title="学科知识整合智能体") as app:
        gr.Markdown("""# 🧠 学科知识整合智能体
        **多教材知识图谱 | 跨教材去重提纯 | RAG 精准问答 | 教师对话优化**

        上传多本教材 → 构建知识图谱 → 智能去重整合 → 问答对话
        """)

        with gr.Tabs():
            # ═══════════ Tab 1: 教材管理 ═══════════
            with gr.Tab("📚 教材管理"):
                with gr.Row():
                    with gr.Column(scale=2):
                        upload_btn = gr.UploadButton(
                            "📤 上传教材文件", file_types=[".pdf", ".md", ".txt", ".docx"],
                            file_count="multiple"
                        )
                    with gr.Column(scale=1):
                        load_sample_btn = gr.Button("📂 加载示例教材", variant="secondary")
                upload_log = gr.Textbox(label="上传日志", lines=5, interactive=False)
                book_list = gr.Markdown(_book_list_md(), every=5)

            # ═══════════ Tab 2: 知识图谱 ═══════════
            with gr.Tab("🗺️ 知识图谱"):
                with gr.Row():
                    book_dropdown = gr.Dropdown(
                        choices=[], label="选择教材", interactive=True
                    )
                    build_btn = gr.Button("🔨 构建图谱", variant="primary")
                    build_all_btn = gr.Button("🔨 全部构建", variant="secondary")

                graph_stats = gr.Markdown("请先选择教材并点击「构建图谱」")
                graph_json = gr.JSON(label="图谱数据 (ECharts 格式)")

                # 图谱可视化占位（图谱数据在 graph_json 中展示）
                graph_html = gr.HTML("""<div style="text-align:center;padding:80px;color:#999;font-size:16px;">
                🗺️ 图谱数据将在右侧 JSON 面板中展示<br/>
                <small>ECharts 可视化需在部署环境中加载（CDN 引入）</small>
                </div>""")

                build_btn.click(
                    lambda bid: build_graph_for_book(bid) if bid else (None, "请选择一个教材"),
                    book_dropdown, [graph_json, graph_stats]
                )

            # ═══════════ Tab 3: 整合 ═══════════
            with gr.Tab("🔀 跨教材整合"):
                integrate_btn = gr.Button("🚀 执行整合分析", variant="primary", size="lg")
                integration_md = gr.Markdown("点击按钮执行跨教材整合分析")
                integrated_graph_json = gr.JSON(label="整合后图谱", visible=False)
                decisions_text = gr.Textbox(label="整合决策详情", lines=10, interactive=False)

                integrate_btn.click(
                    run_integration,
                    None,
                    [integration_md, integrated_graph_json, decisions_text]
                )

            # ═══════════ Tab 4: RAG 问答 ═══════════
            with gr.Tab("💬 RAG 问答"):
                with gr.Row():
                    index_btn = gr.Button("🔍 建立索引", variant="primary")
                    index_status = gr.Textbox(label="索引状态", interactive=False)
                index_btn.click(build_rag_index, None, index_status)

                chatbot_rag = gr.Chatbot(label="RAG 问答", height=400)
                with gr.Row():
                    q_input = gr.Textbox(label="输入问题", placeholder="例如：什么是动作电位？", scale=4)
                    q_btn = gr.Button("提问", variant="primary", scale=1)

                q_btn.click(ask_question, [q_input, chatbot_rag], [chatbot_rag, chatbot_rag])
                q_input.submit(ask_question, [q_input, chatbot_rag], [chatbot_rag, chatbot_rag])

            # ═══════════ Tab 5: 教师对话 ═══════════
            with gr.Tab("👩‍🏫 教师对话"):
                chatbot_dialogue = gr.Chatbot(label="整合方案讨论", height=400)
                with gr.Row():
                    chat_input = gr.Textbox(label="输入消息", placeholder="例如：为什么把炎症和炎症反应合并了？", scale=4)
                    chat_btn = gr.Button("发送", variant="primary", scale=1)

                chat_btn.click(chat_with_teacher, [chat_input, chatbot_dialogue], [chat_input, chatbot_dialogue])
                chat_input.submit(chat_with_teacher, [chat_input, chatbot_dialogue], [chat_input, chatbot_dialogue])

            # ═══════════ Tab 6: 报告 ═══════════
            with gr.Tab("📊 整合报告"):
                report_btn = gr.Button("📝 生成整合报告", variant="primary")
                report_md = gr.Markdown("点击按钮生成完整的整合报告")

                def generate_report():
                    if not _integration:
                        return "⚠️ 请先在「跨教材整合」Tab 中执行整合分析"
                    r = _integration
                    books_info = "\n".join(
                        f"| {b.title} | {b.format} | {b.total_pages} | {b.total_chars:,} | {len(b.chapters)} |"
                        for b in _books.values()
                    )
                    decisions_summary = "\n".join(
                        f"{i+1}. **[{d.action}]** {d.reason}（置信度: {d.confidence:.0%}）"
                        for i, d in enumerate(r.decisions[:10])
                    )
                    return f"""# 学科知识整合报告

> 生成时间: {datetime.now().strftime('%Y-%m-%d %H:%M')}

## 一、整合概览

| 指标 | 数值 |
|------|------|
| 原始教材数 | {len(_books)} |
| 原始总字数 | {r.original_chars:,} |
| 整合后字数 | {r.merged_chars:,} |
| **压缩比** | **{r.compression_ratio:.1%}** |
| 目标 | ≤ 30% |

## 二、教材清单

| 教材名 | 格式 | 页数 | 字数 | 章节数 |
|--------|------|------|------|--------|
{books_info}

## 三、整合决策摘要

- 整合决策总数: {len(r.decisions)}
- 合并 (merge): {sum(1 for d in r.decisions if str(d.action) == 'merge')} 项
- 保留 (keep): {sum(1 for d in r.decisions if str(d.action) == 'keep')} 项
- 删除 (remove): {sum(1 for d in r.decisions if str(d.action) == 'remove')} 项

## 四、知识图谱统计

| 指标 | 数值 |
|------|------|
| 整合前总节点数 | {r.original_node_count} |
| 整合后节点数 | {r.merged_node_count} |

## 五、重点整合案例

{decisions_summary if decisions_summary else '暂无'}

## 六、教学完整性说明

整合过程中通过 Embedding 语义相似度识别重复知识点，优先保留定义最完整的版本。合并后的知识体系覆盖了原始教材的核心内容。

---

*本报告由学科知识整合智能体自动生成*
"""

                report_btn.click(generate_report, None, report_md)

        # ── 跨 Tab 事件绑定（所有组件定义完毕后） ──
        upload_btn.upload(handle_upload, upload_btn, [book_list, upload_log, book_dropdown])
        load_sample_btn.click(load_sample_books, None, [book_list, upload_log, book_dropdown])

        # 底部状态栏
        gr.Markdown("---\n浙江大学未来学习中心 · AI 生态 2026年5月 | 学科知识整合智能体 v1.0")

    return app


if __name__ == "__main__":
    app = create_ui()
    app.queue(max_size=20).launch(
        server_name="127.0.0.1",
        server_port=int(os.getenv("PORT", "7860")),
        share=False,
        show_error=True,
        theme=gr.themes.Soft(primary_hue="blue"),
        css=CSS,
    )
