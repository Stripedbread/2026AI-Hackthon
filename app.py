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

from backend.parser.parser import parse_textbook, TextbookInfo, PARSERS, save_to_cache, load_from_cache, list_cache
from backend.knowledge.extractor import build_knowledge_graph, graph_to_echarts, merge_graphs
from backend.integration.integrator import integrate_graphs, IntegrationResult, compress_book_contents, run_step2_pipeline
from backend.rag.rag_pipeline import TextChunker, VectorIndex, rag_query, RAGResponse
from backend.dialogue.manager import DialogueManager

# ── 启动自检 ──────────────────────────────────
print("=" * 50)
print("[自检] 模块导入完成")

# 网络连通性测试（5秒超时，失败不影响启动）
try:
    import requests
    r = requests.get("https://ms-ens-f8274faf-bcde.api-inference.modelscope.cn/v1/models", 
                     headers={"Authorization": "Bearer ms-b992cd79-197b-42f7-9c1b-d14c0ed0f9b2"},
                     timeout=5)
    print(f"[自检] API 连通性: HTTP {r.status_code}")
except Exception as e:
    print(f"[自检] ⚠️ 网络不通: {type(e).__name__}: {str(e)[:120]}")

# LLM 快速测试（10秒超时，失败不影响启动）
try:
    import signal
    from llm_client import call_llm
    test_reply = call_llm("回复OK", "你只回复OK两个字母", temperature=0.1)
    print(f"[自检] LLM 测试: {test_reply[:80]}")
except Exception as e:
    print(f"[自检] ⚠️ LLM 调用失败: {type(e).__name__}: {str(e)[:200]}")

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
_compression_info: dict = {}  # textbook_id → {"compression_ratio": float, "original_chars": int, "compressed_chars": int, "status": str}

# ── 辅助函数 ──────────────────────────────────

def _book_list_md():
    """生成教材列表 Markdown（含压缩状态）"""
    if not _books:
        return "📭 暂无教材，请上传\n\n> 支持 PDF / Markdown / TXT"
    rows = ["| # | 教材名 | 格式 | 页数 | 字数 | 章节数 | 压缩 | 状态 |",
            "|---|--------|------|------|------|--------|------|------|"]
    for i, (bid, b) in enumerate(_books.items(), 1):
        # 压缩状态
        comp_info = _compression_info.get(bid, {})
        if comp_info:
            ratio = comp_info.get("compression_ratio", 1.0)
            comp_status = f"{ratio*100:.1f}%"
        else:
            comp_status = "⏳ 待压缩"
        status_icon = "✅" if b.status == "completed" else "⏳"
        rows.append(f"| {i} | {b.title} | {b.format} | {b.total_pages} | {b.total_chars:,} | {len(b.chapters)} | {comp_status} | {status_icon} {b.status} |")
    return "\n".join(rows)

# ── Tab 1: 教材上传（含自动压缩） ───────────

def handle_upload(files):
    """处理文件上传（仅解析，不压缩）"""
    if not files:
        return _book_list_md(), "⚠️ 请先选择文件", "⏳ 等待上传..."
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
            book.title = fpath.stem  # 使用原始文件名，而非哈希 ID
            _books[bid] = book
            cache_path = save_to_cache(book)
            results.append(f"✅ {book.title} ({book.format}) — {book.total_pages}页 {book.total_chars:,}字 → 缓存: {cache_path.name}")
        except Exception as e:
            results.append(f"❌ {Path(f.name).name}: {e}")
    return _book_list_md(), "\n".join(results), "🔄 解析完成，等待压缩..."



def handle_upload_and_compress(files, progress=gr.Progress()) -> tuple:
    """
    上传文件 + 解析 + 自动压缩（集成 gr.Progress 进度条）
    上传完成后自动对新增教材执行内容压缩，前端显示进度条。
    """
    global _compression_info

    if not files:
        return _book_list_md(), "⚠️ 请先选择文件", "⏳ 等待上传..."

    # ── Phase 1: 解析上传的文件 ──
    progress(0.0, desc="📤 正在解析文件...")
    results = []
    newly_added = []

    total_files = len(files)
    for fi, f in enumerate(files):
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
            book.title = fpath.stem
            _books[bid] = book
            cache_path = save_to_cache(book)
            results.append(f"✅ {book.title} ({book.format}) — {book.total_pages}页 {book.total_chars:,}字")
            newly_added.append(bid)
        except Exception as e:
            results.append(f"❌ {Path(f.name).name}: {e}")

        progress((fi + 1) / total_files * 0.15, desc=f"📤 解析: {fi+1}/{total_files}")

    if not newly_added:
        return _book_list_md(), "\n".join(results), "⚠️ 没有新文件需要压缩"

    # ── Phase 2: 内容压缩（占 15%→100% 进度） ──
    progress(0.15, desc="🔍 准备压缩...")
    compress_log = []

    # 计算总有效章节数
    all_valid_chs = []
    for bid in newly_added:
        book_dict = _books[bid].to_dict()
        valid = [(bid, c) for c in book_dict.get("chapters", [])
                 if c.get("char_count", 0) > 100]
        all_valid_chs.extend(valid)

    total_chapters = len(all_valid_chs)
    completed = 0

    for bid in newly_added:
        book = _books[bid]
        book_dict = book.to_dict()

        valid_chapters = [c for c in book_dict.get("chapters", [])
                         if c.get("char_count", 0) > 100]
        n_chapters = len(valid_chapters)
        compress_log.append(f"📦 {book.title}: {n_chapters} 个章节待压缩")

        def make_cb(_bid, _title):
            nonlocal completed
            def cb(pct: float, desc: str):
                nonlocal completed
                chapter_progress = completed + pct
                overall = 0.15 + (chapter_progress / max(total_chapters, 1)) * 0.85
                progress(min(overall, 1.0), desc=desc)
            return cb

        cb = make_cb(bid, book.title)

        try:
            compressed = compress_book_contents(book_dict, target_ratio=0.30,
                                                progress_callback=cb)
            orig_c = compressed.get("original_chars", 0)
            comp_c = compressed.get("total_chars", 0)
            ratio = compressed.get("compression_ratio", 1.0)

            _compression_info[bid] = {
                "compression_ratio": ratio,
                "original_chars": orig_c,
                "compressed_chars": comp_c,
                "status": "completed",
            }
            compress_log.append(
                f"✅ {book.title}: {orig_c:,} → {comp_c:,} 字 ({ratio*100:.1f}%)"
            )
        except Exception as e:
            _compression_info[bid] = {
                "compression_ratio": 1.0,
                "original_chars": 0,
                "compressed_chars": 0,
                "status": f"失败: {str(e)[:50]}",
            }
            compress_log.append(f"❌ {book.title}: 压缩失败 - {e}")

        completed += n_chapters

    progress(1.0, desc="✅ 压缩全部完成！")
    return _book_list_md(), "\n".join(results), "\n".join(compress_log)


def run_auto_compression(progress=gr.Progress()) -> tuple:
    """
    对所有已上传但尚未压缩的教材自动执行内容压缩。
    通过 gr.Progress() 在前端显示进度条。
    """
    global _compression_info

    # 找出需要压缩的教材
    pending = [(bid, b) for bid, b in _books.items()
               if bid not in _compression_info]

    if not pending:
        return _book_list_md(), "✅ 所有教材均已压缩完成"

    total = len(pending)
    log_lines = []

    for bi, (bid, book) in enumerate(pending):
        book_dict = book.to_dict()

        # 筛选有效章节
        valid_chapters = [c for c in book_dict.get("chapters", [])
                         if c.get("char_count", 0) > 100]
        n_chapters = len(valid_chapters)
        log_lines.append(f"📦 {book.title}: {n_chapters} 个有效章节待压缩")

        # 定义进度回调（闭包捕获当前 book 的进度）
        def make_callback(_title, _bi, _total, _n_ch):
            def cb(pct: float, desc: str):
                book_progress = (_bi + pct) / _total
                progress(book_progress, desc=f"[{_bi+1}/{_total}] {desc}")
            return cb

        cb = make_callback(book.title, bi, total, n_chapters)

        try:
            compressed = compress_book_contents(book_dict, target_ratio=0.30,
                                                progress_callback=cb)
            orig_c = compressed.get("original_chars", 0)
            comp_c = compressed.get("total_chars", 0)
            ratio = compressed.get("compression_ratio", 1.0)

            _compression_info[bid] = {
                "compression_ratio": ratio,
                "original_chars": orig_c,
                "compressed_chars": comp_c,
                "status": "completed",
            }
            log_lines.append(
                f"✅ {book.title}: {orig_c:,} → {comp_c:,} 字 ({ratio*100:.1f}%)"
            )
        except Exception as e:
            _compression_info[bid] = {
                "compression_ratio": 1.0,
                "original_chars": 0,
                "compressed_chars": 0,
                "status": f"失败: {str(e)[:50]}",
            }
            log_lines.append(f"❌ {book.title}: 压缩失败 - {e}")

    return _book_list_md(), "\n".join(log_lines)


def _refresh_dropdown():
    """刷新教材下拉框 — 作为独立函数确保 Gradio 6.x 正确更新"""
    return gr.update(choices=_book_choices(), value=None)


def load_sample_books():
    """加载 data/textbooks/ 下的示例教材"""
    sample_dir = Path("data/textbooks")
    if not sample_dir.exists():
        return _book_list_md(), "📭 data/textbooks/ 目录不存在，请手动上传教材"
    pdfs = list(sample_dir.glob("*.pdf")) + list(sample_dir.glob("*.md")) + list(sample_dir.glob("*.txt"))
    if not pdfs:
        return _book_list_md(), "📭 data/textbooks/ 下没有教材文件"
    results = []
    for fpath in pdfs:
        try:
            bid = f"book_{uuid.uuid4().hex[:6]}"
            book = parse_textbook(str(fpath), bid)
            _books[bid] = book
            save_to_cache(book)
            results.append(f"✅ {book.title} ({book.format}) — {book.total_pages}页")
        except Exception as e:
            results.append(f"❌ {fpath.name}: {e}")
    return _book_list_md(), "\n".join(results)

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


def _book_choices():
    """返回 (书名, textbook_id) 元组列表，用于下拉框显示"""
    return [(b.title, bid) for bid, b in _books.items()]


def get_book_choices():
    return _book_choices()

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
    reply = _dialogue.chat(sid, message, decisions, books=_books)
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

                # 压缩进度条
                compress_progress = gr.Textbox(
                    label="压缩进度", value="⏳ 等待上传文件...",
                    lines=2, interactive=False, elem_classes=["compress-progress"]
                )
                book_list = gr.Markdown(_book_list_md(), every=5)

            # ═══════════ Tab 2: 知识图谱 ═══════════
            with gr.Tab("🗺️ 知识图谱"):
                with gr.Row():
                    book_dropdown = gr.Dropdown(
                        choices=_book_choices(), label="选择教材", interactive=True
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
        # 使用 .then() 链式调用确保 Gradio 6.x 中下拉框正确刷新
        # 上传 → 自动压缩（带进度条）→ 刷新下拉框
        upload_btn.upload(fn=handle_upload_and_compress, inputs=upload_btn,
                         outputs=[book_list, upload_log, compress_progress])\
                  .then(fn=_refresh_dropdown, outputs=book_dropdown)
        load_sample_btn.click(fn=load_sample_books, outputs=[book_list, upload_log])\
                      .then(fn=lambda: "🔄 正在自动压缩内容...", outputs=compress_progress)\
                      .then(fn=run_auto_compression, outputs=[book_list, compress_progress])\
                      .then(fn=_refresh_dropdown, outputs=book_dropdown)

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
