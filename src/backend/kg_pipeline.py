#!/usr/bin/env python3
"""
知识图谱构建流水线 (KG Pipeline)
━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
严格按照赛题要求：
  1. 为每本教材构建知识图谱（节点 + 关系）
  2. Embedding 预筛选 + LLM 语义判断 跨教材去重整合
  3. 输出整合决策（merge/keep/remove）
  4. 生成可视化 ECharts 数据
  5. 输出整合报告

运行方式：
  python -m src.backend.kg_pipeline            # 从项目根目录
  python src/backend/kg_pipeline.py             # 直接运行
"""

import os, sys, json, time
from pathlib import Path

# 确保导入路径
sys.path.insert(0, str(Path(__file__).parent.parent.parent))
sys.path.insert(0, str(Path(__file__).parent.parent))

from backend.parser.parser import load_from_cache
from backend.knowledge.extractor import (
    build_knowledge_graph, merge_graphs,
    graph_to_echarts, graph_to_echarts_integrated,
)
from backend.integration.integrator import (
    integrate_graphs, IntegrationResult, DecisionAction,
)

OUTPUT_DIR = Path("output")
OUTPUT_DIR.mkdir(parents=True, exist_ok=True)


def print_progress(step: str, pct: float, desc: str = ""):
    """打印进度信息"""
    bar_len = 40
    filled = int(bar_len * pct)
    bar = "█" * filled + "░" * (bar_len - filled)
    print(f"\r[{bar}] {pct*100:5.1f}% | {step}: {desc}", end="", flush=True)
    if pct >= 1.0:
        print()  # 换行


def load_all_cached_books(cache_dir: str = "cache", max_books: int = 7) -> list:
    """加载所有缓存的教材（去重 + 可选限制数量）"""
    cache_path = Path(cache_dir)
    if not cache_path.exists():
        print(f"❌ 缓存目录 {cache_dir} 不存在")
        return []

    book_files = sorted(cache_path.glob("book_*.json"))
    # 过滤掉压缩版本
    book_files = [f for f in book_files if "_compressed" not in f.stem]

    books = []
    seen_titles = set()
    for bf in book_files:
        try:
            with open(bf, "r", encoding="utf-8") as f:
                data = json.load(f)
            title = data.get("title", "").strip()
            # 按教材标题去重（同一教材不同缓存副本）
            if title in seen_titles:
                print(f"  ⏭️ 跳过重复: {title}")
                continue
            seen_titles.add(title)
            books.append(data)
            print(f"  📄 加载: {title} ({data.get('total_chars',0):,} 字, "
                  f"{len(data.get('chapters',[]))} 章)")
        except Exception as e:
            print(f"  ⚠️ 加载失败: {bf.name} - {e}")

    if max_books and len(books) > max_books:
        books = books[:max_books]
        print(f"  ⚠️ 限制为 {max_books} 本教材（可调整 max_books 参数）")
    
    return books


def run_kg_pipeline(books: list = None, use_llm_align: bool = True):
    """
    运行完整的知识图谱构建流水线
    
    Args:
        books: 教材列表（dict），若为 None 则从 cache 加载
        use_llm_align: 是否使用 LLM 进行语义对齐（推荐）
    
    Returns:
        {
            "graphs": [...],          # 各教材图谱
            "merged_graph": {...},    # 合并后的图谱
            "integration": {...},     # 整合结果
            "echarts_before": {...},  # 整合前 ECharts 数据
            "echarts_after": {...},   # 整合后 ECharts 数据
        }
    """
    print("\n" + "=" * 70)
    print("🧠 学科知识图谱构建流水线")
    print("=" * 70)
    
    # ── Step 0: 加载教材 ──
    if books is None:
        print("\n📂 Step 0: 加载缓存的教材...")
        books = load_all_cached_books()
    
    if not books:
        print("❌ 没有教材可处理！请先运行 Step 1 解析教材")
        return None
    
    n_books = len(books)
    total_chars = sum(b.get("total_chars", 0) for b in books)
    print(f"\n📊 共 {n_books} 本教材，总字数 {total_chars:,}")
    for b in books:
        print(f"   - {b.get('title','?')}: {b.get('total_chars',0):,} 字, "
              f"{len(b.get('chapters',[]))} 章")
    
    # ── Step 1: 构建单本教材知识图谱 ──
    print(f"\n{'='*70}")
    print(f"📘 Step 1: 构建单本教材知识图谱 ({n_books} 本)")
    print(f"{'='*70}")
    
    graphs = []
    for bi, book in enumerate(books):
        bid = book.get("textbook_id", f"book_{bi}")
        title = book.get("title", bid)
        
        def make_book_cb(_bi, _title):
            def cb(pct: float, desc: str):
                book_pct = (_bi + pct) / n_books
                print_progress("图谱构建", book_pct, f"[{_bi+1}/{n_books}] {_title}: {desc}")
            return cb
        
        cb = make_book_cb(bi, title)
        graph = build_knowledge_graph(book, progress_callback=cb, verbose=False, max_chapters=50)
        graphs.append(graph)
        
        # 保存图谱到 output/
        kg_path = OUTPUT_DIR / f"{bid}_kg.json"
        with open(kg_path, "w", encoding="utf-8") as f:
            json.dump(graph, f, ensure_ascii=False, indent=2)
        print(f"  💾 图谱已保存: {kg_path}")
    
    # 统计
    total_nodes = sum(g["stats"]["total_nodes"] for g in graphs)
    total_edges = sum(g["stats"]["total_edges"] for g in graphs)
    print(f"\n  📊 图谱统计: {total_nodes} 节点, {total_edges} 关系")
    for g in graphs:
        print(f"     {g['title']}: {g['stats']['total_nodes']} 节点, "
              f"{g['stats']['total_edges']} 关系")
    
    # ── Step 2: 合并图谱 ──
    print(f"\n{'='*70}")
    print(f"🔗 Step 2: 跨教材知识图谱整合（核心步骤）")
    print(f"{'='*70}")
    
    merged_graph = merge_graphs(graphs)
    print(f"  合并前: {total_nodes} 节点, {total_edges} 关系")
    
    # 生成整合前的 ECharts 数据
    echarts_before = graph_to_echarts(merged_graph, max_nodes=300)
    print(f"  📊 整合前 ECharts: {echarts_before['stats']['displayed_nodes']} 节点, "
          f"{echarts_before['stats']['displayed_edges']} 边")
    
    # ── Step 3: 跨教材整合 ──
    print(f"\n  🔍 Step 2a: Embedding 预筛选...")
    print_progress("Embedding", 0.0, "计算语义向量...")
    
    result: IntegrationResult = integrate_graphs(
        graphs, total_chars, target_ratio=0.30, use_llm=use_llm_align,
        progress_callback=lambda pct, desc: print_progress("整合", pct, desc),
    )
    
    # 分类统计决策
    merge_count = sum(1 for d in result.decisions
                      if str(d.action) == 'merge')
    keep_count = sum(1 for d in result.decisions
                     if str(d.action) == 'keep')
    remove_count = sum(1 for d in result.decisions
                       if str(d.action) == 'remove')
    
    print(f"\n  📊 整合结果:")
    print(f"     原始节点: {result.original_node_count}")
    print(f"     整合后节点: {result.merged_node_count}")
    print(f"     整合决策: {len(result.decisions)} 项")
    print(f"       - 合并 (merge): {merge_count} 项")
    print(f"       - 保留 (keep):  {keep_count} 项")
    print(f"       - 删除 (remove): {remove_count} 项")
    print(f"     压缩比: {result.compression_ratio*100:.1f}%")
    
    if result.compression_ratio <= 0.30:
        print(f"     ✅ 压缩比达标！")
    else:
        print(f"     ⚠️ 压缩比超过 30%，可能需要调整参数")
    
    # ── Step 4: 生成整合后可视化数据 ──
    print(f"\n{'='*70}")
    print(f"🎨 Step 3: 生成可视化数据")
    print(f"{'='*70}")
    
    echarts_after = graph_to_echarts_integrated(
        graphs, decisions=result.decisions, mode="merged", max_nodes=300,
    )
    print(f"  📊 整合后 ECharts: {len(echarts_after['nodes'])} 节点, "
          f"{len(echarts_after['links'])} 边, "
          f"{echarts_after.get('removed_count', 0)} 节点标记为已处理")
    
    # ── Step 5: 保存所有结果 ──
    print(f"\n{'='*70}")
    print(f"💾 Step 4: 保存结果")
    print(f"{'='*70}")
    
    # 保存整合结果
    integration_path = OUTPUT_DIR / "integration_result.json"
    with open(integration_path, "w", encoding="utf-8") as f:
        json.dump({
            "original_books": n_books,
            "original_chars": total_chars,
            "original_nodes": result.original_node_count,
            "merged_nodes": result.merged_node_count,
            "compression_ratio": result.compression_ratio,
            "decisions": [
                d.to_dict() if hasattr(d, 'to_dict') else d
                for d in result.decisions
            ],
            "merge_count": merge_count,
            "keep_count": keep_count,
            "remove_count": remove_count,
        }, f, ensure_ascii=False, indent=2)
    print(f"  💾 整合结果: {integration_path}")
    
    # 保存整合前 ECharts
    before_path = OUTPUT_DIR / "echarts_before_integration.json"
    with open(before_path, "w", encoding="utf-8") as f:
        json.dump(echarts_before, f, ensure_ascii=False)
    print(f"  💾 整合前图谱: {before_path}")
    
    # 保存整合后 ECharts
    after_path = OUTPUT_DIR / "echarts_after_integration.json"
    with open(after_path, "w", encoding="utf-8") as f:
        json.dump(echarts_after, f, ensure_ascii=False)
    print(f"  💾 整合后图谱: {after_path}")
    
    # 保存整合决策详情（人类可读）
    report_path = OUTPUT_DIR / "integration_decisions.md"
    _generate_decision_report(result, books, report_path)
    print(f"  💾 决策报告: {report_path}")
    
    # ── 最终总结 ──
    print(f"\n{'='*70}")
    print(f"✅ 知识图谱构建完成！")
    print(f"{'='*70}")
    print(f"  教材数:       {n_books}")
    print(f"  原始总字数:   {total_chars:,}")
    print(f"  原始节点数:   {result.original_node_count}")
    print(f"  整合后节点数: {result.merged_node_count}")
    print(f"  整合决策数:   {len(result.decisions)}")
    print(f"  压缩比:       {result.compression_ratio*100:.1f}%")
    print(f"  输出目录:     {OUTPUT_DIR.absolute()}")
    
    return {
        "graphs": graphs,
        "merged_graph": merged_graph,
        "integration": result,
        "echarts_before": echarts_before,
        "echarts_after": echarts_after,
    }


def _generate_decision_report(result: IntegrationResult, books: list, path: Path):
    """生成整合决策报告（Markdown）"""
    lines = [
        "# 知识图谱整合决策报告",
        "",
        f"生成时间: {time.strftime('%Y-%m-%d %H:%M:%S')}",
        "",
        "## 整合概览",
        "",
        f"| 指标 | 数值 |",
        f"|------|------|",
        f"| 原始教材数 | {len(books)} |",
        f"| 原始总字数 | {sum(b.get('total_chars',0) for b in books):,} |",
        f"| 原始知识点数 | {result.original_node_count} |",
        f"| 整合后知识点数 | {result.merged_node_count} |",
        f"| 整合决策总数 | {len(result.decisions)} |",
        f"| 压缩比 | {result.compression_ratio*100:.1f}% |",
        f"| 目标压缩比 | ≤30% |",
        "",
        "## 教材列表",
        "",
    ]
    
    for b in books:
        lines.append(f"- **{b.get('title','?')}**: {b.get('total_chars',0):,} 字, "
                     f"{len(b.get('chapters',[]))} 章")
    
    lines += [
        "",
        "## 整合决策摘要",
        "",
    ]
    
    merge_count = sum(1 for d in result.decisions if str(d.action) == 'merge')
    keep_count = sum(1 for d in result.decisions if str(d.action) == 'keep')
    remove_count = sum(1 for d in result.decisions if str(d.action) == 'remove')
    
    lines += [
        f"- **合并 (merge)**: {merge_count} 项",
        f"- **保留 (keep)**: {keep_count} 项",
        f"- **删除 (remove)**: {remove_count} 项",
        "",
        "## 重点整合案例",
        "",
    ]
    
    # 选前 10 个高置信度决策
    sorted_decisions = sorted(result.decisions,
                              key=lambda d: d.confidence if hasattr(d, 'confidence') else 0,
                              reverse=True)
    
    for i, d in enumerate(sorted_decisions[:10], 1):
        action = str(d.action)
        reason = d.reason if hasattr(d, 'reason') else ""
        conf = d.confidence if hasattr(d, 'confidence') else 0
        affected = d.affected_nodes if hasattr(d, 'affected_nodes') else []
        lines += [
            f"### {i}. [{action.upper()}] {reason[:60]}",
            f"- **置信度**: {conf:.2f}",
            f"- **涉及节点**: {', '.join(affected[:5])}",
            "",
        ]
    
    lines += [
        "## 教学完整性说明",
        "",
        "整合过程中遵循以下原则确保教学逻辑链条不断裂：",
        "",
        '1. **前置依赖保留**: 如果知识点A是知识点B的前置依赖，合并/删除A时需确保B的定义中包含A的核心内容',
        '2. **概念完整性**: 上位概念被保留后，其包含的下位概念仍可在定义中体现',
        "3. **跨教材互补**: 不同教材对同一概念的描述侧重点不同，整合时保留最完整的版本",
        "4. **逻辑链条**: 关系边(edges)在整合后仍保持连通，教学逻辑路径不断",
    ]
    
    with open(path, "w", encoding="utf-8") as f:
        f.write("\n".join(lines))


# ═══════════════════════════════════════════════
# 主入口
# ═══════════════════════════════════════════════

if __name__ == "__main__":
    print("学科知识图谱构建流水线")
    print("=" * 70)
    
    result = run_kg_pipeline(use_llm_align=True)
    
    if result:
        print("\n✅ 所有步骤完成！")
        print(f"  图谱文件: output/*_kg.json")
        print(f"  ECharts 数据: output/echarts_*.json")
        print(f"  决策报告: output/integration_decisions.md")
    else:
        print("\n❌ 流水线未能完成")
