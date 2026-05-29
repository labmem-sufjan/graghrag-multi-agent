#!/usr/bin/env python3
"""命令行问答入口：调用 run_query，打印路由、chunk 引用与 Critic 结果。"""

from __future__ import annotations

import argparse
import sys


def main() -> None:
    parser = argparse.ArgumentParser(
        description="GraphRAG Multi-Agent 招股书问答（MVP）",
    )
    parser.add_argument("question", nargs="?", help="问题（省略则进入交互模式）")
    parser.add_argument("-q", "--quiet", action="store_true", help="只打印最终答案")
    args = parser.parse_args()

    from src.graph_workflow import run_query

    def run_once(q: str) -> None:
        if not args.quiet:
            print(f"\n问题: {q}\n{'=' * 50}")
        result = run_query(q)
        if not args.quiet:
            print(f"路由: {result.get('route')} ({result.get('route_reason', '')})")
            print(f"引用 chunk: {', '.join(result.get('chunk_ids', [])[:8])}")
            critic_ok = result.get("critic_passed", True)
            print(f"Critic: {'通过' if critic_ok else '未通过'} — {result.get('critic_feedback', '')}")
            print("=" * 50)
        print(result.get("answer", "(无回答)"))

    if args.question:
        run_once(args.question)
        return

    print("GraphRAG Multi-Agent MVP — 输入问题，空行退出")
    while True:
        try:
            q = input("\n你> ").strip()
        except (EOFError, KeyboardInterrupt):
            print("\n再见。")
            break
        if not q:
            break
        run_once(q)


if __name__ == "__main__":
    main()
