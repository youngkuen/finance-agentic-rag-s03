# -*- coding: utf-8 -*-
"""인제스천 그래프.

LangGraph 는 스케줄러가 아니라 제어흐름 + 체크포인트다. 여기서 얻는 것은 두 가지다.
  1. "어떤 문서가 어느 경로로 갔는가"가 상태로 남는다(리포트가 공짜로 나온다).
  2. thread_id 를 파일 해시로 잡으면 같은 문서를 다시 넣어도 건너뛴다(idempotent).

    classify → parse → gate ─┬─ pass ───────────────→ chunk → END
                             ├─ reparse → (pass|ocr|repair)
                             ├─ ocr ────→ (chunk|dead_letter)
                             └─ fail ───→ dead_letter
"""
from __future__ import annotations

import hashlib
from pathlib import Path

from langgraph.graph import END, StateGraph

from . import nodes as N


def build_ingest_graph(checkpointer=None):
    g = StateGraph(N.IngestState)
    for name in ("classify", "parse", "gate", "reparse", "ocr", "llm_repair", "chunk", "dead_letter"):
        g.add_node(name, getattr(N, name))

    g.set_entry_point("classify")
    g.add_edge("classify", "parse")
    g.add_edge("parse", "gate")

    # 게이트 판정이 네 갈래로 갈린다. pass 만 바로 청킹으로 간다.
    g.add_conditional_edges("gate", N.route_after_gate, {
        "pass": "chunk", "reparse": "reparse", "ocr": "ocr", "fail": "dead_letter",
    })

    # 다른 파서로 다시 뽑은 뒤에도 쓸 페이지가 하나도 없으면 LLM 복구가 마지막 수단이다.
    g.add_conditional_edges("reparse", N.route_after_reparse, {
        "pass": "chunk", "ocr": "ocr", "repair": "llm_repair",
    })

    # OCR 캐시에서 건진 페이지나 멀쩡한 본문이 있으면 색인하고, 없으면 사람 검토 큐로.
    g.add_conditional_edges("ocr", N.route_after_ocr, {
        "chunk": "chunk", "fail": "dead_letter",
    })

    g.add_conditional_edges("llm_repair", lambda s: s["route"], {
        "chunk": "chunk", "fail": "dead_letter",
    })

    g.add_edge("chunk", END)
    g.add_edge("dead_letter", END)

    return g.compile(checkpointer=checkpointer)


def thread_id_for(path: Path) -> str:
    """파일 내용 해시. 같은 파일이면 같은 thread 라 체크포인트가 재실행을 건너뛴다."""
    return hashlib.sha256(path.read_bytes()).hexdigest()[:16]
