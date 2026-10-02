# -*- coding: utf-8 -*-
"""인제스천 노드.

원칙: 결정적 파이프라인이 기본이고 LLM 은 실패 브랜치에만 둔다. 문서 하나마다
LLM 을 부르면 비용도 지연도 재현성도 다 잃는다. 여기서 LLM 을 쓰는 곳은
`llm_repair` 하나뿐이고, 그것도 게이트가 재추출까지 실패시켰을 때만 부른다.
"""
from __future__ import annotations

import json
from pathlib import Path
from typing import Any, Literal, TypedDict

from ..llm import get_llm, is_fake
from ..parsing.chunk import chunk_document
from ..parsing.extract import PARSERS, Extraction, extract, extract_with, page_objects, score
from ..parsing.metadata import DocMeta, load_documents
from ..parsing.ocr import load_ocr
from ..parsing.validate import DocReport, validate
from ..settings import get_settings


class IngestState(TypedDict, total=False):
    doc_id: str
    path: str
    meta: dict
    pages: list[str]
    parser: str
    attempts: list[dict]
    report: dict
    ocr_pages: dict
    excluded_pages: list[int]
    degraded_pages: list[int]
    chunks: list[dict]
    route: str
    warnings: list[str]
    llm_repaired: bool


def _meta(doc_id: str) -> DocMeta:
    return load_documents().get(doc_id) or DocMeta(doc_id=doc_id)


def classify(state: IngestState) -> IngestState:
    """형식을 정한다. 확장자가 아니라 첫 바이트로 본다."""
    meta = _meta(state["doc_id"])
    path = Path(state.get("path") or meta.path)
    head = path.open("rb").read(8) if path.exists() else b""
    if head.startswith(b"%PDF-"):
        kind = "pdf"
    elif head.startswith((b"PK\x03\x04", b"\xd0\xcf\x11\xe0")):
        kind = "office"      # hwp/hwpx/zip → 지원하지 않는다
    elif not head:
        kind = "missing"
    else:
        kind = "text"
    return {"path": str(path), "meta": meta.payload(), "route": kind, "warnings": []}


def parse(state: IngestState) -> IngestState:
    """기본 파서로 뽑는다. 라우팅은 게이트 판정을 보고 다음 노드에서 한다."""
    path = Path(state["path"])
    if state["route"] == "pdf":
        ex = extract(path, state["doc_id"])
        return {"pages": ex.pages, "parser": ex.parser, "attempts": ex.attempts}
    if state["route"] == "text":
        raw = path.read_bytes().decode("utf-8", "replace")
        fmt = _meta(state["doc_id"]).format
        if fmt == "html":
            import html as h
            import re
            raw = re.sub(r"<(script|style).*?</\1>", " ", raw, flags=re.S | re.I)
            raw = re.sub(r"<br\s*/?>|</p>|</div>|</tr>|</li>|</h\d>", "\n", raw, flags=re.I)
            raw = h.unescape(re.sub(r"<[^>]+>", " ", raw))
            raw = "\n".join(" ".join(l.split()) for l in raw.splitlines() if l.strip())
        elif fmt == "xml":
            import xml.etree.ElementTree as ET
            raw = "\n".join(t.strip() for t in ET.fromstring(raw).itertext() if t.strip())
        return {"pages": [raw], "parser": fmt or "plain", "attempts": []}
    return {"pages": [], "parser": "none", "attempts": []}


def gate(state: IngestState) -> IngestState:
    """검증 게이트. 여기서 통과 / 재추출 / OCR / 실패가 갈린다."""
    route = state["route"]

    # 지원하지 않는 형식은 검사기에 넣지 않는다. 넣어 봐야 "글자 0자"라는 같은 답만
    # 나오고, 사람 검토 큐에서 "형식 때문에 못 읽음"과 "글자가 깨짐"이 뒤섞인다.
    if route in ("office", "missing"):
        rep = DocReport(doc_id=state["doc_id"], parser=state.get("parser", ""),
                        verdict="fail", reasons=[f"지원하지 않는 형식({route})"],
                        pages=[], usable_pages=0)
        return {"report": rep.to_dict(), "route": "fail"}

    # PDF 만 페이지 객체를 같이 넘긴다. 이미지가 있는 쪽을 알아야 OCR 판정이 선다.
    path = Path(state["path"])
    page_objs = page_objects(path) if route == "pdf" else []
    ex = Extraction(state["doc_id"], state.get("parser", ""), state.get("pages", []))
    rep = validate(ex, page_objs)
    return {"report": rep.to_dict(), "route": rep.verdict}


def reparse(state: IngestState) -> IngestState:
    """다른 파서로 다시 뽑는다. 나쁜 페이지만 갈아 끼운다.

    문서 전체를 바꾸지 않는 이유: 별표15는 492쪽 중 2쪽만 나쁘다. 전체를 다른
    파서로 바꾸면 멀쩡한 490쪽의 품질이 같이 흔들린다.
    """
    path = Path(state["path"])
    bad = set(state["report"].get("bad_pages", []))
    pages = list(state["pages"])
    used = state.get("parser", "")
    fixed: list[int] = []

    for parser in [p for p in PARSERS if p != used]:
        alt = extract_with(path, parser)
        if not alt:
            continue
        for pno in sorted(bad):
            if pno <= len(alt) and score([alt[pno - 1]])["hangul"] > score([pages[pno - 1]])["hangul"]:
                pages[pno - 1] = alt[pno - 1]
                fixed.append(pno)
        if fixed:
            break

    ex = Extraction(state["doc_id"], used, pages)
    rep = validate(ex, page_objects(path))
    warn = state.get("warnings", [])
    if fixed:
        warn = warn + [f"{len(fixed)}쪽을 다른 파서로 교체: {sorted(set(fixed))[:5]}"]
    # 못 고친 페이지를 지우지 않는다. 글자가 깨졌거나 붙어 있어도 근거는 근거다.
    # 지우면 아무것도 못 찾지만, 두면 형태소 검색과 임베딩은 여전히 찾아낸다.
    # 정말 비우는 것은 "글자 자체가 없는" OCR 미확보 페이지뿐이다(ocr 노드에서 처리).
    degraded = [p_.page for p_ in rep.pages if p_.verdict == "reparse"]
    if degraded and rep.usable_pages:
        # 쓸 수 있는 페이지가 없으면 인덱싱이 아니라 LLM 복구로 가므로(route_after_reparse) 이 경고는 붙이지 않는다.
        warn = warn + [f"{len(degraded)}쪽은 품질이 낮은 채로 색인: {degraded[:5]}"]
    return {"pages": pages, "report": rep.to_dict(), "route": rep.verdict, "warnings": warn,
            "degraded_pages": degraded}


def ocr(state: IngestState) -> IngestState:
    """OCR 캐시를 읽어 이미지 페이지를 채운다. 여기서 OCR 을 돌리지 않는다."""
    pages_map, warn = load_ocr(state["doc_id"], Path(state["path"]))
    bad = state["report"].get("bad_pages", [])
    got = [p for p in bad if pages_map.get(p, "").strip()]
    missing = [p for p in bad if not pages_map.get(p, "").strip()]
    usable = state["report"].get("usable_pages", 0)
    # OCR 캐시가 없어도 본문 페이지가 있으면 그것만으로 색인한다.
    route = "chunk" if (got or usable) else "fail"
    if missing:
        warn = warn + [f"OCR 결과가 없어 제외한 페이지: {missing[:5]}"]
    return {"ocr_pages": pages_map, "route": route,
            "warnings": state.get("warnings", []) + warn,
            "excluded_pages": state.get("excluded_pages", []) + missing}


# ── 프롬프트를 채우세요 ──
# (과제) 깨진 페이지 텍스트를 LLM 에게 주고 복원시킨다. 프롬프트에 반드시 넣을 것:
# 내용을 추측해 채우지 말 것, 판독 불가는 [판독불가] 로 표시할 것.
# temperature 0 · 결과 캐시 · llm_repaired 표시는 이미 되어 있다.
def llm_repair(state: IngestState) -> IngestState:
    """마지막 수단. 다른 파서로 다시 추출해도 쓸 수 있는 페이지가 없는 문서의 본문을 LLM 이 복구할 수 있는지 본다.

    temperature 0, 결과는 캐시, 복구했다는 사실을 메타데이터에 남긴다(llm_repaired).
    복구된 청크는 평가에서 따로 세야 한다 — 원문이 아니라 모델이 만든 문장이기 때문이다.
    """
    s = get_settings()
    cache = s.data / "repair_cache" / f"{state['doc_id']}.json"
    if cache.exists():
        data = json.loads(cache.read_text(encoding="utf-8"))
        return {"pages": data["pages"], "llm_repaired": True, "route": "chunk"}

    llm = get_llm("extract", size="main")
    if is_fake(llm):
        return {"route": "fail",
                "warnings": state.get("warnings", []) + ["LLM 키가 없어 복구를 건너뜀"]}

    # 나쁜 페이지만, 그것도 앞부분만 보낸다. 문서 전체를 넣으면 비용이 폭발한다.
    bad = state["report"].get("bad_pages", [])[:3]
    pages = list(state["pages"])
    for pno in bad:
        raw = pages[pno - 1][:3000]
        msg = ("아래는 PDF에서 추출하다 깨진 한국어 금융 문서의 일부다. "
               "읽을 수 있는 문장만 원문 그대로 복원하고, 판독 불가한 부분은 [판독불가]로 표기하라. "
               "내용을 추측해 채우지 마라.\n\n---\n" + raw)
        pages[pno - 1] = llm.invoke(msg).content

    cache.parent.mkdir(parents=True, exist_ok=True)
    cache.write_text(json.dumps({"pages": pages}, ensure_ascii=False), encoding="utf-8")
    return {"pages": pages, "llm_repaired": True, "route": "chunk"}


def chunk(state: IngestState) -> IngestState:
    meta = _meta(state["doc_id"])
    ocr_pages = {int(k): v for k, v in (state.get("ocr_pages") or {}).items()}
    excluded = set(state.get("excluded_pages") or [])
    pages = [("" if (i in excluded and not ocr_pages.get(i)) else p)
             for i, p in enumerate(state.get("pages", []), start=1)]
    cs = chunk_document(pages, meta, ocr_pages=ocr_pages)
    out = []
    for c in cs:
        d = c.to_dict()
        d["meta"]["parser"] = state.get("parser", "")
        d["meta"]["llm_repaired"] = bool(state.get("llm_repaired"))
        d["meta"]["validation"] = state.get("report", {}).get("verdict", "")
        d["meta"]["excluded_pages"] = sorted(excluded)
        d["meta"]["degraded_pages"] = sorted(state.get("degraded_pages") or [])
        out.append(d)
    return {"chunks": out, "route": "done"}


def dead_letter(state: IngestState) -> IngestState:
    """사람 검토 큐. 조용히 버리지 않고 이유를 남긴다."""
    return {"chunks": [], "route": "failed",
            "warnings": state.get("warnings", []) + state["report"].get("reasons", [])}


def route_after_gate(state: IngestState) -> Literal["pass", "reparse", "ocr", "fail"]:
    return state["route"]  # type: ignore[return-value]


def route_after_reparse(state: IngestState) -> Literal["pass", "ocr", "repair"]:
    """재추출 뒤에도 나쁜 페이지가 남을 수 있다. 그래도 쓸 페이지가 있으면 색인한다.

    LLM 복구는 "쓸 페이지가 하나도 없을 때"만 부른다. 멀쩡한 문서의 한 페이지를
    복구하려고 LLM 을 부르면 비용만 늘고 본문에 모델이 지어낸 문장이 섞인다.
    """
    rep = state.get("report", {})
    if rep.get("usable_pages", 0) == 0:
        return "repair"
    return "ocr" if state["route"] == "ocr" else "pass"


def route_after_ocr(state: IngestState) -> Literal["chunk", "fail"]:
    return state["route"]  # type: ignore[return-value]
