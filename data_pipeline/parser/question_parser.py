import os
import fitz
import re
import json
import ollama
from pathlib import Path
from collections import defaultdict
import time

def parse_exam_filename(filename: str, grade):
    # 시작시 초기화
    year = None

    m_year = re.search(r"(\d{2})학년도", filename)
    if m_year:
        year = int(m_year.group(1)) + 2000
    else:
        m_year = re.search(r"(20\d{2})년", filename)
        if m_year:
            year = int(m_year.group(1))
    if year is None:
        return None

    m_month = re.search(r"(\d{1,2})월", filename)
    if m_month:
        month = int(m_month.group(1))
    else:
        # 3) 월이 없으면(예: 수능) 규칙 부여
        if "수능" in filename:
            month = 11   # 수능은 보통 11월로 통일
        else:
            return None

    if grade < 3:
        track = "common"
    else:
        if "미적분" in filename:
            track = "calculus"
        elif "기하" in filename:
            track = "geometry"
        elif "확률과통계" in filename:
            track = "probability"
        else:
            track = "common"

    return {
        "grade": grade,
        "year": year,
        "month": month,
        "track": track
    }

QUESTION_RE = re.compile(r"^\s*(\d{1,2})\.")

NOISE_LINE_PATTERNS = [
    r"^\s*\d{4}학년도",        # 2020학년도...
    r"전국연합학력평가",        # 전국연합학력평가
    r"학력평가\s*문제지",       # 문제지
    r"수학영역",               # 수학영역
    r"제\s*\d+\s*교시",         # 제2 교시
    r"^\s*고\s*1\s*$",          # 고1
    r"^━{5,}\s*$",              # 긴 구분선
]

def strip_noise_lines(s: str) -> str:
    lines = (s or "").splitlines()
    cleaned = []
    for ln in lines:
        t = ln.strip()
        if not t:
            continue
        if any(re.search(pat, t) for pat in NOISE_LINE_PATTERNS):
            continue
        # "페이지번호만 있는 줄" 제거(예: 1, 2, 12 같은 단독 숫자)
        if re.fullmatch(r"\d{1,3}", t):
            continue
        cleaned.append(t)
    return "\n".join(cleaned).strip()

def extract_questions(pdf_path):
    doc = fitz.open(pdf_path)
    questions = {}
    current_q = None

    try:
        for page in doc:
            lines = page.get_text("text").splitlines()
            for line in lines:
                m = QUESTION_RE.match(line)
                if m:
                    qnum = int(m.group(1))
                    current_q = qnum
                    questions[current_q] = line + "\n"
                elif current_q:
                    questions[current_q] += line + "\n"

        return questions
    finally:
        doc.close()


def build_items(pdf_path, filename, grade, kind):
    meta = parse_exam_filename(filename, grade)
    if meta is None:
        return []
    questions = extract_questions(pdf_path)

    # 파일별 폴더 생성(충돌 방지)
    is_solution = (kind == "solution")
    safe_name = filename.replace(".pdf", "")
    assets_dir = Path("output")/ "assets" / "questions" / f"g{grade}"/ safe_name
    assets_by_q = render_exam_images(pdf_path, assets_dir, dpi=150, kind=kind)
    for fp in assets_dir.rglob("*_p*_2.png"):
        try:
            fp.unlink()
        except FileNotFoundError:
            pass

    def drop_header_tiles(paths):
        out = []
        for p in paths:
            # p가 dict인 경우: p["path"]에서 파일명 추출
            if isinstance(p, dict) and "path" in p:
                name = Path(p["path"]).name
                if re.search(r"_p\d+_2\.png$", name):
                    continue
                out.append(p)
                continue

            # p가 Path/str인 경우
            name = p.name if hasattr(p, "name") else Path(str(p)).name
            if re.search(r"_p\d+_2\.png$", name):
                continue
            out.append(p)
        return out

    items = []
    is_common_fn = (lambda q: True) if grade <= 2 else (lambda q: q <= 22)

    for qnum, text in questions.items():
        stem, choices = split_choices(text)

        stem = strip_noise_lines(stem)
        choices = [strip_noise_lines(c) for c in choices]

        raw_assets = assets_by_q.get(qnum, [])

        raw_assets = drop_header_tiles(raw_assets)
        
        item = {
            "id": f"g{grade}_{meta['year']}_{meta['month']}_{meta['track']}_q{qnum:02d}",
            **meta,
            "kind": kind,
            "question_number": qnum,
            "is_common": is_common_fn(qnum),
            "question_text": stem,
            "choices": choices,
            "assets": raw_assets,
            "question_assets": [],   
            "solution_assets": [],   
        }
        if is_solution:
            item["solution_assets"] = raw_assets
        else:
            item["question_assets"] = raw_assets

        items.append(item)

    return items

CHOICE_RE = re.compile(r"(①|②|③|④|⑤)")

def split_choices(text: str):
    """
    returns: (stem, choices_list)
    - stem: 보기 제외한 문제 본문
    - choices_list: ["① ...", "② ...", ...] (없으면 [])
    """
    parts = CHOICE_RE.split(text)
    if len(parts) <= 1:
        return text.strip(), []

    stem = parts[0].strip()
    choices = []
    # parts 구조: [stem, "①", "...", "②", "...", ...]
    for i in range(1, len(parts)-1, 2):
        mark = parts[i]
        body = parts[i+1].strip()
        choices.append(f"{mark} {body}")
    return stem, choices

def save_jsonl(items, out_path):
    out_path = Path(out_path)
    out_path.parent.mkdir(parents=True, exist_ok=True)

    with out_path.open("w", encoding="utf-8") as f:
        for it in items:
            f.write(json.dumps(it, ensure_ascii=False) + "\n")

QNO_RE = re.compile(r"^\s*(\d{1,2})\.\s*$|^\s*(\d{1,2})\.")

def find_question_anchors(doc: fitz.Document):
    anchors = []
    for pno in range(len(doc)):
        page = doc[pno]
        d = page.get_text("dict")

        for block in d.get("blocks", []):
            for line in block.get("lines", []):
                line_text = "".join(
                    span.get("text", "") for span in line.get("spans", [])
                ).strip()

                m = QNO_RE.match(line_text)
                if not m:
                    continue

                qnum = int(m.group(1) or m.group(2))
                x0 = line["bbox"][0]
                y0 = line["bbox"][1]

                anchors.append({
                    "qnum": qnum,
                    "page": pno,
                    "x0": x0,
                    "y0": y0
                })

    anchors.sort(key=lambda a: (a["page"], a["y0"], a["x0"]))
    return anchors

def detect_columns_by_text_blocks(page: fitz.Page, *, content_top=90, content_bottom_margin=90):
    """
    return list of (x_left, x_right) sorted by x_left
    - blocks 기반으로 x0 클러스터를 gap으로 나눠서 2~3컬럼 추정
    """
    w = page.rect.width
    h = page.rect.height
    y_min = content_top
    y_max = h - content_bottom_margin

    blocks = page.get_text("blocks")  # (x0, y0, x1, y1, "text", block_no, block_type)
    xs = []
    for (x0, y0, x1, y1, text, *_rest) in blocks:
        if y1 < y_min or y0 > y_max:
            continue
        if not text or not text.strip():
            continue
        bw = x1 - x0
        bh = y1 - y0
        # 너무 작은 조각/페이지번호 같은 잡음 제거
        if bw < 60 or bh < 10:
            continue
        xs.append(x0)

    if not xs:
        # fallback: 2컬럼 가정
        mid = w / 2
        return [(0, mid), (mid, w)]

    xs_sorted = sorted(xs)

    # x0들 사이 큰 gap 찾기 (컬럼 경계 후보)
    gaps = []
    for a, b in zip(xs_sorted, xs_sorted[1:]):
        gaps.append((b - a, (a + b) / 2))

    # "큰 gap"만 경계로 채택 (경험적으로 12~18% 폭 이상이 잘 먹힘)
    threshold = w * 0.14
    cuts = [pos for (gap, pos) in gaps if gap >= threshold]

    # 경계가 너무 많으면 큰 것 2개만 (최대 3컬럼까지)
    cuts = sorted(cuts)
    if len(cuts) > 2:
        # gap 큰 순으로 2개 선택
        cuts2 = sorted(gaps, key=lambda x: x[0], reverse=True)[:2]
        cuts = sorted([pos for _, pos in cuts2])

    # cuts로 컬럼 구간 생성
    boundaries = [0.0] + cuts + [w]
    cols = [(boundaries[i], boundaries[i+1]) for i in range(len(boundaries)-1)]

    # 너무 좁은 컬럼 제거(오검출 방지)
    cols = [c for c in cols if (c[1] - c[0]) >= w * 0.18] or cols

    return cols

def build_question_segments(doc, anchors, column_mode="auto", pad=6, content_top=90, content_bottom_margin=90):
    from collections import defaultdict
    segs = defaultdict(list)

    def pick_col_from_cols(cols, x0):
        for (l, r) in cols:
            if l <= x0 < r:
                return (l, r)
        return min(cols, key=lambda c: abs((c[0] + c[1]) / 2 - x0))

    def col_rect_auto(pno, top, bottom, x0):
        page = doc[pno]
        pr = page.rect
        top = max(top, content_top)
        bottom = min(bottom, pr.height - content_bottom_margin)

        cols = detect_columns_by_text_blocks(page, content_top=content_top, content_bottom_margin=content_bottom_margin)
        l, r = pick_col_from_cols(cols, x0)
        return fitz.Rect(l, top, r, bottom)

    def col_rect_fixed(pno, top, bottom, x0):
        page = doc[pno]
        pr = page.rect
        top = max(top, content_top)
        bottom = min(bottom, pr.height - content_bottom_margin)

        w = pr.width
        mid = w / 2
        if x0 < mid:
            return fitz.Rect(0, top, mid, bottom)
        else:
            return fitz.Rect(mid, top, w, bottom)

    def make_rect(pno, top, bottom, x0):
        if column_mode == "auto":
            return col_rect_auto(pno, top, bottom, x0)
        else:
            return col_rect_fixed(pno, top, bottom, x0)

    def is_left(pno_local, x0_local):
        w = doc[pno_local].rect.width
        return x0_local < (w / 2)

    for i, a in enumerate(anchors):
        qnum = a["qnum"]
        pno = a["page"]
        x0 = a["x0"]

        page_h = doc[pno].rect.height
        content_bottom = page_h - content_bottom_margin

        y0 = max(a["y0"] - pad, content_top)
        cur_left = is_left(pno, x0)

        next_pno = pno
        next_y0 = content_bottom

        for j in range(i + 1, len(anchors)):
            b = anchors[j]
            bp = b["page"]
            bx0 = b["x0"]
            by0 = max(b["y0"] - pad, content_top)

            if bp != pno:
                next_pno = bp
                next_y0 = by0
                break

            if is_left(pno, bx0) == cur_left:
                next_pno = bp
                next_y0 = by0
                break

        next_y0 = min(next_y0, doc[next_pno].rect.height - content_bottom_margin)

        if next_pno == pno:
            rect = make_rect(pno, y0, next_y0, x0)
            segs[qnum].append((pno, rect))
        else:
            rect0 = make_rect(pno, y0, content_bottom, x0)
            segs[qnum].append((pno, rect0))

            for mid_p in range(pno + 1, next_pno):
                mid_h = doc[mid_p].rect.height
                rect_mid = make_rect(mid_p, content_top, mid_h - content_bottom_margin, x0)
                segs[qnum].append((mid_p, rect_mid))

            rect_last = make_rect(next_pno, content_top, next_y0, x0)
            segs[qnum].append((next_pno, rect_last))

    return segs


def _is_valid_rect(rect: fitz.Rect, min_size: float = 5.0) -> bool:
    # Rect 자체가 깨지거나(역전/0), 너무 작으면 저장할 의미가 없으니 스킵
    if rect is None:
        return False
    if rect.x0 >= rect.x1 or rect.y0 >= rect.y1:
        return False
    if rect.width <= min_size or rect.height <= min_size:
        return False
    return True

QNO_RE_SOL = re.compile(r"^\s*(\d{1,2})\.\s*(?:\[[^\]]+\])?")

def find_solution_anchors(doc: fitz.Document):
    anchors = []
    for pno in range(len(doc)):
        page = doc[pno]
        d = page.get_text("dict")
        for block in d.get("blocks", []):
            for line in block.get("lines", []):
                line_text = "".join(span.get("text", "") for span in line.get("spans", [])).strip()
                m = QNO_RE_SOL.match(line_text)
                if not m:
                    continue
                qnum = int(m.group(1))
                x0 = line["bbox"][0]
                y0 = line["bbox"][1]
                anchors.append({"qnum": qnum, "page": pno, "x0": x0, "y0": y0})
    return anchors

def build_solution_segments_flow(doc, anchors, pad=12, content_top=140, content_bottom_margin=140):
    from collections import defaultdict

    def col_idx(page_rect, x0):
        return 0 if x0 < page_rect.width / 2 else 1

    def col_rect(page_rect, col, top, bottom):
        w = page_rect.width
        mid = w / 2
        if col == 0:
            return fitz.Rect(0, top, mid, bottom)
        return fitz.Rect(mid, top, w, bottom)

    # flow 정렬: page -> col -> y
    anchors2 = []
    for a in anchors:
        pr = doc[a["page"]].rect
        anchors2.append({**a, "col": col_idx(pr, a["x0"])})
    anchors2.sort(key=lambda a: (a["page"], a["col"], a["y0"]))

    segs = defaultdict(list)

    for i, a in enumerate(anchors2):
        qnum, pno, col = a["qnum"], a["page"], a["col"]
        page = doc[pno]
        pr = page.rect
        top = max(a["y0"] - pad, content_top)
        bottom_limit = pr.height - content_bottom_margin

        # 다음 앵커 (없으면 문서 끝으로)
        if i + 1 < len(anchors2):
            b = anchors2[i + 1]
        else:
            b = None

        if b and b["page"] == pno and b["col"] == col:
            bottom = min(max(b["y0"] - pad, content_top), bottom_limit)
            rect = col_rect(pr, col, top, bottom) & pr
            segs[qnum].append((pno, rect))

        elif b and b["page"] == pno and b["col"] != col:
            # 같은 페이지에서 왼->오 넘어감: 2조각
            rect0 = (col_rect(pr, col, top, bottom_limit) & pr)
            segs[qnum].append((pno, rect0))

            # 다음이 다른 컬럼이므로, 그 컬럼의 content_top -> next_y
            next_top = content_top
            next_bottom = min(max(b["y0"] - pad, content_top), bottom_limit)
            rect1 = (col_rect(pr, b["col"], next_top, next_bottom) & pr)
            segs[qnum].append((pno, rect1))

        else:
            # 페이지 넘어감: MVP는 "현재 컬럼의 남은 부분"만 잡고, 다음 페이지는 다음 앵커에서 시작
            rect = col_rect(pr, col, top, bottom_limit) & pr
            segs[qnum].append((pno, rect))

    return segs

def render_exam_images(pdf_path: str, out_dir: Path, *, dpi: int = 200, kind: str = "question"):
    out_dir = Path(out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    doc = fitz.open(pdf_path)

    # 정책 결정
    if kind == "solution":
        anchors = find_solution_anchors(doc)
        column_mode = "auto"
        pad, content_top, content_bottom_margin = 10, 120, 120
        asset_type = "solution_image"
    else:
        anchors = find_question_anchors(doc)
        column_mode = "fixed"
        pad, content_top, content_bottom_margin = 6, 90, 90
        asset_type = "question_image"

    if not anchors:
        doc.close()
        return {}

    segs = build_question_segments(doc, anchors, column_mode, pad, content_top, content_bottom_margin)
    assets_by_q = {}

    client = ollama.Client(timeout=None)

    for qnum, rects in segs.items():
        assets = []
        # idx가 1인 첫 번째 조각만 처리하도록 제한
        for idx, (pno, rect) in enumerate(rects, start=1):
            if idx > 1: 
                continue  # *_p*_2.png 이상의 파일 생성을 원천 차단

            page = doc[pno]
            rect = rect & page.rect
            if not _is_valid_rect(rect):
                continue

            pix = page.get_pixmap(clip=rect, dpi=dpi)
            if pix.width <= 0 or pix.height <= 0:
                continue

            img_path = out_dir / f"{asset_type}_q{qnum:02d}_p{pno+1}_{idx}.png"
            pix.save(str(img_path))

            # Ollama 비전 모델 로직
            question_text_llm = ""
            try:
                img_path_str = str(img_path.resolve())
                
                response = client.chat(
                    model='qwen3.5:9b',
                    messages=[{
                        'role': 'user',
                        'content': "이 이미지의 수학 문제를 LaTeX를 사용해 텍스트로 변환해줘. 다른 설명은 생략하고 문제 내용만 출력해.",
                        'images': [str(img_path)]
                    }],
                options={'num_predict': 4096, 'temperature': 0} # 결과의 일관성을 위해 temp 0 설정
                )
                content = response.get('message', {}).get('content', "")
                if not content.strip():
                    print(f"  [Warning] Q{qnum} 응답이 비어있습니다. (DPI 확인 필요)")
                question_text_llm = response.get('message', {}).get('content', "").strip()
            except Exception as e:
                print(f"  [Error] LLM 처리 실패 (Q{qnum}): {e}")
                question_text_llm = "변환 실패"

            assets.append({
                "type": asset_type,
                "path": str(img_path).replace("\\", "/"),
                "page": pno + 1,
                "text_llm": question_text_llm
            })
            print("GPU 휴식을 위해 잠시 멈춥니다.")
            time.sleep(1.5)

        assets_by_q[qnum] = assets

    doc.close()
    return assets_by_q

ROOT = Path("data")

pdf_paths = sorted({p.resolve() for p in ROOT.rglob("*.pdf")})  # set으로 중복 제거

print("CWD:", os.getcwd())
for pdf_path in pdf_paths:
    filename = pdf_path.name

    if "고1" in pdf_path.parts:
        grade = 1
    # 시간 관계상 고1만 처리 고2 고3은 시간이 되면 추가 예정
    # elif "고2" in pdf_path.parts:
    #     grade = 2
    # elif "고3" in pdf_path.parts:
    #     grade = 3

    else:
        print("SKIP(no grade):", pdf_path)
        continue

    # 문제(문항) PDF만 처리: 해설 PDF는 별도 스크립트에서 처리
    if "해설" in filename:
        print("SKIP(solution pdf):", filename)
        continue

    kind = "question"

    print("PARSING:", filename, grade)

    meta = parse_exam_filename(filename, grade)
    if meta is None:
        print("  -> SKIP(meta parse fail)")
        continue

    items = build_items(str(pdf_path), filename, grade, kind)
    out_path = Path("output") / "jsonl" / "questions" / f"g{grade}" / f"{meta['year']}_{meta['month']}_{meta['track']}_{kind}.jsonl"
    out_path.parent.mkdir(parents=True, exist_ok=True)
    save_jsonl(items, out_path)
    print("GPU 온도를 낮추기 위해 10초간 긴 휴식을 가집니다...")
    time.sleep(10)

    print("WRITE TO:", os.path.abspath(out_path))

    print(f"  -> {len(items)} questions parsed")