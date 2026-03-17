import os
import json
import sys
from pathlib import Path
import django

# =========================================================
# Django setup
# =========================================================
BASE_DIR = Path(__file__).resolve().parents[1]  # scripts/ 아래라고 가정
sys.path.append(str(BASE_DIR))

os.environ.setdefault("DJANGO_SETTINGS_MODULE", "makeQnS.settings")
django.setup()

from preprocessing.models import MockExamQuestion, QuestionImage  # 기존 파일과 동일

# =========================================================
# Helpers
# =========================================================
def model_field_names(model_cls):
    # DB 모델 필드명이 실제로 존재하는지 검사해서 안전하게 defaults 구성
    return {f.name for f in model_cls._meta.get_fields() if hasattr(f, "name")}

MOCK_FIELDS = model_field_names(MockExamQuestion)

def safe_defaults(**kwargs):
    """MockExamQuestion에 실제 존재하는 필드만 defaults로 남긴다."""
    return {k: v for k, v in kwargs.items() if k in MOCK_FIELDS}

def iter_jsonl_files(dir_path: Path):
    if not dir_path.exists():
        return
    for p in sorted(dir_path.rglob("*.jsonl")):
        yield p

def extract_asset_paths_from_question_row(data: dict):
    """
    question jsonl은 보통
    - question_assets: [{"type": "...", "path": "...", "page": 1}, ...]
    - 또는 assets: 위와 같은 dict list
    를 갖는다.
    """
    paths = []

    # 우선순위: question_assets -> assets
    candidates = data.get("question_assets") or data.get("assets") or []

    for a in candidates:
        if isinstance(a, dict) and "path" in a:
            paths.append(a["path"])
        elif isinstance(a, str):
            paths.append(a)

    return paths

def extract_asset_paths_from_solution_row(data: dict):
    """
    solution jsonl은 보통
    - solution_assets: ["assets/solutions/.../q01.png"]
    """
    assets = data.get("solution_assets") or []
    out = []
    for a in assets:
        if isinstance(a, str):
            out.append(a)
        elif isinstance(a, dict) and "path" in a:
            out.append(a["path"])
    return out

def make_abs_path(rel_or_abs: str):
    """
    jsonl에는 보통 상대 경로가 들어있음.
    - 상대경로면 BASE_DIR 기준으로 절대경로로 변환
    - 이미 절대경로면 그대로
    """
    p = Path(rel_or_abs)
    if p.is_absolute():
        return str(p)
    return str((BASE_DIR / rel_or_abs).resolve())

# =========================================================
# Importers
# =========================================================
def import_questions(question_jsonl_root: Path):
    print(f"\n[IMPORT] questions from: {question_jsonl_root}")
    count = 0

    for jsonl_path in iter_jsonl_files(question_jsonl_root):
        print(f"  - reading: {jsonl_path}")

        with jsonl_path.open("r", encoding="utf-8") as f:
            for line in f:
                if not line.strip():
                    continue
                data = json.loads(line)

                qid = data.get("id")
                if not qid:
                    continue

                grade = data.get("grade")
                year = data.get("year")
                month = data.get("month")

                # question_text/choices
                q_text = data.get("question_text", "")
                choices = data.get("choices", [])

                defaults = safe_defaults(
                    grade=grade,
                    year=year,
                    month=month,
                    raw_text=q_text,           # 기존 import가 raw_text에 넣던 방식 유지
                    # 만약 모델에 choices 같은 필드가 있으면 여기에 추가 가능
                    choices=choices,
                )

                question_obj, _ = MockExamQuestion.objects.update_or_create(
                    question_id=qid,
                    defaults=defaults
                )

                # question images
                img_paths = extract_asset_paths_from_question_row(data)

                for rel_path in img_paths:
                    abs_path = make_abs_path(rel_path)
                    QuestionImage.objects.get_or_create(
                        question=question_obj,
                        image_path=abs_path
                    )

                count += 1

    print(f"[DONE] questions imported rows: {count}")


def import_solutions(solution_jsonl_root: Path):
    print(f"\n[IMPORT] solutions from: {solution_jsonl_root}")
    count = 0
    updated_text = 0
    updated_assets = 0

    for jsonl_path in iter_jsonl_files(solution_jsonl_root):
        print(f"  - reading: {jsonl_path}")

        with jsonl_path.open("r", encoding="utf-8") as f:
            for line in f:
                if not line.strip():
                    continue
                data = json.loads(line)

                qid = data.get("id")
                if not qid:
                    continue

                grade = data.get("grade")
                year = data.get("year")
                month = data.get("month")

                sol_text = data.get("solution_text")
                sol_assets = extract_asset_paths_from_solution_row(data)

                # 문제 레코드가 없으면 만들어 둔다(해설부터 들어와도 깨지지 않게)
                defaults = safe_defaults(
                    grade=grade,
                    year=year,
                    month=month,
                )

                question_obj, _ = MockExamQuestion.objects.update_or_create(
                    question_id=qid,
                    defaults=defaults
                )

                # 1) solution_text 필드가 모델에 있으면 저장
                if "solution_text" in MOCK_FIELDS and sol_text is not None:
                    question_obj.solution_text = sol_text
                    updated_text += 1

                # 2) solution_assets 필드가 모델에 있으면 저장(JSONField/텍스트필드 상관없이)
                #    없으면 "이미지 경로는 jsonl에만 존재" 상태로 두고 넘어감
                if "solution_assets" in MOCK_FIELDS and sol_assets:
                    # 상대경로 그대로 저장(권장). 필요하면 abs로 바꾸려면 make_abs_path 사용
                    question_obj.solution_assets = sol_assets
                    updated_assets += 1

                # 3) 저장
                if ("solution_text" in MOCK_FIELDS and sol_text is not None) or ("solution_assets" in MOCK_FIELDS and sol_assets):
                    question_obj.save()

                count += 1

    print(f"[DONE] solutions imported rows: {count} | text updated: {updated_text} | assets updated: {updated_assets}")


def main():
    # ✅ 너 프로젝트 기준 기본 경로(권장)
    # questions jsonl: output/questions/jsonl/g1/...
    # solutions jsonl: output/jsonl/solutions/g1/...  (solution_parser_test.py에서 쓰는 구조)
    candidates_questions = [
        BASE_DIR / "output" / "jsonl" / "question",
        BASE_DIR / "output" / "jsonl" / "question",  # 유지
    ]
    candidates_solutions = [
        BASE_DIR / "output" / "jsonl" / "solutions",
        BASE_DIR / "output" / "solutions" / "jsonl",  # 혹시 이렇게 저장한 경우 대비
        BASE_DIR / "output" / "solutions" / "jsonl",  # 유지
    ]

    q_root = next((p for p in candidates_questions if p.exists()), None)
    s_root = next((p for p in candidates_solutions if p.exists()), None)

    if not q_root:
        print("❌ questions jsonl 폴더를 찾지 못했습니다. (예: output/questions/jsonl)")
    else:
        import_questions(q_root)

    if not s_root:
        print("❌ solutions jsonl 폴더를 찾지 못했습니다. (예: output/jsonl/solutions)")
    else:
        import_solutions(s_root)


if __name__ == "__main__":
    main()
