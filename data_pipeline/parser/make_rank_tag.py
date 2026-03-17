import json
import ollama
from pathlib import Path
from tqdm import tqdm
import time
import re

# 설정
JSONL_DIR = Path("output") / "jsonl"
question_DIR = JSONL_DIR / "questions" / "g1"
solution_DIR = JSONL_DIR / "solutions_latex" / "고1"
FINAL_OUT_DIR = Path("output/final_integrated")
FINAL_OUT_DIR.mkdir(parents=True, exist_ok=True)
# MODEL_NAME = "qwen3:8b"  # 텍스트 분석용
MODEL_NAME = "qwen3:4b"  # 8b가 무겁다면 사용


def get_difficulty_from_latex(question_text, solution_text):
    prompt = f"""
    아래 수학 문제와 해설을 분석하여 난이도를 1~5단계로 평가하고 핵심 키워드를 뽑아주세요.
    1: 공식 대입형 (하)
    2: 기본 응용 (중하)
    3: 복합 개념 적용 (중)
    4: 준킬러/심화 (상)
    5: 킬러 문항 (최상)

    문제 내용:
    {question_text}

    해설 내용:
    {solution_text} 

    답변은 반드시 아래 JSON 형식으로만 출력하세요:
    {{"difficulty": 1~5숫자, "tags": ["개념1", "개념2"], "type": "객관식/주관식"}}
    """
    
    try:
        # Ollama 호출
        response = ollama.chat(model=MODEL_NAME, messages=[{'role': 'user', 'content': prompt}])
        result = response['message']['content'].strip()
        
        # 1. 정규표현식으로 JSON 부분만 추출 (괄호 추가로 no such group 에러 방지)
        json_match = re.search(r'(\{.*\})', result, re.DOTALL)
        
        if json_match:
            json_str = json_match.group(1)
            
            # 2. 마크다운 태그 제거
            json_str = json_str.replace('```json', '').replace('```', '').strip()
            
            # 3. 따옴표 교정 (작은따옴표 -> 큰따옴표)
            # Expecting property name enclosed in double quotes 에러 방어
            if "'" in json_str and '"' not in json_str:
                json_str = json_str.replace("'", '"')
            
            # 4. JSON 파싱
            return json.loads(json_str)
        else:
            print(f"  ⚠️ JSON 패턴을 찾지 못함. (응답: {result[:30]}...)")
            return {"difficulty": 3, "tags": ["미분류"], "type": "미분류"}
    except Exception as e:
        # 에러가 발생해도 루프가 멈추지 않도록 기본값 반환
        print(f"  ❌ 에러 발생: {e}")
        return {"difficulty": 3, "tags": ["분석오류"], "type": "미분류"}
    
def load_jsonl_to_dict(path, key_field="question_no"):
    data_dict = {}
    if not path.exists():
        print(f"⚠️ 파일을 찾을 수 없음: {path}")
        return data_dict
    
    with open(path, 'r', encoding='utf-8') as f:
        for line in f:
            line = line.strip()
            if not line: continue
            try:
                item = json.loads(line)
                # 1. 여러 후보 키 중에서 번호를 찾음 (해설용: question_no, 문제용: question_number)
                raw_val = item.get(key_field) or item.get("question_no") or item.get("question_number")
                
                if raw_val is not None:
                    # 2. '1', 1.0, '01' 등 어떤 형식이든 순수 '정수 1'로 변환
                    clean_key = int(float(str(raw_val).strip()))
                    data_dict[clean_key] = item # 이제 숫자 1이 키가 됨
            except (json.JSONDecodeError, ValueError) as e:
                continue
    return data_dict

def extract_date_key(filename):
    """파일명에서 2020_11 같은 연도_월 키만 추출합니다."""
    # 숫자 4자리(연도)와 1~2자리(월)를 찾아냄
    match = re.search(r'(\d{4}).*?(\d{1,2})', filename)
    if match:
        return f"{match.group(1)}_{int(match.group(2)):02d}" # 예: 2020_11
    return None

def process_integration():
    q_files = list(question_DIR.glob("*.jsonl"))
    s_files = list(solution_DIR.glob("*.jsonl")) # 해설 파일 리스트 미리 확보
    
    # 해설 파일들을 { "2020_11": Path } 형태의 딕셔너리로 만듦
    s_map = {}
    for sf in s_files:
        key = extract_date_key(sf.name)
        if key: s_map[key] = sf

    for q_file_path in q_files:
        q_key = extract_date_key(q_file_path.name)
        s_file_path = s_map.get(q_key) # 연도_월 키로 해설 파일 찾기
        
        if not s_file_path:
            print(f"⚠️ 매칭되는 해설 파일을 찾을 수 없음: {q_file_path.name}")
            continue

        print(f"🧐 매칭 성공! [{q_key}] 통합 중: {q_file_path.name} <-> {s_file_path.name}")
        
        # 문제 파일 로드 (question_number 기준)
        questions = load_jsonl_to_dict(q_file_path, key_field="question_number")

        # 해설 파일 로드 (question_no 기준)
        solutions = load_jsonl_to_dict(s_file_path, key_field="question_no")
                
        final_rows = []
        # 문제 번호 순서대로 처리
        for q_num in sorted(questions.keys(), key=int):
            q_data = questions[q_num]
            s_data = solutions.get(q_num, {})

            # 1. 문제 데이터에서 year, month 가져오기
            year = q_data.get('year')
            month = q_data.get('month')
            
            # [추가] 만약 해설 데이터에만 정보가 있고 문제에 없다면 q_key(2020_11)에서 추출
            if not year or not month:
                year, month = q_key.split('_')

            q_text = ""
            # 1순위: assets 리스트 안의 text_llm (가장 깨끗한 라텍스 결과물)
            assets = q_data.get("assets")
            
            if isinstance(assets, list) and len(assets) > 0:
                # 첫 번째 에셋(dict)을 가져옴
                first_asset = assets[0]
                # 여기서 text_llm이 있는지 확인
                q_text = first_asset.get("text_llm", "").strip()
            
            if q_text:
                print(f"  ✨ {q_num}번: 라텍스 문제 텍스트 추출 성공")
            else:
                # 라텍스 없으면 원본 가져오기
                q_text = q_data.get("question_text", q_data.get("raw_text", ""))
                print(f"  ⚠️ {q_num}번: 라텍스 텍스트를 찾지 못해 원본 사용")
            s_text = s_data.get("latex_text", "")
            if not s_text:
                s_text = s_data.get("raw_text", "")

            # 2. LLM 분석 (난이도/태그)
            meta = get_difficulty_from_latex(q_text, s_text)
            
            # 3. 이미지 경로 생성 (분리된 year, month 활용)
            current_year_month = f"{year}_{int(month):02d}"
            img_idx = (int(q_num) - 1) // 4 + 1
            image_path = f"data/고1/해설/{current_year_month}/{img_idx:02d}.png"

            # 4. 최종 통합 데이터 구성 (year, month 개별 저장)
            integrated_item = {
                "id": f"{current_year_month}_{q_num}",
                "year": int(year),
                "month": int(month),
                "question_no": int(q_num),
                "question_text": q_text,
                "solution_text": s_text,
                "metadata": {
                    "difficulty": meta.get("difficulty", 3),
                    "tags": meta.get("tags", []),
                    "type": meta.get("type", "미분류"),
                    "image_path": image_path
                }
            }
            final_rows.append(integrated_item)
            print(f"  - {q_num}번 완료 GPU 냉각을 위해 잠시 휴식")
            time.sleep(5)

        # 최종 파일 저장
        output_path = FINAL_OUT_DIR / f"{q_file_path.stem}_final.jsonl"
        with open(output_path, 'w', encoding='utf-8') as f:
            for row in final_rows:
                f.write(json.dumps(row, ensure_ascii=False) + "\n")
        print(f"\n✅ 저장 완료: {output_path.name}")
        print("GPU냉각을 위해 잠시 휴식")
        time.sleep(10)

if __name__ == "__main__":
    process_integration()