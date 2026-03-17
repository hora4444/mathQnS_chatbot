import json
import re
import ollama
import time
from pathlib import Path

# --- 설정 ---
RAW_ROOT = Path("output/markdown")           # v2_res.md (원본) 폴더
FINAL_ROOT = Path("output/markdown/solution_latex")   # _final.md (라텍스본) 폴더
OUT_JSONL = Path("output/jsonl/solutions/g1")    # 최종 결과 저장 폴더

def split_by_question(content):
    """
    다양한 접두사(#, -, 공백)와 상관없이 '번호. [출제의도]'를 기준으로 분리합니다.
    """
    # 1. 정규표현식: 
    # [ \t\-\#]* : 줄 시작의 샵, 대시, 공백 등을 모두 허용
    # (\d+) : 문항 번호 캡처 (그룹 1)
    # \.\s*\[출제[ \s]*의도\] : 마침표 뒤에 '출제의도' (공백 허용) 키워드
    # (.*?) : 다음 문항 번호 전까지 모든 내용 캡처 (그룹 2)
    pattern = re.compile(
        r'(?:^|\n)[ \t\-\#]*(\d+)\.\s*\[출제[ \s]*의도\](.*?)(?=\n[ \t\-\#]*\d+\.\s*\[출제[ \s]*의도\]|\Z)', 
        re.DOTALL
    )
    
    matches = pattern.findall(content)
    
    # 2. 딕셔너리 생성 (중복 번호 발생 시 내용 병합 처리)
    extracted = {}
    for q_num_str, q_text in matches:
        q_num = int(q_num_str)
        # 만약 같은 번호가 또 나오면(파편화), 기존 내용 뒤에 붙여줌
        if q_num in extracted:
            extracted[q_num] += "\n\n" + q_text.strip()
        else:
            extracted[q_num] = q_text.strip()
            
    # 3. 디버깅을 위해 번호순으로 정렬된 리스트 확인
    sorted_nums = sorted(extracted.keys())
    print(f"    [DEBUG] 추출된 번호 리스트 (총 {len(sorted_nums)}개): {sorted_nums}")
    
    # 정수 키를 다시 문자열 키로 변환하여 반환 (기존 로직 호환)
    return {str(k): v for k, v in extracted.items()}

def is_hallucinated(raw_text, latex_text):
    """기초적인 품질 검사 (True일 경우 LLM 검수 필요)"""
    if not latex_text: return True
    if len(latex_text) < 10: return True  # 너무 짧으면 누락 가능성
    if latex_text.count('$') % 2 != 0: return True  # LaTeX 문법 오류
    return False

def verify_with_qwen(raw_text, latex_text, q_num, prev_num, latex_dict, max_q_num):

    if q_num == max_q_num:
        extra_fragments = []
        for k, v in latex_dict.items():
            if int(k) > max_q_num:
                extra_fragments.append(f"\n\n[추가 파편 - 원래 {q_num}번 내용]:\n{v}")
        if extra_fragments:
            latex_text += "".join(extra_fragments)

    """Ollama(Qwen3)를 사용한 최종 수식 및 번호 검수"""
    prompt = f"""
    당신은 수학 해설 전문 검토관입니다. 
    이전 문항이 {prev_num}번이었으므로, 이번 문항은 반드시 {q_num}번이어야 합니다.
    
    [원본 텍스트]: {raw_text}
    [1차 교정본 및 파편]: {latex_text}
    
    지시사항:
    1. 깨진 특수문자를 완벽한 LaTeX($...$)로 수정하세요.
    2. 문항 번호를 ### {q_num}. 으로 확정하세요.
    3. 원본에 없는 환각(가짜 해설)은 과감히 삭제하세요.
    4. 잘린 해설이나 파편화된 문장들은 [원본 텍스트]를 바탕으로 복원하고, 
    하나의 자연스러운 해설 문장으로 통합하여 재구성하세요.
    
    최종 수정된 마크다운 내용만 출력하세요.
    """
    try:
        response = ollama.chat(model='qwen3:8b', messages=[
            {'role': 'user', 'content': prompt}
        ])
        return response['message']['content'].strip()
    except Exception as e:
        print(f"      [!] Ollama 에러: {e}")
        return latex_text # 에러 시 일단 기존 것 유지

def save_as_jsonl(data_list, file_path):
    """리스트 데이터를 JSONL 파일로 저장"""
    file_path.parent.mkdir(parents=True, exist_ok=True)
    with open(file_path, "w", encoding="utf-8") as f:
        for entry in data_list:
            f.write(json.dumps(entry, ensure_ascii=False) + "\n")

def main():
    # 처리할 파일 목록 (라텍스본 기준)
    latex_files = list(FINAL_ROOT.glob("*_final.md"))
    
    for latex_path in latex_files:
        # 대응하는 원본 파일 찾기 (v2_res.md)
        raw_name = latex_path.name.replace("_final.md", ".md")
        raw_path = RAW_ROOT / raw_name
        
        if not raw_path.exists():
            print(f"[!] 원본 파일을 찾을 수 없음: {raw_name}")
            continue

        print(f"[*] 검증 및 변환 시작: {latex_path.name}")
        
        with open(raw_path, "r", encoding="utf-8") as f: raw_content = f.read()
        with open(latex_path, "r", encoding="utf-8") as f: latex_content = f.read()

        raw_dict = split_by_question(raw_content)
        latex_dict = split_by_question(latex_content)

        valid_q_nums = sorted([int(k) for k in raw_dict.keys()])
        max_q_num = max(valid_q_nums) # 보통 30

        final_jsonl_items = []
        last_num = 0

        for q_num in valid_q_nums:
            raw_text = raw_dict.get(str(q_num), "")
            latex_text = latex_dict.get(str(q_num), "")

            # 조건 없이 모든 문항을 Qwen3에게 검수받음
            print(f"    [*] {q_num}번 문항 전수 검수 중...")
            final_text = verify_with_qwen(raw_text, latex_text, q_num, last_num, latex_dict, max_q_num)

            final_jsonl_items.append({
                "id": f"{latex_path.stem}_{q_num}",
                "solution_text": final_text,
                "metadata": {"q_num": q_num, "source": latex_path.name}
            })
            last_num = q_num

            print("GPU과부화 방지용 10초 휴식")
            time.sleep(10)

        # 결과 저장
        out_path = OUT_JSONL / f"{latex_path.stem}.jsonl"
        save_as_jsonl(final_jsonl_items, out_path)
        print(f"[*] 완료: {out_path.name} ({len(final_jsonl_items)}개 문항)")

if __name__ == "__main__":
    main()