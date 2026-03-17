import json
import ollama
from pathlib import Path
import shutil
import re
import time
import io
from PIL import Image

BASE_DIR = Path(r"C:\ai\source\soloproject")
JSONL_DIR = BASE_DIR / "output" / "jsonl" / "questions" / "g1"
# MODEL_NAME = 'qwen3.5:4b' # 4b 모델 권장
MODEL_NAME = 'qwen3.5:9b' # 2회차용

client = ollama.Client(timeout=None)

def clean_llm_result(text):
    if not text: return ""
    # 1. <think> 태그와 그 안의 내용 삭제
    text = re.sub(r'<think>.*?</think>', '', text, flags=re.DOTALL)
    # 2. ```latex ... ``` 마크다운 블록 제거 및 내용만 추출
    text = re.sub(r'```(?:latex|markdown|)\n?(.*?)\n?```', r'\1', text, flags=re.DOTALL)
    return text.strip()

def preprocess_and_merge_assets(data):
    """
    assets와 question_assets를 통합하고 중복을 제거하는 헬퍼 함수
    """
    # 두 자산 리스트 합치기
    all_assets = data.get('assets', []) + data.get('question_assets', [])
    
    unique_assets = {}
    for asset in all_assets:
        path = asset.get('path')
        if not path: continue
        
        if path not in unique_assets:
            unique_assets[path] = asset
        else:
            # 이미 존재한다면 text_llm이 있는 데이터를 우선 보존
            if not unique_assets[path].get('text_llm') and asset.get('text_llm'):
                unique_assets[path] = asset
    
    # 통합된 리스트를 assets에 넣고 question_assets는 제거
    data['assets'] = list(unique_assets.values())
    if 'question_assets' in data:
        del data['question_assets']
        
    return data

def repair_all_jsonls():
    jsonl_files = list(JSONL_DIR.glob("*.jsonl"))
    
    if not jsonl_files:
        print(f"JSONL 파일을 찾을 수 없습니다: {JSONL_DIR}")
        return

    for file_path in jsonl_files:
        print(f"\n--- 파일 작업 시작: {file_path.name} ---")
        
        # 백업 생성
        backup_path = file_path.with_suffix(".jsonl.bak")
        shutil.copy(file_path, backup_path)
        
        updated_lines = []
        repaired_count = 0
        skipped_count = 0
        processed_images = {} # 캐싱용
        
        with open(file_path, 'r', encoding='utf-8') as f:
            for line in f:
                data = json.loads(line)
                
                # [1단계] 데이터 구조 통합 실행
                data = preprocess_and_merge_assets(data)
                
                # [2단계] 통합된 assets에 대해서만 LLM 작업 수행
                if 'assets' in data:
                    for asset in data['assets']:
                        if asset.get('type') == 'question_image':

                            img_path_str = asset.get('path')
                            if not img_path_str: 
                                continue

                            if img_path_str in processed_images:
                                asset['text_llm'] = processed_images[img_path_str]
                                continue

                            # 1. 일단 기존에 작업된 텍스트를 가져옵니다.
                            text_result = asset.get('text_llm', "").strip()

                            # 2. [보정 조건 정의] 이 조건 중 하나라도 해당하면 'is_bad'는 True가 됩니다.
                            is_bad = (
                                not text_result or                  # 내용이 아예 없거나
                                len(text_result) < 40 or            # 내용이 너무 짧거나
                                "<think>" in text_result or          # <think> 태그가 남아있거나
                                "To solve" in text_result or        # "To solve this..." 같은 영문 서술이 있거나
                                "I understand" in text_result or      # 모델의 혼잣말이 섞여 있는 경우
                                "the " in text_result               # "the" 같은 불필요한 영어 단어가 포함된 경우
                            )

                            # 3. [스마트 스킵] 내용이 존재하면서 동시에 '나쁘지 않을(Good)' 때만 스킵합니다.
                            if text_result and not is_bad:
                                skipped_count += 1
                                # print(f"  [-] 건너뜀 (이미 양호함): {data['id']}")
                                continue
                            # 실제 이미지 경로 확인
                            full_img_path = BASE_DIR / asset['path']
                            if not full_img_path.exists():
                                print(f"  [!] 파일을 찾을 수 없음: {full_img_path}")
                                continue
                            
                            # LLM 호출
                            try:
                                success = False
                                current_result = ""
                                
                                # 원본 이미지 열기
                                with Image.open(full_img_path) as img:
                                    img = img.convert('RGB') # 안정성을 위해 RGB 변환
                                    width, height = img.size
                                    
                                    print(f"  [>] {data['id']} 변환 중... ({full_img_path.name})")
                                    for attempt in range(6): # 0:원본, 1~5:크롭 시도
                                        if attempt > 0:
                                            print(f"      [!] 재시도 {attempt}: 하단 {attempt*10}% 제거 후 다시 읽는 중...")
                                            # 하단 제거 (높이를 줄임)
                                            new_height = int(height * (1 - (attempt * 0.1)))
                                            active_img = img.crop((0, 0, width, new_height))
                                            time.sleep(2)
                                        else:
                                            active_img = img

                                        # 이미지를 바이트로 변환하여 Ollama에 전달
                                        img_byte_arr = io.BytesIO()
                                        active_img.save(img_byte_arr, format='PNG')
                                        img_bytes = img_byte_arr.getvalue()

                                        response = client.chat(
                                            model=MODEL_NAME,
                                            messages=[{
                                                'role': 'user',
                                                'content': """이미지의 수학 문제를 보고 LaTeX 형식의 텍스트로 변환하세요.
                                                [규칙]
                                                1. 다른 설명 없이 오직 문제 내용만 출력할 것. 해설은 다른 파일로 작성하기에 해설을 작성해선 안됨.
                                                2. 수식은 $...$ 또는 $$...$$를 사용할 것.
                                                3. 결과는 반드시 한국어 기반으로 작성할 것.
                                                4. 마크다운 기호(```latex 등)는 생략하고 텍스트만 출력할 것.""",
                                                'images': [img_bytes]
                                            }]
                                        )
                                        text_result = response['message']['content'].strip()
                                        cleaned_text = clean_llm_result(text_result)

                                        is_still_bad = (
                                            not cleaned_text or                      # 1. 내용이 아예 없거나
                                            "$" not in cleaned_text or               # 2. 수학 문제인데 수식 기호($)가 하나도 없거나
                                            len(re.findall(r'[가-힣]', cleaned_text)) < 2 or # 3. 한국어 설명이 너무 적거나 (수식만 덜렁 있는 경우 방지)
                                            "<think>" in cleaned_text or             # 4. 생각 과정이 포함되었거나
                                            any(word in cleaned_text for word in ["I understand", "Sure", "The image"]) # 5. 모델의 잡담 포함
                                        )

                                        if not is_still_bad:
                                            success = True
                                            asset['text_llm'] = cleaned_text
                                            processed_images[img_path_str] = cleaned_text
                                            break # 성공하면 루프 탈출
                                        time.sleep(1)

                                repaired_count += 1

                                if success:
                                    print(f"  [√] 성공: {data['id']} (시도 횟수: {attempt})")
                                else:
                                    print(f"  [X] 실패: {data['id']} (5회 크롭 후에도 품질 부적합)")
                                    print(f"  [실패 원인] 길이: {len(cleaned_text)}, 수식포함: {'$' in cleaned_text}")

                                print("  [wait] GPU 냉각을 위해 잠시 쉽니다...")
                                time.sleep(1.5)
                            except Exception as e:
                                print(f"  [!] 오류 발생 ({data['id']}): {e}")
                                
                        if repaired_count % 10 == 0 and repaired_count > 0:
                            print("--- 10문항 처리 완료: 10초간 GPU 휴식 ---")
                            time.sleep(10)
                
                updated_lines.append(data)
        
        # 파일 저장
        with open(file_path, 'w', encoding='utf-8') as f:
            for line in updated_lines:
                f.write(json.dumps(line, ensure_ascii=False) + '\n')
        
        print(f"--- {file_path.name} 완료! (보정: {repaired_count}건 / 건너뜀: {skipped_count}건) ---\n")

if __name__ == "__main__":
    repair_all_jsonls()