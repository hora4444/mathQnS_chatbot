import os
import requests
import re
import json
from pathlib import Path
from bs4 import BeautifulSoup
from dotenv import load_dotenv

load_dotenv()
UPSTAGE_API_KEY = os.getenv("UPSTAGE_KEY")

BASE_DIR = Path("data") / "고1" / "해설"
OUT_ROOT = Path("output") / "jsonl" / "solutions" /"고1"
OUT_ROOT.mkdir(parents=True, exist_ok=True)

def call_upstage_with_image(image_path):
    """
        qwen3-vl:8b로는 아쉬워서 변경
        3/9
        qwen3.5:9b로는 테스트 해보지 않았으나 현 시점에서 변경이 무의미하다고 판단
    """
    url = "https://api.upstage.ai/v1/document-ai/document-parse"
    headers = {"Authorization": f"Bearer {UPSTAGE_API_KEY}"}
    try:
        with open(image_path, "rb") as f:
            files = {"document": f}
            data = {"model": "document-parse-260128", "output_formats": '["html"]'}
            response = requests.post(url, headers=headers, files=files, data=data)
            response.raise_for_status()
            return response.json().get("content", {}).get("html", "")
    except Exception as e:
        print(f"\n    [!] API 오류: {e}")
        return ""

def split_combined_html(full_html):
    """[수정] 줄 중간에 번호가 있어도 정확히 절단하는 로직"""
    soup = BeautifulSoup(full_html, 'html.parser')
    
    # 모든 태그의 텍스트를 하나로 합침
    all_text = ""
    for element in soup.find_all(['p', 'div', 'span', 'br']):
        all_text += element.get_text(separator=" ", strip=True) + "\n"

    # 정규표현식으로 '번호. [출제의도]'의 모든 위치를 찾음
    pattern = re.compile(r'(\d+)\.\s*\[출제\s*의도\]')
    matches = list(pattern.finditer(all_text))
    extracted = {}

    for i in range(len(matches)):
        start_idx = matches[i].start()
        q_num = matches[i].group(1)
        
        # 다음 번호가 나오기 전까지가 현재 문항의 내용
        if i + 1 < len(matches):
            end_idx = matches[i+1].start()
            content = all_text[start_idx:end_idx].strip()
        else:
            content = all_text[start_idx:].strip()
            
        extracted[q_num] = content
    return extracted

def main():
    if not BASE_DIR.exists(): return

    for folder in BASE_DIR.iterdir():
        if not folder.is_dir(): continue
        
        print(f"\n[*] 폴더 분석: {folder.name}")
        image_files = sorted(list(folder.glob("*.png")))
        if not image_files: continue
        
        # [수정] 모든 이미지의 HTML을 먼저 다 합칩니다
        combined_html = ""
        for img_path in image_files:
            print(f"    - {img_path.name} 인식 중...", end="\r")
            combined_html += call_upstage_with_image(img_path) + "\n"
        
        # [수정] 다 합쳐진 HTML을 한 번에 쪼갭니다
        raw_dict = split_combined_html(combined_html)
        
        if raw_dict:
            output_file = OUT_ROOT / f"{folder.name}_raw.jsonl"
            with open(output_file, "w", encoding="utf-8") as f:
                for q_num in sorted(map(int, raw_dict.keys())):
                    line = {
                        "year_month": folder.name,
                        "question_no": q_num,
                        "raw_text": raw_dict[str(q_num)],
                        "status": "raw"
                    }
                    f.write(json.dumps(line, ensure_ascii=False) + "\n")
            print(f"\n    [OK] {output_file.name} 저장 완료!")

if __name__ == "__main__":
    main()