import os
import time
import json
from pathlib import Path
from dotenv import load_dotenv
from openai import OpenAI

load_dotenv()
UPSTAGE_API_KEY = os.getenv("UPSTAGE_KEY")

client = OpenAI(
    api_key=UPSTAGE_API_KEY,
    base_url="https://api.upstage.ai/v1/solar"
)

# 입력 폴더(기존 md)와 출력 폴더 설정
IN_ROOT = Path("output") / "jsonl" / "solutions" / "고1"
OUT_ROOT = Path("output") / "jsonl" / "solutions_latex" / "고1"
OUT_ROOT.mkdir(parents=True, exist_ok=True)

def fix_latex_with_solar(raw_text):
    if not raw_text.strip():
        return ""
        
    prompt = f"""
    다음은 OCR로 추출되어 깨진 글자가 포함된 수학 해설 텍스트입니다. 
    1. '' 같은 깨진 특수문자를 올바른 수학 기호($x^2$ 등)로 복원하세요.
    2. 모든 수식은 반드시 $...$ 또는 $$...$$를 사용하여 LaTeX 형식으로 작성하세요.
    3. 한글 설명과 문제 번호는 그대로 유지하세요.
    4. 텍스트에 문항 번호가 빠져 있다면 문맥에 맞게 번호(예: 30.)를 보충해 주세요.

    텍스트:
    {raw_text}
    """
    
    try:
        response = client.chat.completions.create(
            model="solar-pro3",
            messages=[{"role": "user", "content": prompt}],
            temperature=0
        )
        return response.choices[0].message.content
    except Exception as e:
        print(f"    [!] 에러 발생: {e}")
        return raw_text

def main():
    raw_files = list(IN_ROOT.glob("*_raw.jsonl"))

    for raw_path in raw_files:
        print(f"[*] 라텍싱 시작: {raw_path.name}")
        
        output_items = []
        
        with open(raw_path, "r", encoding="utf-8") as f:
            lines = f.readlines()

        for line in lines:
            data = json.loads(line)
            q_num = data["question_no"]
            raw_text = data["raw_text"]
            
            print(f"    - {q_num}번 문항 변환 중...", end="\r")
            
            # Solar 모델로 라텍싱 수행
            latex_text = fix_latex_with_solar(raw_text)
            
            # 결과 데이터 구성
            data["latex_text"] = latex_text
            data["status"] = "latex_completed"
            output_items.append(data)
            
            # API 호출 간격 조절 (필요 시)
            time.sleep(0.1)

        # 2. 결과 저장 (_latex.jsonl)
        output_path = OUT_ROOT / raw_path.name.replace("_raw.jsonl", "_latex.jsonl")
        with open(output_path, "w", encoding="utf-8") as f:
            for item in output_items:
                f.write(json.dumps(item, ensure_ascii=False) + "\n")
        
        print(f"\n[OK] {output_path.name} 저장 완료!")

if __name__ == "__main__":
    main()