import fitz  # PyMuPDF
import json
import re
from pathlib import Path

def extract_intent_from_pdf(pdf_path):
    doc = fitz.open(pdf_path)
    all_text = ""
    
    # 1. PDF 전체 텍스트를 블록 단위로 추출 (3열 꼬임 방지)
    for page in doc:
        blocks = page.get_text("blocks")
        # 블록의 좌표(y0, x0) 순서로 정렬하여 읽기 순서 최적화
        blocks.sort(key=lambda b: (b[1], b[0])) 
        for b in blocks:
            all_text += b[4] + "\n"
    
    # 2. 정규표현식으로 "[출제의도] ..." 문구 찾기
    # 문항 번호와 출제의도를 쌍으로 묶습니다.
    # 예: "1. [출제의도] 다항식의 계산..."
    pattern = re.compile(r'(\d+)\.\s*\[출제의도\]\s*(.*?)(?=\n\d+\.|\n\[|\Z)', re.DOTALL)
    matches = pattern.findall(all_text)
    
    intent_map = {}
    for num, intent in matches:
        # 줄바꿈 제거 및 공백 정리
        clean_intent = intent.replace('\n', ' ').strip()
        intent_map[int(num)] = clean_intent
        
    return intent_map

def save_intents_to_json(pdf_dir):
    pdf_dir = Path(pdf_dir)
    pdf_files = list(pdf_dir.glob("*.pdf"))
    
    for pdf_file in pdf_files:
        print(f"📄 처리 중: {pdf_file.name}")
        try:
            intents = extract_intent_from_pdf(pdf_file)
            
            # JSON 파일로 저장 (파일명_intent.json)
            output_path = pdf_file.with_name(f"{pdf_file.stem}_intent.json")
            with open(output_path, 'w', encoding='utf-8') as f:
                json.dump(intents, f, ensure_ascii=False, indent=4)
            
            print(f"✅ 추출 완료: {len(intents)}개 문항 발견 -> {output_path.name}")
        except Exception as e:
            print(f"❌ {pdf_file.name} 처리 중 오류: {e}")

if __name__ == "__main__":
    # 해설지 PDF 파일들이 모여있는 폴더 경로를 입력하세요
    target_dir = "./path_to_your_pdfs" 
    save_intents_to_json(target_dir)