import ollama
import os

image_path = r"C:\ai\source\soloproject\data\고1\raw_pngs\2020학년도3월학평(서울)해설.png"

# 이미지 파일 읽기
with open(image_path, 'rb') as f:
    image_data = f.read()

try:
    response = ollama.chat(
        model='qwen3-vl:4b',
        messages=[{
            'role': 'user',
            # 지시사항을 영어로 간결하게 변경
            'content': """
            이 이미지에서 1번 문항의 해설만 추출해줘. 
            1. [출제의도] 내용을 정확히 적을 것.
            2. 풀이 과정의 수식은 줄바꿈을 유지하며 상세히 적을 것.
            3. 수식 기호(분수, 곱셈 등)가 틀리지 않도록 주의할 것.
            """,
            'images': [image_data]
        }],
        stream=True 
    )

    print("AI 분석 중 (결과가 나오기까지 잠시 기다려주세요)...")
    print("-" * 30)

    # 응답이 오는지 확인하기 위한 플래그
    has_content = False
    for chunk in response:
        content = chunk.get('message', {}).get('content', '')
        if content:
            print(content, end='', flush=True)
            has_content = True

    if not has_content:
        print("\n[알림] 모델로부터 받은 텍스트가 없습니다. 프롬프트를 변경하거나 모델을 다시 로드해보세요.")

    print("\n" + "-" * 30)
    print("분석 완료!")

except Exception as e:
    print(f"\n오류 발생: {e}")