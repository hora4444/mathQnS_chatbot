from django.db import models

class MockExamQuestion(models.Model):
    # 문제 식별 정보
    question_id = models.CharField(max_length=100, unique=True)
    grade = models.IntegerField(choices=[(1, '고1'), (2, '고2'), (3, '고3')])
    year = models.IntegerField()
    month = models.IntegerField()
    subject = models.CharField(max_length=50, null=True, blank=True) # 선택과목
    track = models.CharField(max_length=20, default="common")  # common/calculus/geometry/probability 등
    question_number = models.IntegerField(null=True, blank=True)  # 1~30

    # 텍스트 데이터 (수선 전/후)
    raw_text = models.TextField() # JSONL의 깨진 텍스트
    cleaned_latex = models.TextField(null=True, blank=True) # Nougat 수선 결과

    # 객관식 보기(없으면 [])
    choices = models.JSONField(default=list, blank=True)

    # 해설(텍스트/OCR + 이미지)
    solution_text = models.TextField(null=True, blank=True)
    solution_assets = models.JSONField(default=list, blank=True)  # ["assets/solutions/.../q01.png"]
    
    # 추가 메타데이터
    score = models.IntegerField(default=0)
    answer = models.CharField(max_length=10, null=True, blank=True)

    def __str__(self):
        return self.question_id

class QuestionImage(models.Model):
    # 1:N 관계 (한 문제에 여러 이미지가 있을 수 있음)
    question = models.ForeignKey(MockExamQuestion, on_delete=models.CASCADE, related_name='images')
    image_path = models.CharField(max_length=500) # 절대 경로 저장용
    
    def __str__(self):
        return f"Image for {self.question.question_id}"