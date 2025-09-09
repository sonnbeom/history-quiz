# History Quiz - 역사 인물 유사도 기반 퀴즈 시스템

CSV 데이터를 기반으로 역사 인물의 메타데이터를 구축하고, OpenAI 임베딩 모델을 사용하여 인물 간 유사도를 측정하는 퀴즈 시스템입니다.

## 🎯 프로젝트 개요

이 프로젝트는 다음과 같은 과정을 통해 역사 인물 퀴즈를 구현합니다:

1. **인물 메타데이터 구축**: CSV 데이터를 구조화된 메타데이터로 변환
2. **임베딩용 문장 생성**: 인물 정보를 자연어 문장으로 변환
3. **벡터화**: OpenAI 임베딩 모델을 사용하여 문장을 벡터로 변환
4. **유사도 측정**: 코사인 유사도 등을 사용하여 인물 간 유사도 계산
5. **퀴즈 시스템**: 랜덤 인물 선택 및 유사도 기반 피드백 제공

## 📁 프로젝트 구조

```
history_quiz/
├── csv/
│   └── 인물정리_v2_0902.csv          # 인물 데이터 CSV 파일
├── person_metadata.py                # 인물 메타데이터 구축 모듈
├── embedding_sentence_generator.py   # 임베딩용 문장 생성 모듈
├── openai_embedding.py              # OpenAI 임베딩 및 벡터화 모듈
├── similarity_analyzer.py           # 유사도 측정 및 분석 모듈
├── history_quiz_demo.py             # 통합 데모 스크립트
├── requirements.txt                 # 의존성 패키지 목록
├── README.md                        # 프로젝트 설명서
└── PRD_History_Quiz.md              # 제품 요구사항 문서
```

## 🚀 설치 및 설정

### 1. 의존성 패키지 설치

```bash
pip install -r requirements.txt
```

### 2. OpenAI API 키 설정

환경변수로 OpenAI API 키를 설정합니다:

```bash
# Windows
set OPENAI_API_KEY=your-api-key-here

# Linux/Mac
export OPENAI_API_KEY=your-api-key-here
```

또는 `.env` 파일을 생성하여 설정:

```
OPENAI_API_KEY=your-api-key-here
```

### 3. CSV 데이터 확인

`csv/인물정리_v2_0902.csv` 파일이 올바른 위치에 있는지 확인합니다.

## 🎮 사용 방법

### 대화형 데모 실행

```bash
python history_quiz_demo.py
```

실행 후 메뉴에서 "1. 대화형 데모"를 선택하면 사용자가 직접 인물을 추측할 수 있습니다.

### 자동화 데모 실행

```bash
python history_quiz_demo.py
```

실행 후 메뉴에서 "2. 자동화 데모"를 선택하면 자동으로 추측 테스트를 수행합니다.

### 개별 모듈 테스트

각 모듈을 개별적으로 테스트할 수 있습니다:

```bash
# 메타데이터 구축 테스트
python person_metadata.py

# 임베딩 문장 생성 테스트
python embedding_sentence_generator.py

# OpenAI 임베딩 테스트
python openai_embedding.py

# 유사도 분석 테스트
python similarity_analyzer.py
```

## 📊 주요 기능

### 1. 인물 메타데이터 구축 (`person_metadata.py`)

- CSV 데이터를 구조화된 메타데이터로 변환
- 인물 검색 및 랜덤 선택 기능
- JSON 형태로 메타데이터 저장/로드

### 2. 임베딩용 문장 생성 (`embedding_sentence_generator.py`)

- 인물 정보를 자연어 문장으로 변환
- 다양한 문장 유형 지원 (기본, 업적, 정치적, 경력, 지역, 종합)
- 여러 문장 조합 생성

### 3. OpenAI 임베딩 처리 (`openai_embedding.py`)

- OpenAI 임베딩 모델 사용 (text-embedding-3-small, text-embedding-3-large 등)
- 임베딩 캐싱으로 성능 최적화
- 배치 처리 지원

### 4. 유사도 분석 (`similarity_analyzer.py`)

- 다양한 유사도 메트릭 (코사인, 유클리드, 맨하탄, 상관계수)
- 상세한 유사도 분석 및 인사이트 제공
- 유사도 행렬 시각화

### 5. 통합 퀴즈 시스템 (`history_quiz_demo.py`)

- 랜덤 인물 선택
- 실시간 유사도 계산 및 피드백
- 힌트 시스템
- 결과 저장 및 분석

## 🔧 설정 옵션

### 임베딩 모델 선택

`openai_embedding.py`에서 사용할 임베딩 모델을 선택할 수 있습니다:

```python
# 가장 저렴한 모델 (1536차원)
embedding_manager = OpenAIEmbeddingManager("text-embedding-3-small")

# 더 정확한 모델 (3072차원)
embedding_manager = OpenAIEmbeddingManager("text-embedding-3-large")

# 이전 모델 (1536차원)
embedding_manager = OpenAIEmbeddingManager("text-embedding-ada-002")
```

### 문장 생성 옵션

`embedding_sentence_generator.py`에서 문장 유형을 선택할 수 있습니다:

- `basic`: 기본 정보 문장
- `achievement`: 업적 중심 문장
- `political`: 정치적 성향 문장
- `career`: 경력 중심 문장
- `region`: 활동 지역 문장
- `comprehensive`: 종합 문장 (기본값)

## 📈 성능 최적화

### 캐싱 시스템

- 임베딩 벡터 캐싱으로 API 호출 최소화
- 인물 메타데이터 캐싱
- 자동 저장/로드 기능

### 배치 처리

- 여러 텍스트를 한 번에 처리
- API 호출 제한 고려한 지연 처리

## 🐛 문제 해결

### 일반적인 문제

1. **OpenAI API 키 오류**
   ```
   ❌ OpenAI API 키가 설정되지 않았습니다.
   ```
   → 환경변수 `OPENAI_API_KEY`를 올바르게 설정했는지 확인

2. **CSV 파일을 찾을 수 없음**
   ```
   ❌ CSV 파일을 찾을 수 없습니다.
   ```
   → `csv/인물정리_v2_0902.csv` 파일이 올바른 위치에 있는지 확인

3. **임베딩 생성 실패**
   ```
   ❌ 임베딩 생성 실패
   ```
   → API 키가 유효한지, 인터넷 연결이 정상인지 확인

### 로그 확인

각 모듈은 상세한 로그를 제공하므로 오류 발생 시 로그를 확인하세요.

## 📝 라이선스

이 프로젝트는 교육 및 연구 목적으로 제작되었습니다.

## 🤝 기여

버그 리포트나 기능 제안은 이슈로 등록해주세요.

## 📞 문의

프로젝트에 대한 문의사항이 있으시면 이슈를 통해 연락해주세요.
