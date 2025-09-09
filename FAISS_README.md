# History Quiz FAISS 버전 - 무료 로컬 임베딩 시스템

FAISS와 로컬 임베딩 모델을 사용하여 OpenAI API 비용 없이 역사 인물 퀴즈 시스템을 구현합니다.

## 🆚 **OpenAI vs FAISS 비교**

| 특징 | OpenAI 임베딩 | FAISS + 로컬 모델 |
|------|---------------|-------------------|
| **비용** | 💰 API 호출 비용 발생 | 🆓 완전 무료 |
| **인터넷** | 🌐 항상 연결 필요 | 📱 오프라인 실행 가능 |
| **속도** | 🐌 API 호출 지연 | ⚡ 로컬 처리로 빠름 |
| **정확도** | 🎯 매우 높음 | 📊 양호함 (모델에 따라) |
| **데이터 보안** | 🔒 클라우드 전송 | 🛡️ 로컬 처리 |
| **설정** | 🔧 API 키만 필요 | ⚙️ 모델 다운로드 필요 |
| **확장성** | 📈 API 제한 있음 | 🚀 무제한 처리 |

## 🚀 **FAISS 버전 설치 및 설정**

### 1. 의존성 패키지 설치

```bash
# FAISS 버전용 패키지 설치
pip install faiss-cpu sentence-transformers torch pandas numpy scikit-learn

# 또는 requirements.txt 사용
pip install -r requirements.txt
```

### 2. 사용 가능한 한국어 임베딩 모델

#### **추천 모델들:**

1. **`jhgan/ko-sroberta-multitask`** (기본값)
   - 한국어 최적화 모델
   - 768차원
   - 다양한 한국어 태스크에 특화

2. **`snunlp/KR-SBERT-V40K-klueNLI-augSTS`**
   - 한국어 SBERT 모델
   - 768차원
   - 자연어 추론에 특화

3. **`sentence-transformers/paraphrase-multilingual-MiniLM-L12-v2`**
   - 다국어 지원 모델
   - 384차원 (가장 가벼움)
   - 한국어 포함 다국어 처리

### 3. GPU 사용 (선택사항)

GPU가 있다면 더 빠른 처리가 가능합니다:

```bash
# GPU 버전 설치
pip uninstall faiss-cpu
pip install faiss-gpu

# CUDA 지원 PyTorch 설치
pip install torch torchvision torchaudio --index-url https://download.pytorch.org/whl/cu118
```

## 🎮 **사용 방법**

### FAISS 버전 데모 실행

```bash
python history_quiz_faiss_demo.py
```

### 개별 모듈 테스트

```bash
# FAISS 임베딩 테스트
python faiss_embedding.py
```

## 📊 **FAISS 버전의 주요 기능**

### 1. **로컬 임베딩 처리** (`faiss_embedding.py`)

- **완전 무료**: API 비용 없음
- **오프라인 실행**: 인터넷 연결 불필요 (초기 모델 다운로드 후)
- **빠른 처리**: 로컬 GPU/CPU 사용
- **데이터 보안**: 모든 데이터가 로컬에서 처리

### 2. **FAISS 벡터 검색**

- **고속 검색**: 대용량 벡터 데이터베이스에서 빠른 검색
- **메모리 효율**: 압축된 인덱스 사용
- **확장성**: 수만 개의 벡터도 빠르게 처리

### 3. **캐싱 시스템**

- **임베딩 캐싱**: 한 번 계산된 임베딩 재사용
- **인덱스 저장**: FAISS 인덱스를 파일로 저장
- **자동 로드**: 프로그램 재시작 시 자동으로 캐시 로드

## 🔧 **설정 옵션**

### 모델 선택

```python
# 한국어 최적화 모델 (기본값)
faiss_manager = FAISSEmbeddingManager("jhgan/ko-sroberta-multitask")

# 한국어 SBERT 모델
faiss_manager = FAISSEmbeddingManager("snunlp/KR-SBERT-V40K-klueNLI-augSTS")

# 다국어 모델 (가벼움)
faiss_manager = FAISSEmbeddingManager("sentence-transformers/paraphrase-multilingual-MiniLM-L12-v2")
```

### GPU 사용 설정

```python
# GPU 사용 확인
import torch
print(f"CUDA 사용 가능: {torch.cuda.is_available()}")

# GPU 메모리 확인
if torch.cuda.is_available():
    print(f"GPU 메모리: {torch.cuda.get_device_properties(0).total_memory / 1024**3:.1f}GB")
```

## 📈 **성능 비교**

### 처리 속도 (예상)

| 작업 | OpenAI API | FAISS (CPU) | FAISS (GPU) |
|------|------------|-------------|-------------|
| 단일 임베딩 | 1-2초 | 0.1-0.3초 | 0.05-0.1초 |
| 100개 임베딩 | 100-200초 | 10-30초 | 5-10초 |
| 벡터 검색 | 0.5-1초 | 0.001-0.01초 | 0.0001-0.001초 |

### 정확도 (예상)

| 모델 | 한국어 정확도 | 다국어 지원 | 모델 크기 |
|------|---------------|-------------|-----------|
| OpenAI text-embedding-3-small | 95% | 우수 | 클라우드 |
| jhgan/ko-sroberta-multitask | 90% | 한국어 특화 | 1.2GB |
| KR-SBERT-V40K-klueNLI-augSTS | 88% | 한국어 특화 | 1.2GB |
| paraphrase-multilingual-MiniLM | 85% | 우수 | 470MB |

## 🐛 **문제 해결**

### 일반적인 문제

1. **모델 다운로드 실패**
   ```
   ❌ 모델 로딩 실패
   ```
   → 인터넷 연결 확인, 방화벽 설정 확인

2. **메모리 부족**
   ```
   ❌ CUDA out of memory
   ```
   → CPU 버전 사용: `faiss-cpu` 설치

3. **FAISS 인덱스 오류**
   ```
   ❌ FAISS 인덱스가 구축되지 않았습니다
   ```
   → 인덱스 파일 삭제 후 재구축

### 성능 최적화 팁

1. **GPU 사용**: CUDA 지원 GPU가 있다면 `faiss-gpu` 사용
2. **배치 처리**: 여러 텍스트를 한 번에 처리
3. **캐싱 활용**: 한 번 계산된 임베딩 재사용
4. **모델 선택**: 용도에 맞는 모델 선택 (속도 vs 정확도)

## 📁 **생성되는 파일들**

### 캐시 파일
- `faiss_embeddings_cache_*.pkl`: 임베딩 캐시
- `faiss_index_*.faiss`: FAISS 인덱스
- `person_names_*.json`: 인물 이름 매핑
- `person_faiss_embeddings.json`: 인물 임베딩 데이터

### 결과 파일
- `quiz_faiss_results.json`: 퀴즈 결과
- `automated_faiss_demo_results.json`: 자동화 데모 결과

## 🎯 **사용 시나리오**

### 1. **개발/테스트 환경**
- API 비용 없이 개발 및 테스트
- 오프라인 환경에서 작업
- 빠른 반복 개발

### 2. **프로덕션 환경**
- 대용량 데이터 처리
- 높은 처리량 요구
- 데이터 보안 중요

### 3. **교육/연구 목적**
- 임베딩 모델 학습
- 벡터 검색 알고리즘 연구
- 비용 없는 실험

## 🔄 **OpenAI 버전과의 호환성**

FAISS 버전은 OpenAI 버전과 동일한 인터페이스를 제공합니다:

```python
# OpenAI 버전
from openai_embedding import OpenAIEmbeddingManager
manager = OpenAIEmbeddingManager("text-embedding-3-small")

# FAISS 버전 (동일한 사용법)
from faiss_embedding import FAISSEmbeddingManager
manager = FAISSEmbeddingManager("jhgan/ko-sroberta-multitask")

# 동일한 메서드 사용
embedding = manager.get_embedding("텍스트")
similarity = manager.calculate_similarity(emb1, emb2)
```

## 📞 **지원**

FAISS 버전에 대한 문의사항이나 문제가 있으시면 이슈를 통해 연락해주세요.

---

**💡 팁**: 처음 실행 시 모델 다운로드에 시간이 걸릴 수 있습니다. 이후에는 캐시를 사용하여 빠르게 실행됩니다!
