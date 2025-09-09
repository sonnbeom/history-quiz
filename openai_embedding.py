"""
OpenAI 임베딩 모델을 사용한 벡터화 모듈
인물 정보를 임베딩 벡터로 변환하고 유사도를 계산합니다.
"""

import openai
import numpy as np
import json
import pickle
from typing import List, Dict, Tuple, Optional, Any
from sklearn.metrics.pairwise import cosine_similarity
import os
from datetime import datetime
import time

# OpenAI API 키 설정 (환경변수에서 가져오기)
# export OPENAI_API_KEY="your-api-key-here"
openai.api_key = os.getenv("OPENAI_API_KEY")


class OpenAIEmbeddingManager:
    """OpenAI 임베딩 관리 클래스"""
    
    def __init__(self, model_name: str = "text-embedding-3-small"):
        """
        초기화
        
        Args:
            model_name: 사용할 OpenAI 임베딩 모델명
                       - text-embedding-3-small (가장 저렴, 1536차원)
                       - text-embedding-3-large (더 정확, 3072차원)
                       - text-embedding-ada-002 (이전 모델, 1536차원)
        """
        self.model_name = model_name
        self.embeddings_cache = {}  # 임베딩 캐시
        self.cache_file = f"embeddings_cache_{model_name.replace('-', '_')}.pkl"
        
        # 모델별 차원 정보
        self.model_dimensions = {
            "text-embedding-3-small": 1536,
            "text-embedding-3-large": 3072,
            "text-embedding-ada-002": 1536
        }
        
        # 캐시 로드
        self.load_cache()
    
    def get_embedding(self, text: str, use_cache: bool = True) -> Optional[List[float]]:
        """
        텍스트를 임베딩 벡터로 변환
        
        Args:
            text: 임베딩할 텍스트
            use_cache: 캐시 사용 여부
            
        Returns:
            Optional[List[float]]: 임베딩 벡터 또는 None
        """
        if not text or not text.strip():
            return None
        
        # 캐시 확인
        if use_cache and text in self.embeddings_cache:
            return self.embeddings_cache[text]
        
        try:
            # OpenAI API 호출
            response = openai.embeddings.create(
                model=self.model_name,
                input=text.strip()
            )
            
            embedding = response.data[0].embedding
            
            # 캐시에 저장
            if use_cache:
                self.embeddings_cache[text] = embedding
            
            return embedding
            
        except Exception as e:
            print(f"임베딩 생성 실패: {e}")
            return None
    
    def get_embeddings_batch(self, texts: List[str], 
                           batch_size: int = 100) -> List[Optional[List[float]]]:
        """
        여러 텍스트를 배치로 임베딩 변환
        
        Args:
            texts: 임베딩할 텍스트 리스트
            batch_size: 배치 크기
            
        Returns:
            List[Optional[List[float]]]: 임베딩 벡터 리스트
        """
        embeddings = []
        
        for i in range(0, len(texts), batch_size):
            batch_texts = texts[i:i + batch_size]
            batch_embeddings = []
            
            for text in batch_texts:
                embedding = self.get_embedding(text)
                batch_embeddings.append(embedding)
                
                # API 호출 제한을 위한 지연
                time.sleep(0.1)
            
            embeddings.extend(batch_embeddings)
            print(f"배치 처리 완료: {i + len(batch_texts)}/{len(texts)}")
        
        return embeddings
    
    def calculate_similarity(self, embedding1: List[float], 
                           embedding2: List[float]) -> float:
        """
        두 임베딩 벡터 간의 코사인 유사도 계산
        
        Args:
            embedding1: 첫 번째 임베딩 벡터
            embedding2: 두 번째 임베딩 벡터
            
        Returns:
            float: 코사인 유사도 (0~1)
        """
        if not embedding1 or not embedding2:
            return 0.0
        
        try:
            # numpy 배열로 변환
            vec1 = np.array(embedding1).reshape(1, -1)
            vec2 = np.array(embedding2).reshape(1, -1)
            
            # 코사인 유사도 계산
            similarity = cosine_similarity(vec1, vec2)[0][0]
            
            return float(similarity)
            
        except Exception as e:
            print(f"유사도 계산 실패: {e}")
            return 0.0
    
    def find_most_similar(self, target_embedding: List[float], 
                         candidate_embeddings: Dict[str, List[float]], 
                         top_k: int = 5) -> List[Tuple[str, float]]:
        """
        가장 유사한 임베딩들 찾기
        
        Args:
            target_embedding: 기준 임베딩 벡터
            candidate_embeddings: 후보 임베딩 벡터들 (이름: 벡터)
            top_k: 상위 k개 반환
            
        Returns:
            List[Tuple[str, float]]: (이름, 유사도) 튜플 리스트
        """
        similarities = []
        
        for name, embedding in candidate_embeddings.items():
            if embedding:
                similarity = self.calculate_similarity(target_embedding, embedding)
                similarities.append((name, similarity))
        
        # 유사도 순으로 정렬
        similarities.sort(key=lambda x: x[1], reverse=True)
        
        return similarities[:top_k]
    
    def save_cache(self):
        """임베딩 캐시를 파일로 저장"""
        try:
            with open(self.cache_file, 'wb') as f:
                pickle.dump(self.embeddings_cache, f)
            print(f"임베딩 캐시 저장 완료: {self.cache_file}")
        except Exception as e:
            print(f"캐시 저장 실패: {e}")
    
    def load_cache(self):
        """파일에서 임베딩 캐시 로드"""
        try:
            if os.path.exists(self.cache_file):
                with open(self.cache_file, 'rb') as f:
                    self.embeddings_cache = pickle.load(f)
                print(f"임베딩 캐시 로드 완료: {len(self.embeddings_cache)}개 항목")
        except Exception as e:
            print(f"캐시 로드 실패: {e}")
            self.embeddings_cache = {}
    
    def clear_cache(self):
        """임베딩 캐시 초기화"""
        self.embeddings_cache = {}
        if os.path.exists(self.cache_file):
            os.remove(self.cache_file)
        print("임베딩 캐시 초기화 완료")
    
    def get_cache_stats(self) -> Dict[str, Any]:
        """캐시 통계 정보 반환"""
        return {
            "cache_size": len(self.embeddings_cache),
            "model_name": self.model_name,
            "model_dimensions": self.model_dimensions.get(self.model_name, "unknown"),
            "cache_file": self.cache_file
        }


class PersonEmbeddingManager:
    """인물 임베딩 관리 클래스"""
    
    def __init__(self, embedding_manager: OpenAIEmbeddingManager):
        """
        초기화
        
        Args:
            embedding_manager: OpenAI 임베딩 관리자
        """
        self.embedding_manager = embedding_manager
        self.person_embeddings = {}  # 인물별 임베딩 저장
        self.embeddings_file = "person_embeddings.json"
    
    def create_person_embedding(self, person_name: str, 
                              sentences: List[str]) -> Optional[List[float]]:
        """
        인물의 여러 문장을 하나의 임베딩으로 결합
        
        Args:
            person_name: 인물 이름
            sentences: 인물에 대한 문장들
            
        Returns:
            Optional[List[float]]: 결합된 임베딩 벡터
        """
        if not sentences:
            return None
        
        # 각 문장의 임베딩 생성
        sentence_embeddings = []
        for sentence in sentences:
            embedding = self.embedding_manager.get_embedding(sentence)
            if embedding:
                sentence_embeddings.append(embedding)
        
        if not sentence_embeddings:
            return None
        
        # 평균 임베딩 계산 (문장들의 평균)
        avg_embedding = np.mean(sentence_embeddings, axis=0).tolist()
        
        # 인물 임베딩 저장
        self.person_embeddings[person_name] = {
            "embedding": avg_embedding,
            "sentences": sentences,
            "created_at": datetime.now().isoformat()
        }
        
        return avg_embedding
    
    def get_person_embedding(self, person_name: str) -> Optional[List[float]]:
        """
        인물의 임베딩 벡터 가져오기
        
        Args:
            person_name: 인물 이름
            
        Returns:
            Optional[List[float]]: 임베딩 벡터
        """
        if person_name in self.person_embeddings:
            return self.person_embeddings[person_name]["embedding"]
        return None
    
    def compare_persons(self, person1_name: str, person2_name: str) -> float:
        """
        두 인물 간의 유사도 계산
        
        Args:
            person1_name: 첫 번째 인물 이름
            person2_name: 두 번째 인물 이름
            
        Returns:
            float: 유사도 점수 (0~1)
        """
        embedding1 = self.get_person_embedding(person1_name)
        embedding2 = self.get_person_embedding(person2_name)
        
        if not embedding1 or not embedding2:
            return 0.0
        
        return self.embedding_manager.calculate_similarity(embedding1, embedding2)
    
    def find_similar_persons(self, target_person: str, 
                           top_k: int = 5) -> List[Tuple[str, float]]:
        """
        특정 인물과 유사한 인물들 찾기
        
        Args:
            target_person: 기준 인물 이름
            top_k: 상위 k개 반환
            
        Returns:
            List[Tuple[str, float]]: (인물명, 유사도) 튜플 리스트
        """
        target_embedding = self.get_person_embedding(target_person)
        if not target_embedding:
            return []
        
        # 모든 인물의 임베딩 수집
        candidate_embeddings = {}
        for person_name, data in self.person_embeddings.items():
            if person_name != target_person:
                candidate_embeddings[person_name] = data["embedding"]
        
        return self.embedding_manager.find_most_similar(
            target_embedding, candidate_embeddings, top_k
        )
    
    def save_embeddings(self):
        """인물 임베딩을 파일로 저장"""
        try:
            with open(self.embeddings_file, 'w', encoding='utf-8') as f:
                json.dump(self.person_embeddings, f, ensure_ascii=False, indent=2)
            print(f"인물 임베딩 저장 완료: {self.embeddings_file}")
        except Exception as e:
            print(f"임베딩 저장 실패: {e}")
    
    def load_embeddings(self):
        """파일에서 인물 임베딩 로드"""
        try:
            if os.path.exists(self.embeddings_file):
                with open(self.embeddings_file, 'r', encoding='utf-8') as f:
                    self.person_embeddings = json.load(f)
                print(f"인물 임베딩 로드 완료: {len(self.person_embeddings)}명")
        except Exception as e:
            print(f"임베딩 로드 실패: {e}")
            self.person_embeddings = {}
    
    def get_embedding_stats(self) -> Dict[str, Any]:
        """임베딩 통계 정보 반환"""
        return {
            "total_persons": len(self.person_embeddings),
            "model_name": self.embedding_manager.model_name,
            "model_dimensions": self.embedding_manager.model_dimensions.get(
                self.embedding_manager.model_name, "unknown"
            )
        }


def main():
    """메인 함수 - 테스트용"""
    # API 키 확인
    if not openai.api_key:
        print("OpenAI API 키가 설정되지 않았습니다.")
        print("환경변수 OPENAI_API_KEY를 설정해주세요.")
        return
    
    # 임베딩 관리자 초기화
    embedding_manager = OpenAIEmbeddingManager("text-embedding-3-small")
    person_manager = PersonEmbeddingManager(embedding_manager)
    
    # 테스트 문장들
    test_sentences = [
        "이순신은 조선 중기의 무신으로 1545년에 태어나 1598년에 사망했습니다.",
        "이순신의 주요 업적은 한산도 대첩, 명량대첩, 노량해전 등의 해전에서 승리한 것입니다.",
        "이순신은 성리학 기반의 무장으로 임진왜란 시기 조선 수군을 이끌었습니다."
    ]
    
    # 임베딩 생성 테스트
    print("임베딩 생성 테스트...")
    embedding = embedding_manager.get_embedding(test_sentences[0])
    if embedding:
        print(f"임베딩 차원: {len(embedding)}")
        print(f"임베딩 샘플: {embedding[:5]}...")
    
    # 인물 임베딩 생성 테스트
    print("\n인물 임베딩 생성 테스트...")
    person_embedding = person_manager.create_person_embedding("이순신", test_sentences)
    if person_embedding:
        print(f"인물 임베딩 차원: {len(person_embedding)}")
    
    # 유사도 계산 테스트
    print("\n유사도 계산 테스트...")
    if embedding and person_embedding:
        similarity = embedding_manager.calculate_similarity(embedding, person_embedding)
        print(f"유사도: {similarity:.4f}")
    
    # 캐시 통계
    print("\n캐시 통계:")
    stats = embedding_manager.get_cache_stats()
    for key, value in stats.items():
        print(f"{key}: {value}")
    
    # 임베딩 저장
    person_manager.save_embeddings()
    embedding_manager.save_cache()


if __name__ == "__main__":
    main()
