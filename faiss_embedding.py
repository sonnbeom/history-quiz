"""
FAISS 기반 임베딩 및 벡터 검색 모듈
로컬 임베딩 모델을 사용하여 비용 없이 벡터화 및 유사도 검색을 수행합니다.
"""

import numpy as np
import faiss
import pickle
import json
from typing import List, Dict, Tuple, Optional, Any
from sentence_transformers import SentenceTransformer
import torch
from datetime import datetime
import os
from sklearn.metrics.pairwise import cosine_similarity


class FAISSEmbeddingManager:
    """FAISS 기반 임베딩 관리 클래스"""
    
    def __init__(self, model_name: str = "jhgan/ko-sroberta-multitask"):
        """
        초기화
        
        Args:
            model_name: 사용할 한국어 임베딩 모델명
                       - jhgan/ko-sroberta-multitask (한국어 최적화, 768차원)
                       - snunlp/KR-SBERT-V40K-klueNLI-augSTS (한국어 SBERT, 768차원)
                       - sentence-transformers/paraphrase-multilingual-MiniLM-L12-v2 (다국어, 384차원)
        """
        self.model_name = model_name
        self.embeddings_cache = {}  # 임베딩 캐시
        self.cache_file = f"faiss_embeddings_cache_{model_name.replace('/', '_')}.pkl"
        
        # 모델 초기화
        print(f"임베딩 모델 로딩 중: {model_name}")
        try:
            self.model = SentenceTransformer(model_name)
            self.embedding_dim = self.model.get_sentence_embedding_dimension()
            print(f"✅ 모델 로딩 완료 - 차원: {self.embedding_dim}")
        except Exception as e:
            print(f"❌ 모델 로딩 실패: {e}")
            raise
        
        # FAISS 인덱스 초기화
        self.faiss_index = None
        self.person_names = []  # 인덱스와 매핑되는 인물 이름들
        self.index_file = f"faiss_index_{model_name.replace('/', '_')}.faiss"
        self.names_file = f"person_names_{model_name.replace('/', '_')}.json"
        
        # 캐시 및 인덱스 로드
        self.load_cache()
        self.load_faiss_index()
    
    def get_embedding(self, text: str, use_cache: bool = True) -> Optional[np.ndarray]:
        """
        텍스트를 임베딩 벡터로 변환
        
        Args:
            text: 임베딩할 텍스트
            use_cache: 캐시 사용 여부
            
        Returns:
            Optional[np.ndarray]: 임베딩 벡터 또는 None
        """
        if not text or not text.strip():
            return None
        
        # 캐시 확인
        if use_cache and text in self.embeddings_cache:
            return self.embeddings_cache[text]
        
        try:
            # 로컬 모델로 임베딩 생성
            embedding = self.model.encode(text, convert_to_numpy=True)
            
            # 캐시에 저장
            if use_cache:
                self.embeddings_cache[text] = embedding
            
            return embedding
            
        except Exception as e:
            print(f"임베딩 생성 실패: {e}")
            return None
    
    def get_embeddings_batch(self, texts: List[str]) -> List[Optional[np.ndarray]]:
        """
        여러 텍스트를 배치로 임베딩 변환
        
        Args:
            texts: 임베딩할 텍스트 리스트
            
        Returns:
            List[Optional[np.ndarray]]: 임베딩 벡터 리스트
        """
        try:
            # 배치 임베딩 생성
            embeddings = self.model.encode(texts, convert_to_numpy=True, show_progress_bar=True)
            
            # 캐시에 저장
            for text, embedding in zip(texts, embeddings):
                self.embeddings_cache[text] = embedding
            
            return [embedding for embedding in embeddings]
            
        except Exception as e:
            print(f"배치 임베딩 생성 실패: {e}")
            return [None] * len(texts)
    
    def build_faiss_index(self, person_embeddings: Dict[str, np.ndarray]):
        """
        FAISS 인덱스 구축
        
        Args:
            person_embeddings: 인물별 임베딩 딕셔너리
        """
        if not person_embeddings:
            print("❌ 임베딩 데이터가 없습니다.")
            return
        
        # 임베딩 벡터들을 배열로 변환
        embeddings_list = []
        self.person_names = []
        
        for person_name, embedding in person_embeddings.items():
            if embedding is not None:
                embeddings_list.append(embedding)
                self.person_names.append(person_name)
        
        if not embeddings_list:
            print("❌ 유효한 임베딩이 없습니다.")
            return
        
        # numpy 배열로 변환
        embeddings_array = np.vstack(embeddings_list).astype('float32')
        
        # FAISS 인덱스 생성 (L2 거리 기반)
        self.faiss_index = faiss.IndexFlatL2(self.embedding_dim)
        
        # 인덱스에 벡터 추가
        self.faiss_index.add(embeddings_array)
        
        print(f"✅ FAISS 인덱스 구축 완료: {len(embeddings_list)}개 벡터")
        
        # 인덱스 저장
        self.save_faiss_index()
    
    def search_similar(self, query_embedding: np.ndarray, top_k: int = 5) -> List[Tuple[str, float]]:
        """
        FAISS를 사용한 유사 벡터 검색
        
        Args:
            query_embedding: 검색할 쿼리 벡터
            top_k: 상위 k개 반환
            
        Returns:
            List[Tuple[str, float]]: (인물명, 거리) 튜플 리스트
        """
        if self.faiss_index is None or len(self.person_names) == 0:
            print("❌ FAISS 인덱스가 구축되지 않았습니다.")
            return []
        
        try:
            # 쿼리 벡터를 2D 배열로 변환
            query_vector = query_embedding.reshape(1, -1).astype('float32')
            
            # FAISS 검색 (거리 기반)
            distances, indices = self.faiss_index.search(query_vector, top_k)
            
            results = []
            for i, (distance, idx) in enumerate(zip(distances[0], indices[0])):
                if idx < len(self.person_names):
                    # 거리를 유사도로 변환 (0~1 스케일)
                    similarity = 1 / (1 + distance)
                    results.append((self.person_names[idx], similarity))
            
            return results
            
        except Exception as e:
            print(f"FAISS 검색 실패: {e}")
            return []
    
    def calculate_similarity(self, embedding1: np.ndarray, embedding2: np.ndarray) -> float:
        """
        두 임베딩 벡터 간의 코사인 유사도 계산
        
        Args:
            embedding1: 첫 번째 임베딩 벡터
            embedding2: 두 번째 임베딩 벡터
            
        Returns:
            float: 코사인 유사도 (0~1)
        """
        if embedding1 is None or embedding2 is None:
            return 0.0
        
        try:
            # 코사인 유사도 계산
            similarity = cosine_similarity([embedding1], [embedding2])[0][0]
            return float(similarity)
            
        except Exception as e:
            print(f"유사도 계산 실패: {e}")
            return 0.0
    
    def save_faiss_index(self):
        """FAISS 인덱스를 파일로 저장"""
        try:
            if self.faiss_index is not None:
                faiss.write_index(self.faiss_index, self.index_file)
                
                # 인물 이름들도 저장
                with open(self.names_file, 'w', encoding='utf-8') as f:
                    json.dump(self.person_names, f, ensure_ascii=False, indent=2)
                
                print(f"✅ FAISS 인덱스 저장 완료: {self.index_file}")
        except Exception as e:
            print(f"❌ FAISS 인덱스 저장 실패: {e}")
    
    def load_faiss_index(self):
        """파일에서 FAISS 인덱스 로드"""
        try:
            if os.path.exists(self.index_file) and os.path.exists(self.names_file):
                self.faiss_index = faiss.read_index(self.index_file)
                
                with open(self.names_file, 'r', encoding='utf-8') as f:
                    self.person_names = json.load(f)
                
                print(f"✅ FAISS 인덱스 로드 완료: {len(self.person_names)}개 벡터")
        except Exception as e:
            print(f"❌ FAISS 인덱스 로드 실패: {e}")
            self.faiss_index = None
            self.person_names = []
    
    def save_cache(self):
        """임베딩 캐시를 파일로 저장"""
        try:
            with open(self.cache_file, 'wb') as f:
                pickle.dump(self.embeddings_cache, f)
            print(f"✅ 임베딩 캐시 저장 완료: {self.cache_file}")
        except Exception as e:
            print(f"❌ 캐시 저장 실패: {e}")
    
    def load_cache(self):
        """파일에서 임베딩 캐시 로드"""
        try:
            if os.path.exists(self.cache_file):
                with open(self.cache_file, 'rb') as f:
                    self.embeddings_cache = pickle.load(f)
                print(f"✅ 임베딩 캐시 로드 완료: {len(self.embeddings_cache)}개 항목")
        except Exception as e:
            print(f"❌ 캐시 로드 실패: {e}")
            self.embeddings_cache = {}
    
    def clear_cache(self):
        """임베딩 캐시 초기화"""
        self.embeddings_cache = {}
        if os.path.exists(self.cache_file):
            os.remove(self.cache_file)
        print("✅ 임베딩 캐시 초기화 완료")
    
    def get_cache_stats(self) -> Dict[str, Any]:
        """캐시 통계 정보 반환"""
        return {
            "cache_size": len(self.embeddings_cache),
            "model_name": self.model_name,
            "embedding_dimensions": self.embedding_dim,
            "faiss_index_size": len(self.person_names) if self.person_names else 0,
            "cache_file": self.cache_file,
            "index_file": self.index_file
        }


class PersonFAISSEmbeddingManager:
    """인물 FAISS 임베딩 관리 클래스"""
    
    def __init__(self, faiss_manager: FAISSEmbeddingManager):
        """
        초기화
        
        Args:
            faiss_manager: FAISS 임베딩 관리자
        """
        self.faiss_manager = faiss_manager
        self.person_embeddings = {}  # 인물별 임베딩 저장
        self.embeddings_file = "person_faiss_embeddings.json"
    
    def create_person_embedding(self, person_name: str, 
                              sentences: List[str]) -> Optional[np.ndarray]:
        """
        인물의 여러 문장을 하나의 임베딩으로 결합
        
        Args:
            person_name: 인물 이름
            sentences: 인물에 대한 문장들
            
        Returns:
            Optional[np.ndarray]: 결합된 임베딩 벡터
        """
        if not sentences:
            return None
        
        # 각 문장의 임베딩 생성
        sentence_embeddings = []
        for sentence in sentences:
            embedding = self.faiss_manager.get_embedding(sentence)
            if embedding is not None:
                sentence_embeddings.append(embedding)
        
        if not sentence_embeddings:
            return None
        
        # 평균 임베딩 계산 (문장들의 평균)
        avg_embedding = np.mean(sentence_embeddings, axis=0)
        
        # 인물 임베딩 저장
        self.person_embeddings[person_name] = {
            "embedding": avg_embedding.tolist(),  # JSON 직렬화를 위해 리스트로 변환
            "sentences": sentences,
            "created_at": datetime.now().isoformat()
        }
        
        return avg_embedding
    
    def get_person_embedding(self, person_name: str) -> Optional[np.ndarray]:
        """
        인물의 임베딩 벡터 가져오기
        
        Args:
            person_name: 인물 이름
            
        Returns:
            Optional[np.ndarray]: 임베딩 벡터
        """
        if person_name in self.person_embeddings:
            embedding_list = self.person_embeddings[person_name]["embedding"]
            return np.array(embedding_list)
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
        
        if embedding1 is None or embedding2 is None:
            return 0.0
        
        return self.faiss_manager.calculate_similarity(embedding1, embedding2)
    
    def find_similar_persons(self, target_person: str, 
                           top_k: int = 5) -> List[Tuple[str, float]]:
        """
        특정 인물과 유사한 인물들 찾기 (FAISS 사용)
        
        Args:
            target_person: 기준 인물 이름
            top_k: 상위 k개 반환
            
        Returns:
            List[Tuple[str, float]]: (인물명, 유사도) 튜플 리스트
        """
        target_embedding = self.get_person_embedding(target_person)
        if target_embedding is None:
            return []
        
        # FAISS를 사용한 빠른 검색
        return self.faiss_manager.search_similar(target_embedding, top_k)
    
    def build_faiss_index(self):
        """인물 임베딩들로 FAISS 인덱스 구축"""
        if not self.person_embeddings:
            print("❌ 인물 임베딩이 없습니다.")
            return
        
        # 인물별 임베딩을 numpy 배열로 변환
        person_embeddings_dict = {}
        for person_name, data in self.person_embeddings.items():
            embedding_list = data["embedding"]
            person_embeddings_dict[person_name] = np.array(embedding_list)
        
        # FAISS 인덱스 구축
        self.faiss_manager.build_faiss_index(person_embeddings_dict)
    
    def save_embeddings(self):
        """인물 임베딩을 파일로 저장"""
        try:
            with open(self.embeddings_file, 'w', encoding='utf-8') as f:
                json.dump(self.person_embeddings, f, ensure_ascii=False, indent=2)
            print(f"✅ 인물 임베딩 저장 완료: {self.embeddings_file}")
        except Exception as e:
            print(f"❌ 임베딩 저장 실패: {e}")
    
    def load_embeddings(self):
        """파일에서 인물 임베딩 로드"""
        try:
            if os.path.exists(self.embeddings_file):
                with open(self.embeddings_file, 'r', encoding='utf-8') as f:
                    self.person_embeddings = json.load(f)
                print(f"✅ 인물 임베딩 로드 완료: {len(self.person_embeddings)}명")
        except Exception as e:
            print(f"❌ 임베딩 로드 실패: {e}")
            self.person_embeddings = {}
    
    def get_embedding_stats(self) -> Dict[str, Any]:
        """임베딩 통계 정보 반환"""
        return {
            "total_persons": len(self.person_embeddings),
            "model_name": self.faiss_manager.model_name,
            "embedding_dimensions": self.faiss_manager.embedding_dim,
            "faiss_index_built": self.faiss_manager.faiss_index is not None
        }


def main():
    """메인 함수 - 테스트용"""
    # FAISS 임베딩 관리자 초기화
    faiss_manager = FAISSEmbeddingManager("jhgan/ko-sroberta-multitask")
    person_manager = PersonFAISSEmbeddingManager(faiss_manager)
    
    # 테스트 문장들
    test_sentences = [
        "이순신은 조선 중기의 무신으로 1545년에 태어나 1598년에 사망했습니다.",
        "이순신의 주요 업적은 한산도 대첩, 명량대첩, 노량해전 등의 해전에서 승리한 것입니다.",
        "이순신은 성리학 기반의 무장으로 임진왜란 시기 조선 수군을 이끌었습니다."
    ]
    
    # 임베딩 생성 테스트
    print("임베딩 생성 테스트...")
    embedding = faiss_manager.get_embedding(test_sentences[0])
    if embedding is not None:
        print(f"임베딩 차원: {embedding.shape}")
        print(f"임베딩 샘플: {embedding[:5]}...")
    
    # 인물 임베딩 생성 테스트
    print("\n인물 임베딩 생성 테스트...")
    person_embedding = person_manager.create_person_embedding("이순신", test_sentences)
    if person_embedding is not None:
        print(f"인물 임베딩 차원: {person_embedding.shape}")
    
    # 유사도 계산 테스트
    print("\n유사도 계산 테스트...")
    if embedding is not None and person_embedding is not None:
        similarity = faiss_manager.calculate_similarity(embedding, person_embedding)
        print(f"유사도: {similarity:.4f}")
    
    # FAISS 인덱스 구축 테스트
    print("\nFAISS 인덱스 구축 테스트...")
    person_manager.build_faiss_index()
    
    # FAISS 검색 테스트
    if person_embedding is not None:
        print("\nFAISS 검색 테스트...")
        similar_persons = person_manager.find_similar_persons("이순신", 3)
        for person_name, similarity in similar_persons:
            print(f"  {person_name}: {similarity:.4f}")
    
    # 통계 정보
    print("\n통계 정보:")
    stats = faiss_manager.get_cache_stats()
    for key, value in stats.items():
        print(f"{key}: {value}")
    
    # 임베딩 저장
    person_manager.save_embeddings()
    faiss_manager.save_cache()


if __name__ == "__main__":
    main()
