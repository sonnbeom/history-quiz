"""
유사도 측정 및 비교 분석 모듈
인물 간의 유사도를 분석하고 상세한 비교 결과를 제공합니다.
"""

import numpy as np
from typing import List, Dict, Tuple, Optional, Any
from sklearn.metrics.pairwise import cosine_similarity, euclidean_distances
from sklearn.decomposition import PCA
import matplotlib.pyplot as plt
import seaborn as sns
from person_metadata import PersonMetadata
from openai_embedding import PersonEmbeddingManager, OpenAIEmbeddingManager


class SimilarityAnalyzer:
    """유사도 분석 클래스"""
    
    def __init__(self, person_embedding_manager: PersonEmbeddingManager):
        """
        초기화
        
        Args:
            person_embedding_manager: 인물 임베딩 관리자
        """
        self.person_manager = person_embedding_manager
        self.embedding_manager = person_embedding_manager.embedding_manager
    
    def calculate_detailed_similarity(self, person1_name: str, 
                                    person2_name: str) -> Dict[str, Any]:
        """
        두 인물 간의 상세한 유사도 분석
        
        Args:
            person1_name: 첫 번째 인물 이름
            person2_name: 두 번째 인물 이름
            
        Returns:
            Dict[str, Any]: 상세한 유사도 분석 결과
        """
        embedding1 = self.person_manager.get_person_embedding(person1_name)
        embedding2 = self.person_manager.get_person_embedding(person2_name)
        
        if not embedding1 or not embedding2:
            return {"error": "임베딩을 찾을 수 없습니다."}
        
        # 다양한 유사도 메트릭 계산
        cosine_sim = cosine_similarity([embedding1], [embedding2])[0][0]
        euclidean_dist = euclidean_distances([embedding1], [embedding2])[0][0]
        
        # 정규화된 유클리드 거리 (0~1 스케일)
        max_possible_dist = np.sqrt(len(embedding1))
        normalized_euclidean = 1 - (euclidean_dist / max_possible_dist)
        
        # 맨하탄 거리
        manhattan_dist = np.sum(np.abs(np.array(embedding1) - np.array(embedding2)))
        max_manhattan = np.sum(np.abs(np.array(embedding1))) + np.sum(np.abs(np.array(embedding2)))
        normalized_manhattan = 1 - (manhattan_dist / max_manhattan) if max_manhattan > 0 else 0
        
        # 피어슨 상관계수
        correlation = np.corrcoef(embedding1, embedding2)[0][1]
        if np.isnan(correlation):
            correlation = 0
        
        return {
            "person1": person1_name,
            "person2": person2_name,
            "cosine_similarity": float(cosine_sim),
            "euclidean_distance": float(euclidean_dist),
            "normalized_euclidean": float(normalized_euclidean),
            "manhattan_distance": float(manhattan_dist),
            "normalized_manhattan": float(normalized_manhattan),
            "correlation": float(correlation),
            "overall_similarity": float(cosine_sim)  # 기본값은 코사인 유사도
        }
    
    def analyze_similarity_breakdown(self, person1_name: str, 
                                   person2_name: str) -> Dict[str, Any]:
        """
        유사도 분석의 세부 요소별 분석
        
        Args:
            person1_name: 첫 번째 인물 이름
            person2_name: 두 번째 인물 이름
            
        Returns:
            Dict[str, Any]: 세부 분석 결과
        """
        # 인물 데이터 가져오기
        person1_data = self.person_manager.person_embeddings.get(person1_name)
        person2_data = self.person_manager.person_embeddings.get(person2_name)
        
        if not person1_data or not person2_data:
            return {"error": "인물 데이터를 찾을 수 없습니다."}
        
        # 각 문장별 유사도 분석
        sentences1 = person1_data.get("sentences", [])
        sentences2 = person2_data.get("sentences", [])
        
        sentence_similarities = []
        for i, sent1 in enumerate(sentences1):
            for j, sent2 in enumerate(sentences2):
                emb1 = self.embedding_manager.get_embedding(sent1)
                emb2 = self.embedding_manager.get_embedding(sent2)
                
                if emb1 and emb2:
                    sim = cosine_similarity([emb1], [emb2])[0][0]
                    sentence_similarities.append({
                        "sentence1_index": i,
                        "sentence2_index": j,
                        "sentence1": sent1,
                        "sentence2": sent2,
                        "similarity": float(sim)
                    })
        
        # 평균 문장 유사도
        avg_sentence_sim = np.mean([s["similarity"] for s in sentence_similarities]) if sentence_similarities else 0
        
        return {
            "person1": person1_name,
            "person2": person2_name,
            "sentence_similarities": sentence_similarities,
            "average_sentence_similarity": float(avg_sentence_sim),
            "total_sentence_pairs": len(sentence_similarities)
        }
    
    def get_similarity_ranking(self, target_person: str, 
                             top_k: int = 10) -> List[Dict[str, Any]]:
        """
        특정 인물과의 유사도 순위
        
        Args:
            target_person: 기준 인물 이름
            top_k: 상위 k개 반환
            
        Returns:
            List[Dict[str, Any]]: 유사도 순위 리스트
        """
        similar_persons = self.person_manager.find_similar_persons(target_person, top_k)
        
        ranking = []
        for person_name, similarity in similar_persons:
            detailed_analysis = self.calculate_detailed_similarity(target_person, person_name)
            ranking.append({
                "rank": len(ranking) + 1,
                "person_name": person_name,
                "similarity": similarity,
                "detailed_analysis": detailed_analysis
            })
        
        return ranking
    
    def find_most_dissimilar(self, target_person: str, 
                           top_k: int = 5) -> List[Tuple[str, float]]:
        """
        특정 인물과 가장 다른 인물들 찾기
        
        Args:
            target_person: 기준 인물 이름
            top_k: 상위 k개 반환
            
        Returns:
            List[Tuple[str, float]]: (인물명, 차이도) 튜플 리스트
        """
        target_embedding = self.person_manager.get_person_embedding(target_person)
        if not target_embedding:
            return []
        
        # 모든 인물과의 유사도 계산
        similarities = []
        for person_name, data in self.person_manager.person_embeddings.items():
            if person_name != target_person:
                embedding = data["embedding"]
                similarity = self.embedding_manager.calculate_similarity(target_embedding, embedding)
                similarities.append((person_name, similarity))
        
        # 유사도가 낮은 순으로 정렬 (가장 다른 인물들)
        similarities.sort(key=lambda x: x[1])
        
        return similarities[:top_k]
    
    def analyze_similarity_distribution(self) -> Dict[str, Any]:
        """
        전체 인물들의 유사도 분포 분석
        
        Returns:
            Dict[str, Any]: 유사도 분포 분석 결과
        """
        person_names = list(self.person_manager.person_embeddings.keys())
        similarities = []
        
        # 모든 인물 쌍의 유사도 계산
        for i, person1 in enumerate(person_names):
            for j, person2 in enumerate(person_names[i+1:], i+1):
                sim = self.person_manager.compare_persons(person1, person2)
                similarities.append(sim)
        
        if not similarities:
            return {"error": "유사도 데이터가 없습니다."}
        
        similarities = np.array(similarities)
        
        return {
            "total_pairs": len(similarities),
            "mean_similarity": float(np.mean(similarities)),
            "median_similarity": float(np.median(similarities)),
            "std_similarity": float(np.std(similarities)),
            "min_similarity": float(np.min(similarities)),
            "max_similarity": float(np.max(similarities)),
            "percentile_25": float(np.percentile(similarities, 25)),
            "percentile_75": float(np.percentile(similarities, 75))
        }
    
    def create_similarity_matrix(self, person_names: Optional[List[str]] = None) -> np.ndarray:
        """
        유사도 행렬 생성
        
        Args:
            person_names: 포함할 인물 이름 리스트 (None이면 전체)
            
        Returns:
            np.ndarray: 유사도 행렬
        """
        if person_names is None:
            person_names = list(self.person_manager.person_embeddings.keys())
        
        n = len(person_names)
        similarity_matrix = np.zeros((n, n))
        
        for i, person1 in enumerate(person_names):
            for j, person2 in enumerate(person_names):
                if i == j:
                    similarity_matrix[i][j] = 1.0  # 자기 자신과의 유사도는 1
                else:
                    sim = self.person_manager.compare_persons(person1, person2)
                    similarity_matrix[i][j] = sim
        
        return similarity_matrix
    
    def visualize_similarity_matrix(self, person_names: Optional[List[str]] = None, 
                                  save_path: Optional[str] = None):
        """
        유사도 행렬 시각화
        
        Args:
            person_names: 포함할 인물 이름 리스트
            save_path: 저장할 파일 경로
        """
        if person_names is None:
            person_names = list(self.person_manager.person_embeddings.keys())
        
        similarity_matrix = self.create_similarity_matrix(person_names)
        
        plt.figure(figsize=(12, 10))
        sns.heatmap(similarity_matrix, 
                   xticklabels=person_names, 
                   yticklabels=person_names,
                   annot=True, 
                   cmap='YlOrRd', 
                   fmt='.3f',
                   cbar_kws={'label': '유사도'})
        
        plt.title('인물 간 유사도 행렬')
        plt.xlabel('인물')
        plt.ylabel('인물')
        plt.xticks(rotation=45, ha='right')
        plt.yticks(rotation=0)
        plt.tight_layout()
        
        if save_path:
            plt.savefig(save_path, dpi=300, bbox_inches='tight')
            print(f"유사도 행렬 이미지 저장: {save_path}")
        
        plt.show()
    
    def get_similarity_insights(self, person1_name: str, person2_name: str) -> Dict[str, Any]:
        """
        두 인물 간의 유사도에 대한 인사이트 제공
        
        Args:
            person1_name: 첫 번째 인물 이름
            person2_name: 두 번째 인물 이름
            
        Returns:
            Dict[str, Any]: 인사이트 결과
        """
        detailed_sim = self.calculate_detailed_similarity(person1_name, person2_name)
        breakdown = self.analyze_similarity_breakdown(person1_name, person2_name)
        
        similarity = detailed_sim.get("overall_similarity", 0)
        
        # 유사도 해석
        if similarity >= 0.8:
            interpretation = "매우 유사함"
            description = "두 인물은 매우 유사한 특성을 가지고 있습니다."
        elif similarity >= 0.6:
            interpretation = "유사함"
            description = "두 인물은 어느 정도 유사한 특성을 가지고 있습니다."
        elif similarity >= 0.4:
            interpretation = "보통"
            description = "두 인물은 일부 유사한 특성을 가지고 있습니다."
        elif similarity >= 0.2:
            interpretation = "다름"
            description = "두 인물은 상당히 다른 특성을 가지고 있습니다."
        else:
            interpretation = "매우 다름"
            description = "두 인물은 매우 다른 특성을 가지고 있습니다."
        
        return {
            "person1": person1_name,
            "person2": person2_name,
            "similarity_score": similarity,
            "interpretation": interpretation,
            "description": description,
            "detailed_analysis": detailed_sim,
            "breakdown_analysis": breakdown
        }


def main():
    """메인 함수 - 테스트용"""
    # 임베딩 관리자 초기화
    embedding_manager = OpenAIEmbeddingManager("text-embedding-3-small")
    person_manager = PersonEmbeddingManager(embedding_manager)
    
    # 유사도 분석기 초기화
    analyzer = SimilarityAnalyzer(person_manager)
    
    # 테스트용 인물 임베딩 생성
    test_persons = {
        "이순신": [
            "이순신은 조선 중기의 무신으로 1545년에 태어나 1598년에 사망했습니다.",
            "이순신의 주요 업적은 한산도 대첩, 명량대첩, 노량해전 등의 해전에서 승리한 것입니다.",
            "이순신은 성리학 기반의 무장으로 임진왜란 시기 조선 수군을 이끌었습니다."
        ],
        "정도전": [
            "정도전은 조선 전기의 문신으로 1342년에 태어나 1398년에 사망했습니다.",
            "정도전의 주요 업적은 조선경국전, 경제문감 저술과 한양 천도 기획입니다.",
            "정도전은 혁명파 사대부로 성리학을 추종했습니다."
        ],
        "이황": [
            "이황은 조선 중기의 문신으로 1501년에 태어나 1570년에 사망했습니다.",
            "이황의 주요 업적은 성학십도 저술과 도산서원 건립입니다.",
            "이황은 동인에 속해 성리학 주리론을 추종했습니다."
        ]
    }
    
    # 인물 임베딩 생성
    for person_name, sentences in test_persons.items():
        person_manager.create_person_embedding(person_name, sentences)
    
    # 유사도 분석 테스트
    print("유사도 분석 테스트")
    print("=" * 50)
    
    # 두 인물 간 상세 유사도 분석
    detailed_sim = analyzer.calculate_detailed_similarity("이순신", "정도전")
    print(f"이순신 vs 정도전 상세 분석:")
    for key, value in detailed_sim.items():
        print(f"  {key}: {value}")
    
    print("\n" + "=" * 50)
    
    # 유사도 순위
    ranking = analyzer.get_similarity_ranking("이순신", 3)
    print("이순신과 유사한 인물 순위:")
    for item in ranking:
        print(f"  {item['rank']}. {item['person_name']}: {item['similarity']:.4f}")
    
    print("\n" + "=" * 50)
    
    # 유사도 분포 분석
    distribution = analyzer.analyze_similarity_distribution()
    print("전체 유사도 분포:")
    for key, value in distribution.items():
        print(f"  {key}: {value}")
    
    print("\n" + "=" * 50)
    
    # 인사이트 분석
    insights = analyzer.get_similarity_insights("이순신", "정도전")
    print("유사도 인사이트:")
    print(f"  해석: {insights['interpretation']}")
    print(f"  설명: {insights['description']}")
    print(f"  유사도 점수: {insights['similarity_score']:.4f}")


if __name__ == "__main__":
    main()
