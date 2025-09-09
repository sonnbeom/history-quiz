"""
History Quiz 데모 스크립트
전체 과정을 통합하여 인물 메타데이터 구축부터 유사도 측정까지의 완전한 워크플로우를 보여줍니다.
"""

import os
import random
import json
from typing import List, Dict, Optional, Tuple
from person_metadata import PersonMetadataBuilder, PersonMetadata
from embedding_sentence_generator import EmbeddingSentenceGenerator
from openai_embedding import OpenAIEmbeddingManager, PersonEmbeddingManager
from similarity_analyzer import SimilarityAnalyzer


class HistoryQuizDemo:
    """History Quiz 데모 클래스"""
    
    def __init__(self, csv_file_path: str, openai_api_key: Optional[str] = None):
        """
        초기화
        
        Args:
            csv_file_path: CSV 파일 경로
            openai_api_key: OpenAI API 키 (환경변수에서 가져오지 않을 경우)
        """
        self.csv_file_path = csv_file_path
        
        # OpenAI API 키 설정
        if openai_api_key:
            os.environ["OPENAI_API_KEY"] = openai_api_key
        
        # 각 모듈 초기화
        self.metadata_builder = PersonMetadataBuilder(csv_file_path)
        self.sentence_generator = EmbeddingSentenceGenerator()
        self.embedding_manager = OpenAIEmbeddingManager("text-embedding-3-small")
        self.person_embedding_manager = PersonEmbeddingManager(self.embedding_manager)
        self.similarity_analyzer = SimilarityAnalyzer(self.person_embedding_manager)
        
        # 데이터 저장소
        self.persons_metadata: List[PersonMetadata] = []
        self.quiz_target_person: Optional[PersonMetadata] = None
        
    def setup_data(self) -> bool:
        """
        데이터 설정 및 초기화
        
        Returns:
            bool: 설정 성공 여부
        """
        print("=" * 60)
        print("History Quiz 데이터 설정 시작")
        print("=" * 60)
        
        try:
            # 1. CSV 데이터 로드 및 메타데이터 구축
            print("1. 인물 메타데이터 구축 중...")
            self.persons_metadata = self.metadata_builder.build_metadata()
            
            if not self.persons_metadata:
                print("❌ 메타데이터 구축 실패")
                return False
            
            print(f"✅ {len(self.persons_metadata)}명의 인물 메타데이터 구축 완료")
            
            # 2. 임베딩 생성 (샘플 인물들)
            print("\n2. 인물 임베딩 생성 중...")
            sample_persons = random.sample(self.persons_metadata, min(10, len(self.persons_metadata)))
            
            for person in sample_persons:
                # 임베딩용 문장 생성
                sentences = self.sentence_generator.generate_multiple_sentences(person, 3)
                
                # 임베딩 생성
                embedding = self.person_embedding_manager.create_person_embedding(
                    person.name_korean, sentences
                )
                
                if embedding:
                    print(f"  ✅ {person.name_korean} 임베딩 생성 완료")
                else:
                    print(f"  ❌ {person.name_korean} 임베딩 생성 실패")
            
            print(f"✅ {len(sample_persons)}명의 인물 임베딩 생성 완료")
            
            return True
            
        except Exception as e:
            print(f"❌ 데이터 설정 실패: {e}")
            return False
    
    def start_quiz(self) -> bool:
        """
        퀴즈 시작
        
        Returns:
            bool: 퀴즈 시작 성공 여부
        """
        print("\n" + "=" * 60)
        print("History Quiz 시작!")
        print("=" * 60)
        
        try:
            # 랜덤 인물 선택
            self.quiz_target_person = self.metadata_builder.get_random_person()
            
            if not self.quiz_target_person:
                print("❌ 퀴즈 대상 인물을 찾을 수 없습니다.")
                return False
            
            print(f"🎯 퀴즈 대상 인물이 선택되었습니다!")
            print(f"   이름: {self.quiz_target_person.name_korean} ({self.quiz_target_person.name_chinese})")
            print(f"   시대: {self.quiz_target_person.era}")
            print(f"   신분: {self.quiz_target_person.social_status}")
            
            # 대상 인물의 임베딩 생성 (아직 없다면)
            if not self.person_embedding_manager.get_person_embedding(self.quiz_target_person.name_korean):
                sentences = self.sentence_generator.generate_multiple_sentences(
                    self.quiz_target_person, 3
                )
                self.person_embedding_manager.create_person_embedding(
                    self.quiz_target_person.name_korean, sentences
                )
                print(f"✅ {self.quiz_target_person.name_korean} 임베딩 생성 완료")
            
            return True
            
        except Exception as e:
            print(f"❌ 퀴즈 시작 실패: {e}")
            return False
    
    def guess_person(self, guessed_name: str) -> Dict[str, any]:
        """
        인물 추측 및 유사도 분석
        
        Args:
            guessed_name: 추측한 인물 이름
            
        Returns:
            Dict[str, any]: 추측 결과 및 유사도 분석
        """
        print(f"\n🔍 '{guessed_name}' 추측 중...")
        
        try:
            # 추측한 인물 찾기
            guessed_person = self.metadata_builder.get_person_by_name(guessed_name)
            
            if not guessed_person:
                return {
                    "success": False,
                    "message": f"'{guessed_name}' 인물을 찾을 수 없습니다.",
                    "similarity": 0.0
                }
            
            # 추측한 인물의 임베딩 생성 (아직 없다면)
            if not self.person_embedding_manager.get_person_embedding(guessed_person.name_korean):
                sentences = self.sentence_generator.generate_multiple_sentences(
                    guessed_person, 3
                )
                self.person_embedding_manager.create_person_embedding(
                    guessed_person.name_korean, sentences
                )
            
            # 유사도 계산
            similarity = self.person_embedding_manager.compare_persons(
                self.quiz_target_person.name_korean, 
                guessed_person.name_korean
            )
            
            # 상세 분석
            detailed_analysis = self.similarity_analyzer.calculate_detailed_similarity(
                self.quiz_target_person.name_korean, 
                guessed_person.name_korean
            )
            
            # 정답 여부 확인
            is_correct = guessed_person.name_korean == self.quiz_target_person.name_korean
            
            result = {
                "success": True,
                "is_correct": is_correct,
                "guessed_person": {
                    "name": guessed_person.name_korean,
                    "chinese_name": guessed_person.name_chinese,
                    "era": guessed_person.era,
                    "achievements": guessed_person.main_achievements
                },
                "target_person": {
                    "name": self.quiz_target_person.name_korean,
                    "chinese_name": self.quiz_target_person.name_chinese,
                    "era": self.quiz_target_person.era,
                    "achievements": self.quiz_target_person.main_achievements
                },
                "similarity": similarity,
                "detailed_analysis": detailed_analysis,
                "message": self._generate_feedback_message(similarity, is_correct)
            }
            
            return result
            
        except Exception as e:
            return {
                "success": False,
                "message": f"추측 처리 중 오류 발생: {e}",
                "similarity": 0.0
            }
    
    def _generate_feedback_message(self, similarity: float, is_correct: bool) -> str:
        """
        피드백 메시지 생성
        
        Args:
            similarity: 유사도 점수
            is_correct: 정답 여부
            
        Returns:
            str: 피드백 메시지
        """
        if is_correct:
            return "🎉 정답입니다! 축하합니다!"
        
        if similarity >= 0.8:
            return f"🔥 매우 유사합니다! (유사도: {similarity:.2%})"
        elif similarity >= 0.6:
            return f"👍 유사합니다! (유사도: {similarity:.2%})"
        elif similarity >= 0.4:
            return f"🤔 어느 정도 유사합니다. (유사도: {similarity:.2%})"
        elif similarity >= 0.2:
            return f"😅 조금 다릅니다. (유사도: {similarity:.2%})"
        else:
            return f"❌ 많이 다릅니다. (유사도: {similarity:.2%})"
    
    def get_hint(self) -> str:
        """
        힌트 제공
        
        Returns:
            str: 힌트 메시지
        """
        if not self.quiz_target_person:
            return "퀴즈가 시작되지 않았습니다."
        
        hints = [
            f"시대: {self.quiz_target_person.era}",
            f"신분: {self.quiz_target_person.social_status}",
            f"활동 지역: {self.quiz_target_person.activity_region}",
            f"주요 업적: {self.quiz_target_person.main_achievements}",
            f"정치 파벌: {self.quiz_target_person.political_faction}",
            f"학파/사상: {self.quiz_target_person.school_thought}"
        ]
        
        # 빈 힌트 제거
        valid_hints = [hint for hint in hints if ":" in hint and hint.split(":", 1)[1].strip()]
        
        if valid_hints:
            return random.choice(valid_hints)
        else:
            return "추가 힌트를 제공할 수 없습니다."
    
    def get_similar_persons(self, top_k: int = 5) -> List[Dict[str, any]]:
        """
        유사한 인물들 찾기
        
        Args:
            top_k: 상위 k개 반환
            
        Returns:
            List[Dict[str, any]]: 유사한 인물 리스트
        """
        if not self.quiz_target_person:
            return []
        
        similar_persons = self.similarity_analyzer.get_similarity_ranking(
            self.quiz_target_person.name_korean, top_k
        )
        
        return similar_persons
    
    def show_quiz_summary(self):
        """퀴즈 요약 정보 표시"""
        if not self.quiz_target_person:
            print("퀴즈가 시작되지 않았습니다.")
            return
        
        print("\n" + "=" * 60)
        print("퀴즈 요약")
        print("=" * 60)
        
        print(f"🎯 정답: {self.quiz_target_person.name_korean} ({self.quiz_target_person.name_chinese})")
        print(f"📅 시대: {self.quiz_target_person.era}")
        print(f"👤 신분: {self.quiz_target_person.social_status}")
        print(f"🏛️ 활동 지역: {self.quiz_target_person.activity_region}")
        print(f"📚 주요 업적: {self.quiz_target_person.main_achievements}")
        print(f"🏛️ 정치 파벌: {self.quiz_target_person.political_faction}")
        print(f"📖 학파/사상: {self.quiz_target_person.school_thought}")
        
        # 유사한 인물들 표시
        print(f"\n🔍 유사한 인물들:")
        similar_persons = self.get_similar_persons(3)
        for person in similar_persons:
            print(f"  - {person['person_name']}: {person['similarity']:.2%}")
    
    def save_results(self, filename: str = "quiz_results.json"):
        """
        퀴즈 결과 저장
        
        Args:
            filename: 저장할 파일명
        """
        if not self.quiz_target_person:
            print("저장할 퀴즈 결과가 없습니다.")
            return
        
        results = {
            "quiz_target": {
                "name": self.quiz_target_person.name_korean,
                "chinese_name": self.quiz_target_person.name_chinese,
                "era": self.quiz_target_person.era,
                "achievements": self.quiz_target_person.main_achievements
            },
            "similar_persons": self.get_similar_persons(5),
            "total_persons": len(self.persons_metadata),
            "embedding_stats": self.person_embedding_manager.get_embedding_stats()
        }
        
        try:
            with open(filename, 'w', encoding='utf-8') as f:
                json.dump(results, f, ensure_ascii=False, indent=2)
            print(f"✅ 퀴즈 결과 저장 완료: {filename}")
        except Exception as e:
            print(f"❌ 결과 저장 실패: {e}")


def interactive_demo():
    """대화형 데모 실행"""
    print("🎮 History Quiz 대화형 데모")
    print("=" * 60)
    
    # CSV 파일 경로 확인
    csv_path = "csv/인물정리_v2_0902.csv"
    if not os.path.exists(csv_path):
        print(f"❌ CSV 파일을 찾을 수 없습니다: {csv_path}")
        return
    
    # OpenAI API 키 확인
    if not os.getenv("OPENAI_API_KEY"):
        print("❌ OpenAI API 키가 설정되지 않았습니다.")
        print("환경변수 OPENAI_API_KEY를 설정하거나 직접 입력해주세요.")
        api_key = input("OpenAI API 키를 입력하세요 (또는 Enter로 건너뛰기): ").strip()
        if not api_key:
            print("API 키 없이 데모를 실행할 수 없습니다.")
            return
    else:
        api_key = None
    
    # 데모 초기화
    demo = HistoryQuizDemo(csv_path, api_key)
    
    # 데이터 설정
    if not demo.setup_data():
        print("❌ 데이터 설정에 실패했습니다.")
        return
    
    # 퀴즈 시작
    if not demo.start_quiz():
        print("❌ 퀴즈 시작에 실패했습니다.")
        return
    
    print("\n🎯 이제 인물을 추측해보세요!")
    print("💡 '힌트'를 입력하면 힌트를 받을 수 있습니다.")
    print("💡 '요약'을 입력하면 퀴즈 요약을 볼 수 있습니다.")
    print("💡 '종료'를 입력하면 퀴즈를 종료합니다.")
    
    guess_count = 0
    max_guesses = 10
    
    while guess_count < max_guesses:
        print(f"\n📝 추측 {guess_count + 1}/{max_guesses}")
        user_input = input("인물 이름을 입력하세요: ").strip()
        
        if user_input.lower() == '종료':
            break
        elif user_input.lower() == '힌트':
            hint = demo.get_hint()
            print(f"💡 힌트: {hint}")
            continue
        elif user_input.lower() == '요약':
            demo.show_quiz_summary()
            continue
        elif not user_input:
            print("❌ 이름을 입력해주세요.")
            continue
        
        # 추측 처리
        result = demo.guess_person(user_input)
        
        if not result["success"]:
            print(f"❌ {result['message']}")
            continue
        
        guess_count += 1
        
        # 결과 표시
        print(f"\n📊 추측 결과:")
        print(f"   추측한 인물: {result['guessed_person']['name']} ({result['guessed_person']['chinese_name']})")
        print(f"   시대: {result['guessed_person']['era']}")
        print(f"   유사도: {result['similarity']:.2%}")
        print(f"   메시지: {result['message']}")
        
        # 정답 확인
        if result["is_correct"]:
            print(f"\n🎉 축하합니다! {guess_count}번 만에 정답을 맞추셨습니다!")
            break
        else:
            # 상세 분석 표시
            detailed = result["detailed_analysis"]
            print(f"   코사인 유사도: {detailed.get('cosine_similarity', 0):.4f}")
            print(f"   상관계수: {detailed.get('correlation', 0):.4f}")
    
    # 퀴즈 종료
    print(f"\n🏁 퀴즈 종료!")
    demo.show_quiz_summary()
    demo.save_results()


def automated_demo():
    """자동화된 데모 실행"""
    print("🤖 History Quiz 자동화 데모")
    print("=" * 60)
    
    # CSV 파일 경로 확인
    csv_path = "csv/인물정리_v2_0902.csv"
    if not os.path.exists(csv_path):
        print(f"❌ CSV 파일을 찾을 수 없습니다: {csv_path}")
        return
    
    # OpenAI API 키 확인
    if not os.getenv("OPENAI_API_KEY"):
        print("❌ OpenAI API 키가 설정되지 않았습니다.")
        print("환경변수 OPENAI_API_KEY를 설정해주세요.")
        return
    
    # 데모 초기화
    demo = HistoryQuizDemo(csv_path)
    
    # 데이터 설정
    if not demo.setup_data():
        print("❌ 데이터 설정에 실패했습니다.")
        return
    
    # 퀴즈 시작
    if not demo.start_quiz():
        print("❌ 퀴즈 시작에 실패했습니다.")
        return
    
    # 자동 추측 테스트
    print(f"\n🎯 자동 추측 테스트 시작")
    
    # 몇 가지 인물로 추측 테스트
    test_guesses = ["이순신", "정도전", "이황", "박지원", "김정희"]
    
    for guess in test_guesses:
        print(f"\n🔍 '{guess}' 추측 테스트...")
        result = demo.guess_person(guess)
        
        if result["success"]:
            print(f"   결과: {result['message']}")
            print(f"   유사도: {result['similarity']:.2%}")
            
            if result["is_correct"]:
                print(f"   🎉 정답!")
                break
        else:
            print(f"   ❌ {result['message']}")
    
    # 퀴즈 요약
    demo.show_quiz_summary()
    
    # 결과 저장
    demo.save_results("automated_demo_results.json")


def main():
    """메인 함수"""
    print("History Quiz 데모 프로그램")
    print("=" * 60)
    print("1. 대화형 데모 (사용자가 직접 추측)")
    print("2. 자동화 데모 (자동으로 추측 테스트)")
    print("3. 종료")
    
    while True:
        choice = input("\n선택하세요 (1-3): ").strip()
        
        if choice == "1":
            interactive_demo()
            break
        elif choice == "2":
            automated_demo()
            break
        elif choice == "3":
            print("프로그램을 종료합니다.")
            break
        else:
            print("❌ 잘못된 선택입니다. 1-3 중에서 선택해주세요.")


if __name__ == "__main__":
    main()
