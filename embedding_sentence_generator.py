"""
임베딩용 문장 생성 모듈
인물 메타데이터를 기반으로 임베딩 모델에 최적화된 자연어 문장을 생성합니다.
"""

from typing import List, Dict, Any
from person_metadata import PersonMetadata
import re


class EmbeddingSentenceGenerator:
    """임베딩용 문장 생성 클래스"""
    
    def __init__(self):
        """초기화"""
        self.sentence_templates = {
            'basic_info': "{name}은(는) {era} {social_status}로 {birth_year}년에 태어나 {death_year}년에 사망했습니다.",
            'achievements': "{name}의 주요 업적은 {achievements}입니다.",
            'political': "{name}은(는) {political_faction}에 속하며 {school_thought}를 추종했습니다.",
            'career': "{name}은(는) {career_path}을 통해 {highest_position}까지 올랐습니다.",
            'region': "{name}은(는) {activity_region}에서 주로 활동했습니다.",
            'comprehensive': "{name}은(는) {era}의 {social_status}로 {activity_region}에서 활동하며 {achievements} 등의 업적을 남겼습니다. {political_faction}에 속해 {school_thought}를 추종했습니다."
        }
    
    def clean_text(self, text: str) -> str:
        """
        텍스트 정리 함수
        
        Args:
            text: 정리할 텍스트
            
        Returns:
            str: 정리된 텍스트
        """
        if not text or text == "-":
            return ""
        
        # 특수 문자 제거 및 정리
        cleaned = re.sub(r'[^\w\s가-힣(),·「」『』]', ' ', str(text))
        cleaned = re.sub(r'\s+', ' ', cleaned).strip()
        
        return cleaned
    
    def format_years(self, birth_year: int, death_year: int) -> str:
        """
        연도 포맷팅 함수
        
        Args:
            birth_year: 출생연도
            death_year: 사망연도
            
        Returns:
            str: 포맷팅된 연도 문자열
        """
        if birth_year and death_year:
            return f"{birth_year}년~{death_year}년"
        elif birth_year:
            return f"{birth_year}년 출생"
        elif death_year:
            return f"{death_year}년 사망"
        else:
            return "연도 미상"
    
    def generate_basic_sentence(self, person: PersonMetadata) -> str:
        """
        기본 정보 문장 생성
        
        Args:
            person: 인물 메타데이터
            
        Returns:
            str: 생성된 기본 정보 문장
        """
        template = self.sentence_templates['basic_info']
        
        return template.format(
            name=person.name_korean,
            era=self.clean_text(person.era),
            social_status=self.clean_text(person.social_status),
            birth_year=person.birth_year or "미상",
            death_year=person.death_year or "미상"
        )
    
    def generate_achievement_sentence(self, person: PersonMetadata) -> str:
        """
        업적 문장 생성
        
        Args:
            person: 인물 메타데이터
            
        Returns:
            str: 생성된 업적 문장
        """
        template = self.sentence_templates['achievements']
        
        achievements = self.clean_text(person.main_achievements)
        if not achievements:
            achievements = "구체적인 업적 기록이 없음"
        
        return template.format(
            name=person.name_korean,
            achievements=achievements
        )
    
    def generate_political_sentence(self, person: PersonMetadata) -> str:
        """
        정치적 성향 문장 생성
        
        Args:
            person: 인물 메타데이터
            
        Returns:
            str: 생성된 정치적 성향 문장
        """
        template = self.sentence_templates['political']
        
        faction = self.clean_text(person.political_faction)
        thought = self.clean_text(person.school_thought)
        
        if not faction:
            faction = "정치적 성향 미상"
        if not thought:
            thought = "사상적 배경 미상"
        
        return template.format(
            name=person.name_korean,
            political_faction=faction,
            school_thought=thought
        )
    
    def generate_career_sentence(self, person: PersonMetadata) -> str:
        """
        경력 문장 생성
        
        Args:
            person: 인물 메타데이터
            
        Returns:
            str: 생성된 경력 문장
        """
        template = self.sentence_templates['career']
        
        career_path = self.clean_text(person.career_path)
        position = self.clean_text(person.highest_position)
        
        if not career_path:
            career_path = "경력 경로 미상"
        if not position:
            position = "관직 미상"
        
        return template.format(
            name=person.name_korean,
            career_path=career_path,
            highest_position=position
        )
    
    def generate_region_sentence(self, person: PersonMetadata) -> str:
        """
        활동 지역 문장 생성
        
        Args:
            person: 인물 메타데이터
            
        Returns:
            str: 생성된 활동 지역 문장
        """
        template = self.sentence_templates['region']
        
        region = self.clean_text(person.activity_region)
        if not region:
            region = "활동 지역 미상"
        
        return template.format(
            name=person.name_korean,
            activity_region=region
        )
    
    def generate_comprehensive_sentence(self, person: PersonMetadata) -> str:
        """
        종합 문장 생성 (가장 상세한 문장)
        
        Args:
            person: 인물 메타데이터
            
        Returns:
            str: 생성된 종합 문장
        """
        template = self.sentence_templates['comprehensive']
        
        # 각 필드 정리
        era = self.clean_text(person.era)
        social_status = self.clean_text(person.social_status)
        region = self.clean_text(person.activity_region)
        achievements = self.clean_text(person.main_achievements)
        faction = self.clean_text(person.political_faction)
        thought = self.clean_text(person.school_thought)
        
        # 빈 값 처리
        if not era:
            era = "시대 미상"
        if not social_status:
            social_status = "신분 미상"
        if not region:
            region = "활동 지역 미상"
        if not achievements:
            achievements = "주요 업적 미상"
        if not faction:
            faction = "정치적 성향 미상"
        if not thought:
            thought = "사상적 배경 미상"
        
        return template.format(
            name=person.name_korean,
            era=era,
            social_status=social_status,
            activity_region=region,
            achievements=achievements,
            political_faction=faction,
            school_thought=thought
        )
    
    def generate_all_sentences(self, person: PersonMetadata) -> Dict[str, str]:
        """
        모든 유형의 문장 생성
        
        Args:
            person: 인물 메타데이터
            
        Returns:
            Dict[str, str]: 문장 유형별 생성된 문장들
        """
        sentences = {
            'basic': self.generate_basic_sentence(person),
            'achievement': self.generate_achievement_sentence(person),
            'political': self.generate_political_sentence(person),
            'career': self.generate_career_sentence(person),
            'region': self.generate_region_sentence(person),
            'comprehensive': self.generate_comprehensive_sentence(person)
        }
        
        return sentences
    
    def generate_embedding_sentence(self, person: PersonMetadata, 
                                  sentence_type: str = 'comprehensive') -> str:
        """
        임베딩용 문장 생성 (메인 함수)
        
        Args:
            person: 인물 메타데이터
            sentence_type: 문장 유형 ('basic', 'achievement', 'political', 
                          'career', 'region', 'comprehensive')
            
        Returns:
            str: 생성된 임베딩용 문장
        """
        if sentence_type == 'basic':
            return self.generate_basic_sentence(person)
        elif sentence_type == 'achievement':
            return self.generate_achievement_sentence(person)
        elif sentence_type == 'political':
            return self.generate_political_sentence(person)
        elif sentence_type == 'career':
            return self.generate_career_sentence(person)
        elif sentence_type == 'region':
            return self.generate_region_sentence(person)
        elif sentence_type == 'comprehensive':
            return self.generate_comprehensive_sentence(person)
        else:
            # 기본값은 종합 문장
            return self.generate_comprehensive_sentence(person)
    
    def generate_multiple_sentences(self, person: PersonMetadata, 
                                  max_sentences: int = 3) -> List[str]:
        """
        여러 문장 조합 생성 (더 풍부한 임베딩을 위해)
        
        Args:
            person: 인물 메타데이터
            max_sentences: 최대 문장 수
            
        Returns:
            List[str]: 생성된 문장 리스트
        """
        all_sentences = self.generate_all_sentences(person)
        
        # 우선순위에 따라 문장 선택
        priority_order = ['comprehensive', 'achievement', 'political', 'career', 'basic', 'region']
        
        selected_sentences = []
        for sentence_type in priority_order:
            if len(selected_sentences) >= max_sentences:
                break
            
            sentence = all_sentences.get(sentence_type, "")
            if sentence and sentence not in selected_sentences:
                selected_sentences.append(sentence)
        
        return selected_sentences


def main():
    """메인 함수 - 테스트용"""
    from person_metadata import PersonMetadataBuilder
    
    # 메타데이터 로드
    builder = PersonMetadataBuilder("csv/인물정리_v2_0902.csv")
    persons = builder.build_metadata()
    
    if persons:
        # 문장 생성기 초기화
        generator = EmbeddingSentenceGenerator()
        
        # 테스트 인물 선택
        test_person = builder.get_person_by_name("이순신")
        if test_person:
            print(f"테스트 인물: {test_person.name_korean}")
            print("=" * 50)
            
            # 모든 문장 유형 생성
            all_sentences = generator.generate_all_sentences(test_person)
            for sentence_type, sentence in all_sentences.items():
                print(f"{sentence_type}: {sentence}")
                print()
            
            # 여러 문장 조합 생성
            print("여러 문장 조합:")
            multiple_sentences = generator.generate_multiple_sentences(test_person, 3)
            for i, sentence in enumerate(multiple_sentences, 1):
                print(f"{i}. {sentence}")


if __name__ == "__main__":
    main()
