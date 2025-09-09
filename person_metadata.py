"""
인물 메타데이터 구축 모듈
CSV 데이터를 기반으로 인물의 상세 정보를 구조화하여 저장합니다.
"""

import pandas as pd
import json
from typing import Dict, List, Optional, Any
from dataclasses import dataclass, asdict
import re


@dataclass
class PersonMetadata:
    """인물 메타데이터 클래스"""
    # 기본 정보
    name_korean: str  # 한글 이름
    name_chinese: str  # 한자 이름
    courtesy_name: str  # 자
    pen_name: str  # 호
    birth_year: Optional[int]  # 출생연도
    death_year: Optional[int]  # 사망연도
    activity_period: str  # 활동 연대 범위
    era: str  # 활동 시대
    
    # 배경 정보
    family_clan: str  # 본관
    activity_region: str  # 활동 지역
    highest_position: str  # 최고 관직
    position_type: str  # 관직/직위명
    social_status: str  # 신분
    career_path: str  # 입신 경로
    
    # 업적 및 사상
    main_achievements: str  # 메인 업적
    political_faction: str  # 정치 파벌
    school_thought: str  # 학파/사상
    exam_rounds: str  # 출제 회차
    
    def to_dict(self) -> Dict[str, Any]:
        """딕셔너리로 변환"""
        return asdict(self)


class PersonMetadataBuilder:
    """인물 메타데이터 구축 클래스"""
    
    def __init__(self, csv_file_path: str):
        """
        초기화
        
        Args:
            csv_file_path: CSV 파일 경로
        """
        self.csv_file_path = csv_file_path
        self.persons_data: List[PersonMetadata] = []
        
    def load_csv_data(self) -> pd.DataFrame:
        """
        CSV 데이터 로드
        
        Returns:
            pandas DataFrame: 로드된 CSV 데이터
        """
        try:
            # CSV 파일 읽기 (인코딩 문제 해결)
            df = pd.read_csv(self.csv_file_path, encoding='utf-8-sig')
            
            # 빈 행 제거
            df = df.dropna(subset=['이름(한글)'])
            
            # 중복 제거 (같은 인물의 다른 출제 회차)
            df = df.drop_duplicates(subset=['이름(한글)', '이름(한자)'], keep='first')
            
            print(f"CSV 데이터 로드 완료: {len(df)}명의 인물")
            return df
            
        except Exception as e:
            print(f"CSV 데이터 로드 실패: {e}")
            return pd.DataFrame()
    
    def clean_data(self, value: Any) -> str:
        """
        데이터 정리 함수
        
        Args:
            value: 정리할 값
            
        Returns:
            str: 정리된 문자열
        """
        if pd.isna(value) or value == '-':
            return ""
        
        # 문자열로 변환
        cleaned = str(value).strip()
        
        # 특수 문자 정리
        cleaned = re.sub(r'[^\w\s가-힣(),·「」『』]', '', cleaned)
        
        return cleaned
    
    def parse_years(self, year_str: str) -> Optional[int]:
        """
        연도 파싱 함수
        
        Args:
            year_str: 연도 문자열
            
        Returns:
            Optional[int]: 파싱된 연도 또는 None
        """
        if pd.isna(year_str) or year_str == '-':
            return None
        
        try:
            # 소수점 제거 후 정수로 변환
            year = float(year_str)
            return int(year) if not pd.isna(year) else None
        except (ValueError, TypeError):
            return None
    
    def build_metadata(self) -> List[PersonMetadata]:
        """
        메타데이터 구축
        
        Returns:
            List[PersonMetadata]: 구축된 메타데이터 리스트
        """
        df = self.load_csv_data()
        
        if df.empty:
            print("CSV 데이터가 비어있습니다.")
            return []
        
        persons = []
        
        for _, row in df.iterrows():
            try:
                # 기본 정보 추출
                metadata = PersonMetadata(
                    name_korean=self.clean_data(row['이름(한글)']),
                    name_chinese=self.clean_data(row['이름(한자)']),
                    courtesy_name=self.clean_data(row['자']),
                    pen_name=self.clean_data(row['호']),
                    birth_year=self.parse_years(row['출생연도']),
                    death_year=self.parse_years(row['사망연도']),
                    activity_period=self.clean_data(row['활동 연대 범위']),
                    era=self.clean_data(row['활동 시대']),
                    family_clan=self.clean_data(row['본관']),
                    activity_region=self.clean_data(row['활동 지역']),
                    highest_position=self.clean_data(row['최고 관직']),
                    position_type=self.clean_data(row['관직/직위명']),
                    social_status=self.clean_data(row['신분']),
                    career_path=self.clean_data(row['입신 경로']),
                    main_achievements=self.clean_data(row['메인 업적']),
                    political_faction=self.clean_data(row['정치 파벌']),
                    school_thought=self.clean_data(row['학파/사상']),
                    exam_rounds=self.clean_data(row['출제 회차'])
                )
                
                # 유효한 데이터만 추가
                if metadata.name_korean:
                    persons.append(metadata)
                    
            except Exception as e:
                print(f"메타데이터 구축 실패 - 행 {_}: {e}")
                continue
        
        self.persons_data = persons
        print(f"메타데이터 구축 완료: {len(persons)}명")
        return persons
    
    def get_person_by_name(self, name: str) -> Optional[PersonMetadata]:
        """
        이름으로 인물 검색
        
        Args:
            name: 검색할 이름 (한글 또는 한자)
            
        Returns:
            Optional[PersonMetadata]: 찾은 인물 메타데이터 또는 None
        """
        for person in self.persons_data:
            if (person.name_korean == name or 
                person.name_chinese == name or
                name in person.name_korean or
                name in person.name_chinese):
                return person
        return None
    
    def get_random_person(self) -> Optional[PersonMetadata]:
        """
        랜덤 인물 선택
        
        Returns:
            Optional[PersonMetadata]: 랜덤으로 선택된 인물 메타데이터
        """
        import random
        
        if not self.persons_data:
            return None
        
        return random.choice(self.persons_data)
    
    def save_metadata(self, output_file: str = "person_metadata.json"):
        """
        메타데이터를 JSON 파일로 저장
        
        Args:
            output_file: 출력 파일명
        """
        try:
            metadata_dict = [person.to_dict() for person in self.persons_data]
            
            with open(output_file, 'w', encoding='utf-8') as f:
                json.dump(metadata_dict, f, ensure_ascii=False, indent=2)
            
            print(f"메타데이터 저장 완료: {output_file}")
            
        except Exception as e:
            print(f"메타데이터 저장 실패: {e}")
    
    def load_metadata(self, input_file: str = "person_metadata.json") -> List[PersonMetadata]:
        """
        JSON 파일에서 메타데이터 로드
        
        Args:
            input_file: 입력 파일명
            
        Returns:
            List[PersonMetadata]: 로드된 메타데이터 리스트
        """
        try:
            with open(input_file, 'r', encoding='utf-8') as f:
                metadata_dict = json.load(f)
            
            persons = []
            for data in metadata_dict:
                person = PersonMetadata(**data)
                persons.append(person)
            
            self.persons_data = persons
            print(f"메타데이터 로드 완료: {len(persons)}명")
            return persons
            
        except Exception as e:
            print(f"메타데이터 로드 실패: {e}")
            return []


def main():
    """메인 함수 - 테스트용"""
    # 메타데이터 구축
    builder = PersonMetadataBuilder("csv/인물정리_v2_0902.csv")
    persons = builder.build_metadata()
    
    if persons:
        # 랜덤 인물 선택 테스트
        random_person = builder.get_random_person()
        if random_person:
            print(f"\n랜덤 선택된 인물: {random_person.name_korean} ({random_person.name_chinese})")
            print(f"시대: {random_person.era}")
            print(f"주요 업적: {random_person.main_achievements}")
        
        # 메타데이터 저장
        builder.save_metadata()
        
        # 특정 인물 검색 테스트
        test_person = builder.get_person_by_name("이순신")
        if test_person:
            print(f"\n검색된 인물: {test_person.name_korean}")
            print(f"관직: {test_person.highest_position}")
            print(f"학파: {test_person.school_thought}")


if __name__ == "__main__":
    main()
