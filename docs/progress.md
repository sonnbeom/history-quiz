# 진행 상황 — 한국사 인물 맞히기 퀴즈 콘텐츠

새 세션에서는 이 파일만 읽고 이어서 진행한다. 이번 범위는 **콘텐츠만** (개발은 나중에 따로).

## 작업 규칙 (사용자 지시)
- 한 번에 한 단계만. 단계가 끝나면 5줄 이내로 보고(한 일 / 바뀐 파일 / 확인 방법 / 다음 단계 / 문제·결정 필요)하고 멈춘다. 사용자 승인 전에는 다음 단계로 가지 않는다.
- 단계마다 이 파일을 갱신하고 git 커밋한다.
- 범위 밖 작업은 하지 않고 보고에 제안으로만 적는다.
- 속성 값은 content/schema.md 목록 안에서만. 목록 변경은 보고에 제안 → 승인 후 반영.

## 에이전트 (.claude/agents/)
| 에이전트 | 산출물 | 비고 |
|---|---|---|
| exam-researcher | content/research/exam_figures.md | WebSearch, WebFetch |
| figure-db-builder | content/schema.md, content/figures.json | model: sonnet |
| hint-writer | content/hints.json | 힌트 5개, 1번 가장 어려움 |
| fact-checker | content/review.md | WebSearch, WebFetch, 독립 검수, 파일 수정 안 함 |

참고: 에이전트 파일은 만든 세션에서는 바로 로드되지 않는다. 그 세션에서는 general-purpose 에이전트에게 해당 정의 파일을 읽고 따르게 해서 실행했다. 새 세션에서는 이름으로 바로 호출하면 된다.

## 단계
- [x] 1. exam-researcher → exam_figures.md (2026-11-26 완료, 사용자 승인 대기)
- [x] 2. figure-db-builder → schema.md (2026-11-26)
- [x] 3. 샘플 10명: figure-db-builder → hint-writer → fact-checker 반복 (2026-11-26 완료, 4차 검수 통과)
- [x] 4. 나머지 인물 힌트·검수 — 전 묶음 완료(119명 검수 통과, 2026-11-28)
- [ ] 5. 최종 정리: 전체 인물 수, 시대별 분포, 자주 나온 검수 문제, 사용자가 직접 확인할 출처 이용 조건

## TODO (새 세션은 여기부터) — 2026-11-28 기준
현재 작업 방식(2026-11-28 변경: 사용량은 고려하지 않음, 묶음 하나 ①~④를 끝까지 하고 채팅 보고 후 멈춤. 이전: 토큰을 아끼려 해서 **작업 하나(아래 체크 한 칸)씩만 하고 기록·커밋·push 후 멈춘다.** 사용자가 "이어서 해"라고 하면 다음 칸을 한다. 묶음 하나 ≈ 28만 토큰, 약 15분.)
각 묶음 순서: ① hint-writer(힌트 작성, url은 figures에 있는 것만, 없으면 null) → ② fact-checker 1차 검수(인물당 대백과 1항목만 열람, review.md에 "4단계 3x 1차 검수" 절) → ③ figure-db-builder(figures.json만)·hint-writer(hints.json만) 병렬 수정 → ④ fact-checker 재검수(지적 항목만, 새 열람 최소) → `node tools/validate.js` 오류 0 확인.

- [x] 3a 10명 (척준경~의천) — 검수 통과
- [x] 3b ① 힌트 작성 — ji-nul, yo-se, gwanghaegun, jeong-seon, bak-je-ga, chae-je-gong, yeonsangun, gwon-yul, sin-rip, gim-si-min
- [x] 3b ② 1차 검수 — review.md "4단계 3b 1차 검수" 절, B1-1~B1-13(미해결 12, B1-3 기록용)
- [x] 3b ③ 수정 → ④ 재검수 — 2차 B2-1~3 → 3차 확인, 미해결 0 (총 90명)
- [x] 3c·4a·4b ① 힌트 작성(29명, d1fd8f4)
- [x] 3c ②~④ — joseon-taejong, jo-byeong-gap, jeon-bong-jun, bak-gyu-su, an-chang-ho, yi-sang-jae, bang-jeong-hwan, na-cheol, jo-so-ang, an-hui-je
- [x] 4a ②~④ — hwang-sang-gyu, baek-nam-un, jang-myeon, na-un-gyu, choe-chi-won, balhae-mu, munju, jijeung, gim-chun-chu, munmu
- [x] 4b ②~④ — hyegong, gim-ji-jeong, wonseong, jinseong, goryeo-yejong, goryeo-wonjong, seunghwahu-on, gim-yun-hu, seong-hyeon
- [ ] 5단계 최종 정리 (위 5번 항목 내용) → 단계 5 체크 ← **다음 할 일**
- 남은 자잘한 것: 미등록 confusable id 4개(gim-heum-dol, danjong, choe-seung-ro, seo-hui), in-hu confusable gongmin 시기 불일치 — 5단계 때 정리 방침 결정.

## 결정된 사항
- 2026-11-26: 사용자가 단계별 승인 없이 끝까지 진행하라고 지시했다("승인 없이 계속, 모두 허용"). 단계마다 progress 기록과 커밋은 계속한다. schema 변경 제안은 오케스트레이터가 판단해 반영하고 여기에 기록한다.
- 대상: 수능 기준 학생. 목표 인물 100~150명. 인물당 힌트 5개.
- 조사 기간: 기본 3년(수능 2025~2027학년도 본시험, 한능검 심화 2024~2026 시행), 인물 100명 미만이면 5년. 6·9월 모의평가 제외.

## 단계별 기록
### 1단계 (2026-11-26)
- 3년 범위로 충분했다(5년 확장 안 함). 수능 3개 회차, 한능검 심화 12개 회차(제69~80회).
- 고유 인물 359명. 2회 이상 출제된 한국인은 119명이다.
- 상위 인물: 박정희 10회, 왕건 7회, 김구·김원봉·이승만·최우·영조 각 6회.
- 약점: 모든 자료가 제3자 사본이다. 공식 사이트는 자바스크립트로 막혀 받지 못했다.
  - 한능검 문항 번호 `?`, 추론한 인물 `추`는 대조가 필요하다.
  - 수능 2027 일부 인물은 정답표 없이 추론했다.
  - 경계 인물의 횟수는 ±1 정도 오차가 있을 수 있다.
- 저작권: 평가원 기출은 무단 복제·배포를 금지한다(공공누리 표시 없음). 국편은 "ALL RIGHTS RESERVED"다. 상업적 이용·변형 조건은 두 기관 모두 확인 못했다.
### 2단계 (2026-11-26)
- schema.md 작성. 시대 10개(사용자 예시 그대로), 국가 18개(통일신라는 신라로 통합), 역할 12개(복수 1~3), 분야 10개(복수 1~3), active_period는 대표 활동 기간(+approx).
- 오케스트레이터 결정: 외국인은 수록 제외 / 수록 후보 = 2회 이상 출제 한국인 119명, 빈도순 / 샘플 10명은 상위 빈도 중 시대 고르게 / hints.json 형식을 schema 7절에 명시.
- 보류된 변경 제안(schema.md 끝 "변경 제안"): 선사+고조선 통합, 개항기→근대 명칭, 후삼국 별도 시대. 기본안 유지.
### 3단계 (2026-11-26)
- 샘플 10명: bak-jeong-hui, wang-geon, gim-gu, choe-u, yeongjo, yi-seong-gye, joseon-gojong, jang-bo-go, gwanggaeto, gim-jeong-hui. 검수 4차에 걸쳐 미해결 0건.
- 결정: 고종 id는 joseon-gojong(고려 고종 충돌, schema 접두 목록에 추가). 별칭 "광무제" 삭제. 건국자 활동 시기 예외(즉위 전 대표 활동부터, schema 5절). 광개토 시기는 능비 기준 391~412.
- 검사 도구: `node tools/validate.js`(허용 값, 형식, achievements 2~5, 힌트 내 이름·별칭, confusable 양방향 경고).
- 흐름: builder는 sources url null → fact-checker가 열람 URL을 review.md에 제시 → builder·writer가 반영. 검수 지적에서 얻은 규칙은 에이전트 정의에 추가함.
- 개발 때 참고: 힌트 source.title에 인물명이 들어 있으므로 출처는 정답 공개 후에만 보여야 한다.
- 남은 수록 후보: 2회 이상 출제 한국인 119명 중 109명 → 30/30/30/19명 단위로 진행.
### 4단계 1묶음 (2026-11-26) — 30명, 검수 3차 통과 (총 40명)
- 추가: gim-won-bong, yi-seung-man, ji-cheong-cheon, gongmin, jeongjo, joseon-sukjong, sinmun, gim-gyu-sik, heungseon-daewongun, goryeo-sukjong, sejo, sejong, gwangjong, gyeon-hwon, gim-ok-gyun, min-gyeom-ho, yang-heon-su, gim-hong-jip, no-tae-u, gim-dae-jung, gim-yeong-sam, jeong-do-jeon, yi-ja-gyeom, sunjong, dae-jo-yeong, baekje-seong, goryeo-seongjong, gung-ye, eo-jae-yeon, heo-wi
- 결정: 힌트 출처 형식을 `source`→`sources` 배열로 변경(schema 7절, validate.js, hint-writer 반영). 흥선 대원군 별칭 "대원군" 유지(다른 대원군 수록 시 재검토). 같은 묘호 이름 표기는 "숙종(조선)"/"숙종(고려)" 식(고려 고종 수록 시 "고종"→"고종(조선)"으로 정리). 노태우는 대백과 인물 항목이 없어 대통령기록관 약력 + 대백과 사건 항목 사용.
- 다음 2묶음 후보: 3회 출제 중 표 40번(박상진)부터 — 양기탁, 민영환, 최익현, 안중근, 최영 등.
### 4단계 2묶음 (2026-11-27 완료, 검수 3차 통과 — 총 70명)
- figures.json에 30명 추가(총 70명): bak-sang-jin ~ jeong-jung-bu(exam_figures 3-1절 40~70번). 힌트 작성 중 사용량 한도로 중단 → 재개.
- 사용자 지시(2026-11-27): 사용량이 많아지면 멈춘다. /usage의 current가 50%를 넘으면 중단. (Claude는 /usage를 직접 볼 수 없으므로 묶음 단계마다 멈출 수 있는 지점에서 커밋해 둔다.)
- 2026-11-27: 사용자 전역 설정에 토큰 가드 추가(~/.claude/token-guard.sh, statusline.sh). 프롬프트 1개에서 5시간 사용량 +10%p 이상이거나 사용량이 50% 이상이면 자동으로 멈춘다. 멈추면 progress.md의 "다음 할 일"부터 이어간다. → 2026-11-27 사용자 요청으로 토큰 가드·상태줄 모두 삭제. 이제 자동 중단은 없고, 10명 단위로 끊어서 멈춘다.
- 2묶음 결정: 우거왕 start null(재위 시작 미상), 최익현 state 대한제국, 전두환은 노태우처럼 대통령기록관 약력+대백과 사건 항목. 4번에서 확정되던 6명은 4번을 겹치게 고침(양기탁·묘청·송시열은 허용). 최익현 5번 "흥선 대원군", 박영효 5번 "김홍집" 언급은 허용.
- 다음 할 일: 3묶음(exam_figures 3-1절의 남은 2회 이상 한국인 약 19명) figure-db-builder → hint-writer → fact-checker → 수정 반복 → 5단계 최종 정리.
### 4단계 3·4묶음 (진행 중, 2026-11-27)
- figures.json에 남은 49명(exam_figures 3-1절 #71~#119)을 한 번에 추가 → 총 119명. 외국인(#120~#124) 제외. (앞서 "남은 19명"이라 적은 것은 계산 착오: 샘플 이후 109명 = 30+30+30+19.)
- 30명 단위 규칙을 지키려고 힌트·검수는 나눠서 진행: 3묶음 = cheok-jun-gyeong ~ an-hui-je(30명), 4묶음 = hwang-sang-gyu ~ seong-hyeon(19명).
- 확인 필요: in-hu(인후) 이력 불확실 — 확인 못하면 수록 제외 검토. 미등록 confusable id 4개 남음(gim-heum-dol, danjong, choe-seung-ro, seo-hui).
- 사용자 지시(2026-11-27): 토큰을 아끼려고 작업을 더 잘게 나눈다. 한 번에 10명씩만 힌트 작성 → 검수 → 수정 → 커밋 → 멈춤. "이어서 해"를 들으면 다음 10명을 진행한다.
  - 3a: cheok-jun-gyeong, jo-wi-chong, yun-gwan, il-yeon, gim-bu-sik, yi-seung-hyu, choe-mu-seon, in-hu, gi-cheol, ui-cheon
  - 3b: ji-nul, yo-se, gwanghaegun, jeong-seon, bak-je-ga, chae-je-gong, yeonsangun, gwon-yul, sin-rip, gim-si-min
  - 3c: joseon-taejong, jo-byeong-gap, jeon-bong-jun, bak-gyu-su, an-chang-ho, yi-sang-jae, bang-jeong-hwan, na-cheol, jo-so-ang, an-hui-je
  - 4a: hwang-sang-gyu, baek-nam-un, jang-myeon, na-un-gyu, choe-chi-won, balhae-mu, munju, jijeung, gim-chun-chu, munmu
  - 4b: hyegong, gim-ji-jeong, wonseong, jinseong, goryeo-yejong, goryeo-wonjong, seunghwahu-on, gim-yun-hu, seong-hyeon
- 2026-11-27: 3a 힌트 작성 완료(힌트 있는 인물 80명, validate 오류 0). url은 전부 null. 검수 전.
  - 검수 때 볼 것: in-hu는 5번까지 기철과 구별 안 됨(figures 정보 부족) → 이력 확인해 5번 보강 또는 수록 제외. 최무선 4번이 한 단계 빨리 확정. 출처 표제 미확인: 진포대첩, 무신정변, 교장, 묘청의 난, 이자겸의 난.
- 2026-11-27: 3a 1차 검수 완료 → review.md "4단계 3a 1차 검수" 절, 지적 C1-1~C1-16 미해결. in-hu는 대백과 E0047135로 확인됨(몽골 출신 겁령구, 1250~1311) → 수록 유지, C1-12·C1-13대로 보강.
- 2026-11-27: 3a 완료 — C1-1~16 반영, 2차 검수 미해결 0건(총 80명 검수 통과). 참고: in-hu confusable의 gongmin은 시기 불일치, 충렬왕 수록 시 교체.
- 진행 방식(3a에서 확인): 힌트 작성(약 4.7만 토큰) → 1차 검수(약 12만, 인물당 대백과 1항목) → builder·writer 병렬 수정(약 7만) → 재검수(지적 항목만).
- 2026-11-27: 3b 힌트 작성 완료(힌트 있는 인물 90명, validate 오류 0). url은 bak-je-ga 4번 '규장각' 외 전부 null. 검수 전.
  - 검수 때 볼 것: figures에 없는 지식으로 쓴 부분(요세 4·5 법화경·참회, 광해군 4 분조, 연산군 4 사초·폐비, 권율 2·3·5 이치·도원수, 김시민 3 수령 등). 정선·연산군 4번은 confusable과 안 겹침. 권율·신립·김시민 1번 같은 문장(의도).
  - 기존 문제(범위 밖): yi-won-ik 힌트 3에 "광해군" 이름이 들어 있음 → gwanghaegun 수록으로 DB 인물 이름 노출. 3b 검수 때 같이 수정.
- 2026-11-28: 3b 1차 검수 완료 → B1-1~B1-13. 핵심: 10명 url 전부 null(B1-1, 열람 URL 표 있음), 광해군 3번에서 조기 확정(B1-2 순서 재배치안), yi-won-ik 3번 '광해군' 대체 문장(B1-9), 요세 5번 근거 없는 표현(B1-4), 권율 roles에 문신·관료 추가(B1-6), 표제 차이 탄금대전투·진주전투·백련사(B1-5·7·8, url 채울 때 본문 열람 필요), 박제가·정선 활동 시기(B1-10·11). validate 오류 0·경고 117(url 없음).
  - 환경 메모: PATH에 node·python 없음 → validate는 Adobe 동봉 node v20.18.0으로 실행됨. 홈(C:\Users\FAST)에서 시작한 세션은 프로젝트 에이전트를 이름으로 못 부름 → history-quiz 폴더에서 세션을 시작하거나 general-purpose에게 정의 파일을 읽혀 실행.
- (완료) 3b ③ 수정(builder는 figures.json만, writer는 hints.json만, 병렬) → ④ 재검수 → validate → 커밋 후 멈춤. 이후 3c → 4a → 4b → 5단계.
- 2026-11-28: 3b 완료 — B1 반영(builder·writer 병렬) → 2차 검수 B2-1~3(1차 수정안 결함 2건 포함: 광해군 3번 "17세기 전반"이 숙종·영조 모두 제거, 이원익 "재상에 올랐다" 사실 오류) → 오케스트레이터가 수정안대로 반영·3차 확인, 미해결 0. yo-se 결사 출처는 '백련사결사' E0070731(E0022065는 절 항목). validate 오류 0·경고 58.
  - 사용자 지시(2026-11-28): 사용량은 고려하지 않는다. 차근차근 묶음 단위로 끝내고 채팅으로 보고.
  - 환경: hint-writer는 셸이 없어 validate를 못 돌림 → 오케스트레이터가 돌린다.
- (완료) 3c ① 이후는 아래 기록 참고.
- 2026-11-28: 사용자 지시로 3c·4a·4b를 한꺼번에 진행. 병렬 방식: 묶음마다 에이전트가 content/tmp/의 임시 파일(hints_3c.json 등, review_3c.md 등)에만 쓰고 오케스트레이터가 hints.json·review.md에 합친다(같은 파일 동시 수정 충돌 방지). ③ 수정은 builder 1명·writer 1명이 세 묶음을 한꺼번에.
  - ① 힌트 작성 3건 병렬 완료 → hints.json에 합침(119명 전원), validate 오류 0·경고 172(대부분 url null). 커밋 d1fd8f4.
  - ② 1차 검수 3건(지적 번호 3c=D1, 4a=E1, 4b=F1)을 병렬로 시작했으나 세션 종료로 중단, 결과 파일 없음, 재개 불가 → 처음부터 다시 돌려야 한다.
  - 작성자들이 남긴 검수 확인 요청(나철 3번 을사오적 처단 계획, 전봉준 4번 청일전쟁 정세, 박규수 3번 이하 확정, 황상규 4번 김원봉과 폭탄 반입, 발해 무왕 4번 독자 연호, 원성왕 3·4번 김지정과 겹침, 예종 3번 사실상 확정, 김윤후 5번 노비 문서 소각, 표제 확인: '태종'·'예종'·'원종'·'승화후 온'·'백산상회' 등)은 검수 프롬프트에 다시 넣는다.
- (완료) 3c·4a·4b ②~④.
- 2026-11-28: 3c·4a·4b ②~④ 완료. 1차 지적 40건(D1 13, E1 16, F1 11)을 반영했다(56d9204). 2차에서 새 지적 3건(D2-1, E2-1, F2-1)이 나왔고 오케스트레이터가 반영했다. validate 오류 0건, 경고 0건. 29명 검수 통과로 총 119명 전원 통과.
  - 오케스트레이터 판단: 안창호 state "해당 없음", 김춘추 active_period는 재위 기간 654~661, 장면 국무총리는 연도 없이 "6·25 전쟁 중", 방정환 D1-8 대안 미채택.
  - 진행 중 5시간 사용량 한도에 걸려 hint-writer가 한 번 멈췄다. 한도가 풀린 뒤 이어서 끝냈다.
  - 5단계에 넣을 것: 출처는 정답 공개 뒤에만 보여 준다(전 인물 공통, title에 이름 있음). 미등록 confusable id 4개(gim-heum-dol, danjong, choe-seung-ro, seo-hui). in-hu와 공민왕의 시기 불일치. jo-so-ang 별칭 '조용은' 추가 여부(선택).
- 다음 할 일: 5단계 최종 정리.
