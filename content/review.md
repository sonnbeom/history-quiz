# 콘텐츠 검수 기록 (review.md)

- 검수자: 독립 검수(reviewer). figures.json, hints.json, schema.md는 읽기만 했다.
- 대조 출처: 한국민족문화대백과사전(encykorea.aks.ac.kr) 항목을 WebFetch로 직접 열어 확인. "열람"은 본문을 열어 해당 문장을 확인한 것, "검색만"은 검색 결과 요약으로만 본 것.

---

## 1차 검수 (bak-jeong-hui, wang-geon, gim-gu, choe-u, yeongjo, yi-seong-gye, gojong, jang-bo-go, gwanggaeto, gim-jeong-hui / 2026-11-26)

### 1-1. 기계 검사 결과

`node tools/validate.js` 실행 결과(종료 코드 0):

- 인물 10명, 힌트 있는 인물 10명, **오류 0건**.
- 경고 10건: 10명 모두 "확인된 source url 없음".
- 미등록 confusable id 26개: jeon-du-hwan, yi-seung-man, no-tae-u, gyeon-hwon, gung-ye, gwangjong, gim-won-bong, gim-gyu-sik, yeo-un-hyeong, choe-chung-heon, jeong-jung-bu, yi-ui-bang, jeongjo, joseon-sukjong, gwanghaegun, jeong-do-jeon, joseon-taejong, choe-yeong, heungseon-daewongun, sunjong, gim-ok-gyun, choe-chi-won, jangsu, gogugwon, jeong-seon, bak-je-ga (아직 DB에 없는 인물. 형식은 모두 id 규칙에 맞음).

validate.js는 이름·별칭 **전체 문자열**만 찾는다. 그래서 검수자가 따로 스크립트를 돌려 이름·별칭의 두 글자 조각, 띄어 쓴 낱말, 주요 고유 표지(광개토·영락·추사·완당·청해진·연잉·광무·백범·최이 등)를 힌트에서 찾았다. 걸린 것은 "대통령"(박정희 1·2), "고려"(왕건 5), "조선"(이성계 1), "황제"(고종 5), "청해진"(장보고 5), "영락"(광개토 5), "세한도"(김정희 5)였다. 모두 일반 명사이거나 업적 이름이고 이름·별칭·묘호·호가 아니어서 **항목 6 위반 없음**으로 판단했다. 눈으로도 50개 힌트를 모두 확인했다.

- era, state, roles, fields 값: 10명 모두 schema.md 허용 목록 안에 있음(항목 2 통과).
- exam_count: 10명 모두 exam_figures.md 3-1절 횟수와 일치(박정희 10, 왕건 7, 김구·최우·영조 6, 이성계·고종 5, 장보고·김정희 4, 광개토대왕 3).

### 1-2. 작가가 "통설"로 표시한 내용 대조 결과

| 인물 / 힌트 | 내용 | 판정 | 근거 |
|---|---|---|---|
| wang-geon 2 | 송악 호족 출신, 다른 군주 밑에서 장수 | 사실로 확인. 백과사전: "개성의 해상 세력가인 왕융(용건)과 한씨 부인 사이에서 태어났다", "궁예의 휘하에서 혁혁한 공을 세워 시중이 되었다". 다만 난이도 문제 있음(아래 표 W-3) | https://encykorea.aks.ac.kr/Article/E0059032 (열람) |
| yi-seong-gye 2 | 홍건적·왜구 격퇴 | 사실로 확인. "홍건적이 침입해 수도가 함락되자 이듬 해 정월 친병 2,000명을 거느리고 수도탈환작전에 참가", "황산대첩으로 알려질 만큼 혁혁한 것" | https://encykorea.aks.ac.kr/Article/E0059033 (열람) |
| jang-bo-go 3 | 산둥 법화원 건립 | 사실로 확인. 산둥 적산촌에 법화원을 세우고 지원해 재당 신라인의 정신적 중심이 됨 | https://encykorea.aks.ac.kr/Article/E0048474 (열람) |
| yeongjo 3 | 이인좌의 난이 왕위 정통성 문제 | 사실로 확인. 1728년 소론 강경파·남인 일부가 "왕으로서의 정통성을 인정하지 않았다"는 명분, 경종 독살 관련 혐의 | https://encykorea.aks.ac.kr/Article/E0037669 (열람) |
| yeongjo 5 | 군포 2필→1필 | 사실로 확인. "1750년(영조 26)에 균역법이 실시되어 종래 2필의 군포를 1필로 줄여 받도록" | https://encykorea.aks.ac.kr/Article/E0037669 (열람) |

### 1-3. 문제 항목

| # | 인물 id | 대상 | 점검 항목 | 이유 | 수정안 | 근거 URL | 상태 |
|---|---|---|---|---|---|---|---|
| G-1 | gojong | hint 1 | 4 | "19세기 후반부터 20세기 초까지 나라를 다스린 군주"는 고종 하나뿐이다. 순종은 1907~1910 재위라 19세기가 없고, 흥선 대원군은 군주가 아니다. **첫 힌트만으로 답이 정해진다.** | "왕실 종친 집안 출신으로, 개항을 전후한 시기 조정 정치의 중심에 있었다."(흥선 대원군도 들어맞음) 등 시기·지위가 겹치는 문장으로 교체 | https://encykorea.aks.ac.kr/Article/E0003939 | 미해결 |
| G-2 | gojong | hint 2 | 3, 4 | "재위 중에"가 들어가 군주로 한정되므로 강화도 조약·임오군란·갑신정변 시기의 군주 = 고종으로 확정된다. 1번을 고쳐도 2번에서 바로 확정. | "재위 중에"를 빼고 "강화도 조약, 임오군란, 갑신정변 같은 사건이 일어난 시기에 정치의 중심에 있었다."(흥선 대원군·김옥균도 들어맞음) | https://encykorea.aks.ac.kr/Article/E0003939 | 미해결 |
| G-3 | gojong | hint 3, achievements[0] | 1 (확인 필요) | "아버지가 섭정"은 백과사전 안에서도 표현이 갈린다. 고종 항목은 "조대비가 수렴청정을 하였다"(공식 섭정은 조대비, 대원군은 실권), 이하응 항목은 "섭정을 맡아 조선왕조를 실질적으로 통치". 틀렸다고 단정하지 않으나 정확하게 하려면 표현을 바꾸는 것이 안전. | "즉위 초에는 아버지가 실권을 쥐고 정치를 이끌었다." | https://encykorea.aks.ac.kr/Article/E0003939 , https://encykorea.aks.ac.kr/Article/E0046384 (검색만) | 미해결(확인 필요) |
| G-4 | gojong | hint 4 source | 8 | 원수부 설치는 표시한 출처인 백과사전 '고종' 항목에 나오지 않는다. 아관 파천과 1897년 2월 환궁은 나온다. | hint 4 source에 '원수부' 항목 추가 | https://encykorea.aks.ac.kr/Article/E0040727 (열람: "대한제국 때 설치되었던 황제 직속의 최고 군통수기관", 1899년 설치) | 미해결 |
| G-5 | gojong | name | 8 (확인 필요) | "고종"은 고려 고종(23대, 1213~1259, 최우 집권·몽골 침입기 왕)과 묘호가 같다. schema.md 접두 필수 목록에 고종이 빠져 있어 지금은 규칙 위반은 아니다. 하지만 고려 고종이 나중에 들어오거나 오답 입력 "고종"을 판정할 때 충돌한다. 백과사전도 두 항목을 따로 둔다(E0003938 고려 고종, E0003939 조선 고종). | schema.md 담당에 "고종"을 접두 필수 묘호 목록에 넣을지 변경 제안 요청. id는 바꾸지 않는다는 규칙이 있으므로 지금 결정 필요 | https://encykorea.aks.ac.kr/Article/E0003938 (검색만), https://encykorea.aks.ac.kr/Article/E0003939 | 미해결(확인 필요) |
| G-6 | gojong | aliases "광무제" | 8 (확인 필요) | "광무제"는 보통 후한 광무제(유수)를 가리킨다. 외국인은 수록 제외라 DB 충돌은 없지만 학생 입력 판정에서 혼동 여지가 있다. 한국사 서술에서는 "광무황제"가 더 흔하다. | "광무황제"로 바꾸거나 함께 둠 | 확인 못함(사전 항목 미열람) | 미해결(확인 필요) |
| G-7 | gojong | sources url | 8 | url null. | 아래 1-6 source url 수정안 | 아래 | 미해결 |
| W-1 | wang-geon | achievements[0] | 1 | "918년 고려를 세우고 송악(개경)을 도읍으로 삼았다"는 부정확하다. 918년 철원 포정전에서 즉위했고 개성 천도는 919년 1월이다. | "918년 고려를 세우고 이듬해 송악(개경)으로 도읍을 옮겼다." | https://encykorea.aks.ac.kr/Article/E0059032 (열람: "919년(태조 2) 1월에 개성으로 도읍을 옮겼다") | 미해결 |
| W-2 | wang-geon | sources | 8 | url null. 그리고 출처 제목 "한국민족문화대백과사전 '왕건'"은 실제 항목 제목과 다르다. 실제 항목 제목은 '태조'다. | title을 "한국민족문화대백과사전 '태조(고려)'"로 바꾸고 url 기입(1-6) | https://encykorea.aks.ac.kr/Article/E0059032 | 미해결 |
| W-3 | wang-geon | hint 2 | 3, 4 | 내용은 사실이다. 그런데 "송악"이 들어가 2번에서 사실상 확정된다. 견훤·궁예는 송악 호족이 아니어서 이 힌트에 들어맞지 않는다. 송악을 빼면 견훤(상주 호족 아자개의 아들, 신라 장수 출신)에게도 들어맞아 혼동 인물 구분 힌트가 된다. | "지방 호족 집안 출신으로, 처음에는 다른 군주 밑에서 장수로 활약했다." | https://encykorea.aks.ac.kr/Article/E0059032 | 미해결 |
| W-4 | wang-geon | hint 4 source | 8 | 만부교 낙타 사건, 발해 유민 수용, 북진 정책은 표시한 출처인 '태조' 항목 본문에 없다(본문에서 '발해·거란·낙타·북진'을 찾지 못함). 내용 자체는 사실이다. | hint 4 source를 '만부교사건' 항목으로 바꾸고, 발해 유민·북진은 '북진 정책' 항목을 추가 | https://encykorea.aks.ac.kr/Article/E0017609 (열람: 942년 낙타 50필, "낙타는 모두 만부교 아래에서 굶겨 죽였다"), https://encykorea.aks.ac.kr/Article/E0024747 (검색만), https://encykorea.aks.ac.kr/Article/E0011926 (검색만, 934년 대광현 우대) | 미해결 |
| P-1 | bak-jeong-hui | sources url | 8 | url null. | 1-6 | https://encykorea.aks.ac.kr/Article/E0021134 | 미해결 |
| P-2 | bak-jeong-hui | hint 4 source | 8 | 경부 고속도로는 '박정희' 항목에 나오지 않는다(경제개발 5개년 계획은 나옴: "1962년 제1차 경제개발5개년계획을 수립, 추진"). | hint 4 source에 '경부고속도로' 항목 추가(1968.2.1 착공, 1970.7.7 완공, 제2차 경제개발 5개년 계획 기간) | https://encykorea.aks.ac.kr/Article/E0002460 (열람) | 미해결 |
| Y-1 | yeongjo | sources url | 8 | url null. | 1-6 | https://encykorea.aks.ac.kr/Article/E0037669 | 미해결 |
| Y-2 | yeongjo | hint 4 source | 8 | 탕평비는 '영조' 항목 본문에 나오지 않는다(준천은 나옴: 1760년 개천 준설). | hint 4 source에 '탕평비' 항목 추가(1742년 성균관 반수교 위에 건립) | https://encykorea.aks.ac.kr/Article/E0058858 (열람) | 미해결 |
| L-1 | yi-seong-gye | sources | 8 | url null. 출처 제목 '이성계'도 실제 항목 제목('태조')과 다르다. | title "한국민족문화대백과사전 '태조(조선)'", url 기입(1-6) | https://encykorea.aks.ac.kr/Article/E0059033 | 미해결 |
| L-2 | yi-seong-gye | active_period | 1 (확인 필요) | 1388~1398은 schema.md 규칙과 맞지 않는다. schema는 "국왕: 재위 기간"이므로 1392~1398이어야 한다. 무장 활동까지 넣으면 시작은 홍건적 격퇴 때(1362년 개경 탈환)가 된다. 1388(위화도 회군)은 두 기준 어디에도 맞지 않는다. | figure-db-builder가 기준을 정해 1392~1398 또는 1362~1398 중 하나로 정하고 이유를 적음 | https://encykorea.aks.ac.kr/Article/E0059033 ("재위: 1392년~1398년") | 미해결(확인 필요) |
| L-3 | yi-seong-gye | hint 1+2 | 4 (확인 필요) | 1번만으로는 정도전·태종 등에게 들어맞아 문제없다. 하지만 1번(고려 말~조선 초)과 2번(홍건적·왜구를 물리친 무장)을 합치면 사실상 하나로 정해진다. 2번만 보면 최영에게도 들어맞지만 최영은 1388년에 죽어 1번에서 빠진다. 점진 공개 방식이면 2단계에서 확정되는 셈이다. | 1번을 "고려 말에 활동한 무장이다" 등으로 바꿔 최영과 겹치게 하거나, 지금대로 두되 난이도 곡선을 오케스트레이터가 판단 | https://encykorea.aks.ac.kr/Article/E0057469 (검색만, 최영의 홍건적 격퇴) | 미해결(확인 필요) |
| K-1 | gwanggaeto | sources url | 8 | url null. | 1-6 | https://encykorea.aks.ac.kr/Article/E0005057 | 미해결 |
| K-2 | gwanggaeto | hint 2 | 3, 4 | "4세기 말에서 5세기 초까지 재위하며 활발한 정복 활동"은 장수왕(413~491), 고국원왕(331~371), 고국양왕(384~391), 소수림왕(371~384)이 모두 빠지므로 2번에서 확정된다. 혼동 인물(장수왕·고국원왕)에게 들어맞지 않는다. | 연대를 빼고 "활발한 정복 활동으로 나라의 영토를 크게 넓혔다."(장수왕·태조왕·미천왕도 해당) | https://encykorea.aks.ac.kr/Article/E0005057 | 미해결 |
| K-3 | gwanggaeto | hint 5 source | 8 | "아들이 세운 거대한 비석"의 건립 주체(장수왕)는 '광개토왕' 항목에 나오지 않는다(414년 능비 건립만 나옴). | hint 5 source에 '광개토왕릉비' 항목 추가 | https://encykorea.aks.ac.kr/Article/E0005058 (열람: "414년(장수왕 3)", "제20대 장수왕에 의해 건립되었다") | 미해결 |
| J-1 | jang-bo-go | sources url | 8 | url null. | 1-6 | https://encykorea.aks.ac.kr/Article/E0048474 | 미해결 |
| J-2 | jang-bo-go | roles | 2 (확인 필요) | 백과사전은 장보고를 당·신라·일본 해상 무역을 장악한 통일신라 상인(무역가)으로 정의한다. "무신·장수"는 청해진대사로서 맞지만 무역가 면이 역할에 드러나지 않는다. schema는 상인을 "기타"로 둔다. 허용 값 위반은 아니다. | ["무신·장수", "기타"] 검토. 바꾸지 않아도 무방하고 fields "경제"가 보완함 | https://encykorea.aks.ac.kr/Article/E0048474 | 미해결(확인 필요) |
| C-1 | choe-u | sources url | 8 | url null. | 1-6 | https://encykorea.aks.ac.kr/Article/E0057502 | 미해결 |
| U-1 | gim-gu | sources url | 8 | url null. | 1-6 | https://encykorea.aks.ac.kr/Article/E0008753 | 미해결 |
| H-1 | gim-jeong-hui | sources url | 8 | url null. | 1-6 | https://encykorea.aks.ac.kr/Article/E0010428 | 미해결 |
| H-2 | gim-jeong-hui | active_period | 1 (확인 필요) | start 1819(문과 급제, 34세)는 achievements의 북한산비 고증(1816년 조사, 1817년 재확인)보다 늦다. 1809년 연행(옹방강·완원 교유)도 그보다 앞선다. "대표 활동 시작" 기준이면 1816년 이하가 맞다. | start를 1816(또는 1809)으로 바꾸는 것을 검토 | https://encykorea.aks.ac.kr/Article/E0010428 (열람: 1809년 연행, 34세 급제), https://encykorea.aks.ac.kr/Article/E0024796 (검색만, 1816년 조사) | 미해결(확인 필요) |

모든 힌트(50개)의 source.url도 null이다. 각 힌트에 1-6의 인물 항목 url을 넣고, 위 G-4·W-4·P-2·Y-2·K-3은 보조 항목 url을 넣어야 한다. 별도 항목으로 세지 않고 각 인물의 sources url 항목에 포함했다.

### 1-4. 난이도·혼동 인물 판단 (항목 3·4·5)

"확정 단계"는 1번부터 차례로 공개할 때 답이 하나로 좁혀지는 힌트 번호다. 혼동 인물이 DB에 없어도 실존 인물을 기준으로 판단했다.

| 인물 id | 1번 너무 쉬움? | 확정 단계 | 4·5번이 혼동 인물에게도 맞나 | 5번까지 보면 하나? | 비고 |
|---|---|---|---|---|---|
| bak-jeong-hui | 아니오(역대 대통령 모두) | 3(한일 협정·베트남 파병) | 4번 "경제 개발 5개년 계획"은 전두환·노태우 때도 이어졌지만(경제사회발전 5개년 계획) 경부 고속도로와 함께라 박정희뿐. 5번은 박정희뿐 | 예 | 2번은 전두환에게도 들어맞아 좋음 |
| wang-geon | 아니오(견훤·궁예도 해당) | 2(송악) → W-3 | 4·5번 혼동 인물에 안 맞음(정상) | 예 | |
| gim-gu | 아니오(김원봉·김규식·이승만) | 4 | 3번(남북 협상)은 김규식에게도 정확히 맞아 좋음. 4·5번 김구뿐 | 예 | 문제 없음 |
| choe-u | 아니오(무신 집권자 전부) | 4(정방 설치) | 4·5번 최우뿐 | 예 | 3번 "아버지" 판단: 1~3번은 최항(아버지 최우 계승, 몽골 침입기)과 최의에게도 들어맞는다. 논리적으로 3번에서 확정되지 않으므로 **너무 이르지 않다**. 다만 최충헌→최우 승계가 잘 알려져 있어 실제로는 3번에서 맞히는 학생이 많을 것 |
| yeongjo | 아니오(숙종·정조·광해군) | 3(이인좌의 난) | 4·5번 영조뿐 | 예 | 2번 탕평은 정조에게도 맞아 좋음 |
| yi-seong-gye | 아니오(정도전·태종) | 1+2 누적 → L-3 | 4번 위화도 회군은 조민수(좌군도통사)도 함께했으나 "정권 장악"까지는 이성계. 5번 이성계뿐(한양 천도는 태종도 1405년에 했으나 "1392년 첫 국왕"으로 구분) | 예 | |
| gojong | **예** → G-1 | 1 | 4·5번 고종뿐 | 예 | 3번 "아버지" 판단: 흥선 대원군과 고종 관계가 잘 알려져 있어 3번 자체가 확정 힌트다. 1·2번을 고치면 3번 확정은 다른 인물(박정희·영조 3번 확정)과 같은 수준이라 허용 가능. 1·2번을 고치지 않으면 순서와 관계없이 1번에서 끝난다 |
| jang-bo-go | 아니오(신라 하대 전부) | 2~3 | 4·5번 장보고뿐 | 예 | 2번은 최치원(당에서 문인 관료)과 "군인"으로 갈림. 정년도 맞으나 DB 대상 아님 |
| gwanggaeto | 아니오(고구려 왕 전부) | 2 → K-2 | 4·5번 광개토뿐 | 예 | 3번 "한강 이북 성 탈취"는 장수왕(475년 한성 함락)과 일부 겹쳐 좋음 |
| gim-jeong-hui | 아니오(박제가 등) | 3~4 | 4·5번 김정희뿐 | 예 | 2번(청 학문 수용·실사구시)은 박제가에게도 맞아 좋음. 김정희가 어려서 박제가에게 배웠다는 백과사전 서술과도 맞음 |

### 1-5. 기출 문장·사료 번역문 복제 여부 (항목 7)

- 저장소에는 기출 원문이 없고 exam_figures.md에는 출제 맥락 요약만 있다. 그래서 **기출 원문과 글자 단위 대조는 하지 못했다(확인 필요)**. 맥락 요약과 대조한 결과, 힌트는 교과서식 핵심어(균역법, 청해진, 정방, 훈요 10조 등)를 쓰지만 문장은 새로 쓴 것으로 보인다.
- 널리 알려진 사료 번역문과 비교했다.
  - 왕건 4번은 『고려사』 만부교 기사 번역("사신들을 섬으로 유배하고 낙타는 모두 만부교 아래에서 굶겨 죽였다", 백과사전 인용)과 핵심어가 같지만 문장 구조가 다르고 짧은 사실 서술이다. 복제 아님.
  - 최우 4번은 『고려사』 "자기 집에 정방을 두고 백관의 인사를 다루었다"의 의역 수준이다. 복제 아님.
  - 광개토 4번은 능비 영락 10년 기사("보기 5만을 보내 신라를 구원")를 옮기지 않았다.
  - 장보고 5번은 『삼국사기』 청해진 설치 청원문을 옮기지 않았다.
  - 김구 3번은 「삼천만 동포에게 읍고함」 문구를 쓰지 않았다.
  - 김정희 2번의 "사실에 근거해 진리를 찾는"은 실사구시의 표준 풀이로, 특정 문장을 복제한 것이 아니다.
- 결론: 항목 7 위반 없음(기출 원문 대조는 확인 필요로 남김).

### 1-6. source url 수정안 (figures.json sources, hints.json source)

| 인물 id | 주 항목(열람 확인) | 보조 항목 | 비고 |
|---|---|---|---|
| bak-jeong-hui | 박정희 https://encykorea.aks.ac.kr/Article/E0021134 | 경부고속도로 https://encykorea.aks.ac.kr/Article/E0002460 (열람) | 5·16 항목 https://encykorea.aks.ac.kr/Article/E0038494 (검색만) |
| wang-geon | 태조(고려) https://encykorea.aks.ac.kr/Article/E0059032 | 만부교사건 https://encykorea.aks.ac.kr/Article/E0017609 (열람), 북진 정책 https://encykorea.aks.ac.kr/Article/E0024747 (검색만) | 제목을 '태조'로 고칠 것 |
| gim-gu | 김구 https://encykorea.aks.ac.kr/Article/E0008753 | 남북협상 https://encykorea.aks.ac.kr/Article/E0011936 (검색만) | 아명 창암, 본명 창수, 호 백범 확인 |
| choe-u | 최우 https://encykorea.aks.ac.kr/Article/E0057502 | 정방 https://encykorea.aks.ac.kr/Article/E0050231 (검색만) | 개명 최이, 1219 계승, 1225 정방, 야별초, 1232 강화 천도, 1249 사망 확인 |
| yeongjo | 영조 https://encykorea.aks.ac.kr/Article/E0037669 | 탕평비 https://encykorea.aks.ac.kr/Article/E0058858 (열람) | 연잉군, 이인좌의 난, 준천, 균역법 확인 |
| yi-seong-gye | 태조(조선) https://encykorea.aks.ac.kr/Article/E0059033 | — | 제목을 '태조'로 고칠 것. 즉위 후 이름 이단(李旦) 확인 |
| gojong | 고종 https://encykorea.aks.ac.kr/Article/E0003939 | 원수부 https://encykorea.aks.ac.kr/Article/E0040727 (열람) | E0003938은 고려 고종이므로 혼동 주의 |
| jang-bo-go | 장보고 https://encykorea.aks.ac.kr/Article/E0048474 | 청해진 https://encykorea.aks.ac.kr/Article/E0056856 (검색만) | 궁복·궁파, 828 청해진, 법화원, 신무왕, 846 염장 확인 |
| gwanggaeto | 광개토왕 https://encykorea.aks.ac.kr/Article/E0005057 | 광개토왕릉비 https://encykorea.aks.ac.kr/Article/E0005058 (열람) | 담덕, 영락 확인. 재위는 정의 391~413, 개설 391~412(능비 기준)로 항목 안에서 갈림. figures 값 391~413은 정의와 일치 |
| gim-jeong-hui | 김정희 https://encykorea.aks.ac.kr/Article/E0010428 | 실사구시 https://encykorea.aks.ac.kr/Article/E0033652 (열람), 서울 북한산 신라 진흥왕 순수비 https://encykorea.aks.ac.kr/Article/E0024795 (검색만) | 1809 연행, 1840~1848 제주 유배, 1844 세한도, 1856 사망 확인 |

확인 못함: "광무제/광무황제" 별칭의 백과사전 근거(G-6). 우리역사넷 항목은 이번 회차에서 열지 않았다.

### 1-7. 별칭 점검 (항목 8)

- 10명 사이에 별칭 충돌 없음(validate.js 확인).
- DB 밖 실존 인물과 겹칠 여지: "고종"(고려 고종, G-5), "광무제"(후한 광무제, G-6). "이단"은 일반 명사(異端)와 같지만 인물 충돌은 아니라 문제로 보지 않음. "호태왕"은 광개토왕 시호 약칭으로 백과사전 확인.

### 1차 회차 요약

- 문제 항목 수(인물별): gojong 7, wang-geon 4, yi-seong-gye 3, gwanggaeto 3, bak-jeong-hui 2, yeongjo 2, jang-bo-go 2, gim-jeong-hui 2, gim-gu 1, choe-u 1. **합계 27건**(이 가운데 확인 필요 7건: G-3, G-5, G-6, L-2, L-3, J-2, H-2).
- 문제 없음으로 통과한 id: **없음**. gim-gu, choe-u는 sources url 기입(U-1, C-1)만 남았다. url만 넣으면 통과.
- 남은 미해결: **27건**.
