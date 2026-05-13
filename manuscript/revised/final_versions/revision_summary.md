# Revision Summary

## 주요 수정 내용

이번 통합본은 `manuscript/revised/`의 section별 revise markdown을 하나의 제출용 draft 흐름으로 재구성하였다. 기존 방법별 나열 구조를 줄이고, “언론이 재현하는 공직사회 표상(media representation of the civil service)”과 “누적적 재구성(cumulative reconfiguration)”을 중심축으로 Abstract, Introduction, Theory, Methods, Results, Discussion, Policy Implication, Conclusion을 정렬하였다.

주요 변경은 다음과 같다.

- 분석 단위를 “공직사회 자체”가 아니라 “공직사회 관련 언론 담론”으로 명확히 제한하였다.
- Introduction, Methods, Results, Discussion, Conclusion에서 measurement-inference boundary를 반복적으로 명시하였다.
- “공직사회 변화”, “구성원 경험”, “공정성 인식”, “조직문화 변화”처럼 실제 내부 상태를 직접 측정한 듯한 표현을 “언론 담론에서 재현되는 변화”, “구성원 경험 관련 의제”, “공정성 관련 의제의 담론적 가시성”, “조직문화 관련 언론 의제”로 조정하였다.
- Concept cluster analysis를 본문 Results의 중심 증거로 배치하고, TF-IDF/log-odds/LDA/네트워크 분석은 보조 검증 및 triangulation evidence로 재정리하였다.
- COVID robustness와 3시기 분석은 causal identification이 아니라 lexical shock 점검과 temporal differentiation 전략으로 설명하였다.
- Policy implication은 실제 조직 진단이나 직접 처방이 아니라 정책 커뮤니케이션 및 공적 담론 관리 차원의 함의로 낮추었다.

## Construct Validity 대응 방식

Reviewer2의 construct validity 비판에 대응하기 위해 원고 전반에서 다음 경계를 명확히 했다.

- 본 연구는 실제 공무원의 인식, 태도, 경험, 조직문화 상태를 직접 측정하지 않는다.
- 본 연구의 자료는 언론 기사이며, 분석 결과는 언론 담론에서 공직사회가 어떻게 의제화되고 재현되는지를 보여준다.
- TF-IDF, log-odds, LDA, concept cluster, 공출현 네트워크는 실제 조직 상태의 직접 지표가 아니라 담론적 가시성, 주제 구성, 의제 결합 구조를 보여주는 텍스트 기반 지표이다.

## Measurement-Inference Boundary 수정 방향

측정과 추론의 경계를 다음과 같이 정리하였다.

- 측정 대상: 언론 기사에 나타난 키워드, 토픽, 개념군 빈도, 공출현 관계
- 추론 대상: 공직사회 관련 언론 담론의 의제 구조와 표상 방식
- 추론하지 않는 대상: 실제 공무원의 인식, 경험, 태도, 이직 의도, 조직문화 상태

Temporal split은 정부교체 효과와 코로나19 효과를 인과적으로 분리하는 설계가 아니라, 코로나 이전부터 관찰되는 의제와 코로나 이후 급격히 부각되는 의제를 시간적으로 구분하기 위한 temporal differentiation으로 설명하였다.

Concept cluster는 공정성, 조직문화, 성과주의/NPM, 디지털 전환, 방역·위기 대응 의제의 dictionary-based proxy로 정의하였다. 따라서 결과는 개념군별 “담론 가시성”으로 해석하도록 조정하였다.

## Overclaiming 완화 전략

다음 표현을 일관되게 조정하였다.

- “공직사회가 변화했다” -> “공직사회 관련 언론 담론이 재구성되었다”
- “구성원의 경험/인식” -> “구성원 경험 관련 의제” 또는 “공정성 관련 의제”
- “변화시켰다/영향을 미쳤다” -> “담론적 가시성을 높였다”, “의제화되었다”, “결합되었다”
- “확인하였다/보여준다” -> 문맥에 따라 “관찰되었다”, “시사한다”, “해석할 수 있다”
- “정책 처방” -> “정책 커뮤니케이션 및 공적 담론 관리 차원의 함의”

## Reviewer 대응 논리

Reviewer2에 대해서는 “언론 기사로 실제 공직사회 내부 상태를 측정한 것이 아니다”라는 경계를 명시적으로 제시한다. 본 연구의 기여는 실제 조직행태를 대체 측정하는 데 있지 않고, 공직사회가 공적 담론장에서 어떤 의제와 프레임으로 재현되는지를 장기 시계열로 분석하는 데 있다.

Reviewer3에 대해서는 분석이 많아 보이는 문제를 줄이기 위해 각 분석의 역할을 구분하였다.

- TF-IDF/log-odds: 시기별 차별 어휘와 의제 가시성 확인
- LDA: 언론 담론의 주제 묶음 확인
- Concept cluster: 이론적 개념군별 담론 가시성 비교
- COVID robustness: 코로나 lexical shock 의존성 점검
- 3시기 분석: 정부교체 이후 담론 재구성과 코로나 이후 의제 재배열의 temporal differentiation
- 네트워크 분석: 기존 의제와 새로운 의제의 결합 구조 확인

## Narrative Coherence 점검

통합본의 중심 narrative는 다음과 같이 정리된다.

1. 박근혜 정부 시기 공직사회 관련 언론 담론은 제도개혁·성과관리 의제와 상대적으로 밀접하게 결합되어 있었다.
2. 문재인 정부 전기에는 공정성, 조직문화, 디지털 전환 관련 의제가 코로나 이전부터 부각되기 시작했다.
3. 코로나19 이후에는 방역·위기 대응 담론이 급격히 부각되었고, 디지털 전환 담론도 강화되었다.
4. 코로나 lexical shock 제거 후에도 주요 구조는 유지되어, 전체 결과가 코로나 관련 단어 빈도 급증에만 의존하지 않음을 보였다.
5. 따라서 공직사회 관련 언론 담론은 제도개혁에서 조직문화로 단순 대체된 것이 아니라, 정부교체 이후 시작된 담론 재구성이 코로나 국면에서 누적적으로 재배열·증폭된 과정으로 해석된다.

## Abstract-Conclusion Alignment

Abstract와 Conclusion 모두 동일한 메시지를 공유하도록 정렬하였다.

- 분석 단위: 공직사회 내부 상태가 아니라 언론 담론
- 핵심 결과: 공정성·디지털은 코로나 이전부터 증가, 방역은 코로나 이후 급증, 성과주의/NPM은 약화
- 핵심 해석: 정부교체와 코로나19의 단일 인과 효과가 아니라 누적적 재구성
- 한계: 실제 인식·경험 검증은 설문, 면접, 행정자료와의 triangulation 필요

## Policy Implication Overreach 점검

정책 함의는 “조직 진단” 또는 “정책 처방”이 아니라 “정책 커뮤니케이션 및 공적 담론 관리” 차원으로 제한하였다. 실제 공무원의 공정성 인식, 조직 경험, 이탈 원인을 직접 설명하지 않고, 해당 의제들이 언론 담론에서 어떤 문제틀로 재현되는지에 대한 함의로 조정하였다.

## 부록 이동 대상

본문의 reviewer readability를 높이기 위해 다음 자료는 appendix로 이동하는 것이 적절하다.

- seed sensitivity 상세
- full TF-IDF tables
- full log-odds tables
- dictionary full list
- monthly trend 전체 figure 및 표
- robustness full outputs
- LDA coherence 및 seed별 topic 상세
- 네트워크 전체 그림 및 중심성 전체표
