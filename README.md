# ALD Al2O3 표면 반응 모델링

TMA와 H2O를 순차 공급할 때 표면종의 흡착, 반응, 탈착을 계산하고 사이클별 Al 증가량을 GPC로 환산하는 mean-field 모델이다. 문헌 반응계수, MMAOH 중간체와 온도별 최대 OH 자리 수 경험식을 사용한다.

## 모델 설명

[모델의 식, 계수, 가정과 검증](현재모델_식_계수_출처_검증_2026-10-13.md)에 전체 정의를 정리했다.

한 사이클은 TMA 공급, 퍼지, H2O 공급, 퍼지로 구성된다. OH, 흡착 TMA, DMA, MMAOH, MMA와 각 물 흡착 상태를 추적한다. 첫 표면만 전부 OH로 설정하고 이후 잔류 표면종을 다음 사이클로 넘긴다. 물 반응에 의해 Me가 제거되고 OH가 재생된다.

최대 자리 수는 온도 경험식으로 지정한다. 별도 수화/탈수산화 동역학은 보류 상태다. 유량이나 유동장을 풀지 않고 표면 전구체 분압을 입력한다.

## 계산과 결과

- [실행 코드](experiments/2026-10-13_gakis_comparison/run.py)
- [입력값과 출처](experiments/2026-10-13_gakis_comparison/02_파라미터_조사.md)
- [조건별 결과 CSV](experiments/2026-10-13_gakis_comparison/results.csv)
- [문헌 비교 지표](experiments/2026-10-13_gakis_comparison/metrics.csv)

기준 비교는 자리 수지 B, 두 전구체 각각 0.1 Torr, 물 공급 100ms, TMA 공급 25/60ms, 막 밀도 3.5 g/cm^3이다. 500사이클 평균 GPC를 사용하며 300 C는 550사이클이다.

Gakis 실험 12점에 대한 평균 절대 상대오차는 5.80%, RMSE는 0.0649 angstrom/cycle이다. 최대 상대오차는 125 C의 15.7%다. 표면 분압과 종별 점유수는 가정이며 이 수치를 독립 검증 정확도로 해석하지 않는다.

스크립트를 실행하면 자리 수지 A/B 및 네 분압 조건을 함께 계산하고 해당 실험 폴더의 출력 파일을 갱신한다. Python의 numpy, scipy, matplotlib가 필요하다. Windows 그래프에는 Malgun Gothic을 사용한다.

## 자료 구성

| 위치 | 내용 |
|---|---|
| 현재모델_식_계수_출처_검증_2026-10-13.md | 이 페이지에서 설명하는 기준 모델의 전체 정의 |
| experiments/2026-10-13_gakis_comparison/ | 기준 비교 코드, 계수, 결과 |
| experiments/2026-10-13_single_source/ | Gakis 원문과 실험 추출 데이터 |
| src/, 01_모델설계.md, 02_파라미터_조사.md, 03_결과.md, 03_결과/ | 별도의 Phase별 모델과 산출 자료. 식과 검증 수치는 각 모델에만 적용 |
| experiments/의 다른 폴더 | 조건별 분석과 비교안. 채택 범위는 각 문서 참조 |

## 문헌

Gakis et al., Chemical Engineering Science, DOI [10.1016/j.ces.2018.09.037](https://doi.org/10.1016/j.ces.2018.09.037). 새 양자화학 계산은 수행하지 않는다.
