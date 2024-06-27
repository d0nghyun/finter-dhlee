# 코드 요약

## 목적
이 코드는 한국 주식 시장에서 특정 조건에 따라 포지션을 계산하고 할당하는 퀀트 모델을 구현합니다.

## 주요 전략 및 알고리즘
1. **데이터 수집 및 처리**:
   - `ContentFactory`를 사용하여 주식 가격 데이터(`price_close`)와 보너스 발행 데이터(`bonus_issue`)를 가져옵니다.
   - 보너스 발행 데이터를 정리하여 시작 날짜와 종료 날짜를 추출합니다.

2. **거래일 계산**:
   - 주어진 날짜 범위 내에서 가장 가까운 거래일을 찾는 함수를 정의합니다.
   - 보너스 발행 시작일과 종료일에 대해 가장 가까운 거래일을 계산합니다.

3. **포지션 설정**:
   - 시작일과 종료일에 대한 플래그를 설정하고 이를 기반으로 포지션을 계산합니다.
   - 부채비율(`debt-to-market`)을 기준으로 주식을 랭킹하고, 상위 50%에 해당하는 주식만 포지션에 포함시킵니다.
   - 포지션 크기를 조정하여 총 할당 금액이 1억 원을 넘지 않도록 합니다.

4. **포지션 이동**:
   - 최종 포지션을 하루 뒤로 이동시켜 포지션을 설정합니다.

## 데이터 소스
- `ContentFactory`를 통해 `price_close`, `bonus_issue`, `debt-to-market` 데이터를 사용합니다.

## 요약
이 코드는 한국 주식 시장에서 보너스 발행 정보를 기반으로 포지션을 설정하고, 부채비율을 고려하여 포지션을 조정하는 퀀트 모델입니다. 주식 가격 데이터와 보너스 발행 데이터를 활용하여 거래일을 계산하고, 포지션을 설정 및 조정하여 최종 포지션을 반환합니다.