# 코드 요약

## 목적
이 코드는 한국 주식 시장에서 특정 조건에 따라 포지션을 계산하고 할당하는 퀀트 모델입니다. 주로 보너스 발행 데이터와 유동성 노출 데이터를 사용하여 포지션을 결정합니다.

## 주요 전략 및 알고리즘
1. **보너스 발행 데이터 처리**:
   - `ContentFactory`를 사용하여 보너스 발행 데이터를 가져옵니다.
   - 보너스 발행 데이터를 정리하고, 시작 날짜와 종료 날짜를 계산합니다.

2. **거래일 계산**:
   - 주어진 날짜 범위 내에서 가장 가까운 거래일을 찾는 함수를 정의합니다.
   - 시작 날짜와 종료 날짜에 대해 가장 가까운 거래일을 찾습니다.

3. **포지션 플래그 설정**:
   - 시작 날짜와 종료 날짜에 대해 플래그를 설정합니다.
   - 시작 플래그는 1, 종료 플래그는 -1로 설정합니다.

4. **포지션 계산**:
   - 시작 플래그와 종료 플래그를 사용하여 포지션을 계산합니다.
   - 유동성 노출 데이터를 사용하여 포지션을 조정합니다.
   - 포지션을 0.02e8로 스케일링하고, 총 할당이 1e8을 초과하는 경우 비례 배분합니다.

5. **포지션 이동**:
   - 최종 포지션을 하루 뒤로 이동시켜 반환합니다.

## 데이터 소스
- `ContentFactory`를 통해 가져오는 데이터:
  - `price_close`: 주식 종가 데이터
  - `content.dart.api.disclosure.bonus_issue.1d`: 보너스 발행 데이터
  - `content.handa.dataguide.factor.liquidity_exposure.1d`: 유동성 노출 데이터

## 요약
이 코드는 한국 주식 시장에서 보너스 발행과 유동성 노출 데이터를 기반으로 포지션을 계산하는 퀀트 모델입니다. 거래일을 기준으로 시작과 종료 플래그를 설정하고, 유동성 노출 데이터를 사용하여 포지션을 조정합니다. 최종 포지션은 하루 뒤로 이동시켜 반환됩니다.