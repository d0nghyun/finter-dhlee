# Quant Model Summary

## 목적
이 코드는 한국 주식 시장에서 특정 주식의 보너스 발행 정보를 기반으로 포지션을 계산하는 퀀트 모델입니다. 주식의 보너스 발행일 전후의 거래일을 고려하여 포지션을 설정하고, 안전 노출 지표를 사용하여 포지션을 조정합니다.

## 주요 전략 및 알고리즘
1. **보너스 발행 정보 활용**:
   - `ContentFactory`를 사용하여 보너스 발행 데이터를 가져옵니다.
   - 보너스 발행일 전후의 거래일을 계산하여 포지션을 설정합니다.

2. **거래일 계산**:
   - `iter_trading_days`와 `TradingDay`를 사용하여 주어진 기간 내의 거래일 리스트를 생성합니다.
   - 보너스 발행일 전후의 가장 가까운 거래일을 찾는 함수를 정의합니다.

3. **포지션 설정**:
   - 보너스 발행일 전후의 거래일에 따라 포지션 플래그를 설정합니다.
   - 안전 노출 지표(`safety_exposure`)를 사용하여 포지션을 조정합니다.
   - 포지션의 총합이 일정 금액을 초과하지 않도록 조정합니다.

## 데이터 소스
- `ContentFactory("raw", _start, end)`: 원시 데이터
- `ContentFactory("kr_stock", _start, end)`: 한국 주식 데이터
- `price_close`: 주식 종가 데이터
- `content.dart.api.disclosure.bonus_issue.1d`: 보너스 발행 데이터
- `content.handa.dataguide.factor.safety_exposure.1d`: 안전 노출 지표 데이터

## 요약
이 퀀트 모델은 한국 주식 시장에서 보너스 발행 정보를 기반으로 포지션을 설정하고, 안전 노출 지표를 사용하여 포지션을 조정합니다. 거래일 계산과 포지션 조정 알고리즘을 통해 안정적인 포지션을 유지하며, 포지션의 총합이 일정 금액을 초과하지 않도록 관리합니다.