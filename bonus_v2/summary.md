# Quant Model Summary

## 목적
이 코드는 한국 주식 시장에서 특정 주식의 포지션을 계산하여 알파 전략을 구현하는 것을 목적으로 합니다. 주식의 보너스 발행 정보를 활용하여 포지션을 설정하고, 단기 반전 지표를 사용하여 포지션을 조정합니다.

## 주요 전략 및 알고리즘
1. **보너스 발행 정보 활용**:
   - `ContentFactory`를 사용하여 보너스 발행 데이터를 가져옵니다.
   - 보너스 발행 날짜를 기준으로 시작일과 종료일을 설정합니다.
   - 가장 가까운 거래일을 찾는 함수를 통해 시작일과 종료일을 조정합니다.

2. **포지션 설정**:
   - 보너스 발행 시작일에 포지션을 설정하고, 종료일에 포지션을 해제합니다.
   - `start_flag`와 `end_flag`를 사용하여 포지션을 설정합니다.

3. **단기 반전 지표 활용**:
   - `descriptor` 데이터를 가져와 백분위 순위로 변환하고, 상위 50%에 해당하는 주식만 선택합니다.
   - 선택된 주식에 대해 포지션을 조정합니다.

4. **포지션 조정**:
   - 포지션을 0.02e8 단위로 설정하고, 총 포지션이 1e8을 초과할 경우 비율에 맞게 조정합니다.
   - 최종 포지션을 하루 뒤로 이동시켜 설정합니다.

## 데이터 소스
- `ContentFactory`를 통해 `raw` 및 `kr_stock` 데이터를 가져옵니다.
- `price_close`: 주식 종가 데이터
- `content.dart.api.disclosure.bonus_issue.1d`: 보너스 발행 데이터
- `content.handa.dataguide.descriptor.short-term-reversal.1d`: 단기 반전 지표 데이터

## 요약
이 코드는 한국 주식 시장에서 보너스 발행 정보를 활용하여 특정 주식의 포지션을 설정하고, 단기 반전 지표를 사용하여 포지션을 조정하는 알파 전략을 구현합니다. 주식의 시작일과 종료일을 가장 가까운 거래일로 조정하고, 포지션을 설정한 후 단기 반전 지표를 통해 최종 포지션을 조정합니다.