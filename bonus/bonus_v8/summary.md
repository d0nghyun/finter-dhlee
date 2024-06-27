# 코드 요약

## 목적
이 코드는 한국 주식 시장에서 특정 주식의 포지션을 계산하여 알파 전략을 구현하는 것을 목적으로 합니다.

## 주요 전략 및 알고리즘
1. **데이터 수집 및 처리**:
   - `ContentFactory`를 사용하여 주식 가격 데이터(`price_close`)와 보너스 발행 데이터(`bonus_issue`)를 가져옵니다.
   - 보너스 발행 데이터를 정리하여 시작 날짜와 종료 날짜를 추출합니다.

2. **거래일 계산**:
   - 주어진 기간 내의 거래일 리스트를 생성합니다.
   - 특정 날짜에 가장 가까운 거래일을 찾는 함수를 정의하여, 보너스 발행 시작일과 종료일에 대한 거래일을 계산합니다.

3. **포지션 계산**:
   - 시작일과 종료일에 대한 플래그를 설정하여 포지션을 계산합니다.
   - 성장 노출 지표(`growth_exposure`)를 사용하여 주식의 순위를 매기고, 상위 50%에 해당하는 주식만 선택합니다.
   - 포지션 크기를 조정하여 총 포지션이 일정 금액을 초과하지 않도록 합니다.

4. **포지션 이동**:
   - 최종 포지션을 하루 이동시켜 다음 날의 포지션을 설정합니다.

## 데이터 소스
- `price_close`: 주식 종가 데이터
- `bonus_issue`: 보너스 발행 데이터
- `growth_exposure`: 성장 노출 지표 데이터

## 요약
이 코드는 한국 주식 시장에서 보너스 발행 정보를 기반으로 특정 주식의 포지션을 계산하여 알파 전략을 구현합니다. 주식의 성장 노출 지표를 사용하여 상위 50%의 주식을 선택하고, 포지션 크기를 조정하여 리스크를 관리합니다. 최종 포지션은 하루 이동시켜 다음 날의 포지션을 설정합니다.