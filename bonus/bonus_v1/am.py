from datetime import datetime, timedelta

import numpy as np
import pandas as pd

from finter import BaseAlpha
from finter.calendar import TradingDay, iter_trading_days
from finter.data import ContentFactory


class Alpha(BaseAlpha):
    def get(self, start, end):
        # 시작 날짜 설정
        _start = 20000101

        # ContentFactory 객체 생성
        cf_raw = ContentFactory("raw", _start, end)
        cf_kr = ContentFactory("kr_stock", _start, end)

        # 가격과 보너스 데이터 가져오기
        price = cf_kr.get_df("price_close")
        self.price = price
        bonus = cf_raw.get_df("content.dart.api.disclosure.bonus_issue.1d")

        # 보너스 데이터 정리
        # bonus = bonus.unstack().dropna()
        bonus = bonus.reset_index().melt(id_vars='index', value_vars=bonus.columns).dropna().set_index(['index', 'variable'])['value']

        # 날짜 데이터 처리
        end_date = pd.to_datetime(
            bonus.apply(lambda x: x["nstk_asstd"]).replace("-", np.nan),
            format="%Y년 %m월 %d일",
        )
        end_date = end_date.dt.date

        # 날짜 데이터프레임 생성
        end_date_df = end_date.reset_index()
        end_date_df.columns = "start_date", "ccid", "end_date"

        # 시작 날짜 데이터 처리
        end_date_df["start_date"] = end_date_df["start_date"].dt.date

        # 거래일 리스트 생성
        trading_days_list = [
            i.date()
            for i in iter_trading_days(
                TradingDay.day_delta(_start, n=-21, exchange="krx"), end
            )
        ]

        # 가장 가까운 거래일을 찾는 함수
        def find_nearest_trading_day(trading_days, target_date, days_prior):
            nearest_day = None
            nearest_index = -1
            for index, day in enumerate(trading_days):
                if day <= target_date:
                    if (
                        nearest_day is None
                        or target_date - day < target_date - nearest_day
                    ):
                        nearest_day = day
                        nearest_index = index

            # 가장 가까운 거래일 n일 전 날짜 찾기
            prior_index = nearest_index - days_prior
            if prior_index >= 0:
                prior_day = trading_days[prior_index]
            else:
                prior_day = None
            return prior_day

        # 가장 가까운 거래일 찾기
        end_date_df["end_trading_day"] = end_date_df["end_date"].apply(
            lambda x: find_nearest_trading_day(trading_days_list, x, 3)
        )
        end_date_df["start_trading_day"] = end_date_df["start_date"].apply(
            lambda x: find_nearest_trading_day(trading_days_list, x, 0)
        )

        # Flag 설정
        end_date_df["start_flag"] = 1
        end_date_df["end_flag"] = -1

        # 데이터프레임 변환
        start_flag = end_date_df.pivot(
            index="start_trading_day", columns="ccid", values="start_flag"
        ).reindex(price.index)
        end_flag = end_date_df.pivot(
            index="end_trading_day", columns="ccid", values="end_flag"
        ).reindex(price.index)

        # 포지션 계산
        position = start_flag.fillna(end_flag).ffill().replace(-1, np.nan)
        position = position * 0.02e8
        allocate = position.sum(axis=1)
        position[allocate >= 1e8] = position[allocate >= 1e8].div(
            allocate / 1e8, axis=0
        )

        # 포지션 이동
        position = position.shift().fillna(0)

        return position
