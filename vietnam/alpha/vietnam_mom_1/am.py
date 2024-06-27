import numpy as np
from datetime import datetime, timedelta
from finter import BaseAlpha
from finter.data import ContentFactory
from finter.calendar import TradingDay, iter_days

class Alpha(BaseAlpha):  # BaseAlpha Class를 상속받아옴 (필수)
    def get(self, start, end):  # start와 end를 입력받아서 start ~ end에 해당하는 포지션을 반환

        cf = ContentFactory('raw', int((datetime.strptime(str(start), "%Y%m%d") - timedelta(days=365 * 3)).strftime("%Y%m%d")), end)
        self.close_price = cf.get_df("content.spglobal.compustat.price_volume.vnm-stock-price_close.1d")

        ind = self.close_price.index

        self.close_price = self.close_price.dropna(how='all')

        # 252일 수익률
        self.price_change = self.close_price.pct_change().rolling(252).sum()

        # 252일 수익률의 cross-sectional rank
        self.rank = self.price_change.rank(pct=True, axis=1)

        # 252일 수익률의 cross-sectional rank
        self.rank = self.rank.rolling(252).mean()

        # rank의 상위 20%
        self.signal = self.rank[self.rank < 0.5]

        # eqaul weight로 투자
        self.position = self.signal.div(self.signal.sum(axis=1), axis=0) * 1e8

        self.position = self.position.fillna(-1).reindex(ind).ffill().replace(-1, np.nan)


        # 종가데이터는 다음날부터 사용가능하므로 shift 1
        position = self.position.shift(1).fillna(0)

        return position
