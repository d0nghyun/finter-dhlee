from datetime import datetime, timedelta
import numpy as np
import pandas as pd
from finter import BasePortfolio

model_info = {
    "exchange": "vnm",             # Korean Exchange (KRX)
    "universe": "compustat",       # Universe for the model
    "instrument_type": "stock",    # Instrument type, e.g., stock
    "freq": "1d",                  # Frequency of data, daily
    "position_type": "target",     # Position type, target positions
    "type": "portfolio",           # Model type, portfolio
}

class Portfolio(BasePortfolio):
    alpha_set = {f"vnm.compustat.stock.ldh0127.mom_{i}" for i in range(1, 5)}
    alpha_set = alpha_set | set(["vnm.compustat.stock.ldh0127.vietnam_mom_2"])

    def alpha_loader(self, start, end):
        return self.get_alpha_position_loader(
            start, end,
            model_info["exchange"],
            model_info["universe"],
            model_info["instrument_type"],
            model_info["freq"],
            model_info["position_type"],
        )

    def get(self, start, end):
        self.start_dt, self.end_dt = datetime.strptime(str(start), "%Y%m%d"), datetime.strptime(str(end), "%Y%m%d")
        _start = int((self.start_dt - timedelta(days=252 * 3)).strftime("%Y%m%d"))
        
        cf = ContentFactory('vn_stock', _start, end)
        self.cf  = cf
        gics = cf.get_df('gics')
        gics.columns = gics.columns + '01W'

        price = cf.get_df('price_close')


        ret = price.pct_change(fill_method=None)
        ret = ret[ret.abs().rank(axis=1, pct=True)<0.99]
        sector = gics.fillna(0).applymap(lambda x: str(x)[:2])
        unique_sectors = set(sector.values.flatten())

        sector_momentum = pd.DataFrame()

        for sector_code in unique_sectors:
            if sector_code != '0':  # 미분류 섹터 제외
                sector_mask = (sector == sector_code)
                sector_momentum[f'Sector_{sector_code}'] = (sector_mask * ret).mean(axis=1)

        sector_momentum = sector_momentum.rolling(252).apply(lambda x: x.sum())
        top_2_sectors = sector_momentum.rank(axis=1, ascending=False) <= 1

        # 상위 2개 섹터에 속하는 기업들의 마스크 생성
        top_2_mask = pd.DataFrame(False, index=price.index, columns=price.columns)

        for date in top_2_mask.index:
            top_sectors = top_2_sectors.loc[date][top_2_sectors.loc[date]].index
            for top_sector in top_sectors:
                sector_code = top_sector.split('_')[1]
                top_2_mask.loc[date] |= (sector.loc[date] == sector_code)


        
        alpha_loader = self.alpha_loader(_start, end)
        alpha_dict = {}
        for i in self.alpha_set:
            alpha_dict[i] = alpha_loader.get_alpha(i)
            
        positive_dfs = []

        for i, df in alpha_dict.items():
            positive_df = df>0
            positive_dfs.append(positive_df)

        result_df = pd.concat(positive_dfs).groupby(level=0).sum()
        signal = result_df[result_df>=2]
        
        signal *= top_2_mask.replace(False, np.nan).ffill(limit=21).shift()
        

        pf = signal.div(signal.sum(axis=1).replace(0, 1), axis=0) * 1e8
        return pf.fillna(0).loc[str(start): str(end)]
