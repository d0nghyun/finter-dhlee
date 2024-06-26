from datetime import datetime, timedelta

import pandas as pd
from finter import BasePortfolio

model_info = {
    "exchange": "krx",             # Korean Exchange (KRX)
    "universe": "krx",             # Universe for the model
    "instrument_type": "stock",    # Instrument type, e.g., stock
    "freq": "1d",                  # Frequency of data, daily
    "position_type": "target",     # Position type, target positions
    "type": "portfolio",           # Model type, portfolio
}

class Portfolio(BasePortfolio):
    alpha_set = {f"krx.krx.stock.ldh0127.bonus_v{i}" for i in range(1, 15)}

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

        alpha_loader = self.alpha_loader(_start, end)
        alpha_dict = {}
        for i in self.alpha_set:
            alpha_dict[i] = alpha_loader.get_alpha(i)
        pf = sum(map(lambda x: x.fillna(0), alpha_dict.values()))

        pf = pf.div(pf.sum(axis=1), axis=0) * 1e8
        return pf
