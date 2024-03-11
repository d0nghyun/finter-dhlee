from datetime import datetime, timedelta
from finter.framework_model import BaseAlpha

def calculate_previous_start_date(start_date, lookback_days):
    start = datetime.strptime(str(start_date), "%Y%m%d")
    previous_start = start - timedelta(days=lookback_days)
    return int(previous_start.strftime("%Y%m%d"))

# Define the lookback period in days
LOOKBACK_DAYS = 365

# Alpha class inheriting from BaseAlpha
class Alpha(BaseAlpha):
    # Method to generate alpha
    def get(self, start, end):
        # Calculate the start date for data retrieval
        pre_start = calculate_previous_start_date(start, LOOKBACK_DAYS)
        # Retrieve daily closing prices
        self.close = self.get_cm(
            "content.fnguide.ftp.price_volume.price_close.1d"
        ).get_df(pre_start, end)
        # Calculate momentum
        momentum_21d = self.close.pct_change(21, fill_method=None)
        # Rank stocks by momentum
        stock_rank = momentum_21d.rank(pct=True, axis=1)
        # Select top 10% of stocks
        stock_top10 = stock_rank[stock_rank<=0.1]
        # Apply rolling mean to smooth data
        stock_top10_rolling = stock_top10.rolling(21).apply(lambda x: x.mean())
        # Normalize and scale to position sizes
        stock_ratio = stock_top10_rolling.div(stock_top10_rolling.sum(axis=1), axis=0)
        position = stock_ratio * 1e8
        # Shift positions to avoid look-ahead bias
        alpha = position.shift(1)
        return alpha.loc[str(start): str(end)]

