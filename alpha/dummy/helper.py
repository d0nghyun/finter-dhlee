from datetime import datetime, timedelta


def calculate_previous_start_date(start_date, lookback_days):
    start = datetime.strptime(str(start_date), "%Y%m%d")
    previous_start = start - timedelta(days=lookback_days)
    return int(previous_start.strftime("%Y%m%d"))
