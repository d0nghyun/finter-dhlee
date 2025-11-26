# CLAUDE.md - AI Assistant Guide for finter-dhlee

## Repository Overview

This is a **quantitative trading model repository** containing alpha and portfolio strategies for stock markets. The codebase uses the `finter` framework for building systematic trading models.

### Markets Covered
- **Korean Stock Market (KRX)**: Bonus issue event-driven strategies
- **Vietnam Stock Market (VNM)**: Momentum-based strategies using Compustat data

## Directory Structure

```
finter-dhlee/
├── bonus/                          # Korean market alpha models
│   ├── bonus_v1/ to bonus_v14/     # 14 alpha model versions
│   │   ├── am.py                   # Alpha model implementation
│   │   ├── model_meta.json         # Model metadata
│   │   └── summary.md              # Strategy documentation
│   └── portfolio/
│       └── bonus_v1/               # Portfolio combining bonus alphas
│           ├── pf.py               # Portfolio implementation
│           ├── model_meta.json
│           └── summary.md
├── vietnam/                        # Vietnam market models
│   ├── alpha/
│   │   └── vietnam_mom_1/          # Momentum alpha model
│   │       ├── am.py
│   │       ├── model_meta.json
│   │       └── summary.md
│   └── portfolio/
│       └── vn_1/ to vn_3/          # 3 portfolio versions
│           ├── pf.py
│           ├── model_meta.json
│           └── summary.md
└── README.md
```

## File Conventions

### Alpha Models (`am.py`)
- Extend `BaseAlpha` class from `finter`
- Implement `get(self, start, end)` method returning position DataFrame
- Class name: `Alpha`

```python
from finter import BaseAlpha

class Alpha(BaseAlpha):
    def get(self, start, end):
        # Implementation returning position DataFrame
        return position
```

### Portfolio Models (`pf.py`)
- Extend `BasePortfolio` class from `finter`
- Define `alpha_set` for component alphas
- Implement `get(self, start, end)` method
- Class name: `Portfolio`

```python
from finter import BasePortfolio

model_info = {
    "exchange": "krx",
    "universe": "krx",
    "instrument_type": "stock",
    "freq": "1d",
    "position_type": "target",
    "type": "portfolio",
}

class Portfolio(BasePortfolio):
    alpha_set = {f"krx.krx.stock.ldh0127.bonus_v{i}" for i in range(1, 15)}

    def get(self, start, end):
        # Implementation returning position DataFrame
        return pf
```

### Model Metadata (`model_meta.json`)
Required fields:
- `exchange`: Market exchange (`"krx"`, `"vnm"`)
- `universe`: Universe name (`"krx"`, `"compustat"`)
- `instrument_type`: Usually `"stock"`
- `freq`: Data frequency (`"1d"` for daily)
- `position_type`: Position type (`"target"`)
- `type`: Model type (`"alpha"` or `"portfolio"`)
- `nickname`: Model identifier (e.g., `"bonus_v1"`)
- `model_dir`: Directory path relative to repo root

For alpha models, include:
- `exposure`: `"long_only"` or other exposure type

### Summary Files (`summary.md`)
Korean-language documentation including:
- Strategy purpose and goals
- Algorithm description
- Data sources used
- Key implementation details

## Key Framework Components

### Data Access
```python
from finter.data import ContentFactory

# For Korean stock data
cf_kr = ContentFactory("kr_stock", start, end)
price = cf_kr.get_df("price_close")

# For Vietnam stock data
cf_vn = ContentFactory("vn_stock", start, end)
price = cf_vn.get_df("price_close")

# For raw data access
cf_raw = ContentFactory("raw", start, end)
bonus_data = cf_raw.get_df("content.dart.api.disclosure.bonus_issue.1d")
```

### Calendar Utilities
```python
from finter.calendar import TradingDay, iter_trading_days

# Get trading days
trading_days = iter_trading_days(start, end)

# Date offset
offset_date = TradingDay.day_delta(date, n=-21, exchange="krx")
```

### Alpha Loading in Portfolios
```python
alpha_loader = self.get_alpha_position_loader(
    start, end,
    exchange, universe, instrument_type, freq, position_type
)
alpha_df = alpha_loader.get_alpha("krx.krx.stock.ldh0127.bonus_v1")
```

## Data Conventions

### Date Format
- Use integers in `YYYYMMDD` format: `20240101`, `20231231`
- Convert with: `datetime.strptime(str(date), "%Y%m%d")`

### Position Scaling
- Standard position value: `1e8` (100 million)
- Normalize positions: `pf = pf.div(pf.sum(axis=1), axis=0) * 1e8`

### DataFrame Structure
- Index: DatetimeIndex (dates)
- Columns: Instrument identifiers (CCIDs)
- Values: Position weights or signals

## Common Patterns

### Signal Generation
```python
# Rank-based signal
rank = values.rank(axis=1, pct=True)
signal = rank[rank > 0.8]  # Top 20%

# Equal weight normalization
position = signal.div(signal.sum(axis=1), axis=0) * 1e8
```

### Lookback Data Loading
```python
# Load extra historical data for calculations
_start = int((datetime.strptime(str(start), "%Y%m%d") - timedelta(days=252 * 3)).strftime("%Y%m%d"))
```

### Position Shift
```python
# Shift positions by 1 day (use next-day data)
position = position.shift(1).fillna(0)
```

## Model Naming Convention

Alpha identifier format: `{exchange}.{universe}.{instrument_type}.{user_id}.{model_name}`

Example: `krx.krx.stock.ldh0127.bonus_v1`

## Development Guidelines

1. **Always include** `model_meta.json` with complete metadata
2. **Always include** `summary.md` documenting the strategy
3. **Use standard class names**: `Alpha` for `am.py`, `Portfolio` for `pf.py`
4. **Return DataFrames** from `get()` method with proper index/columns
5. **Shift positions** when using same-day data (close prices available next day)
6. **Scale to 1e8** for standard position sizing
7. **Handle NaN values** appropriately (fillna, dropna)

## Testing Notes

- Models receive `start` and `end` as integer dates
- Expected return: pandas DataFrame with position values
- Position values should sum to approximately 1e8 per row (when normalized)
