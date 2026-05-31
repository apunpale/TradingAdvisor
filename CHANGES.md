# Recent Changes & Fixes

## Monthly Contribution Support
Added support for monthly recurring investments alongside the initial investment.

### Changes Made:

#### 1. **Src/portfolio.py**
- Added `contributions` field to track all contributions with dates
- Added `contribute(amount, date)` method to record external cash deposits

#### 2. **Src/backtest.py**
- Added `monthly_contribution` parameter to `run_backtest()`
- Deposits monthly contribution on the first trading day of each new month
- Updated `run_backtest_with_benchmark()` to pass through `monthly_contribution`

#### 3. **Src/dashboard.py**
- Added "Monthly contribution (£)" input field in the UI
- Displays total contributions as a metric in the results
- Enhanced `benchmark_return()` function with robust error handling:
  - Checks if ticker is None before attempting to access panel data
  - Handles TypeError and KeyError exceptions gracefully
  - Validates that prices are positive before calculating returns
- Fixed relative performance graph and rolling alpha/beta sections:
  - Added null checks for benchmark_ticker
  - Wrapped chart rendering in try-except block
  - Shows info message when no benchmark is available
  - Prevents crashes when benchmark data is missing

## Key Improvements:

### Robustness
- Dashboard no longer crashes if benchmark data is unavailable
- Graceful fallback when benchmark ticker is None
- Better error messages for users when benchmark comparison is disabled

### Visibility
- Total contributions are now displayed in the backtest results
- Contributions are tracked and recorded for analysis
- Users can see the impact of regular investments

## Testing
All changes have been verified with:
- Unit tests (pytest) - all passing
- Integration tests with monthly contributions
- Benchmark calculation tests with various scenarios

## How to Use:

### Backtest with Monthly Contributions:
1. Open the Streamlit dashboard:
   ```bash
   streamlit run Src/dashboard.py
   ```

2. Configure the backtest:
   - Enter "Initial investment" (one-time)
   - Enter "Monthly contribution" (recurring)
   - Select date range and tickers
   - Optionally select a benchmark

3. Results show:
   - Final portfolio value (including contributions)
   - Total return (including contribution impact)
   - Monthly contribution timeline in the contributions field
   - Benchmark comparison (if available)

### Portfolio Value Calculation:
The portfolio value includes:
- Initial investment
- All accumulated monthly contributions
- Returns on all invested capital

### Monthly Contribution Timing:
- Contributions are deposited on the first trading day of each month
- They can be immediately invested if buy signals are triggered that day
- Contributions are recorded with their exact dates
