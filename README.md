# crypto-signals

Rule-based BUY / HOLD / SELL alerts for BTC, ETH and SOL. Alert only: it never places orders.

    pip install -r requirements.txt
    python daily.py                                  # today's signals, logged to journal.csv
    python daily.py trade SOL buy 100 --units 0.95   # record a trade you made (money, coins)
    python daily.py skip SOL                         # record a skipped monthly buy
    python backtest.py                               # the plan vs buying the full deposit monthly
    python backtest.py --start 2023-01-01
    python alts.py                                   # altcoin status: SELL / WATCH / HOLD / WEAK / DEAD
    python alts.py --backtest                        # altcoin sell rules vs just holding
    python weekly.py                                 # full weekly report (core + altcoins)
    python -m pytest tests

The plan (all numbers in `rules.yaml`): deposit a fixed amount each month, make a regular
buy, buy more when the market is fearful, skip the buy when it is high, keep unspent money
as a reserve for extreme-fear dips, and sell in tranches only in greed and only in profit.

- `data.py`: daily prices from Coinbase's public API and the Fear & Greed index, cached in `data/`.
- `strategy.py`: the decision logic shared by the backtest and the daily run.
- `alts.py`: altcoin tier. No buy signals; it flags strength to sell into and coins that are fading or dead.
- `weekly.py`: the report sent by the weekly email.
- `output/`: backtest tables, trade lists and charts.

The backtest fills each signal at the next day's open with fees. Fear & Greed data starts
February 2018; Coinbase SOL prices start June 2021.
