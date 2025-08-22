#!/usr/bin/env python3
"""Simple stock trading simulator with momentum-based rebalancing."""

import argparse
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt


def run_simulator(args):
    """Run the trading simulation using parameters from ``args``.
    """
    # ------------------------------------------------------------------
    # Load and filter price data for the desired date range. Each column
    # corresponds to a ticker symbol and each row to a trading day.
    # ------------------------------------------------------------------
    pricerows = pd.read_csv(args.price_file, index_col=0, parse_dates=True)
    pricerows = pricerows.loc[args.start_date:args.end_date]
    tickers = pricerows.columns.tolist()

    # ------------------------------------------------------------------
    # Set up starting portfolio values.
    #   equity      - total account equity at the start of the simulation
    #   shares_held - mapping of ticker -> number of shares currently held
    # ------------------------------------------------------------------
    equity = args.init_equity
    shares_held = {t: 0.0 for t in tickers}

    # ``total_contrib`` tracks dollars actually invested in the market.
    total_contrib = 0.0

    # Track actual dollars spent or received per ticker so that
    # total invested reflects real cash flows instead of the intended
    # allocation.
    cumulative_buys = {t: 0.0 for t in tickers}
    cumulative_sold = {t: 0.0 for t in tickers}

    if args.debug:
        print(
            f"Simulation from {pricerows.index[0].date()} to {pricerows.index[-1].date()}"
        )
        print(
            f"Initial equity: {equity:.2f}, "
            f"invest fraction={args.init_invest_fraction}, "
        )

    # ------------------------------------------------------------------
    # Invest the specified fraction of equity on the first trading day.
    #   initial_invest_amt  - total dollars to invest initially
    #                In B & H comparison bh_invest_amt does not change
    #   cash         - uninvested portion of equity
    #   per_stock    - equal-weight amount invested into each ticker
    #   cumulative_buys/sold track dollars spent/received over time
    # ------------------------------------------------------------------
    first_pricerows = pricerows.iloc[0]
    initial_invest_amt = equity * args.init_invest_fraction
    per_stock = initial_invest_amt / len(tickers)
    cash = equity  # Start with all equity as cash and reduce as each ticker is bought
    for t in tickers:
        p = first_pricerows[t]
        if pd.isna(p) or p <= 0:
            continue
        initial_shares = per_stock // p
        spent = per_stock
        shares_held[t] = initial_shares
        cumulative_buys[t] += spent
        total_contrib += spent
        cash -= spent
        if args.debug:
            print(f"Init buy {initial_shares:.2f} {t} at {p:.2f}")

    # Track month of last contribution so we know when to add more cash.
    last_contrib_month = pricerows.index[0].month
    # Curves for plotting and analysis after the simulation runs.
    # ``invest_curve`` will hold the running total of contributed capital.
    cash_curve, invest_curve, equity_curve, bh_curve, dates = [], [], [], [], []

    # Buy & hold baseline: used for comparison; holds initial shares forever
    bh_shares = {t: shares_held[t] for t in tickers}
    bh_invested_each = per_stock

    #-------------------------------------------------------------------
    # TRADING
    #-------------------------------------------------------------------
    for i, date in enumerate(pricerows.index):
        price = pricerows.iloc[i]

        # --------------------------------------------------------------
        # Add monthly contribution when the calendar month changes.
        # --------------------------------------------------------------
        if date.month != last_contrib_month:
            cash += args.monthly_contribution
            last_contrib_month = date.month
            if args.debug:
                print(
                    f"{date.date()} contrib {args.monthly_contribution:.2f}, cash={cash:.2f}"
                )

        # --------------------------------------------------------------
        # Rebalance every N trading days. Compare current price to the
        # price at the previous rebalance to determine momentum.
        # --------------------------------------------------------------
        if i % args.rebalance_days == 0 and i >= args.rebalance_days:
            for t in tickers:
                p = price[t]
                prev_price = pricerows[t].iloc[i - args.rebalance_days]
                if any(pd.isna(x) or x <= 0 for x in (p, prev_price)):
                    continue

                pct_change = (p - prev_price) / prev_price * 100.0
                hval = shares_held[t] * p  # current holding value in dollars

                if pct_change >= args.buy_threshold:
                    buy_amt = hval * args.buy_fraction
                    shares_to_buy = min(buy_amt // p, cash // p)
                    spent = shares_to_buy
                    if shares_to_buy > 0:
                        shares_held[t] += shares_to_buy
                        cash -= spent
                        cumulative_buys[t] += spent
                        total_contrib += spent
                        if args.debug:
                            print(f"{date.date()} BUY {shares_to_buy:.2f} {t} @ {p:.2f}")
                elif pct_change <= -args.sell_threshold:
                    sell_amt = hval * args.sell_fraction
                    shares_to_sell = sell_amt // p
                    if shares_to_sell > 0:
                        shares_held[t] -= shares_to_sell
                        cash += shares_to_sell * p
                        cumulative_sold[t] += shares_to_sell * p
                        total_contrib -= spent
                        if args.debug:
                            print(f"{date.date()} SELL {shares_to_sell:.2f} {t} @ {p:.2f}")
    
        # Record portfolio metrics for plotting.
        dates.append(date)
        invest_curve.append(total_contrib)
        cash_curve.append(cash)

        portfolio_val = sum(shares_held[t]*price[t] for t in tickers if not pd.isna(price[t]))
        total_equity = cash + portfolio_val
        equity_curve.append(total_equity)
        
        # Follow equity in the B&H case
        bh_val = sum(bh_shares[t]*price[t] for t in tickers if not pd.isna(price[t]))
        bh_curve.append(bh_val)


    # --------------------------------------------------------------
    # Final performance summary
    # --------------------------------------------------------------
    final_equity = equity_curve[-1]
    final_invest = invest_curve[-1]  # Same as total_contrib final value
    net_gain = final_equity - final_invest
    pct_gain = 100 * (final_equity / final_invest - 1) if final_invest > 0 else 0

    print(f"\nFinal equity: ${final_equity:,.2f}")
    print(f"Total invested: ${final_invest:,.2f}")
    print(f"Net gain: ${net_gain:,.2f}")
    print(f"% gain: {pct_gain:.2f}%")

    # --------------------------------------------------------------
    # Plot equity curves and per-stock performance
    # --------------------------------------------------------------
    fig, axs = plt.subplots(3, 1, figsize=(10, 12))

    axs[0].plot(dates, equity_curve, label="Strategy", color="tab:blue")
    axs[0].plot(dates, bh_curve, label="Buy & Hold", color="tab:orange")
    axs[0].plot(dates, invest_curve, label="Investment", color="black")
    axs[0].plot(dates, cash_curve, label="Cash", color="gold")
    axs[0].set_title("Equity Curve")
    axs[0].legend()

    equity_arr = np.array(equity_curve)
    invest_arr = np.array(invest_curve)
    pct_curve = np.zeros_like(equity_arr)
    valid = invest_arr > 0
    pct_curve[valid] = equity_arr[valid] / invest_arr[valid] - 1

    bh_arr = np.array(bh_curve)
    if initial_invest_amt > 0:
        pct_bh = bh_arr / initial_invest_amt - 1
    else:
        pct_bh = np.zeros_like(bh_arr)

    axs[1].plot(dates, pct_curve * 100, label="Strategy", color="tab:blue")
    axs[1].plot(dates, pct_bh * 100, label="Buy & Hold", color="tab:orange")
    axs[1].set_title("% Gain")
    axs[1].legend()

    # Collect per-stock gains for scatter plot
    dollar_gains, pct_gains = [], []
    bh_dollar_gains, bh_pct_gains = [], []

    for t in tickers:
        final_hold = shares_held[t] * pricerows[t].iloc[-1]
        final_value = cumulative_sold[t] + final_hold
        invested = cumulative_buys[t]

        if invested > 0:
            strat_dgain = final_value - invested
            strat_pgain = 100.0 * (final_value / invested - 1.0)
        else:
            strat_dgain = 0
            strat_pgain = 0

        dollar_gains.append(strat_dgain)
        pct_gains.append(strat_pgain)

        end_price = pricerows[t].iloc[-1]
        if not pd.isna(end_price):
            bh_val_each = bh_shares[t] * end_price
            bh_dgain = bh_val_each - bh_invested_each
            bh_pgain = 100.0 * (bh_val_each / bh_invested_each - 1.0)
        else:
            bh_dgain, bh_pgain = 0, 0

        bh_dollar_gains.append(bh_dgain)
        bh_pct_gains.append(bh_pgain)

        # Report per-ticker results for reference
        print(f"strat_dgain: {strat_dgain:.2f} ")
        print(f"bh_dgain: {bh_dgain:.2f} ")
        print(f"bh_pgain: {bh_pgain:.2f} ")
        print(f"bh_val_each: {bh_val_each:.2f} ")
        print(f"bh_invested_each: {bh_invested_each:.2f} ")

    axs[2].scatter(dollar_gains, pct_gains, color="tab:blue", label="Strategy")
    axs[2].scatter(bh_dollar_gains, bh_pct_gains, color="tab:orange", label="Buy & Hold")
    axs[2].set_xlabel("Gain/Loss $")
    axs[2].set_ylabel("% Gain")
    axs[2].set_title("Per-stock gain vs % gain")
    axs[2].legend()

    plt.tight_layout()
    plt.show()


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--price-file", type=str, default="ndx100_prices.csv",
                        help="CSV file containing historical price data")
    parser.add_argument("--init-equity", type=float, default=100000,
                        help="Starting account equity in dollars")
    parser.add_argument("--init-invest-fraction", type=float, default=1.0,
                        help="Fraction of initial equity invested on day one")
    parser.add_argument("--buy-fraction", type=float, default=0.05,
                        help="Fraction of current holdings to buy when threshold met")
    parser.add_argument("--sell-fraction", type=float, default=0.05,
                        help="Fraction of holdings to sell when threshold met")
    parser.add_argument("--buy-threshold", type=float, default=5.0,
                        help="Percent price increase over the rebalance period required to trigger a buy")
    parser.add_argument("--sell-threshold", type=float, default=5.0,
                        help="Percent price decrease over the rebalance period required to trigger a sell")
    parser.add_argument("--rebalance-days", type=int, default=5,
                        help="Trading days between rebalancing checks")
    parser.add_argument("--monthly-contribution", type=float, default=1000,
                        help="Cash added at the start of each month")
    parser.add_argument("--start-date", type=str, default="2006-01-03",
                        help="Simulation start date (YYYY-MM-DD)")
    parser.add_argument("--end-date", type=str, default="2023-12-31",
                        help="Simulation end date (YYYY-MM-DD)")
    parser.add_argument("--debug", action="store_true")
    args = parser.parse_args()

    run_simulator(args)

