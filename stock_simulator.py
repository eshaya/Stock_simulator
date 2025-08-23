#!/usr/bin/env python3
"""Simple stock trading simulator with momentum-based rebalancing."""

import argparse
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import sys

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

    # Load QQQ data for comparison
    try:
        qqq_pricerows = pd.read_csv(args.qqq_price_file, index_col=0, parse_dates=True)
        # Align QQQ data to the main dataframe's index to ensure they share the same time scale
        qqq_prices = qqq_pricerows['Close'].reindex(pricerows.index, method='ffill').bfill()
    except FileNotFoundError:
        print(f"Error: QQQ price file '{args.qqq_price_file}' not found. Skipping QQQ comparison.")
        qqq_prices = None
    except (KeyError, IndexError) as e:
        print(f"Error reading '{args.qqq_price_file}': A required column was not found or the file is malformed. {e}")
        qqq_prices = None

    # ------------------------------------------------------------------
    # Set up starting portfolio values.
    #   equity      - total account equity at the start of the simulation
    #   shares_held - mapping of ticker -> number of shares currently held
    # ------------------------------------------------------------------
    equity = args.init_equity
    shares_held = {t: 0.0 for t in tickers}
 
    # cash_contrib sums the monthly contributions over time
    cash_contrib = 0.0  

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
    # Initial investment logic.
    # Try to buy each stock on the first day it has a valid price,
    # within a 10-day window. If no price is found, drop the stock.
    # ------------------------------------------------------------------
    initial_invest_amt = equity * args.init_invest_fraction
    cash = equity

    tradable_tickers = []
    initial_purchases = {} # t -> (shares, price, date)
    for t in tickers:
        # Find the first valid price within the first 10 days
        first_10_days = pricerows[t].iloc[:10]
        first_valid_price_series = first_10_days.dropna()
        if not first_valid_price_series.empty:
            first_valid_date = first_valid_price_series.index[0]
            first_valid_price = first_valid_price_series.iloc[0]
            if first_valid_price > 0:
                tradable_tickers.append(t)
                initial_purchases[t] = {'price': first_valid_price, 'date': first_valid_date}
                continue
        # If we get here, the stock is untradable
        print(f"Warning: No valid price for {t} within the first 10 days. Dropping.")

    # Update tickers list and recalculate per-stock allocation
    tickers = tradable_tickers
    per_stock = initial_invest_amt / len(tickers) if tickers else 0

    bh_invested = {} # Store actual B&H investment per ticker
    # Perform initial buys
    for t in tickers:
        p = initial_purchases[t]['price']
        if pd.isna(p) or p <=0: # Should not happen due to above logic, but as a safeguard
            print(f"Error: Invalid price {p} for {t} during initial buy. Exiting.")
            sys.exit(1)
        shares_to_buy = per_stock // p
        spent = shares_to_buy * p
        shares_held[t] += shares_to_buy
        cumulative_buys[t] += spent
        cash -= spent

        # For buy & hold comparison
        bh_invested[t] = spent
        if args.debug:
            print(f"Init buy {shares_to_buy:.2f} {t} at {p:.2f} on {initial_purchases[t]['date'].date()}")

    # Track month of last contribution so we know when to add more cash.
    last_contrib_month = pricerows.index[0].month

    # Curves for plotting and analysis after the simulation runs.
    # ``invest_curve`` will hold the running total of contributed capital.
    cash_curve, invest_curve, equity_curve, bh_curve, qqq_bh_curve, dates = [], [], [], [], [], []

    # Buy & hold baseline: used for comparison; holds initial shares forever
    bh_shares = {t: shares_held[t] for t in tickers}

    # QQQ Buy & Hold baseline
    qqq_bh_shares = 0
    if qqq_prices is not None:
        first_qqq_price = qqq_prices.iloc[0]
        if not pd.isna(first_qqq_price) and first_qqq_price > 0:
            qqq_bh_shares = initial_invest_amt / first_qqq_price
        else:
            print("Warning: Invalid starting price for QQQ. Skipping QQQ comparison.")
            qqq_prices = None  # Disable QQQ tracking

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
            cash_contrib += args.monthly_contribution
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
                    shares_to_buy = min(buy_amt, cash) // p
                    spent = shares_to_buy * p
                    if shares_to_buy > 0:
                        shares_held[t] += shares_to_buy
                        cash -= spent
                        cumulative_buys[t] += spent
                        if args.debug:
                            print(f"{date.date()} BUY {shares_to_buy:.2f} {t} @ {p:.2f}")
                elif pct_change <= -args.sell_threshold:
                    sell_amt = hval * args.sell_fraction
                    shares_to_sell = sell_amt // p
                    received = shares_to_sell * p
                    if shares_to_sell > 0:
                        shares_held[t] -= shares_to_sell
                        cash += received
                        cumulative_sold[t] += received
                        if args.debug:
                            print(f"{date.date()} SELL {shares_to_sell:.2f} {t} @ {p:.2f}")
    
        # Record portfolio metrics for plotting.
        dates.append(date)
        # The "Net Invested" curve shows the net capital deployed in the market.
        net_invested = sum(cumulative_buys.values()) - sum(cumulative_sold.values())
        invest_curve.append(net_invested)
        cash_curve.append(cash)

        portfolio_val = sum(shares_held[t]*price[t] for t in tickers if not pd.isna(price[t]))
        # Total equity is cash + (portfolio value) - (cash contributions from external sources)
        total_equity = cash - cash_contrib + portfolio_val
        equity_curve.append(total_equity)
        
        # Follow equity in the B&H case
        bh_val = sum(bh_shares[t]*price[t] for t in tickers if not pd.isna(price[t]))
        bh_curve.append(bh_val)

        # Follow equity in the QQQ B&H case
        if qqq_prices is not None:
            qqq_price_today = qqq_prices.loc[date]
            if not pd.isna(qqq_price_today):
                qqq_bh_curve.append(qqq_bh_shares * qqq_price_today)
            else: # If price is missing, carry forward the last value
                qqq_bh_curve.append(qqq_bh_curve[-1] if qqq_bh_curve else 0)


    # --------------------------------------------------------------
    # Final performance summary
    # --------------------------------------------------------------
    final_equity = equity_curve[-1]
    # Investment is the initial equity; contributions are treated as a loan to be "paid back".
    final_invest = args.init_equity
    net_gain = final_equity - final_invest
    pct_gain = 100 * (net_gain / final_invest) if final_invest > 0 else 0

    print("\n--- Strategy Performance ---")
    print(f"Final equity:   ${final_equity:,.2f}")
    print(f"Initial equity: ${final_invest:,.2f}")
    print(f"Net gain:       ${net_gain:,.2f}")
    print(f"% gain:         {pct_gain:.2f}%")

    # B&H NDX100 Performance
    final_bh_equity = bh_curve[-1]
    # B&H is benchmarked against the same initial investment amount as the strategy.
    bh_total_invested = sum(bh_invested.values())
    bh_net_gain = final_bh_equity - bh_total_invested
    bh_pct_gain = 100 * (bh_net_gain / bh_total_invested) if bh_total_invested > 0 else 0
    print("\n--- Buy & Hold (NDX100) ---")
    print(f"Final equity:   ${final_bh_equity:,.2f}")
    print(f"Net gain:       ${bh_net_gain:,.2f}")
    print(f"% gain:         {bh_pct_gain:.2f}%")

    # QQQ B&H Performance
    if qqq_prices is not None:
        final_qqq_equity = qqq_bh_curve[-1]
        qqq_net_gain = final_qqq_equity - initial_invest_amt
        qqq_pct_gain = 100 * (final_qqq_equity / initial_invest_amt - 1) if initial_invest_amt > 0 else 0
        print("\n--- Buy & Hold (QQQ) ---")
        print(f"Final equity:   ${final_qqq_equity:,.2f}")
        print(f"Net gain:       ${qqq_net_gain:,.2f}")
        print(f"% gain:         {qqq_pct_gain:.2f}%")

    # --------------------------------------------------------------
    # Plot equity curves and per-stock performance
    # --------------------------------------------------------------
    fig, axs = plt.subplots(3, 1, figsize=(10, 12))

    equity_arr = np.array(equity_curve)
    # The percentage gain is based on the initial capital, consistent with the "loan" model.
    pct_curve = np.zeros_like(equity_arr)
    if args.init_equity > 0:
        pct_curve = (equity_arr / args.init_equity) - 1

    bh_arr = np.array(bh_curve)
    if bh_total_invested > 0:
        pct_bh = bh_arr / bh_total_invested - 1
    else:
        pct_bh = np.zeros_like(bh_arr)

    # Plot of performances
    if qqq_prices is not None:
        axs[0].plot(dates, qqq_bh_curve, label="Buy & Hold (QQQ)", color="black", linewidth=2.5)
    
    axs[0].plot(dates, equity_curve, label="Strategy", color="tab:blue")
    axs[0].plot(dates, bh_curve, label="Buy & Hold (NDX100)", color="tab:orange")
    
    axs[0].plot(dates, invest_curve, label="Net Invested", color="gray", linestyle="--")
    axs[0].plot(dates, cash_curve, label="Cash", color="gold")
    axs[0].set_title("Portfolio Value")
    axs[0].legend()

    # Plot percentage gains
    axs[1].plot(dates, pct_curve * 100, label="Strategy", color="tab:blue")
    axs[1].plot(dates, pct_bh * 100, label="Buy & Hold (NDX100)", color="tab:orange")
    if qqq_prices is not None:
        pct_qqq_bh = (np.array(qqq_bh_curve) / initial_invest_amt - 1) * 100
        axs[1].plot(dates, pct_qqq_bh, label="Buy & Hold (QQQ)", color="black", linewidth=2.5)

    axs[1].set_title("Percentage Gain")
    axs[1].legend()

    # Collect per-stock gains for scatter plot
    dollar_gains, pct_gains = [], []
    bh_dollar_gains, bh_pct_gains = [], []

    for t in tickers:
        end_price = pricerows[t].iloc[-1]
        final_hold = shares_held[t] * end_price
        final_value = cumulative_sold[t] + final_hold
        invested = cumulative_buys[t]

        if invested > 0:
            strat_dgain = final_value - invested
            strat_pgain = 100.0 * (strat_dgain / invested)
        else:
            strat_dgain = 0
            strat_pgain = 0

        dollar_gains.append(strat_dgain)
        pct_gains.append(strat_pgain)

        bh_invested_t = bh_invested.get(t, 0)
        if not pd.isna(end_price) and bh_invested_t > 0:
            bh_val_each = bh_shares[t] * end_price
            bh_dgain = bh_val_each - bh_invested_t
            bh_pgain = 100.0 * (bh_dgain / bh_invested_t)
        else:
            bh_dgain, bh_pgain = 0, 0

        bh_dollar_gains.append(bh_dgain)
        bh_pct_gains.append(bh_pgain)

        # Report per-ticker results for reference
        print(f"ticker: {t:s} ")
        print(f"strat_dgain: {strat_dgain:.2f} ")
        print(f"invested: {invested:.2f} ")
        print(f"strat_pgain: {strat_pgain:.2f} ")
        print(f"bh_dgain: {bh_dgain:.2f} ")
        print(f"bh_pgain: {bh_pgain:.2f} ")
        print(f"bh_invested_each: {bh_invested_t:.2f} ")

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
    parser.add_argument("--qqq-price-file", type=str, default="qqq_prices.csv",
                        help="CSV file for QQQ B&H comparison")
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
