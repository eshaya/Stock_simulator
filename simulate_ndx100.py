# !/usr/bin/env python3
import argparse
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt

def run_simulator(args):
    # Load price data
    pricerows = pd.read_csv(args.price_file, index_col=0, parse_dates=True)
    pricerows = pricerows.loc[args.start_date:args.end_date]
    tickers = pricerows.columns.tolist()

    # Initial equity and cash
    equity = args.init_equity
    cash = equity * (1.0 - args.init_invest_fraction)
    shares_held = {t: 0.0 for t in tickers}

    if args.debug:
        print(f"Simulation from {pricerows.index[0].date()} to {pricerows.index[-1].date()}")
        print(f"Initial equity: {equity:.2f}, "
              f"invest fraction={args.init_invest_fraction}, "
              f"cash={cash:.2f}")

    # First-day initial investment
    first_pricerows = pricerows.iloc[0]
    invest_amt = equity * args.init_invest_fraction
    total_invest = invest_amt  
    per_stock = invest_amt / len(tickers)
    cumulative_buys = {t: per_stock for t in tickers}
    cumulative_sold = {t: 0 for t in tickers}
    for t in tickers:
        p = first_pricerows[t]
        if pd.isna(p) or p <= 0:
            continue
        initial_shares = per_stock // p
        shares_held[t] += initial_shares
        cash += per_stock - initial_shares * p  # leftover fractional
        if args.debug:
            print(f"Init buy {initial_shares:.2f} {t} at {p:.2f}")

    last_contrib_month = pricerows.index[0].month
    invest_curve, equity_curve, bh_curve, dates = [], [], [], []

    # Buy & hold baseline (never changes)
    bh_shares = {t: shares_held[t] for t in tickers}
    bh_invested_each = per_stock


    # Simulation loop
    for i, date in enumerate(pricerows.index):
        price = pricerows.iloc[i]

        # Monthly contribution
        if date.month != last_contrib_month:
            cash += args.monthly_contribution
            last_contrib_month = date.month
            if args.debug:
                print(f"{date.date()} contrib {args.monthly_contribution:.2f}, cash={cash:.2f}")

        # Rebalance every N days
        if i % args.rebalance_days == 0 and i > 0:
            for t in tickers:
                p = price[t]
                if pd.isna(p) or p <= 0: continue
                hval = shares_held[t] * p

                # SELL (momentum mode = sell losers)
                sell_amt = hval * args.sell_fraction
                shares_to_sell = sell_amt // p
                if shares_to_sell > 0:
                    shares_held[t] -= shares_to_sell
                    cash += shares_to_sell * p
                    cumulative_sold[t] += shares_to_sell * p
                    if args.debug:
                        print(f"{date.date()} SELL {shares_to_sell:.2f} {t} @ {p:.2f}")

                # BUY (momentum mode = buy winners)
                buy_amt = hval * args.buy_fraction
                shares_to_buy = min(buy_amt // p, cash // p)
                if shares_to_buy > 0:
                    shares_held[t] += shares_to_buy
                    cash -= shares_to_buy * p
                    cumulative_buys[t] += shares_to_buy * p

                    if args.debug:
                        print(f"{date.date()} BUY {shares_to_buy:.2f} {t} @ {p:.2f}")

        # Compute equity
        port_val = sum(shares_held[t] * price[t] for t in tickers if not pd.isna(price[t]))
        total_equity = cash + port_val
        total_invest = sum(cumulative_buys[t] - cumulative_sold[t] for t in tickers if not pd.isna(price[t]))
        bh_val = sum(bh_shares[t] * price[t] for t in tickers if not pd.isna(price[t]))
        dates.append(date)
        invest_curve.append(total_invest)
        equity_curve.append(total_equity)
        bh_curve.append(bh_val)

    # Final report
    final_equity = equity_curve[-1]
    final_invest = invest_curve[-1]
    net_gain = final_equity - final_invest
    pct_gain = 100 * (final_equity / final_invest - 1) if final_invest > 0 else 0

    print(f"\nFinal equity: ${final_equity:,.2f}")
    print(f"Total invested: ${final_invest:,.2f}")
    print(f"Net gain: ${net_gain:,.2f}")
    print(f"% gain: {pct_gain:.2f}%")

    # Plot results
    fig, axs = plt.subplots(3, 1, figsize=(10, 12))

    # Equity curve
    axs[0].plot(dates, equity_curve, label="Strategy", color="tab:blue")
    axs[0].plot(dates, bh_curve, label="Buy & Hold", color="tab:orange")
    axs[0].set_title("Equity Curve")
    axs[0].legend()

    # % gain
    pct_curve = np.array(equity_curve)/np.array(invest_curve) - 1
    pct_bh = np.array(bh_curve)/invest_amt - 1
    axs[1].plot(dates, pct_curve*100, label="Strategy", color="tab:blue")
    axs[1].plot(dates, pct_bh*100, label="Buy & Hold", color="tab:orange")
    axs[1].set_title("% Gain")
    axs[1].legend()

    # Per-stock comparison
    dollar_gains = []
    pct_gains = []
    bh_dollar_gains = []
    bh_pct_gains = []

    for t in tickers:
        final_hold = shares_held[t] * pricerows[t].iloc[-1]       # whats left at end
        final_value = cumulative_sold[t] + final_hold           # realized + remaining
        invested = cumulative_buys[t]

        # strategy result
        if invested > 0:
            strat_dgain = final_value - invested
            strat_pgain = 100. * (final_value / invested - 1.)
        else:
            strat_dgain = 0
            strat_pgain = 0

        dollar_gains.append(strat_dgain)
        pct_gains.append(strat_pgain)

        # buy & hold result (baseline: equal-weighted initial investment only)
        end_price = pricerows[t].iloc[-1]

        if not pd.isna(end_price):
            bh_val_each = bh_shares[t] * end_price
            bh_dgain = bh_val_each - bh_invested_each
            bh_pgain = 100.0 * (bh_val_each / bh_invested_each - 1.0)
        else:
            bh_dgain, bh_pgain = 0, 0

        bh_dollar_gains.append(bh_dgain)
        bh_pct_gains.append(bh_pgain)
        print(f"strat_dgain: {strat_dgain:.2f} ")
        print(f"bh_dgain: {bh_dgain:.2f} ")
        print(f"bh_dgain: {bh_dgain:.2f} ")
        print(f"bh_pgain: {bh_pgain:.2f} ")
        print(f"bh_val_each: {bh_val_each:.2f} ")
        print(f"bh_invested_each: {bh_invested_each:.2f} ")

    # Scatter plot
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
    parser.add_argument("--price-file", type=str, default="ndx100_prices.csv")
    parser.add_argument("--init-equity", type=float, default=100000)
    parser.add_argument("--init-invest-fraction", type=float, default=1.0)
    parser.add_argument("--buy-fraction", type=float, default=0.05)
    parser.add_argument("--sell-fraction", type=float, default=0.05)
    parser.add_argument("--rebalance-days", type=int, default=5)
    parser.add_argument("--monthly-contribution", type=float, default=1000)
    parser.add_argument("--start-date", type=str, default="2006-01-03")
    parser.add_argument("--end-date", type=str, default="2023-12-31")
    parser.add_argument("--debug", action="store_true")
    args = parser.parse_args()
    
    
    run_simulator(args)
