#!/usr/bin/env python3
"""Simple stock trading simulator with momentum-based rebalancing."""

import argparse
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import sys
import csv

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

    # Track per-stock market value over time for composition plot
    values_df = pd.DataFrame(0.0, index=pricerows.index, columns=tickers)

    # --- open trade log (overwrite each run) ---
    log_fname = args.log_file
    log_fh = open(log_fname, "w", newline='')
    log_writer = csv.writer(log_fh)
    # include shares_held after ticker
    log_writer.writerow(['date', 'ticker', 'shares_held', 'share_change', 'price', 'final_value'])

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
        first_30_days = pricerows[t].iloc[:30]
        first_valid_price_series = first_30_days.dropna()
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

        # Log initial buy (formatted: shares 2dp, price and final value as money)
        log_writer.writerow([
            initial_purchases[t]['date'].date().isoformat(),
            t,
            f"{shares_held[t]:.2f}",
            f"{shares_to_buy:.2f}",
            f"${p:.2f}",
            f"${(shares_held[t] * p):.2f}"
        ])

    # Track month of last contribution so we know when to add more cash.
    last_contrib_month = pricerows.index[0].month

    # Curves for plotting and analysis after the simulation runs.
    # ``invest_curve`` will hold the running total of net capital deployed.
    cash_curve, invest_curve, equity_curve, bh_curve, qqq_bh_curve, dates, cash_contrib_curve = [], [], [], [], [], [], []
    bh_capital_curve, qqq_capital_curve = [], []

    # Buy & hold baseline: used for comparison; holds initial shares forever
    bh_shares = {t: shares_held[t] for t in tickers}
    bh_total_invested = sum(bh_invested.values())

    # QQQ Buy & Hold baseline
    qqq_bh_shares = 0
    qqq_total_invested = 0
    if qqq_prices is not None:
        first_qqq_price = qqq_prices.iloc[0]
        if not pd.isna(first_qqq_price) and first_qqq_price > 0:
            qqq_bh_shares = initial_invest_amt / first_qqq_price
            qqq_total_invested = initial_invest_amt
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

            # B&H and QQQ strategies also invest the monthly contribution
            bh_total_invested += args.monthly_contribution
            per_stock_contrib = args.monthly_contribution / len(tickers) if tickers else 0
            if per_stock_contrib > 0:
                for t in tickers:
                    p = price[t]
                    if not pd.isna(p) and p > 0:
                        shares_to_buy = per_stock_contrib / p
                        bh_shares[t] += shares_to_buy
            
            if qqq_prices is not None:
                qqq_total_invested += args.monthly_contribution
                qqq_price_today = qqq_prices.loc[date]
                if not pd.isna(qqq_price_today) and qqq_price_today > 0:
                    qqq_shares_to_buy = args.monthly_contribution / qqq_price_today
                    qqq_bh_shares += qqq_shares_to_buy

            if args.debug:
                print(
                    f"{date.date()} contrib {args.monthly_contribution:.2f}, cash={cash:.2f}"
                )

        # --------------------------------------------------------------
        # Rebalance every N trading days. On rebalance days, we:
        # 1. Calculate momentum (pct_change) for all tickers.
        # 2. Create lists of potential buys and sells based on thresholds.
        # 3. Sort sells by worst momentum and buys by best momentum.
        # 4. Execute all sells, then execute buys until cash runs out.
        # --------------------------------------------------------------
        if i % args.rebalance_days == 0 and i >= args.rebalance_days:
            potential_trades = []
            for t in tickers:
                p = price[t]
                prev_price = pricerows[t].iloc[i - args.rebalance_days]
                
                if any(pd.isna(x) or x <= 0 for x in (p, prev_price)):
                    continue

                pct_change = (p - prev_price) / prev_price * 100.0
                potential_trades.append({'ticker': t, 'pct_change': pct_change, 'price': p})

            # Separate into buys and sells, then sort them by momentum
            buys = [trade for trade in potential_trades if trade['pct_change'] >= args.buy_threshold]
            sells = [trade for trade in potential_trades if trade['pct_change'] <= -args.sell_threshold]
            
            buys.sort(key=lambda x: x['pct_change'], reverse=True) # Best momentum first
            sells.sort(key=lambda x: x['pct_change'])              # Worst momentum first

            # Execute sells (limit total sell proceeds to at most current cash; allow partial shares)
            initial_cash_before_sells = cash
            remaining_sell_cap = initial_cash_before_sells
            for trade in sells:
                t, p, pct_change = trade['ticker'], trade['price'], trade['pct_change']
                if p <= 0 or pd.isna(p):
                    continue
                hval = shares_held[t] * p
                desired_sell = hval * args.sell_fraction
                # cap proceeds to remaining_sell_cap and compute partial shares accordingly
                allowed_proceeds = min(desired_sell, remaining_sell_cap)
                if allowed_proceeds <= 0.0:
                    continue
                shares_to_sell = allowed_proceeds / p
                # ensure we don't sell more than we hold (numerical safety)
                if shares_to_sell > shares_held[t]:
                    shares_to_sell = shares_held[t]
                received = shares_to_sell * p
                if shares_to_sell > 0.0:
                    shares_held[t] -= shares_to_sell
                    cash += received
                    cumulative_sold[t] += received
                    remaining_sell_cap -= received
                    # log sell (negative share change) with formatting
                    log_writer.writerow([
                        date.date().isoformat(),
                        t,
                        f"{shares_held[t]:.2f}",
                        f"{-shares_to_sell:.2f}",
                        f"${p:.2f}",
                        f"${(shares_held[t] * p):.2f}"
                    ])
                    if args.debug:
                        print(f"{date.date()} SELL {shares_to_sell:.2f} {t} @ {p:.2f} (change: {pct_change:.2f}%)")
                if remaining_sell_cap <= 0.0:
                    if args.debug:
                        print(f"{date.date()} Sell cap reached: limited to initial cash ${initial_cash_before_sells:.2f}")
                    break

            # Execute buys (best performers first)
            for trade in buys:
                t, p, pct_change = trade['ticker'], trade['price'], trade['pct_change']
                hval = shares_held[t] * p
                buy_amt = hval * args.buy_fraction
                shares_to_buy = min(buy_amt, cash) // p
                spent = shares_to_buy * p
                if shares_to_buy > 0:
                    shares_held[t] += shares_to_buy
                    cash -= spent
                    cumulative_buys[t] += spent
                    # log buy (positive share change) with formatting
                    log_writer.writerow([
                        date.date().isoformat(),
                        t,
                        f"{shares_to_buy:.2f}",
                        f"${p:.2f}",
                        f"${(shares_held[t] * p):.2f}"
                    ])
                    if args.debug:
                        print(f"{date.date()} BUY {shares_to_buy:.2f} {t} @ {p:.2f} (change: {pct_change:.2f}%)")
    
        # Record daily portfolio metrics for plotting.
        dates.append(date)
        net_invested = sum(cumulative_buys.values()) - sum(cumulative_sold.values())
        invest_curve.append(net_invested)
        cash_curve.append(cash)
        cash_contrib_curve.append(cash_contrib)
        portfolio_val = sum(shares_held[t]*price[t] for t in tickers if not pd.isna(price[t]))
        # Total equity is the real-time total value of the account.
        total_equity = cash + portfolio_val
        equity_curve.append(total_equity)
        
        # record per-stock market values for this date (use 0 for missing prices)
        for t in tickers:
            p = price[t]
            if pd.isna(p) or p <= 0:
                values_df.at[date, t] = 0.0
            else:
                values_df.at[date, t] = shares_held[t] * p

        # Follow equity in the B&H case
        bh_val = sum(bh_shares[t]*price[t] for t in tickers if not pd.isna(price[t]))
        bh_curve.append(bh_val)
        bh_capital_curve.append(bh_total_invested)

        # Follow equity in the QQQ B&H case
        if qqq_prices is not None:
            qqq_price_today = qqq_prices.loc[date]
            if not pd.isna(qqq_price_today):
                qqq_bh_curve.append(qqq_bh_shares * qqq_price_today)
            else:  # If price is missing, carry forward the last value
                qqq_bh_curve.append(qqq_bh_curve[-1] if qqq_bh_curve else 0)
            qqq_capital_curve.append(qqq_total_invested)
        else:
            # If QQQ is disabled, append 0 or last value to keep lists aligned
            qqq_bh_curve.append(qqq_bh_curve[-1] if qqq_bh_curve else 0)
            qqq_capital_curve.append(qqq_capital_curve[-1] if qqq_capital_curve else 0)

    # --------------------------------------------------------------
    # Final performance summary
    # --------------------------------------------------------------

    final_equity = equity_curve[-1] # Final equity is the last value in the equity curve
    total_capital = args.init_equity + cash_contrib
    net_gain = final_equity - total_capital
    pct_gain = 100 * (net_gain / total_capital) if total_capital > 0 else 0

    print("\n--- Strategy Performance ---")
    print(f"Final equity:   ${final_equity:,.2f}")
    print(f"Total capital:  ${total_capital:,.2f}")
    print(f"Net gain:       ${net_gain:,.2f}")
    print(f"Overall % gain: {pct_gain:.2f}%")

    # B&H NDX100 Performance
    final_bh_equity = bh_curve[-1]
    bh_net_gain = final_bh_equity - bh_total_invested
    bh_pct_gain = 100 * (bh_net_gain / bh_total_invested) if bh_total_invested > 0 else 0
    print("\n--- Buy & Hold (NDX100) ---")
    print(f"Final equity:   ${final_bh_equity:,.2f}")
    print(f"Total capital:  ${bh_total_invested:,.2f}")
    print(f"Net gain:       ${bh_net_gain:,.2f}")
    print(f"% gain:         {bh_pct_gain:.2f}%")

    # QQQ B&H Performance
    if qqq_prices is not None:
        final_qqq_equity = qqq_bh_curve[-1]
        qqq_net_gain = final_qqq_equity - qqq_total_invested
        qqq_pct_gain = 100 * (qqq_net_gain / qqq_total_invested) if qqq_total_invested > 0 else 0
        print("\n--- Buy & Hold (QQQ) ---")
        print(f"Final equity:   ${final_qqq_equity:,.2f}")
        print(f"Total capital:  ${qqq_total_invested:,.2f}")
        print(f"Net gain:       ${qqq_net_gain:,.2f}")
        print(f"% gain:         {qqq_pct_gain:.2f}%")

    # --------------------------------------------------------------
    # Calculate annual performance for plotting
    # --------------------------------------------------------------
    data_dict = {
        'strategy': equity_curve,
        'invested': invest_curve,
        'contrib': cash_contrib_curve,
        'bh_ndx': bh_curve,
        'bh_capital': bh_capital_curve,
    }
    if qqq_prices is not None and qqq_bh_curve:
        data_dict['bh_qqq'] = qqq_bh_curve
        data_dict['qqq_capital'] = qqq_capital_curve

    results_df = pd.DataFrame(data_dict, index=pd.to_datetime(dates))
    annual_df = results_df.resample('YE').last()
    annual_df.index = annual_df.index.year

    # Strategy annual return
    equity_start = annual_df['strategy'].shift(1).fillna(args.init_equity)
    contrib_start = annual_df['contrib'].shift(1).fillna(0)
    contrib_for_year = annual_df['contrib'] - contrib_start
    gain_for_year = annual_df['strategy'] - equity_start
    investment_return = gain_for_year - contrib_for_year
    # Annual % gain is the investment return for the year divided by the equity at the start of the year.
    annual_df['strategy_pct_gain'] = (investment_return / equity_start.replace(0, np.nan)) * 100

    # B&H NDX100 annual return
    bh_initial_investment = sum(bh_invested.values())
    bh_start_equity = annual_df['bh_ndx'].shift(1).fillna(bh_initial_investment)
    bh_start_capital = annual_df['bh_capital'].shift(1).fillna(bh_initial_investment)
    bh_contrib_for_year = annual_df['bh_capital'] - bh_start_capital
    bh_gain_for_year = annual_df['bh_ndx'] - bh_start_equity
    bh_investment_return = bh_gain_for_year - bh_contrib_for_year
    # Annual % gain is the investment return for the year divided by the equity at the start of the year.
    annual_df['bh_ndx_pct_gain'] = (bh_investment_return / bh_start_equity.replace(0, np.nan)) * 100

    # B&H QQQ annual return
    if 'bh_qqq' in annual_df.columns:
        qqq_start_equity = annual_df['bh_qqq'].shift(1).fillna(initial_invest_amt)
        qqq_start_capital = annual_df['qqq_capital'].shift(1).fillna(initial_invest_amt)
        qqq_contrib_for_year = annual_df['qqq_capital'] - qqq_start_capital
        qqq_gain_for_year = annual_df['bh_qqq'] - qqq_start_equity
        qqq_investment_return = qqq_gain_for_year - qqq_contrib_for_year
        # Annual % gain is the investment return for the year divided by the equity at the start of the year.
        annual_df['bh_qqq_pct_gain'] = (qqq_investment_return / qqq_start_equity.replace(0, np.nan)) * 100

    # --------------------------------------------------------------
    # Plot equity curves and per-stock performance
    # --------------------------------------------------------------
    fig, axs = plt.subplots(3, 1, figsize=(10, 14), gridspec_kw={'height_ratios':[3,1,2]})
 
     # Plot of portfolio value over time
    if qqq_prices is not None:
         axs[0].plot(dates, qqq_bh_curve, label="Buy & Hold (QQQ)", color="black", linewidth=2.5)
     
    axs[0].plot(dates, equity_curve, label="Strategy", color="tab:blue")
    axs[0].plot(dates, bh_curve, label="Buy & Hold (NDX100)", color="tab:orange")
     
    # The total capital line shows initial equity plus all contributions.
    total_capital_curve = args.init_equity + np.array(cash_contrib_curve)
    axs[0].plot(dates, total_capital_curve, label="Total Capital", color="gray", linestyle="--")
    axs[0].plot(dates, invest_curve, label="Net Invested", color="green", linestyle=":")
    axs[0].plot(dates, cash_curve, label="Cash", color="gold")
    axs[0].set_title("Portfolio Value")
    axs[0].legend()
     
    # Plot annual percentage gains as a bar chart
    years = annual_df.index
    x = np.arange(len(years))
    width = 0.25
     
    axs[1].bar(x - width, annual_df['strategy_pct_gain'].fillna(0), width, label='Strategy', color="tab:blue")
    axs[1].bar(x, annual_df['bh_ndx_pct_gain'].fillna(0), width, label='Buy & Hold (NDX100)', color="tab:orange")
    if 'bh_qqq_pct_gain' in annual_df.columns:
         axs[1].bar(x + width, annual_df['bh_qqq_pct_gain'].fillna(0), width, label='Buy & Hold (QQQ)', color="black")
 
    axs[1].set_ylabel('% Gain')
    axs[1].set_title('Annual Percentage Gain')
    axs[1].set_xticks(x)
    axs[1].set_xticklabels(years, rotation=45)
    axs[1].legend()
    axs[1].axhline(0, color='grey', linewidth=0.8)
 
    # -------------------------
    # Stacked area "layer cake" of portfolio composition by stock
    # -------------------------
    # Use per-stock values (exclude cash) to compute composition percentages
    comp_df = values_df.fillna(0.0)
    comp_sum = comp_df.sum(axis=1).replace(0.0, np.nan)
    pct_df = (comp_df.div(comp_sum, axis=0) * 100.0).fillna(0.0)

    # Order columns so largest final weights are at bottom of the stack
    final_weights = pct_df.iloc[-1].sort_values(ascending=False)
    cols_order = final_weights.index.tolist()
    pct_df = pct_df[cols_order]
    # Choose colors
    cmap = plt.get_cmap('tab20')
    colors = [cmap(i % cmap.N) for i in range(len(cols_order))]

    # stacked area plot (use dates converted to matplotlib datetime)
    axs[2].stackplot(dates, [pct_df[c].values for c in cols_order], labels=cols_order, colors=colors)
    axs[2].set_ylim(0,100)
    axs[2].set_ylabel('Percent of Stock Portfolio (%)')
    axs[2].set_title('Portfolio Composition by Stock (stacked area)')

    # Show legend for top 10 largest at final date (display right-to-left)
    top_n = 10
    top_cols = final_weights.index[:top_n].tolist()
    # create proxy artists for legend in reversed order so largest appears lower in legend
    from matplotlib.patches import Patch
    legend_patches = []
    for c in top_cols:
        idx = cols_order.index(c)
        legend_patches.append(Patch(facecolor=colors[idx], label=c))
    axs[2].legend(handles=list(reversed(legend_patches)), title=f"Top {top_n} stocks (by final %)", loc='upper left', bbox_to_anchor=(1.01, 1))

    # tighten layout will be done below
    # Collect per-stock gains for scatter plot
    dollar_gains, pct_gains, bh_dollar_gains, bh_pct_gains = [], [], [], []
    portfolio_summary_data = []
    
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

        portfolio_summary_data.append({
            'Ticker': t,
            'Value': final_hold,
            'Cost Basis': invested,
            'Received': cumulative_sold[t],
            'Net Gain': strat_dgain
        })

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

        if args.debug:
            print(f"{t:>5s}: Strat Gain ${strat_dgain:8.2f} ({strat_pgain:6.2f}%) on ${invested:8.2f} | "
                  f"B&H Gain ${bh_dgain:8.2f} ({bh_pgain:6.2f}%) on ${bh_invested_t:8.2f}")

    if portfolio_summary_data:
        summary_df = pd.DataFrame(portfolio_summary_data)
        summary_df = summary_df.set_index('Ticker').sort_values(by='Value', ascending=False)
        with pd.option_context('display.max_rows', None, 'display.float_format', '{:,.2f}'.format):
            print("\n--- Final Portfolio Holdings ---")
            print(summary_df[['Value', 'Cost Basis', 'Received', 'Net Gain']])

    #axs[2].scatter(dollar_gains, pct_gains, color="tab:blue", label="Strategy")
    #axs[2].scatter(bh_dollar_gains, bh_pct_gains, color="tab:orange", label="Buy & Hold")
    #axs[2].set_xlabel("Gain/Loss $")
    #axs[2].set_ylabel("% Gain")
    #axs[2].set_title("Per-stock gain vs % gain")
    #axs[2].legend()

    plt.tight_layout()
    plt.show()

    # close trade log
    log_fh.close()


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--price-file", type=str, default="ndx100_prices.csv",
                        help="CSV file containing historical price data")
    parser.add_argument("--qqq-price-file", type=str, default="qqq_prices.csv",
                        help="CSV file for QQQ B&H comparison")
    parser.add_argument("--init-equity", type=float, default=10000,
                        help="Starting account equity in dollars")
    parser.add_argument("--init-invest-fraction", type=float, default=1.0,
                        help="Fraction of initial equity invested on day one")
    parser.add_argument("--buy-fraction", type=float, default=0.60,
                        help="Fraction of current holdings to buy when threshold met")
    parser.add_argument("--sell-fraction", type=float, default=0.06,
                        help="Fraction of holdings to sell when threshold met")
    parser.add_argument("--buy-threshold", type=float, default=10.0,
                        help="Percent price increase over the rebalance")
    parser.add_argument("--sell-threshold", type=float, default=10.0,
                        help="Percent price decrease to trigger sells")
    parser.add_argument("--rebalance-days", type=int, default=21,
                        help="Number of trading days between rebalances")
    parser.add_argument("--start-date", type=str, default="2000-01-01",
                        help="Start date for simulation (YYYY-MM-DD)")
    parser.add_argument("--end-date", type=str, default="2100-01-01",
                        help="End date for simulation (YYYY-MM-DD)")
    parser.add_argument("--monthly-contribution", type=float, default=0.0,
                        help="Monthly cash contribution added on month change")
    parser.add_argument("--debug", action="store_true",
                        help="Enable debug prints")
    parser.add_argument("--log-file", type=str, default="trade_log.csv",
                        help="CSV file to write trade log (overwritten each run)")
    # allow unknown args from IDEs/notebooks
    args, _unknown = parser.parse_known_args()
    run_simulator(args)