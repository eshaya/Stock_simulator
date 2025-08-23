import pandas as pd
import matplotlib.pyplot as plt
import sys

def plot_stock_histories(csv_path: str, qqq_csv_path: str, normalize: bool = True):
    """
    Reads CSV files of stock prices and plots the price history for each stock,
    highlighting the top 10 performers and the QQQ index.

    Args:
        csv_path (str): The path to the CSV file for NDX 100 stocks.
        qqq_csv_path (str): The path to the CSV file for QQQ.
        normalize (bool): If True, normalizes prices to their starting value.
    """
    try:
        # Read the main CSV file, using the 'Date' column as the index
        # and parsing the dates into datetime objects.
        stock_prices_df = pd.read_csv(csv_path, index_col='Date', parse_dates=True)

        # Read the QQQ data and prepare it for plotting
        qqq_df = pd.read_csv(qqq_csv_path, index_col='Date', parse_dates=True)
        qqq_series = qqq_df['Close'].rename('QQQ')
        # Align QQQ data to the main dataframe's index to ensure they share the same time scale
        qqq_series = qqq_series.reindex(stock_prices_df.index).ffill().bfill()

        if normalize:
            # Drop stocks that have no price on the first day to avoid errors
            stock_prices_df = stock_prices_df.dropna(axis='columns', how='any')
            # Divide all prices by the first day's price to see relative performance
            stock_prices_df = stock_prices_df.div(stock_prices_df.iloc[0])
            # Normalize the QQQ data as well
            qqq_series = qqq_series.div(qqq_series.iloc[0])

        # Get the last available price for each stock and find the top 10
        last_prices = stock_prices_df.iloc[-1]
        top_10_tickers = last_prices.nlargest(10).index

        # Separate tickers into top 10 and others
        other_tickers = stock_prices_df.columns.difference(top_10_tickers)

        # Create a plot
        plt.style.use('seaborn-v0_8-whitegrid')
        fig, ax = plt.subplots(figsize=(16, 9))

        # Plot all "other" stocks in a light grey color with thin lines
        stock_prices_df[other_tickers].plot(ax=ax, legend=False, color='lightgray', linewidth=0.75, zorder=1)

        # Plot the top 10 stocks, which will be automatically added to the legend
        stock_prices_df[top_10_tickers].plot(ax=ax, zorder=2, linewidth=2)

        # Plot the QQQ line with specific styling
        qqq_series.plot(ax=ax, color='black', linewidth=3, label='QQQ', zorder=3)

        # Determine titles and labels based on normalization
        if normalize:
            title = 'Normalized Price History of NDX 100 Stocks (Top 10 & QQQ Highlighted)'
            ylabel = 'Normalized Price (Initial Price = 1.0)'
            legend_title = 'Top 10 Performance & QQQ'
        else:
            title = 'Price History of NDX 100 Stocks (Top 10 & QQQ Highlighted)'
            ylabel = 'Adjusted Close Price (USD)'
            legend_title = 'Top 10 by Last Price & QQQ'

        # Set plot titles and labels for clarity
        ax.set_title(title, fontsize=18)
        ax.set_xlabel('Date', fontsize=14)
        ax.set_ylabel(ylabel, fontsize=14)

        # Manually create the legend to ensure only the top 10 and QQQ are included.
        # The last lines added to the axes correspond to the top 10 stocks and QQQ.
        handles, labels = ax.get_legend_handles_labels()
        num_in_legend = len(top_10_tickers) + 1
        ax.legend(handles[-num_in_legend:], labels[-num_in_legend:], title=legend_title)

        # Improve layout and display the plot
        fig.tight_layout()
        plt.show()

    except FileNotFoundError:
        print(f"Error: The file was not found at {csv_path}", file=sys.stderr) 
    except KeyError as e:
        print(f"Error: A required column {e} was not found in one of the CSV files.", file=sys.stderr)
    except Exception as e:
        print(f"An error occurred: {e}", file=sys.stderr)

if __name__ == '__main__':
    ndx_csv_path = '/home/eshaya/Documents/python/Stock_simulator/ndx100_prices.csv'
    qqq_csv_path = '/home/eshaya/Documents/python/Stock_simulator/qqq_prices.csv'
    # Set to True to see relative performance, False to see absolute prices.
    normalize_prices = True
    plot_stock_histories(ndx_csv_path, qqq_csv_path, normalize=normalize_prices)
