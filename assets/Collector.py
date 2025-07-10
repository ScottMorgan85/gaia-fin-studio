import yfinance as yf
import pandas as pd

class InfoCollector:
    """Safely collect and process market data."""

    @staticmethod
    def get_ticker(ticker_name: str):
        """
        Get a yfinance Ticker object.
        """
        return yf.Ticker(ticker_name)

    @staticmethod
    def get_history(ticker, period=None, interval='1d', start=None, end=None):
        """
        Safely fetch historical data.
        You must set at most 2 of: period, start, end.
        """
        if period and (start or end):
            raise ValueError("Setting period, start and end is nonsense. Set maximum 2 of them.")
        
        if period:
            return ticker.history(period=period, interval=interval)
        else:
            return ticker.history(start=start, end=end, interval=interval)

    @staticmethod
    def get_demo_daily_history(interval='5m'):
        """
        Demo daily history for S&P 500 (^GSPC) as a template.
        """
        return InfoCollector.get_history(
            InfoCollector.get_ticker("^GSPC"),
            period="1d",
            interval=interval,
            start=None,
            end=None
        )

    @staticmethod
    def get_prev_date(stock_info: pd.DataFrame):
        """
        Get the first date in stock history.
        """
        if stock_info.empty:
            return None
        return stock_info.index[0]

    @staticmethod
    def get_daily_info(stock_info: pd.DataFrame, key_info: str):
        """
        Get a single field value for the first row.
        """
        if stock_info.empty:
            return None
        if key_info not in stock_info.columns:
            return None
        return stock_info.loc[stock_info.index[0], key_info]

    @staticmethod
    def download_batch_history(stocks: list, start_time, end_time):
        """
        Batch download multiple tickers.
        """
        if not stocks:
            raise ValueError("No stocks provided for batch history download.")
        history = yf.download(stocks, start=start_time, end=end_time)
        if history.empty:
            return None
        return history
