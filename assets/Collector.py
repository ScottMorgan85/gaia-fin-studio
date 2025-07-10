import yfinance as yf
import pandas as pd



class InfoCollector:
    @staticmethod
    def get_history(ticker_name, period=None, interval='1d', start=None, end=None):
        # 🛑 Prevent accidental triple params
        if period and (start or end):
            raise ValueError("Setting period, start and end is nonsense. Set maximum 2 of them.")

        ticker = yf.Ticker(ticker_name)

        # ✅ Use the correct combo
        if period:
            return ticker.history(period=period, interval=interval)
        else:
            return ticker.history(start=start, end=end, interval=interval)

    @staticmethod
    def get_demo_daily_history(interval='5m'):
        # 🗓️ Safe example: use period *or* start/end, not both
        return InfoCollector.get_history(
            ticker_name="^GSPC",
            period="1d",
            interval=interval,
            start=None,
            end=None
        )
        
    # @staticmethod
    # def get_prev_date(stock_info: pd.DataFrame):
    #     return stock_info.index[0]

    # @staticmethod
    # def get_daily_info(stock_info: pd.DataFrame, key_info: str):
    #     return stock_info.loc[stock_info.index[0], key_info]

    # @staticmethod
    # def download_batch_history(stocks: list, start_time, end_time):
    #     return yfinance.download(stocks, start=start_time, end=end_time)

    @staticmethod
    def get_prev_date(stock_info: pd.DataFrame):
        if stock_info.empty:
            return None
        return stock_info.index[0]

    @staticmethod
    def get_daily_info(stock_info: pd.DataFrame, key_info: str):
        if stock_info.empty:
            return None
        if key_info not in stock_info.columns:
            return None
        return stock_info.loc[stock_info.index[0], key_info]

    @staticmethod
    def download_batch_history(stocks: list, start_time, end_time):
        if not stocks:
            raise ValueError("No stocks provided for batch history download.")
        history = yfinance.download(stocks, start=start_time, end=end_time)
        if history.empty:
            return None
        return history
