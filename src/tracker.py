import json
import logging
import sqlite3
from datetime import datetime
from typing import Dict, List

import pandas as pd
import yfinance as yf

from analysis import TradingBot
from sentiment import fetch_news_for_symbol

logger = logging.getLogger(__name__)


def flatten_columns(df: pd.DataFrame) -> pd.DataFrame:
    """Flatten MultiIndex columns returned by newer versions of yfinance."""
    if isinstance(df.columns, pd.MultiIndex):
        df.columns = df.columns.droplevel(1)
    return df


class StockTracker:
    def __init__(self, db_path: str = "stock_tracker.db"):
        self.conn = sqlite3.connect(db_path)
        self.setup_database()
        self.trading_bot = TradingBot()

    def setup_database(self):
        cursor = self.conn.cursor()

        cursor.execute('''
            CREATE TABLE IF NOT EXISTS watched_stocks (
                symbol TEXT PRIMARY KEY,
                added_date TIMESTAMP,
                last_analysis TIMESTAMP,
                last_recommendation TEXT
            )
        ''')

        cursor.execute('''
            CREATE TABLE IF NOT EXISTS trading_history (
                id INTEGER PRIMARY KEY,
                symbol TEXT,
                date TIMESTAMP,
                action TEXT,
                price FLOAT,
                quantity INTEGER,
                confidence FLOAT,
                reasons TEXT
            )
        ''')

        cursor.execute('''
            CREATE TABLE IF NOT EXISTS analysis_history (
                id INTEGER PRIMARY KEY,
                symbol TEXT,
                date TIMESTAMP,
                technical_signals TEXT,
                news_sentiment FLOAT,
                prediction_accuracy FLOAT
            )
        ''')

        self.conn.commit()

    def add_stock(self, symbol: str) -> bool:
        try:
            cursor = self.conn.cursor()
            cursor.execute(
                "INSERT INTO watched_stocks (symbol, added_date) VALUES (?, ?)",
                (symbol.upper(), datetime.now())
            )
            self.conn.commit()
            return True
        except sqlite3.IntegrityError:
            return False

    def remove_stock(self, symbol: str) -> bool:
        cursor = self.conn.cursor()
        cursor.execute("DELETE FROM watched_stocks WHERE symbol = ?", (symbol.upper(),))
        self.conn.commit()
        return True

    def get_watched_stocks(self) -> List[str]:
        cursor = self.conn.cursor()
        cursor.execute("SELECT symbol FROM watched_stocks")
        return [row[0] for row in cursor.fetchall()]

    def analyze_stock(self, symbol: str) -> Dict:
        try:
            stock = yf.Ticker(symbol)
            df = stock.history(period="1y")
            df = flatten_columns(df)

            if df.empty:
                raise ValueError(f"No data available for {symbol}")

            news = fetch_news_for_symbol(symbol)
            analysis = self.trading_bot.analyze_stock(symbol, df, news)

            cursor = self.conn.cursor()
            cursor.execute('''
                UPDATE watched_stocks
                SET last_analysis = ?, last_recommendation = ?
                WHERE symbol = ?
            ''', (datetime.now(), json.dumps(analysis['recommendation']), symbol))

            cursor.execute('''
                INSERT INTO analysis_history
                (symbol, date, technical_signals, news_sentiment, prediction_accuracy)
                VALUES (?, ?, ?, ?, ?)
            ''', (
                symbol, datetime.now(),
                json.dumps(analysis['technical_signals']),
                analysis['news_sentiment']['recent_sentiment'],
                0.0
            ))

            self.conn.commit()
            return analysis

        except Exception as e:
            logger.error(f"Error analyzing stock {symbol}: {e}")
            return {
                'technical_signals': {},
                'news_sentiment': {
                    'recent_sentiment': 0, 'overall_sentiment': 0,
                    'sentiment_trend': 'Neutral', 'details': []
                },
                'price_predictions': [],
                'overall_confidence': 0,
                'recommendation': {
                    'action': 'HOLD', 'confidence': 0,
                    'reasons': f'Error analyzing stock: {str(e)}'
                }
            }
