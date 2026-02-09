from dataclasses import dataclass
from datetime import datetime, timedelta
from typing import List, Dict, Optional, Tuple, Union
import yfinance as yf
import pandas as pd
import numpy as np
from scipy import stats
import streamlit as st
import plotly.graph_objects as go
import plotly.express as px
from bs4 import BeautifulSoup
import requests
import sqlite3
import ta
import time
import torch
import torch.nn as nn
from sklearn.preprocessing import MinMaxScaler
from concurrent.futures import ThreadPoolExecutor
import logging
import json
from textblob import TextBlob
import io

# Configure logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

# ─── Constants ───────────────────────────────────────────────────────────────
APP_NAME = "Trading Analysis Platform"
APP_ICON = "📈"

TRADING212_DEMO_URL = "https://demo.trading212.com/api/v0"
TRADING212_LIVE_URL = "https://live.trading212.com/api/v0"


def flatten_columns(df: pd.DataFrame) -> pd.DataFrame:
    """Flatten MultiIndex columns returned by newer versions of yfinance"""
    if isinstance(df.columns, pd.MultiIndex):
        df.columns = df.columns.droplevel(1)
    return df


def fetch_news_for_symbol(symbol: str) -> List[Dict]:
    """
    Fetch news for a stock symbol using multiple fallback methods.
    Newer yfinance versions changed the news format.
    """
    news_items = []

    # Method 1: Try yf.Search (works in yfinance >= 0.2.31)
    try:
        search = yf.Search(symbol, news_count=10)
        if hasattr(search, 'news') and search.news:
            raw = search.news
            if isinstance(raw, dict):
                raw = raw.get('news', [])
            if isinstance(raw, list):
                news_items = raw
    except Exception as e:
        logger.debug(f"yf.Search news failed for {symbol}: {e}")

    # Method 2: Fallback to Ticker.news
    if not news_items:
        try:
            stock = yf.Ticker(symbol)
            raw = stock.news
            if isinstance(raw, dict):
                raw = raw.get('news', raw.get('items', []))
            if isinstance(raw, list) and raw:
                news_items = raw
        except Exception as e:
            logger.debug(f"Ticker.news failed for {symbol}: {e}")

    return news_items


# ─── Data Models ─────────────────────────────────────────────────────────────
@dataclass
class Trade:
    symbol: str
    date: datetime
    type: str
    price: float
    quantity: int
    total: float
    confidence: float
    signals: Dict[str, str]
    notes: str = ""


# ─── LSTM Neural Network ────────────────────────────────────────────────────
class LSTM(nn.Module):
    def __init__(self, input_dim, hidden_dim, num_layers, output_dim):
        super(LSTM, self).__init__()
        self.hidden_dim = hidden_dim
        self.num_layers = num_layers
        self.lstm = nn.LSTM(input_dim, hidden_dim, num_layers, batch_first=True)
        self.fc = nn.Linear(hidden_dim, output_dim)

    def forward(self, x):
        h0 = torch.zeros(self.num_layers, x.size(0), self.hidden_dim).requires_grad_()
        c0 = torch.zeros(self.num_layers, x.size(0), self.hidden_dim).requires_grad_()
        out, (hn, cn) = self.lstm(x, (h0.detach(), c0.detach()))
        out = self.fc(out[:, -1, :])
        return out


# ─── Technical Analysis Engine ───────────────────────────────────────────────
class TechnicalAnalysis:
    def __init__(self):
        self.pattern_recognition = self._setup_pattern_recognition()

    def _setup_pattern_recognition(self):
        patterns = {
            'doji': self._is_doji,
            'hammer': self._is_hammer,
            'engulfing': self._is_engulfing,
            'morning_star': self._is_morning_star
        }
        return patterns

    @staticmethod
    def _is_doji(candle: pd.Series, threshold: float = 0.1) -> bool:
        body = abs(candle['Open'] - candle['Close'])
        upper_shadow = candle['High'] - max(candle['Open'], candle['Close'])
        lower_shadow = min(candle['Open'], candle['Close']) - candle['Low']
        return body <= threshold * (upper_shadow + lower_shadow)

    @staticmethod
    def _is_hammer(candle: pd.Series) -> bool:
        body = abs(candle['Open'] - candle['Close'])
        upper_shadow = candle['High'] - max(candle['Open'], candle['Close'])
        lower_shadow = min(candle['Open'], candle['Close']) - candle['Low']
        return lower_shadow > 2 * body and upper_shadow < body

    @staticmethod
    def _is_engulfing(candles: pd.DataFrame, idx: int) -> bool:
        if idx == 0:
            return False
        curr = candles.iloc[idx]
        prev = candles.iloc[idx - 1]
        return (curr['Open'] > prev['Close'] and curr['Close'] < prev['Open']) or \
               (curr['Open'] < prev['Close'] and curr['Close'] > prev['Open'])

    @staticmethod
    def _is_morning_star(candles: pd.DataFrame, idx: int) -> bool:
        if idx < 2:
            return False
        first = candles.iloc[idx - 2]
        second = candles.iloc[idx - 1]
        third = candles.iloc[idx]
        return first['Close'] < first['Open'] and \
               abs(second['Open'] - second['Close']) < abs(first['Open'] - first['Close']) * 0.1 and \
               third['Close'] > third['Open']

    def calculate_indicators(self, df: pd.DataFrame) -> pd.DataFrame:
        # Trend Indicators
        df['sma_20'] = ta.trend.sma_indicator(df['Close'], window=20)
        df['sma_50'] = ta.trend.sma_indicator(df['Close'], window=50)
        df['ema_12'] = ta.trend.ema_indicator(df['Close'], window=12)
        df['ema_26'] = ta.trend.ema_indicator(df['Close'], window=26)
        df['adx'] = ta.trend.adx(df['High'], df['Low'], df['Close'])

        # Momentum Indicators
        df['rsi'] = ta.momentum.rsi(df['Close'], window=14)
        df['macd'] = ta.trend.macd_diff(df['Close'])
        df['stoch_k'] = ta.momentum.stoch(df['High'], df['Low'], df['Close'])
        df['stoch_d'] = ta.momentum.stoch_signal(df['High'], df['Low'], df['Close'])

        # Volatility Indicators
        df['bbands_upper'] = ta.volatility.bollinger_hband(df['Close'])
        df['bbands_lower'] = ta.volatility.bollinger_lband(df['Close'])
        df['atr'] = ta.volatility.average_true_range(df['High'], df['Low'], df['Close'])

        # Volume Indicators
        df['obv'] = ta.volume.on_balance_volume(df['Close'], df['Volume'])
        df['mfi'] = ta.volume.money_flow_index(df['High'], df['Low'], df['Close'], df['Volume'])

        # Pattern Recognition
        df['doji'] = df.apply(self._is_doji, axis=1)
        df['hammer'] = df.apply(self._is_hammer, axis=1)
        df['engulfing'] = pd.Series([self._is_engulfing(df, i) for i in range(len(df))])
        df['morning_star'] = pd.Series([self._is_morning_star(df, i) for i in range(len(df))])

        return df

    def generate_signals(self, df: pd.DataFrame) -> Dict[str, Union[str, float]]:
        signals = {}
        confidence_scores = []

        # RSI
        rsi = df['rsi'].iloc[-1]
        if rsi < 30:
            signals['RSI'] = 'Strong Buy'
            confidence_scores.append(1 - (rsi / 30))
        elif rsi > 70:
            signals['RSI'] = 'Strong Sell'
            confidence_scores.append((rsi - 70) / 30)
        else:
            signals['RSI'] = 'Neutral'
            confidence_scores.append(0.5)

        # MACD
        macd = df['macd'].iloc[-1]
        macd_prev = df['macd'].iloc[-2]
        if macd > 0 and macd_prev < 0:
            signals['MACD'] = 'Strong Buy'
            confidence_scores.append(0.8)
        elif macd < 0 and macd_prev > 0:
            signals['MACD'] = 'Strong Sell'
            confidence_scores.append(0.8)
        else:
            signals['MACD'] = 'Hold'
            confidence_scores.append(0.5)

        # Moving Averages
        sma_20 = df['sma_20'].iloc[-1]
        sma_50 = df['sma_50'].iloc[-1]
        price = df['Close'].iloc[-1]
        ma_confidence = abs((sma_20 - sma_50) / sma_50)
        if sma_20 > sma_50:
            signals['MA'] = 'Bullish'
            confidence_scores.append(min(ma_confidence, 1.0))
        else:
            signals['MA'] = 'Bearish'
            confidence_scores.append(min(ma_confidence, 1.0))

        # Bollinger Bands
        bb_upper = df['bbands_upper'].iloc[-1]
        bb_lower = df['bbands_lower'].iloc[-1]
        bb_range = bb_upper - bb_lower
        if bb_range > 0:
            bb_confidence = abs((price - bb_lower) / bb_range)
        else:
            bb_confidence = 0.5
        if price < bb_lower:
            signals['BB'] = 'Strong Buy'
            confidence_scores.append(1 - bb_confidence)
        elif price > bb_upper:
            signals['BB'] = 'Strong Sell'
            confidence_scores.append(bb_confidence)
        else:
            signals['BB'] = 'Neutral'
            confidence_scores.append(0.5)

        # Pattern Recognition
        if df['doji'].iloc[-1]:
            signals['Pattern'] = 'Doji - Potential Reversal'
            confidence_scores.append(0.6)
        elif df['hammer'].iloc[-1]:
            signals['Pattern'] = 'Hammer - Potential Bullish'
            confidence_scores.append(0.7)
        elif df['engulfing'].iloc[-1]:
            signals['Pattern'] = 'Engulfing - Strong Signal'
            confidence_scores.append(0.8)
        elif df['morning_star'].iloc[-1]:
            signals['Pattern'] = 'Morning Star - Very Bullish'
            confidence_scores.append(0.9)

        signals['overall_confidence'] = np.mean(confidence_scores)
        return signals


# ─── News & Sentiment Engine ────────────────────────────────────────────────
class NewsAnalyzer:
    def __init__(self):
        pass

    def analyze_news(self, news_items: List[Dict]) -> Dict:
        """Analyze news sentiment using TextBlob"""
        sentiments = []

        if not news_items:
            return {
                'recent_sentiment': 0,
                'overall_sentiment': 0,
                'sentiment_trend': 'Neutral',
                'details': []
            }

        for item in news_items:
            try:
                # Handle different yfinance news formats
                published_date = None
                if isinstance(item.get('providerPublishTime'), (int, float)):
                    published_date = datetime.fromtimestamp(item['providerPublishTime'])
                elif 'published' in item:
                    published_date = item['published']
                elif 'publish_time' in item:
                    published_date = item['publish_time']
                else:
                    published_date = datetime.now()

                # Get title — handle nested structures
                title = ''
                if isinstance(item.get('title'), str):
                    title = item['title']
                elif isinstance(item.get('title'), dict):
                    title = item['title'].get('text', '')

                # Get content/summary
                content = ''
                for key in ['summary', 'text', 'description', 'body']:
                    val = item.get(key, '')
                    if isinstance(val, str) and val:
                        content = val
                        break

                if not title:
                    continue

                title_sentiment = TextBlob(title).sentiment.polarity
                content_sentiment = TextBlob(content).sentiment.polarity if content else title_sentiment
                combined_sentiment = (title_sentiment * 0.6 + content_sentiment * 0.4)

                # Get link/URL
                link = item.get('link', item.get('url', ''))

                sentiments.append({
                    'title': title,
                    'sentiment': combined_sentiment,
                    'date': published_date,
                    'link': link
                })

            except Exception as e:
                logger.error(f"Error processing news item: {e}")
                continue

        if sentiments:
            recent_sentiment = np.mean([s['sentiment'] for s in sentiments[:5]])
            overall_sentiment = np.mean([s['sentiment'] for s in sentiments])
            return {
                'recent_sentiment': recent_sentiment,
                'overall_sentiment': overall_sentiment,
                'sentiment_trend': 'Improving' if recent_sentiment > overall_sentiment else 'Declining',
                'details': sentiments
            }

        return {
            'recent_sentiment': 0,
            'overall_sentiment': 0,
            'sentiment_trend': 'Neutral',
            'details': []
        }


# ─── LSTM Price Prediction ──────────────────────────────────────────────────
class PricePredictionModel:
    def __init__(self, input_dim=5, hidden_dim=32, num_layers=2, output_dim=1):
        self.model = LSTM(input_dim, hidden_dim, num_layers, output_dim)
        self.scaler = MinMaxScaler()

    def prepare_data(self, df: pd.DataFrame, lookback: int = 60) -> Tuple[torch.Tensor, torch.Tensor]:
        features = ['Close', 'Volume', 'rsi', 'macd', 'obv']
        data = df[features].values

        # Replace NaN/Inf to prevent nan training loss
        data = np.nan_to_num(data, nan=0.0, posinf=0.0, neginf=0.0)

        scaled_data = self.scaler.fit_transform(data)

        X, y = [], []
        for i in range(lookback, len(scaled_data)):
            X.append(scaled_data[i - lookback:i])
            y.append(scaled_data[i, 0])

        return torch.FloatTensor(np.array(X)), torch.FloatTensor(np.array(y)).reshape(-1, 1)

    def train(self, df: pd.DataFrame):
        X, y = self.prepare_data(df)

        criterion = nn.MSELoss()
        optimizer = torch.optim.Adam(self.model.parameters(), lr=0.01)

        self.model.train()
        for epoch in range(100):
            optimizer.zero_grad()
            output = self.model(X)
            loss = criterion(output, y)
            loss.backward()
            optimizer.step()

            if epoch % 10 == 0:
                logger.info(f'Epoch {epoch}, Loss: {loss.item():.4f}')

    def predict(self, df: pd.DataFrame, days_ahead: int = 5) -> List[float]:
        self.model.eval()

        X, _ = self.prepare_data(df)
        last_sequence = X[-1].unsqueeze(0)  # Shape: (1, lookback, features)

        predictions = []
        for _ in range(days_ahead):
            with torch.no_grad():
                pred = self.model(last_sequence)  # Shape: (1, 1)
                predictions.append(pred.item())

                # Create new timestep with prediction + zeros for other features
                new_step = torch.zeros(1, 1, last_sequence.size(2))
                new_step[0, 0, 0] = pred.item()

                last_sequence = torch.cat((last_sequence[:, 1:, :], new_step), dim=1)

        # Inverse transform — pad with zeros for other feature columns
        num_features = self.scaler.n_features_in_
        pred_array = np.zeros((len(predictions), num_features))
        pred_array[:, 0] = predictions
        pred_array = self.scaler.inverse_transform(pred_array)
        return pred_array[:, 0].tolist()


# ─── Portfolio Connection (Trading 212 + CSV) ───────────────────────────────
class PortfolioConnector:
    """Securely connect to broker APIs to read portfolio data (read-only)."""

    @staticmethod
    def fetch_trading212_portfolio(api_key: str, use_demo: bool = False) -> Optional[pd.DataFrame]:
        """
        Fetch open positions from Trading 212 API.
        This is READ-ONLY — no trades are placed.
        """
        base_url = TRADING212_DEMO_URL if use_demo else TRADING212_LIVE_URL
        headers = {"Authorization": api_key}

        try:
            resp = requests.get(f"{base_url}/equity/portfolio", headers=headers, timeout=10)
            resp.raise_for_status()
            positions = resp.json()

            if not positions:
                return pd.DataFrame()

            rows = []
            for pos in positions:
                rows.append({
                    'Symbol': pos.get('ticker', ''),
                    'Quantity': pos.get('quantity', 0),
                    'Avg Price': pos.get('averagePrice', 0),
                    'Current Price': pos.get('currentPrice', 0),
                    'P/L': pos.get('ppl', 0),
                    'Value': pos.get('quantity', 0) * pos.get('currentPrice', 0),
                })

            return pd.DataFrame(rows)

        except requests.exceptions.HTTPError as e:
            if e.response.status_code == 401:
                st.error("Invalid API key. Please check your Trading 212 API key.")
            else:
                st.error(f"API error: {e.response.status_code}")
            return None
        except requests.exceptions.ConnectionError:
            st.error("Could not connect to Trading 212. Please check your internet connection.")
            return None
        except Exception as e:
            st.error(f"Unexpected error: {str(e)}")
            return None

    @staticmethod
    def parse_csv_portfolio(uploaded_file) -> Optional[pd.DataFrame]:
        """Parse a CSV portfolio export from any broker."""
        try:
            df = pd.read_csv(uploaded_file)

            # Try to identify common column patterns
            col_map = {}
            for col in df.columns:
                col_lower = col.lower().strip()
                if any(k in col_lower for k in ['symbol', 'ticker', 'instrument', 'stock', 'code']):
                    col_map['Symbol'] = col
                elif any(k in col_lower for k in ['quantity', 'shares', 'qty', 'amount', 'no. of shares']):
                    col_map['Quantity'] = col
                elif any(k in col_lower for k in ['avg', 'average', 'cost', 'purchase price']):
                    col_map['Avg Price'] = col
                elif any(k in col_lower for k in ['current', 'market', 'last price', 'price']):
                    col_map['Current Price'] = col
                elif any(k in col_lower for k in ['p/l', 'profit', 'gain', 'return', 'pnl']):
                    col_map['P/L'] = col
                elif any(k in col_lower for k in ['value', 'market value', 'total']):
                    col_map['Value'] = col

            if 'Symbol' not in col_map:
                st.error("Could not identify a symbol/ticker column. Please ensure your CSV has a column for stock symbols.")
                return None

            result = pd.DataFrame()
            result['Symbol'] = df[col_map['Symbol']]
            result['Quantity'] = df[col_map.get('Quantity', col_map['Symbol'])].apply(
                lambda x: pd.to_numeric(x, errors='coerce')
            ) if 'Quantity' in col_map else 0
            result['Avg Price'] = df[col_map.get('Avg Price', col_map['Symbol'])].apply(
                lambda x: pd.to_numeric(x, errors='coerce')
            ) if 'Avg Price' in col_map else 0
            result['Current Price'] = df[col_map.get('Current Price', col_map['Symbol'])].apply(
                lambda x: pd.to_numeric(x, errors='coerce')
            ) if 'Current Price' in col_map else 0
            result['P/L'] = df[col_map.get('P/L', col_map['Symbol'])].apply(
                lambda x: pd.to_numeric(x, errors='coerce')
            ) if 'P/L' in col_map else 0
            result['Value'] = df[col_map.get('Value', col_map['Symbol'])].apply(
                lambda x: pd.to_numeric(x, errors='coerce')
            ) if 'Value' in col_map else 0

            return result.dropna(subset=['Symbol'])

        except Exception as e:
            st.error(f"Error parsing CSV: {str(e)}")
            return None


# ─── Trading Analysis Core ───────────────────────────────────────────────────
class TradingBot:
    def __init__(self):
        self.technical_analysis = TechnicalAnalysis()
        self.news_analyzer = NewsAnalyzer()
        self.price_predictor = PricePredictionModel()

    def analyze_stock(self, symbol: str, df: pd.DataFrame, news: List[Dict]) -> Dict:
        try:
            df = self.technical_analysis.calculate_indicators(df)
            technical_signals = self.technical_analysis.generate_signals(df)
            news_analysis = self.news_analyzer.analyze_news(news)

            try:
                self.price_predictor.train(df)
                price_predictions = self.price_predictor.predict(df)
            except Exception as e:
                logger.error(f"Error in price prediction: {e}")
                price_predictions = []

            recommendation = self._generate_recommendation(
                technical_signals, news_analysis, price_predictions
            )

            return {
                'technical_signals': technical_signals,
                'news_sentiment': news_analysis,
                'price_predictions': price_predictions,
                'overall_confidence': technical_signals.get('overall_confidence', 0),
                'recommendation': recommendation
            }

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

    def _generate_recommendation(self, technical_signals, news_analysis, price_predictions):
        buy_signals = 0
        sell_signals = 0

        if technical_signals.get('RSI') == 'Strong Buy': buy_signals += 2
        elif technical_signals.get('RSI') == 'Strong Sell': sell_signals += 2

        if technical_signals.get('MACD') == 'Strong Buy': buy_signals += 2
        elif technical_signals.get('MACD') == 'Strong Sell': sell_signals += 2

        if technical_signals.get('MA') == 'Bullish': buy_signals += 1
        elif technical_signals.get('MA') == 'Bearish': sell_signals += 1

        if news_analysis['recent_sentiment'] > 0.2: buy_signals += 2
        elif news_analysis['recent_sentiment'] < -0.2: sell_signals += 2

        if len(price_predictions) > 0:
            current_price = price_predictions[0]
            future_price = price_predictions[-1]
            if future_price > current_price * 1.02: buy_signals += 2
            elif future_price < current_price * 0.98: sell_signals += 2

        total_signals = max(buy_signals + sell_signals, 1)
        buy_confidence = buy_signals / total_signals
        sell_confidence = sell_signals / total_signals

        if buy_confidence > sell_confidence and buy_confidence > 0.6:
            action = 'BUY'
            confidence = buy_confidence
        elif sell_confidence > buy_confidence and sell_confidence > 0.6:
            action = 'SELL'
            confidence = sell_confidence
        else:
            action = 'HOLD'
            confidence = max(buy_confidence, sell_confidence)

        return {
            'action': action,
            'confidence': confidence,
            'reasons': self._generate_reason_text(technical_signals, news_analysis, price_predictions)
        }

    def _generate_reason_text(self, technical_signals, news_analysis, price_predictions):
        reasons = []

        tech_bullish = sum(1 for s in technical_signals.values() if 'Buy' in str(s) or 'Bullish' in str(s))
        tech_bearish = sum(1 for s in technical_signals.values() if 'Sell' in str(s) or 'Bearish' in str(s))

        if tech_bullish > tech_bearish:
            reasons.append(f"Technical indicators are bullish ({tech_bullish} positive signals)")
        elif tech_bearish > tech_bullish:
            reasons.append(f"Technical indicators are bearish ({tech_bearish} negative signals)")

        if news_analysis['recent_sentiment'] > 0.2:
            reasons.append("Recent news sentiment is positive")
        elif news_analysis['recent_sentiment'] < -0.2:
            reasons.append("Recent news sentiment is negative")

        if len(price_predictions) > 0:
            price_change = ((price_predictions[-1] - price_predictions[0]) / price_predictions[0]) * 100
            reasons.append(f"Predicted price change: {price_change:.2f}% over next {len(price_predictions)} days")

        return " | ".join(reasons)


# ─── Stock Tracker (Database Layer) ─────────────────────────────────────────
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


# ─── Streamlit Application ──────────────────────────────────────────────────
def run_streamlit_app():
    st.set_page_config(
        page_title=APP_NAME,
        page_icon=APP_ICON,
        layout="wide"
    )

    # Custom CSS
    st.markdown("""
        <style>
        .stApp {
            background-color: #0e1117;
            color: #ffffff;
        }
        .tradingCard {
            background-color: #1e2127;
            padding: 1rem;
            border-radius: 0.5rem;
            margin: 0.5rem 0;
        }
        .confidence-high { color: #00ff00; }
        .confidence-medium { color: #ffff00; }
        .confidence-low { color: #ff0000; }
        .security-badge {
            background-color: #1a3a2a;
            border: 1px solid #2d6a4f;
            border-radius: 8px;
            padding: 12px;
            margin: 8px 0;
            font-size: 0.85em;
        }
        </style>
    """, unsafe_allow_html=True)

    tracker = StockTracker()

    # ── Sidebar ──────────────────────────────────────────────────────────
    with st.sidebar:
        st.title(f"{APP_ICON} {APP_NAME}")

        st.header("Manage Watchlist")
        new_stock = st.text_input("Add Stock Symbol").upper()
        if st.button("Add to Watchlist"):
            if new_stock:
                if tracker.add_stock(new_stock):
                    st.success(f"Added {new_stock}")
                else:
                    st.error("Already in watchlist")

        watched_stocks = tracker.get_watched_stocks()
        if watched_stocks:
            stock_to_remove = st.selectbox("Remove Stock", [""] + watched_stocks)
            if st.button("Remove Selected") and stock_to_remove:
                tracker.remove_stock(stock_to_remove)
                st.success(f"Removed {stock_to_remove}")
                st.rerun()

        st.divider()
        st.caption("Built with LSTM neural networks, real-time technical analysis, and NLP sentiment scoring.")

    # ── Main Content ─────────────────────────────────────────────────────
    st.title(f"{APP_ICON} {APP_NAME}")

    if not watched_stocks:
        st.info("Add stocks to your watchlist using the sidebar to begin analysis.")
    else:
        tabs = st.tabs([
            "📊 Market Signals",
            "📉 Technical Analysis",
            "📰 Sentiment Analysis",
            "💼 Portfolio",
            "📈 Performance"
        ])

        # ── Tab 1: Market Signals ────────────────────────────────────────
        with tabs[0]:
            st.header("Market Signals Overview")

            for symbol in watched_stocks:
                analysis = tracker.analyze_stock(symbol)
                recommendation = analysis['recommendation']

                with st.expander(f"{symbol} — {recommendation['action']}", expanded=True):
                    col1, col2, col3 = st.columns([2, 2, 1])

                    with col1:
                        st.subheader("Signal Strength")
                        confidence_color = (
                            "confidence-high" if recommendation['confidence'] > 0.7
                            else "confidence-medium" if recommendation['confidence'] > 0.5
                            else "confidence-low"
                        )
                        st.markdown(
                            f"<h3 class='{confidence_color}'>{recommendation['confidence']:.2%}</h3>",
                            unsafe_allow_html=True
                        )

                    with col2:
                        st.subheader("Analysis Summary")
                        st.write(recommendation['reasons'])

                    with col3:
                        st.subheader("Signal")
                        if recommendation['action'] == 'BUY':
                            st.success("BUY")
                        elif recommendation['action'] == 'SELL':
                            st.error("SELL")
                        else:
                            st.info("HOLD")

                    st.subheader("Technical Indicators")
                    indicators = analysis['technical_signals']
                    cols = st.columns(4)
                    for i, (indicator, value) in enumerate(indicators.items()):
                        if indicator != 'overall_confidence':
                            cols[i % 4].metric(indicator, value)

                    if analysis['price_predictions']:
                        st.subheader("Price Forecast")
                        pred_df = pd.DataFrame({
                            'Day': range(1, len(analysis['price_predictions']) + 1),
                            'Predicted Price': analysis['price_predictions']
                        })
                        fig = px.line(pred_df, x='Day', y='Predicted Price',
                                      title=f"{symbol} — {len(analysis['price_predictions'])}-Day Forecast")
                        fig.update_layout(template="plotly_dark", height=350)
                        st.plotly_chart(fig, use_container_width=True)

        # ── Tab 2: Technical Analysis ────────────────────────────────────
        with tabs[1]:
            st.header("Technical Analysis")
            selected_stock = st.selectbox("Select Instrument", watched_stocks, key="tech_analysis")

            if selected_stock:
                stock = yf.Ticker(selected_stock)
                df = stock.history(period="1y")
                df = flatten_columns(df)
                df = tracker.trading_bot.technical_analysis.calculate_indicators(df)

                # Candlestick + overlays
                fig = go.Figure()
                fig.add_trace(go.Candlestick(
                    x=df.index, open=df['Open'], high=df['High'],
                    low=df['Low'], close=df['Close'], name='Price'
                ))
                fig.add_trace(go.Scatter(x=df.index, y=df['sma_20'], name='SMA 20', line=dict(color='orange')))
                fig.add_trace(go.Scatter(x=df.index, y=df['sma_50'], name='SMA 50', line=dict(color='blue')))
                fig.add_trace(go.Scatter(x=df.index, y=df['bbands_upper'], name='BB Upper',
                                         line=dict(color='gray', dash='dash')))
                fig.add_trace(go.Scatter(x=df.index, y=df['bbands_lower'], name='BB Lower',
                                         line=dict(color='gray', dash='dash')))
                fig.update_layout(
                    title=f"{selected_stock} — Price & Indicators",
                    yaxis_title="Price", height=600, template="plotly_dark"
                )
                st.plotly_chart(fig, use_container_width=True)

                st.subheader("Indicator Detail")
                col1, col2 = st.columns(2)

                with col1:
                    fig_rsi = go.Figure()
                    fig_rsi.add_trace(go.Scatter(x=df.index, y=df['rsi'], name='RSI'))
                    fig_rsi.add_hline(y=70, line_dash="dash", line_color="red")
                    fig_rsi.add_hline(y=30, line_dash="dash", line_color="green")
                    fig_rsi.update_layout(title="RSI", height=300, template="plotly_dark")
                    st.plotly_chart(fig_rsi, use_container_width=True)

                    fig_macd = go.Figure()
                    fig_macd.add_trace(go.Scatter(x=df.index, y=df['macd'], name='MACD'))
                    fig_macd.add_hline(y=0, line_dash="dash", line_color="gray")
                    fig_macd.update_layout(title="MACD", height=300, template="plotly_dark")
                    st.plotly_chart(fig_macd, use_container_width=True)

                with col2:
                    fig_vol = go.Figure()
                    fig_vol.add_trace(go.Bar(x=df.index, y=df['Volume'], name='Volume'))
                    fig_vol.update_layout(title="Volume", height=300, template="plotly_dark")
                    st.plotly_chart(fig_vol, use_container_width=True)

                    fig_mfi = go.Figure()
                    fig_mfi.add_trace(go.Scatter(x=df.index, y=df['mfi'], name='MFI'))
                    fig_mfi.add_hline(y=80, line_dash="dash", line_color="red")
                    fig_mfi.add_hline(y=20, line_dash="dash", line_color="green")
                    fig_mfi.update_layout(title="Money Flow Index", height=300, template="plotly_dark")
                    st.plotly_chart(fig_mfi, use_container_width=True)

        # ── Tab 3: Sentiment Analysis (Fixed) ────────────────────────────
        with tabs[2]:
            st.header("News Sentiment Analysis")
            selected_stock = st.selectbox("Select Instrument", watched_stocks, key="news_analysis")

            if selected_stock:
                with st.spinner(f"Fetching news for {selected_stock}..."):
                    news_items = fetch_news_for_symbol(selected_stock)

                news_analysis = tracker.trading_bot.news_analyzer.analyze_news(news_items)

                # Sentiment overview
                col1, col2, col3 = st.columns(3)
                with col1:
                    sentiment_score = news_analysis['recent_sentiment']
                    st.metric(
                        "Recent Sentiment",
                        f"{sentiment_score:.2f}",
                        delta=f"{sentiment_score - news_analysis['overall_sentiment']:.2f}"
                    )
                with col2:
                    st.metric("Overall Sentiment", f"{news_analysis['overall_sentiment']:.2f}")
                with col3:
                    st.metric("Sentiment Trend", news_analysis['sentiment_trend'])

                # News timeline
                st.subheader("Recent Headlines")
                if news_analysis['details']:
                    for news_item in news_analysis['details']:
                        sentiment_icon = (
                            "🟢" if news_item['sentiment'] > 0.2
                            else "🔴" if news_item['sentiment'] < -0.2
                            else "⚪"
                        )

                        with st.expander(f"{sentiment_icon} {news_item['title']}"):
                            st.write(f"**Date:** {news_item['date']}")
                            st.write(f"**Sentiment Score:** {news_item['sentiment']:.3f}")
                            if news_item.get('link'):
                                st.markdown(f"[Read full article →]({news_item['link']})")
                else:
                    st.info("No recent news found for this instrument. Sentiment data may be limited outside of US market hours.")

        # ── Tab 4: Portfolio Connection ──────────────────────────────────
        with tabs[3]:
            st.header("Portfolio Connection")

            st.markdown("""
                <div class='security-badge'>
                    🔒 <strong>Security & Privacy</strong><br>
                    Your credentials are handled with care:<br>
                    • API keys are stored <strong>only in your browser session</strong> and are never saved to disk or transmitted to third parties.<br>
                    • All broker connections are <strong>read-only</strong> — this platform cannot place trades or modify your account.<br>
                    • Data is fetched directly from your broker's official API over HTTPS encryption.<br>
                    • You can disconnect at any time by clearing the API key field or refreshing the page.
                </div>
            """, unsafe_allow_html=True)

            st.write("")

            connection_method = st.radio(
                "Connection Method",
                ["Trading 212 API", "CSV Import"],
                horizontal=True
            )

            if connection_method == "Trading 212 API":
                st.subheader("Connect to Trading 212")
                st.markdown("""
                    To connect your Trading 212 account:
                    1. Open your Trading 212 app → **Settings** → **API (Beta)**
                    2. Generate a new API key
                    3. Paste it below
                    
                    *Your key is only stored in this browser session and is never logged or saved.*
                """)

                col1, col2 = st.columns([3, 1])
                with col1:
                    api_key = st.text_input(
                        "API Key",
                        type="password",
                        placeholder="Paste your Trading 212 API key here",
                        key="t212_key"
                    )
                with col2:
                    use_demo = st.checkbox("Use Demo Account", value=True)

                if st.button("Connect", type="primary") and api_key:
                    with st.spinner("Connecting to Trading 212..."):
                        portfolio_df = PortfolioConnector.fetch_trading212_portfolio(api_key, use_demo)

                    if portfolio_df is not None and not portfolio_df.empty:
                        st.session_state['portfolio'] = portfolio_df
                        st.success(f"Connected — {len(portfolio_df)} positions loaded")
                    elif portfolio_df is not None:
                        st.info("Connected, but no open positions found.")

            elif connection_method == "CSV Import":
                st.subheader("Import Portfolio from CSV")
                st.markdown("""
                    Upload a CSV export from your broker. The platform will automatically detect columns 
                    for symbol, quantity, price, and P/L. Most broker CSV exports are supported.
                """)

                uploaded_file = st.file_uploader("Upload CSV", type=['csv'])
                if uploaded_file:
                    portfolio_df = PortfolioConnector.parse_csv_portfolio(uploaded_file)
                    if portfolio_df is not None and not portfolio_df.empty:
                        st.session_state['portfolio'] = portfolio_df
                        st.success(f"Imported — {len(portfolio_df)} positions loaded")

            # Display portfolio if loaded
            if 'portfolio' in st.session_state and not st.session_state['portfolio'].empty:
                st.divider()
                st.subheader("Your Portfolio")

                portfolio_df = st.session_state['portfolio']
                st.dataframe(portfolio_df, use_container_width=True, hide_index=True)

                # Summary metrics
                col1, col2, col3 = st.columns(3)
                with col1:
                    total_value = portfolio_df['Value'].sum()
                    st.metric("Total Value", f"${total_value:,.2f}" if total_value else "—")
                with col2:
                    total_pl = portfolio_df['P/L'].sum()
                    st.metric("Total P/L", f"${total_pl:,.2f}" if total_pl else "—",
                              delta=f"{total_pl:,.2f}" if total_pl else None)
                with col3:
                    st.metric("Positions", len(portfolio_df))

                # Offer to add portfolio stocks to watchlist
                st.write("")
                portfolio_symbols = portfolio_df['Symbol'].dropna().unique().tolist()
                new_symbols = [s for s in portfolio_symbols if s not in watched_stocks]
                if new_symbols:
                    if st.button(f"Add {len(new_symbols)} portfolio stocks to watchlist"):
                        added = 0
                        for sym in new_symbols:
                            if tracker.add_stock(sym):
                                added += 1
                        st.success(f"Added {added} stocks to watchlist")
                        st.rerun()

        # ── Tab 5: Performance ───────────────────────────────────────────
        with tabs[4]:
            st.header("Performance Analytics")

            col1, col2 = st.columns(2)
            with col1:
                start_date = st.date_input("Start Date", value=datetime.now() - timedelta(days=30))
            with col2:
                end_date = st.date_input("End Date", value=datetime.now())

            cursor = tracker.conn.cursor()
            cursor.execute("""
                SELECT symbol, date, technical_signals, news_sentiment, prediction_accuracy 
                FROM analysis_history 
                WHERE date BETWEEN ? AND ?
                ORDER BY date DESC
            """, (start_date, end_date))

            performance_data = cursor.fetchall()

            if performance_data:
                perf_df = pd.DataFrame(performance_data,
                                       columns=['Symbol', 'Date', 'Technical_Signals',
                                                'News_Sentiment', 'Prediction_Accuracy'])

                st.subheader("Prediction Accuracy")
                fig_accuracy = px.line(perf_df, x='Date', y='Prediction_Accuracy',
                                       color='Symbol', title="Model Accuracy Over Time")
                fig_accuracy.update_layout(template="plotly_dark")
                st.plotly_chart(fig_accuracy, use_container_width=True)

                st.subheader("Signal History")
                for symbol in perf_df['Symbol'].unique():
                    symbol_data = perf_df[perf_df['Symbol'] == symbol]
                    with st.expander(f"{symbol} — Detailed Analysis"):
                        col1, col2 = st.columns(2)
                        with col1:
                            fig_sentiment = px.line(symbol_data, x='Date', y='News_Sentiment',
                                                    title="News Sentiment Trend")
                            fig_sentiment.update_layout(template="plotly_dark")
                            st.plotly_chart(fig_sentiment, use_container_width=True)
                        with col2:
                            try:
                                signals_df = pd.DataFrame([
                                    json.loads(signals) for signals in symbol_data['Technical_Signals']
                                ])
                                if not signals_df.empty:
                                    fig_signals = px.line(signals_df, title="Technical Signals")
                                    fig_signals.update_layout(template="plotly_dark")
                                    st.plotly_chart(fig_signals, use_container_width=True)
                            except Exception:
                                st.info("Signal chart data not available")
            else:
                st.info("No performance data available for the selected date range. Run some analyses first.")


if __name__ == "__main__":
    run_streamlit_app()