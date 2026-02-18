import logging
from typing import Dict, List, Union

import numpy as np
import pandas as pd
import ta

from prediction import PricePredictionModel
from sentiment import NewsAnalyzer

logger = logging.getLogger(__name__)


class TechnicalAnalysis:
    def __init__(self):
        self.pattern_recognition = self._setup_pattern_recognition()

    def _setup_pattern_recognition(self):
        return {
            'doji': self._is_doji,
            'hammer': self._is_hammer,
            'engulfing': self._is_engulfing,
            'morning_star': self._is_morning_star
        }

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
        # Trend
        df['sma_20'] = ta.trend.sma_indicator(df['Close'], window=20)
        df['sma_50'] = ta.trend.sma_indicator(df['Close'], window=50)
        df['ema_12'] = ta.trend.ema_indicator(df['Close'], window=12)
        df['ema_26'] = ta.trend.ema_indicator(df['Close'], window=26)
        df['adx'] = ta.trend.adx(df['High'], df['Low'], df['Close'])

        # Momentum
        df['rsi'] = ta.momentum.rsi(df['Close'], window=14)
        df['macd'] = ta.trend.macd_diff(df['Close'])
        df['stoch_k'] = ta.momentum.stoch(df['High'], df['Low'], df['Close'])
        df['stoch_d'] = ta.momentum.stoch_signal(df['High'], df['Low'], df['Close'])

        # Volatility
        df['bbands_upper'] = ta.volatility.bollinger_hband(df['Close'])
        df['bbands_lower'] = ta.volatility.bollinger_lband(df['Close'])
        df['atr'] = ta.volatility.average_true_range(df['High'], df['Low'], df['Close'])

        # Volume
        df['obv'] = ta.volume.on_balance_volume(df['Close'], df['Volume'])
        df['mfi'] = ta.volume.money_flow_index(df['High'], df['Low'], df['Close'], df['Volume'])

        # Candlestick patterns
        df['doji'] = df.apply(self._is_doji, axis=1)
        df['hammer'] = df.apply(self._is_hammer, axis=1)
        df['engulfing'] = pd.Series([self._is_engulfing(df, i) for i in range(len(df))])
        df['morning_star'] = pd.Series([self._is_morning_star(df, i) for i in range(len(df))])

        return df

    def generate_signals(self, df: pd.DataFrame) -> Dict[str, Union[str, float]]:
        signals = {}
        confidence_scores = []

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

        bb_upper = df['bbands_upper'].iloc[-1]
        bb_lower = df['bbands_lower'].iloc[-1]
        bb_range = bb_upper - bb_lower
        bb_confidence = abs((price - bb_lower) / bb_range) if bb_range > 0 else 0.5
        if price < bb_lower:
            signals['BB'] = 'Strong Buy'
            confidence_scores.append(1 - bb_confidence)
        elif price > bb_upper:
            signals['BB'] = 'Strong Sell'
            confidence_scores.append(bb_confidence)
        else:
            signals['BB'] = 'Neutral'
            confidence_scores.append(0.5)

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

        if price_predictions:
            if price_predictions[-1] > price_predictions[0] * 1.02: buy_signals += 2
            elif price_predictions[-1] < price_predictions[0] * 0.98: sell_signals += 2

        total_signals = max(buy_signals + sell_signals, 1)
        buy_confidence = buy_signals / total_signals
        sell_confidence = sell_signals / total_signals

        if buy_confidence > sell_confidence and buy_confidence > 0.6:
            action, confidence = 'BUY', buy_confidence
        elif sell_confidence > buy_confidence and sell_confidence > 0.6:
            action, confidence = 'SELL', sell_confidence
        else:
            action, confidence = 'HOLD', max(buy_confidence, sell_confidence)

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

        if price_predictions:
            price_change = ((price_predictions[-1] - price_predictions[0]) / price_predictions[0]) * 100
            reasons.append(f"Predicted price change: {price_change:.2f}% over next {len(price_predictions)} days")

        return " | ".join(reasons)
