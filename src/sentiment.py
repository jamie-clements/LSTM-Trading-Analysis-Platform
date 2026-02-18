import logging
from datetime import datetime
from typing import Dict, List

import numpy as np
import yfinance as yf
from textblob import TextBlob

logger = logging.getLogger(__name__)


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


class NewsAnalyzer:
    def analyze_news(self, news_items: List[Dict]) -> Dict:
        """Analyze news sentiment using TextBlob."""
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
                published_date = None
                if isinstance(item.get('providerPublishTime'), (int, float)):
                    published_date = datetime.fromtimestamp(item['providerPublishTime'])
                elif 'published' in item:
                    published_date = item['published']
                elif 'publish_time' in item:
                    published_date = item['publish_time']
                else:
                    published_date = datetime.now()

                title = ''
                if isinstance(item.get('title'), str):
                    title = item['title']
                elif isinstance(item.get('title'), dict):
                    title = item['title'].get('text', '')

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
