# LSTM Trading Analysis Platform

![Python](https://img.shields.io/badge/python-3.10+-blue.svg)
![License](https://img.shields.io/badge/license-MIT-green.svg)

A quantitative equity analysis tool combining LSTM-based price forecasting, multi-indicator technical analysis, and NLP sentiment scoring to generate interpretable trading signals via an interactive dashboard.

---

## Problem

Retail and research traders typically rely on fragmented tools — separate platforms for technical charting, news aggregation, and price forecasting. This project unifies those signals into a single, real-time pipeline backed by a neural network model and a rule-based signal ensemble, making the decision logic explicit and auditable.

---

## Technical Approach

| Component | Method |
|---|---|
| Price Forecasting | LSTM (PyTorch), trained on rolling 60-day windows across 5 features |
| Technical Analysis | RSI, MACD, Bollinger Bands, SMA/EMA, ADX, OBV, MFI |
| Candlestick Patterns | Rule-based detection: Doji, Hammer, Engulfing, Morning Star |
| Sentiment Scoring | TextBlob polarity scoring on real-time headlines via yfinance |
| Signal Ensemble | Weighted vote across technical, sentiment, and forecast signals |
| Persistence | SQLite — watchlist and analysis history |

Signals are combined into a confidence-weighted BUY / HOLD / SELL recommendation. The LSTM is retrained on each analysis run using the trailing year of OHLCV data.

---

## Stack

- **PyTorch** — LSTM architecture
- **yfinance** — market data and news ingestion
- **ta** — technical indicator computation
- **TextBlob** — NLP sentiment analysis
- **Streamlit + Plotly** — interactive dashboard and charting
- **SQLite** — local persistence

---

## Getting Started

```bash
git clone https://github.com/yourusername/LSTM-Trading-Analysis-Platform.git
cd LSTM-Trading-Analysis-Platform

python -m venv venv
source venv/bin/activate        # Windows: venv\Scripts\activate

pip install -r requirements.txt
streamlit run src/advisor.py
```

Open [http://localhost:8501](http://localhost:8501) in your browser.

---

## Project Structure

```
LSTM-Trading-Analysis-Platform/
├── src/
│   └── advisor.py          # Core application: models, analysis engine, Streamlit UI
├── requirements.txt
├── .gitignore
└── README.md
```

---

## Disclaimer

This tool is for research and educational purposes only. It does not constitute financial advice. Always consult a qualified financial adviser before making investment decisions.
