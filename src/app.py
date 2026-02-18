import json
import logging
from datetime import datetime, timedelta

import pandas as pd
import plotly.express as px
import plotly.graph_objects as go
import streamlit as st
import yfinance as yf

from config import APP_ICON, APP_NAME
from portfolio import PortfolioConnector
from sentiment import fetch_news_for_symbol
from tracker import StockTracker, flatten_columns

logging.basicConfig(level=logging.INFO)


def run_streamlit_app():
    st.set_page_config(
        page_title=APP_NAME,
        page_icon=APP_ICON,
        layout="wide"
    )

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

    # ── Sidebar ───────────────────────────────────────────────────────────
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

    # ── Main Content ──────────────────────────────────────────────────────
    st.title(f"{APP_ICON} {APP_NAME}")

    if not watched_stocks:
        st.info("Add stocks to your watchlist using the sidebar to begin analysis.")
        return

    tabs = st.tabs([
        "📊 Market Signals",
        "📉 Technical Analysis",
        "📰 Sentiment Analysis",
        "💼 Portfolio",
        "📈 Performance"
    ])

    # ── Tab 1: Market Signals ─────────────────────────────────────────────
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

    # ── Tab 2: Technical Analysis ─────────────────────────────────────────
    with tabs[1]:
        st.header("Technical Analysis")
        selected_stock = st.selectbox("Select Instrument", watched_stocks, key="tech_analysis")

        if selected_stock:
            stock = yf.Ticker(selected_stock)
            df = stock.history(period="1y")
            df = flatten_columns(df)
            df = tracker.trading_bot.technical_analysis.calculate_indicators(df)

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

    # ── Tab 3: Sentiment Analysis ─────────────────────────────────────────
    with tabs[2]:
        st.header("News Sentiment Analysis")
        selected_stock = st.selectbox("Select Instrument", watched_stocks, key="news_analysis")

        if selected_stock:
            with st.spinner(f"Fetching news for {selected_stock}..."):
                news_items = fetch_news_for_symbol(selected_stock)

            news_analysis = tracker.trading_bot.news_analyzer.analyze_news(news_items)

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

    # ── Tab 4: Portfolio Connection ───────────────────────────────────────
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

        if 'portfolio' in st.session_state and not st.session_state['portfolio'].empty:
            st.divider()
            st.subheader("Your Portfolio")

            portfolio_df = st.session_state['portfolio']
            st.dataframe(portfolio_df, use_container_width=True, hide_index=True)

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

            st.write("")
            portfolio_symbols = portfolio_df['Symbol'].dropna().unique().tolist()
            new_symbols = [s for s in portfolio_symbols if s not in watched_stocks]
            if new_symbols:
                if st.button(f"Add {len(new_symbols)} portfolio stocks to watchlist"):
                    added = sum(1 for sym in new_symbols if tracker.add_stock(sym))
                    st.success(f"Added {added} stocks to watchlist")
                    st.rerun()

    # ── Tab 5: Performance ────────────────────────────────────────────────
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
