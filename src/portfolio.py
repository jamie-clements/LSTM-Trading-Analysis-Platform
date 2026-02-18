from typing import Optional

import pandas as pd
import requests
import streamlit as st

from config import TRADING212_DEMO_URL, TRADING212_LIVE_URL


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
