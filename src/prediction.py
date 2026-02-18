import logging
from typing import List, Tuple

import numpy as np
import pandas as pd
import torch
import torch.nn as nn
from sklearn.preprocessing import MinMaxScaler

from models import LSTM

logger = logging.getLogger(__name__)


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
