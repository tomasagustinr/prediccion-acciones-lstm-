# -*- coding: utf-8 -*-
import pandas as pd
import numpy as np
import yfinance as yf
import matplotlib.pyplot as plt
from sklearn.preprocessing import MinMaxScaler
import torch
import torch.nn as nn


# ── Modelo LSTM en PyTorch ───────────────────────────────────────────────────

class LSTMModel(nn.Module):
    def __init__(self, n_features, hidden_size=64):
        super().__init__()
        self.lstm = nn.LSTM(input_size=n_features, hidden_size=hidden_size, batch_first=True)
        self.dropout = nn.Dropout(0.2)
        self.fc = nn.Linear(hidden_size, 1)

    def forward(self, x):
        out, _ = self.lstm(x)
        out = self.dropout(out[:, -1, :])
        return self.fc(out)


# ── Funciones ────────────────────────────────────────────────────────────────

def descargar_datos(ticker):
    df = yf.download(
        tickers=ticker,
        period='max',
        progress=False,
        auto_adjust=False
    )

    if isinstance(df.columns, pd.MultiIndex):
        df.columns = df.columns.droplevel(1)
    df = df.reset_index()
    df['Stock'] = ticker

    if 'Adj Close' not in df.columns:
        if 'Close' in df.columns:
            df['Adj Close'] = df['Close']
        else:
            raise ValueError(f'No se encontró columna de precio de cierre para {ticker}')

    df = df.rename(columns={
        'Open': '1. Open', 'High': '2. High', 'Low': '3. Low',
        'Close': '4. Close', 'Volume': '5. Volume',
        'Adj Close': '7. Adj Close', 'Stock': '6. Stock'
    })
    df = df.drop(columns=['1. Open', '2. High', '3. Low', '4. Close'], errors='ignore')
    return df


def calcular_rsi(series, periodo=14):
    delta = series.diff()
    ganancia = delta.where(delta > 0, 0.0)
    perdida = -delta.where(delta < 0, 0.0)
    avg_ganancia = ganancia.rolling(window=periodo, min_periods=1).mean()
    avg_perdida = perdida.rolling(window=periodo, min_periods=1).mean()
    rs = avg_ganancia / avg_perdida
    rsi = 100 - (100 / (1 + rs))
    return rsi.fillna(50)


def calcular_indicadores(data):
    data = data.copy()
    data['EMA_20'] = data['7. Adj Close'].ewm(span=20, adjust=False).mean()
    data['EMA_50'] = data['7. Adj Close'].ewm(span=50, adjust=False).mean()
    data['EMA_200'] = data['7. Adj Close'].ewm(span=200, adjust=False).mean()
    data['RSI'] = calcular_rsi(data['7. Adj Close'])
    return data


def create_sequences(data, seq_length):
    X, y = [], []
    for i in range(len(data) - seq_length):
        X.append(data[i:i + seq_length])
        y.append(data[i + seq_length, 0])
    return np.array(X), np.array(y)


def entrenar_y_predecir(df_stocks, ticker, fecha_inicio, seq_length=7, dias=7, epochs=100):
    data = df_stocks[df_stocks['6. Stock'] == ticker].copy()
    data.sort_values('Date', inplace=True)
    data.reset_index(drop=True, inplace=True)
    data = calcular_indicadores(data)

    data_lstm = data[data['Date'] >= pd.Timestamp(fecha_inicio)].copy()
    data_lstm.sort_values('Date', ascending=True, inplace=True)
    data_lstm.reset_index(drop=True, inplace=True)

    if len(data_lstm) < seq_length + 5:
        raise ValueError(f'Muy pocos datos desde {fecha_inicio}. Elegi una fecha mas antigua.')

    feature_cols = ['7. Adj Close', '5. Volume', 'RSI', 'EMA_20', 'EMA_50', 'EMA_200']

    scaler = MinMaxScaler()
    scaled_data = scaler.fit_transform(data_lstm[feature_cols])

    scaler_target = MinMaxScaler()
    scaler_target.fit_transform(data_lstm[['7. Adj Close']])

    X, y = create_sequences(scaled_data, seq_length)
    split = int(0.8 * len(X))
    X_train, y_train = X[:split], y[:split]

    n_features = len(feature_cols)

    X_train_t = torch.tensor(X_train, dtype=torch.float32)
    y_train_t = torch.tensor(y_train, dtype=torch.float32).unsqueeze(1)

    # Separar validacion (10% del entrenamiento)
    val_size = max(1, int(0.1 * len(X_train_t)))
    X_val_t = X_train_t[-val_size:]
    y_val_t = y_train_t[-val_size:]
    X_train_t = X_train_t[:-val_size]
    y_train_t = y_train_t[:-val_size]

    model = LSTMModel(n_features=n_features)
    optimizer = torch.optim.Adam(model.parameters())
    criterion = nn.MSELoss()

    # Entrenamiento con EarlyStopping manual
    best_val_loss = float('inf')
    best_state = None
    patience_counter = 0
    patience = 10
    batch_size = 16

    for epoch in range(epochs):
        model.train()
        indices = torch.randperm(len(X_train_t))
        for start in range(0, len(X_train_t), batch_size):
            batch_idx = indices[start:start + batch_size]
            X_batch = X_train_t[batch_idx]
            y_batch = y_train_t[batch_idx]

            optimizer.zero_grad()
            pred = model(X_batch)
            loss = criterion(pred, y_batch)
            loss.backward()
            optimizer.step()

        model.eval()
        with torch.no_grad():
            val_pred = model(X_val_t)
            val_loss = criterion(val_pred, y_val_t).item()

        if val_loss < best_val_loss:
            best_val_loss = val_loss
            best_state = {k: v.clone() for k, v in model.state_dict().items()}
            patience_counter = 0
        else:
            patience_counter += 1
            if patience_counter >= patience:
                break

    if best_state is not None:
        model.load_state_dict(best_state)

    # Forecast: recalcular indicadores en cada paso
    model.eval()
    price_history = data_lstm['7. Adj Close'].values.tolist()
    volume_last = data_lstm['5. Volume'].values[-1]
    last_window = scaled_data[-seq_length:]

    future_preds = []
    for _ in range(dias):
        input_t = torch.tensor(last_window, dtype=torch.float32).unsqueeze(0)
        with torch.no_grad():
            pred_scaled = model(input_t).item()
        pred_price = scaler_target.inverse_transform([[pred_scaled]])[0, 0]
        future_preds.append(pred_price)
        price_history.append(pred_price)

        price_series = pd.Series(price_history)
        new_ema20 = price_series.ewm(span=20, adjust=False).mean().iloc[-1]
        new_ema50 = price_series.ewm(span=50, adjust=False).mean().iloc[-1]
        new_ema200 = price_series.ewm(span=200, adjust=False).mean().iloc[-1]
        new_rsi = calcular_rsi(price_series).iloc[-1]

        new_row = np.array([[pred_price, volume_last, new_rsi, new_ema20, new_ema50, new_ema200]])
        new_row_scaled = scaler.transform(new_row)
        last_window = np.vstack([last_window[1:], new_row_scaled])

    predictions = np.array(future_preds)
    last_date = data_lstm['Date'].iloc[-1]
    next_dates = pd.date_range(start=last_date + pd.Timedelta(days=1), periods=dias)
    df_future = pd.DataFrame({'Date': next_dates, '7. Adj Close': predictions})

    return data, data_lstm, df_future


def graficar(data, data_lstm, df_future, ticker):
    precio_actual = data['7. Adj Close'].iloc[-1]
    ema20_actual  = data['EMA_20'].iloc[-1]
    ema50_actual  = data['EMA_50'].iloc[-1]
    ema200_actual = data['EMA_200'].iloc[-1]
    rsi_actual = data['RSI'].iloc[-1]
    vol_actual = data['5. Volume'].iloc[-1]
    vol_promedio = data['5. Volume'].rolling(20).mean().iloc[-1]

    if precio_actual > ema20_actual and precio_actual > ema50_actual:
        senal_ema = 'Tendencia ALCISTA: precio por encima de EMA 20 y EMA 50'
    elif precio_actual < ema20_actual and precio_actual < ema50_actual:
        senal_ema = 'Tendencia BAJISTA: precio por debajo de EMA 20 y EMA 50'
    else:
        senal_ema = 'Tendencia MIXTA: precio entre EMA 20 y EMA 50'

    if rsi_actual > 70:
        senal_rsi = f'SOBRECOMPRA (RSI {rsi_actual:.1f}) — posible correccion bajista'
    elif rsi_actual < 30:
        senal_rsi = f'SOBREVENTA (RSI {rsi_actual:.1f}) — posible rebote alcista'
    elif rsi_actual > 60:
        senal_rsi = f'Zona ALCISTA (RSI {rsi_actual:.1f})'
    elif rsi_actual < 40:
        senal_rsi = f'Zona BAJISTA (RSI {rsi_actual:.1f})'
    else:
        senal_rsi = f'Zona NEUTRAL (RSI {rsi_actual:.1f})'

    ratio_vol = vol_actual / vol_promedio if vol_promedio > 0 else 0
    if ratio_vol > 1.5:
        senal_vol = f'Volumen ALTO ({ratio_vol:.1f}x promedio 20d) — confirma movimiento'
    elif ratio_vol < 0.5:
        senal_vol = f'Volumen BAJO ({ratio_vol:.1f}x promedio 20d) — movimiento debil'
    else:
        senal_vol = f'Volumen NORMAL ({ratio_vol:.1f}x promedio 20d)'

    print(f'Precio actual: ${precio_actual:.2f} | EMA 20: ${ema20_actual:.2f} | EMA 50: ${ema50_actual:.2f} | EMA 200: ${ema200_actual:.2f}')
    print(f'Senal EMA:     {senal_ema}')
    print(f'Senal RSI:     {senal_rsi}  [sobrecompra >70 | sobreventa <30]')
    print(f'Senal Volumen: {senal_vol}')

    print(f'\n--- Prediccion proximos {len(df_future)} dias ---')
    for _, row in df_future.iterrows():
        print(f"  {row['Date'].strftime('%Y-%m-%d')}:  ${row['7. Adj Close']:.2f}")
    print()

    fig1, ax1 = plt.subplots(figsize=(14, 4))
    ax1.plot(data['Date'], data['7. Adj Close'],
             label='Precio Ajustado (Historico)', color='#1f77b4', linewidth=1.2)
    ax1.set_title(f'Historial del Precio Ajustado de {ticker}', fontsize=13)
    ax1.set_xlabel('Fecha')
    ax1.set_ylabel('Precio (USD)')
    ax1.legend()
    ax1.grid(True, alpha=0.3)
    fig1.tight_layout()
    plt.show()

    fig2, ax2 = plt.subplots(figsize=(14, 4))
    ultimos = data_lstm[['Date', '7. Adj Close', 'EMA_20', 'EMA_50', 'EMA_200']].tail(30)
    ax2.plot(ultimos['Date'], ultimos['7. Adj Close'],
             label='Ultimos 30 dias reales', color='#1f77b4', linewidth=1.2)
    ax2.plot(ultimos['Date'], ultimos['EMA_20'],
             label='EMA 20', color='#ff7f0e', linewidth=1.2, linestyle='--')
    ax2.plot(ultimos['Date'], ultimos['EMA_50'],
             label='EMA 50', color='#2ca02c', linewidth=1.2, linestyle='--')
    ax2.plot(ultimos['Date'], ultimos['EMA_200'],
             label='EMA 200', color='#9467bd', linewidth=1.2, linestyle='--')
    ax2.plot(df_future['Date'], df_future['7. Adj Close'],
             label=f'Prediccion LSTM ({len(df_future)} dias)',
             color='#d62728', linewidth=2, linestyle='--', marker='o', markersize=5)
    ax2.axvline(x=data_lstm['Date'].iloc[-1], color='gray', linestyle=':', linewidth=1,
                label='Hoy')
    ax2.set_title(f'Prediccion LSTM para {ticker} (proximos {len(df_future)} dias)', fontsize=13)
    ax2.set_xlabel('Fecha')
    ax2.set_ylabel('Precio (USD)')
    ax2.legend()
    ax2.grid(True, alpha=0.3)
    fig2.tight_layout()
    plt.show()


# ── Ejecucion ────────────────────────────────────────────────────────────────

if __name__ == '__main__':
    ticker = input('Introduce el ticker (ej: AAPL, TSLA): ').strip().upper()

    if not ticker or not all(c.isalnum() or c == '-' for c in ticker):
        print('Ticker invalido. Usa solo letras, numeros y guiones (ej: AAPL, BRK-B).')
        exit(1)

    fecha = input('Fecha de inicio para el modelo (YYYY-MM-DD, default 2024-01-01): ').strip()
    if not fecha:
        fecha = '2024-01-01'

    print(f'\nDescargando datos de {ticker}...')
    try:
        df_stocks = descargar_datos(ticker)
    except Exception as e:
        print(f'Error al descargar datos: {e}')
        exit(1)

    if df_stocks.empty or ticker not in df_stocks['6. Stock'].values:
        print(f'Ticker "{ticker}" no encontrado. Verifica que sea un simbolo valido.')
        exit(1)

    print('Entrenando modelo LSTM...')
    try:
        data, data_lstm, df_future = entrenar_y_predecir(df_stocks, ticker, fecha)
    except ValueError as e:
        print(f'Error: {e}')
        exit(1)

    print('Listo!\n')
    graficar(data, data_lstm, df_future, ticker)
