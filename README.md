# Prediccion de Precios de Acciones con LSTM (PyTorch)

Modelo de prediccion de precios bursatiles usando una red neuronal LSTM (Long Short-Term Memory) con PyTorch. Descarga datos historicos via `yfinance`, calcula indicadores tecnicos (RSI, EMA, Volumen) y genera predicciones a 7 dias con analisis de senales.

---

## Como usarlo?

### Opcion 1 — Notebook interactivo (.ipynb)

1. Abri `lstm_prediccion_acciones.ipynb` en [Google Colab](https://colab.research.google.com/) o Jupyter Notebook
2. Ejecuta la **celda 1** para instalar las dependencias
3. Ejecuta la **celda 2** para cargar el visualizador
4. Ingresa el ticker de la accion (ej: `TSLA`, `AAPL`, `GOOGL`)
5. Selecciona la fecha de inicio para el entrenamiento
6. Hace click en **Establecer prediccion**

### Opcion 2 — Script por terminal (.py)

```bash
python lstm_prediccion_acciones.py
```

Te pide el ticker y la fecha de inicio por consola.

> Requiere Python 3.12 (PyTorch no soporta Python 3.14 todavia).

---

## Que muestra?

### Senales de analisis

```
Precio actual: $354.72 | EMA 20: $368.07 | EMA 50: $365.36 | EMA 200: $377.29
Senal EMA:     Tendencia BAJISTA: precio por debajo de EMA 20 y EMA 50
Senal RSI:     Zona NEUTRAL (RSI 48.2)  [sobrecompra >70 | sobreventa <30]
Senal Volumen: Volumen ALTO (1.8x promedio 20d) — confirma movimiento

--- Prediccion proximos 7 dias ---
  2026-10-17:  $361.54
  2026-10-18:  $366.04
  2026-10-19:  $369.95
  ...
```

- **Senal EMA**: tendencia alcista, bajista o mixta segun precio vs EMA 20 y EMA 50
- **Senal RSI**: sobrecompra (>70), sobreventa (<30), o zona alcista/bajista/neutral
- **Senal Volumen**: volumen del dia vs promedio de 20 dias (alto confirma movimiento, bajo sugiere debilidad)

### Graficos

**Grafico 1 — Historial completo**
Precio de cierre ajustado desde el inicio de los datos disponibles.

**Grafico 2 — Prediccion LSTM**
Ultimos 30 dias reales + prediccion a 7 dias, con indicadores EMA 20, EMA 50 y EMA 200 superpuestos.

---

## Modelo

- **Arquitectura**: LSTM (64 unidades) → Dropout (0.2) → Dense (1)
- **Framework**: PyTorch
- **Optimizador**: Adam
- **Funcion de perdida**: MSE
- **EarlyStopping**: manual, patience=10, restaura mejores pesos
- **Ventana de secuencia**: 7 dias
- **Forecast**: autoregresivo, recalcula indicadores en cada paso
- **Escalado**: MinMaxScaler (uno para features, otro para target)

### Features de entrada

| Feature | Descripcion |
|---------|-------------|
| Adj Close | Precio de cierre ajustado |
| Volume | Volumen operado |
| RSI (14) | Indice de Fuerza Relativa |
| EMA 20 | Media movil exponencial de 20 dias |
| EMA 50 | Media movil exponencial de 50 dias |
| EMA 200 | Media movil exponencial de 200 dias |

---

## Dependencias

```
yfinance
torch
scikit-learn
pandas
numpy
matplotlib
ipywidgets        # solo para el notebook
```

Instalacion rapida:

```bash
pip install yfinance torch scikit-learn pandas numpy matplotlib
```

> En Google Colab se instalan automaticamente con la primera celda del notebook.

---

## Estructura del repositorio

```
├── lstm_prediccion_acciones.ipynb   # Notebook con visualizador interactivo (widgets)
├── lstm_prediccion_acciones.py     # Script para ejecutar por terminal
└── README.md
```

---

## Advertencia

Este proyecto es de caracter **educativo**. Las predicciones generadas por el modelo no constituyen asesoramiento financiero y no deben usarse para tomar decisiones de inversion.
