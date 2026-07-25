# Pirulito

Motor de seguimiento y puntuación de acciones con datos de **TradingView**.

Combina ocho señales técnicas en un puntaje único de 0 a 100, aplica controles
de riesgo que pueden rebajar el veredicto, y calcula cuántas acciones comprar
según cuánto capital estés dispuesto a arriesgar.

> **Esto no es asesoramiento financiero.** Es una herramienta de análisis. El
> rendimiento pasado no predice el futuro, y ningún puntaje sustituye tu propio
> criterio ni tu tolerancia al riesgo.

---

## Dos formas de usarlo

| | Dónde corre | Para qué sirve |
|---|---|---|
| **Python** (`pirulito/`) | Tu terminal | Escanear cientos de acciones, rankearlas, backtestear |
| **Pine Script** (`pine/pirulito.pine`) | La app de TradingView | Ver el puntaje sobre el gráfico, con alertas |

Ambos implementan el **mismo algoritmo**. Elegí el que te sirva; usar los dos
juntos es lo más práctico: escaneás con Python, mirás los candidatos en el
gráfico con Pine.

---

## Instalación

```bash
git clone <este-repo>
cd Pirulito
pip install -r requirements.txt
```

Solo necesita `requests`. Python 3.9 o superior.

---

## Uso rápido

### Escanear un mercado entero

```bash
python -m pirulito scan --market america --limit 300 --top 20 --only-buys
```

Una sola request a TradingView trae cientos de símbolos con todos los
indicadores ya calculados, así que escanear 300 acciones cuesta casi lo mismo
que escanear una.

```
#  Simbolo  Precio  Puntaje  Veredicto      Confianza  Cobertura
-  -------  ------  -------  -------------  ---------  ---------
1  NVDA     178.42     78.3  COMPRA FUERTE        81%       100%
2  AVGO     342.10     71.9  COMPRA               74%       100%
3  MSFT     512.66     68.4  COMPRA               69%       100%
```

### Analizar acciones puntuales

```bash
python -m pirulito analyze AAPL MSFT NASDAQ:NVDA
```

Muestra el desglose señal por señal, con el porqué de cada una:

```
Desglose de senales
Senal          Valor               Peso  Aporte  Detalle
-------------  -----  -----------  ----  ------  --------------------------------------------
trend          +0.63       |###    0.20  +0.127  Precio 184.54 vs SMA50 (-1.6%) y SMA200 (+9.8%)
momentum       +0.48       |##     0.18  +0.086  Momentum 12-1: +13.1%
risk_adjusted  +0.33       |##     0.14  +0.047  Sharpe +0.35, vol anualizada 65.2%
macd           -0.39     ##|       0.11  -0.043  Histograma MACD -1.346, estable

Plan de posicion
  3 acciones a 184.54 = 553.62 (5.5% del capital)
  Stop 153.93   Objetivo 245.77   Ratio beneficio/riesgo 2.0:1
```

### Otros comandos

```bash
# Con histórico OHLCV completo en vez del snapshot (más señales, más lento)
python -m pirulito analyze AAPL --history --provider tradingview

# Backtest de la estrategia
python -m pirulito backtest AAPL MSFT KO --provider synthetic --years 3 --top-n 3

# Ajustar capital y riesgo por operación
python -m pirulito scan --capital 50000 --risk 0.02

# Exportar
python -m pirulito scan --csv resultados.csv --json resultados.json

# Ver toda la configuración
python -m pirulito config > mi-config.json
python -m pirulito --config mi-config.json scan
```

---

## Cómo funciona el algoritmo

### 1. Ocho señales, cada una entre -1 y +1

| Señal | Peso | Qué mide |
|---|---|---|
| `trend` | 0.20 | Distancia entre SMA50 y SMA200, y del precio a ambas |
| `momentum` | 0.18 | Retorno de 12 meses **excluyendo el último mes** |
| `risk_adjusted` | 0.14 | Ratio de Sharpe |
| `macd` | 0.11 | Nivel del histograma MACD y su pendiente |
| `near_high` | 0.10 | Posición dentro del rango de 52 semanas |
| `rsi` | 0.09 | RSI, premiando fuerza sana y castigando extremos |
| `tv_rating` | 0.07 | El rating técnico propio de TradingView |
| `trend_quality` | 0.06 | R² de la regresión log-precio, o ADX |
| `volume` | 0.05 | Volumen confirmando (o contradiciendo) el precio |

Todas son **continuas, no booleanas**. Una acción 1% arriba de su SMA200 no
puntúa igual que una 30% arriba, y una señal que solo salta entre dos valores
haría temblar el puntaje cada vez que el precio roza un umbral.

Tres decisiones que vale la pena explicar:

- **Momentum 12-1** salta el último mes porque el movimiento más reciente
  tiende a revertir antes de que la tendencia larga se reafirme.
- **RSI tiene su pico en 60**, no en 30. La dirección ya la cubren `trend` y
  `momentum`; acá solo se mide si el movimiento es sostenible. RSI 95 y RSI 15
  son ambos malas noticias.
- **Volumen cambia de signo con el precio.** Un pico de volumen con precio
  subiendo es acumulación; el mismo pico con precio cayendo es distribución.

### 2. Promedio ponderado sobre lo que hay

Si a una señal le faltan datos, su peso se reparte entre las demás en vez de
contarla como neutral. El campo **cobertura** te dice qué porcentaje del peso
total tenía datos reales detrás.

### 3. Controles de riesgo con poder de veto

Un puntaje alto no alcanza. El veredicto **baja un escalón por cada control
que se incumple**:

- Volatilidad anualizada > 75%
- Caída actual desde máximos > 35%
- Volumen en dólares < 1.000.000/día

Una acción con puntaje de COMPRA FUERTE que además es ilíquida y volátil
termina en MANTENER.

### 4. Confianza ≠ puntaje

La **confianza** mide cuánto se ponen de acuerdo las señales entre sí. Ocho
señales tirando en direcciones opuestas dan un puntaje cerca de cero que
*parece* un MANTENER tranquilo pero en realidad es una discusión. Esa
diferencia queda visible.

### 5. Tamaño de posición por riesgo, no por capital

El stop se coloca a 2,5 ATR por debajo de la entrada, y la cantidad de acciones
sale de ahí: si te stopean, perdés siempre el mismo 1% de la cuenta, sea la
acción tranquila o violenta. Un tope aparte evita que una acción muy quieta se
coma la cartera solo porque su stop quedó cerca.

---

## El indicador de TradingView

1. Abrí un gráfico en TradingView → **Pine Editor**
2. Pegá el contenido de `pine/pirulito.pine`
3. **Add to chart**

Vas a ver el puntaje 0-100 en un panel, con las bandas de compra y venta
sombreadas, y una tabla con el desglose completo de señales, la volatilidad
anualizada, la caída desde máximos y el stop sugerido por ATR.

Cuatro alertas listas para configurar: entrada en zona de compra, compra
fuerte, salida de la zona de compra, y entrada en zona de venta.

El indicador también expone su puntaje como columna, así que podés agregarlo al
screener de TradingView y ordenar por él.

**Diferencia con la versión Python:** el Pine no incluye la señal `tv_rating`
(el rating propio de TradingView no es accesible desde Pine Script), así que
reparte ese peso entre las otras ocho.

---

## Sobre los datos: leé esto

TradingView **no tiene una API pública oficial** de datos de mercado. Este
proyecto usa dos rutas no oficiales:

- **`TradingViewScanner`** consulta `scanner.tradingview.com`, el mismo
  endpoint que usa el screener de su web. No requiere cuenta.
- **`TradingViewHistory`** usa `tvdatafeed`, un cliente websocket no oficial.
  Hace falta solo para el backtest y el modo `--history`. **No está en PyPI:**

  ```bash
  pip install git+https://github.com/rongardF/tvdatafeed.git
  ```

Los términos de servicio de TradingView restringen el scraping y la
redistribución de sus datos. Usalo como herramienta personal de investigación,
mantené la frecuencia de requests baja, y si vas a poner dinero real detrás,
contratá un proveedor de datos con licencia.

### Alternativas sin depender de eso

```bash
# CSV exportado del propio TradingView ("Export chart data..." en el gráfico)
python -m pirulito analyze AAPL --history --provider csv --csv-dir ./datos

# Datos sintéticos, para probar el motor sin red
python -m pirulito analyze AAPL --history --provider synthetic
```

---

## Estructura

```
pirulito/
  indicators.py    Matemática pura: SMA, EMA, RSI, MACD, ATR, Sharpe, regresión
  features.py      Capa de medición: normaliza TradingView e histórico a un formato común
  signals.py       Las ocho señales, cada una a [-1, 1]
  scoring.py       Mezcla ponderada -> puntaje -> veredicto
  risk.py          Controles de riesgo y tamaño de posición
  screener.py      Orquestación: traer, puntuar, rankear
  backtest.py      Simulación walk-forward sin sesgo de anticipación
  report.py        Tablas de terminal, JSON, CSV
  cli.py           Interfaz de línea de comandos
  data/            TradingView (screener + histórico), CSV, sintético
pine/
  pirulito.pine    El mismo algoritmo como indicador de TradingView
tests/             146 tests, sin dependencias externas
```

`features.py` es la pieza que hace que todo esto funcione: convierte tanto una
fila del screener de TradingView como un histórico OHLCV al mismo conjunto de
mediciones, así un solo motor de scoring sirve a las dos rutas.

---

## Tests

```bash
python -m unittest discover -s tests -v
```

El test más importante es `test_backtest.py::TestNoLookahead`: verifica que la
decisión tomada en una fecha **no cambia** si después agregás barras futuras.
Sin eso, cualquier backtest miente.

---

## Sobre el backtest

`run_backtest` trunca las series a la fecha de decisión antes de puntuar, cobra
comisión sobre el volumen operado, y compara contra comprar-y-mantener con
pesos iguales sobre el mismo universo.

Aun así: **un backtest que le gana al benchmark en un universo y un período es
evidencia débil.** Tomalo como verificación de que el algoritmo no es
activamente dañino, no como un pronóstico. Sobre datos sintéticos aleatorios la
estrategia gana en algunos universos y pierde en otros, que es exactamente lo
que debería pasar cuando no hay una ventaja real que capturar.

---

## Configuración

Todo lo que el algoritmo trata como criterio ajustable vive en `config.py`:
pesos, ventanas de indicadores, umbrales de veredicto, límites de riesgo y
parámetros de dimensionamiento.

```bash
python -m pirulito config > mi-config.json
# editá mi-config.json
python -m pirulito --config mi-config.json scan
```

Si subís mucho un peso, bajá los otros. Los pesos se normalizan solos, así que
lo que importa es la proporción entre ellos, no que sumen 1.
