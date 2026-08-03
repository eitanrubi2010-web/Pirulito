"""Pirulito en un solo archivo, para Google Colab.

Version condensada del motor: consulta el screener de TradingView, puntua cada
accion con las mismas nueve senales que el paquete completo, aplica los mismos
controles de riesgo y muestra el ranking.

Pensado para pegarse en una sola celda de Colab, asi se puede usar desde un
iPad o cualquier navegador, sin instalar nada. Solo necesita `requests`, que
Colab ya trae.

No es asesoramiento financiero.
"""

import math
from dataclasses import dataclass, field
from typing import Any, Dict, List, Optional

import requests

# ══════════════════════════════════════════════════════════════════════════
#  CONFIGURACION — editá esto y volvé a correr la celda
# ══════════════════════════════════════════════════════════════════════════

MERCADO = "america"        # america, argentina, spain, germany, brazil...
CANTIDAD = 200             # cuantas acciones traer del mercado
MOSTRAR = 25               # cuantas mostrar en el ranking
SOLO_COMPRAS = False       # True = solo COMPRA y COMPRA FUERTE
CAPITAL = 10_000.0         # capital total, para dimensionar posiciones
RIESGO_POR_OPERACION = 0.01  # 1% del capital arriesgado por operacion
DETALLE_DE_LOS_MEJORES = 3   # desglose completo de los N primeros

# Para analizar acciones puntuales en vez de escanear el mercado,
# poné por ejemplo: MIS_ACCIONES = ["AAPL", "MSFT", "NASDAQ:NVDA"]
MIS_ACCIONES: List[str] = []

# ── Pesos de cada senal (se normalizan solos, importa la proporcion) ──
PESOS = {
    "tendencia": 0.20,
    "momentum": 0.18,
    "riesgo_ajustado": 0.14,
    "macd": 0.11,
    "cerca_maximos": 0.10,
    "rsi": 0.09,
    "rating_tv": 0.07,
    "calidad_tendencia": 0.06,
    "volumen": 0.05,
}

# ── Controles de riesgo: cada uno que se incumpla baja un escalon ──
MAX_VOLATILIDAD_ANUAL = 0.75
MAX_CAIDA_DESDE_MAXIMOS = 0.35
MIN_VOLUMEN_DIARIO_USD = 1_000_000.0

# ── Umbrales del veredicto, sobre el puntaje que va de -1 a +1 ──
UMBRAL_COMPRA_FUERTE = 0.35
UMBRAL_COMPRA = 0.15
UMBRAL_REDUCIR = -0.15
UMBRAL_VENTA = -0.35

TASA_LIBRE_DE_RIESGO = 0.04
MULTIPLO_ATR_STOP = 2.5
OBJETIVO_BENEFICIO_RIESGO = 2.0
MAX_PORCENTAJE_POSICION = 0.20

DIAS_HABILES = 252

# ══════════════════════════════════════════════════════════════════════════
#  Utilidades numericas
# ══════════════════════════════════════════════════════════════════════════


def clamp(valor: float, minimo: float = -1.0, maximo: float = 1.0) -> float:
    return max(minimo, min(maximo, valor))


def squash(valor: float, escala: float) -> float:
    """Lleva un numero sin limites al rango [-1, 1].

    `escala` es la magnitud que cae alrededor de 0.76, o sea define que
    cuenta como una lectura "fuerte" para esa medicion en particular.
    """
    return math.tanh(valor / escala)


def _num(valor: Any) -> Optional[float]:
    if valor is None or isinstance(valor, bool):
        return None
    try:
        resultado = float(valor)
    except (TypeError, ValueError):
        return None
    return None if resultado != resultado else resultado  # descarta NaN


def _pct(valor: Any) -> Optional[float]:
    """El screener informa porcentajes como enteros: 2.5 significa 2,5%."""
    numero = _num(valor)
    return None if numero is None else numero / 100.0


# ══════════════════════════════════════════════════════════════════════════
#  Datos: el screener de TradingView
# ══════════════════════════════════════════════════════════════════════════

URL_SCREENER = "https://scanner.tradingview.com/{mercado}/scan"

COLUMNAS = [
    "name", "description", "exchange", "sector", "market_cap_basic",
    "close", "volume", "change",
    "SMA50", "SMA200", "RSI", "MACD.macd", "MACD.signal", "ATR",
    "ADX", "ADX+DI", "ADX-DI", "Volatility.D",
    "Perf.1M", "Perf.3M", "Perf.6M", "Perf.Y",
    "price_52_week_high", "price_52_week_low",
    "average_volume_10d_calc", "average_volume_60d_calc",
    "Recommend.All",
]

CABECERAS = {
    "User-Agent": (
        "Mozilla/5.0 (Macintosh; Intel Mac OS X 10_15_7) AppleWebKit/537.36 "
        "(KHTML, like Gecko) Chrome/125.0 Safari/537.36"
    ),
    "Content-Type": "application/json",
    "Accept": "application/json",
    "Origin": "https://www.tradingview.com",
    "Referer": "https://www.tradingview.com/",
}


def _pedir(cuerpo: Dict[str, Any], mercado: str) -> List[Dict[str, Any]]:
    respuesta = requests.post(
        URL_SCREENER.format(mercado=mercado),
        json=cuerpo,
        headers=CABECERAS,
        timeout=25,
    )
    respuesta.raise_for_status()
    filas = respuesta.json().get("data") or []

    salida = []
    for entrada in filas:
        registro = dict(zip(COLUMNAS, entrada.get("d") or []))
        ticker = entrada.get("s", "")
        registro["ticker"] = ticker
        if ticker and ":" in ticker:
            registro.setdefault("exchange", ticker.split(":", 1)[0])
            registro["name"] = registro.get("name") or ticker.split(":", 1)[1]
        salida.append(registro)
    return salida


def escanear_mercado(mercado: str, cantidad: int) -> List[Dict[str, Any]]:
    """Trae las acciones mas grandes del mercado, ya filtradas por liquidez."""
    cuerpo = {
        "filter": [
            {"left": "type", "operation": "equal", "right": "stock"},
            {"left": "is_primary", "operation": "equal", "right": True},
            {"left": "close", "operation": "greater", "right": 5},
            {
                "left": "average_volume_10d_calc|close",
                "operation": "greater",
                "right": MIN_VOLUMEN_DIARIO_USD,
            },
        ],
        "options": {"lang": "en"},
        "markets": [mercado],
        "symbols": {"query": {"types": []}, "tickers": []},
        "columns": COLUMNAS,
        "sort": {"sortBy": "market_cap_basic", "sortOrder": "desc"},
        "range": [0, max(1, cantidad)],
    }
    return _pedir(cuerpo, mercado)


def consultar_acciones(simbolos: List[str], mercado: str) -> List[Dict[str, Any]]:
    """Trae solo los simbolos pedidos. Aceptan 'AAPL' o 'NASDAQ:AAPL'."""
    tickers = []
    for simbolo in simbolos:
        limpio = simbolo.strip().upper()
        tickers.append(limpio if ":" in limpio else f"NASDAQ:{limpio}")
    cuerpo = {
        "symbols": {"tickers": tickers, "query": {"types": []}},
        "columns": COLUMNAS,
    }
    return _pedir(cuerpo, mercado)


# ══════════════════════════════════════════════════════════════════════════
#  Mediciones
# ══════════════════════════════════════════════════════════════════════════


@dataclass
class Medicion:
    """Todo lo que el algoritmo necesita saber de una accion."""

    simbolo: str
    precio: float
    nombre: Optional[str] = None
    sector: Optional[str] = None
    sma_rapida: Optional[float] = None
    sma_lenta: Optional[float] = None
    rsi: Optional[float] = None
    macd_hist: Optional[float] = None
    momentum: Optional[float] = None
    retorno_anual: Optional[float] = None
    volatilidad_anual: Optional[float] = None
    sharpe: Optional[float] = None
    caida_actual: Optional[float] = None
    atr: Optional[float] = None
    maximo_52s: Optional[float] = None
    minimo_52s: Optional[float] = None
    volumen_relativo: Optional[float] = None
    cambio_reciente: Optional[float] = None
    volumen_usd: Optional[float] = None
    adx: Optional[float] = None
    di_mas: Optional[float] = None
    di_menos: Optional[float] = None
    rating_tv: Optional[float] = None


def medir(fila: Dict[str, Any]) -> Medicion:
    """Convierte una fila del screener en mediciones normalizadas."""
    simbolo = fila.get("name") or fila.get("ticker") or "?"
    precio = _num(fila.get("close"))
    if precio is None or precio <= 0:
        raise ValueError(f"{simbolo}: sin precio utilizable")

    m = Medicion(
        simbolo=simbolo,
        precio=precio,
        nombre=fila.get("description"),
        sector=fila.get("sector"),
        sma_rapida=_num(fila.get("SMA50")),
        sma_lenta=_num(fila.get("SMA200")),
        rsi=_num(fila.get("RSI")),
        atr=_num(fila.get("ATR")),
        maximo_52s=_num(fila.get("price_52_week_high")),
        minimo_52s=_num(fila.get("price_52_week_low")),
        adx=_num(fila.get("ADX")),
        di_mas=_num(fila.get("ADX+DI")),
        di_menos=_num(fila.get("ADX-DI")),
        rating_tv=_num(fila.get("Recommend.All")),
    )

    linea = _num(fila.get("MACD.macd"))
    senal = _num(fila.get("MACD.signal"))
    if linea is not None and senal is not None:
        m.macd_hist = linea - senal

    anual = _pct(fila.get("Perf.Y"))
    mensual = _pct(fila.get("Perf.1M"))
    if anual is not None:
        m.retorno_anual = anual
        # Momentum 12-1: el retorno del año sin el ultimo mes, porque ese
        # ultimo tramo tiende a revertir antes de que siga la tendencia larga.
        if mensual is not None and mensual > -1.0:
            m.momentum = (1.0 + anual) / (1.0 + mensual) - 1.0
        else:
            m.momentum = anual
    if mensual is not None:
        m.cambio_reciente = mensual

    vol_diaria = _pct(fila.get("Volatility.D"))
    if vol_diaria is not None and vol_diaria > 0:
        m.volatilidad_anual = vol_diaria * math.sqrt(DIAS_HABILES)
        if m.retorno_anual is not None:
            m.sharpe = (m.retorno_anual - TASA_LIBRE_DE_RIESGO) / m.volatilidad_anual

    if m.maximo_52s and m.maximo_52s > 0:
        m.caida_actual = max(0.0, (m.maximo_52s - precio) / m.maximo_52s)

    vol10 = _num(fila.get("average_volume_10d_calc"))
    vol60 = _num(fila.get("average_volume_60d_calc"))
    if vol10 is not None:
        m.volumen_usd = precio * vol10
        if vol60 and vol60 > 0:
            m.volumen_relativo = vol10 / vol60

    return m


# ══════════════════════════════════════════════════════════════════════════
#  Las nueve senales, cada una entre -1 y +1
# ══════════════════════════════════════════════════════════════════════════


@dataclass
class Senal:
    nombre: str
    puntaje: float
    peso: float
    motivo: str

    @property
    def aporte(self) -> float:
        return self.puntaje * self.peso


def _senales(m: Medicion) -> List[Senal]:
    salida: List[Senal] = []

    def agregar(nombre: str, puntaje: Optional[float], motivo: str) -> None:
        if puntaje is None:
            return
        peso = PESOS.get(nombre, 0.0)
        if peso > 0:
            salida.append(Senal(nombre, clamp(puntaje), peso, motivo))

    # Tendencia: distancia entre medias — la idea del cruce dorado, pero
    # medida como distancia para que se degrade suave en vez de disparar una
    # sola vez — mas cuanto se extendio el precio por encima de cada una.
    if m.sma_rapida and m.sma_lenta and m.sma_rapida > 0 and m.sma_lenta > 0:
        puntaje = (
            0.45 * squash((m.sma_rapida - m.sma_lenta) / m.sma_lenta, 0.06)
            + 0.35 * squash((m.precio - m.sma_lenta) / m.sma_lenta, 0.10)
            + 0.20 * squash((m.precio - m.sma_rapida) / m.sma_rapida, 0.05)
        )
        agregar(
            "tendencia",
            puntaje,
            f"Precio vs SMA50 {(m.precio / m.sma_rapida - 1) * 100:+.1f}%, "
            f"vs SMA200 {(m.precio / m.sma_lenta - 1) * 100:+.1f}%",
        )

    if m.momentum is not None:
        agregar(
            "momentum",
            squash(m.momentum, 0.25),
            f"Retorno 12-1: {m.momentum * 100:+.1f}%",
        )

    # Sharpe: el retorno solo premia lo que mas se movio; dividirlo por la
    # volatilidad pregunta cuanto de ese movimiento hubo que aguantar.
    if m.sharpe is not None:
        vol = (
            f", vol {m.volatilidad_anual * 100:.0f}%"
            if m.volatilidad_anual is not None
            else ""
        )
        agregar("riesgo_ajustado", squash(m.sharpe, 1.0), f"Sharpe {m.sharpe:+.2f}{vol}")

    if m.macd_hist is not None and m.precio > 0:
        agregar(
            "macd",
            squash(m.macd_hist / m.precio, 0.01),
            f"Histograma MACD {m.macd_hist:+.3f} "
            f"({m.macd_hist / m.precio * 100:+.2f}% del precio)",
        )

    # Cerca de maximos: las acciones pegadas a su maximo de 52 semanas
    # historicamente siguen rindiendo; un descuento profundo suele significar
    # que algo se rompio, no que algo esta barato.
    if (
        m.maximo_52s
        and m.minimo_52s is not None
        and m.maximo_52s > 0
        and m.maximo_52s != m.minimo_52s
    ):
        posicion = (m.precio - m.minimo_52s) / (m.maximo_52s - m.minimo_52s)
        caida = (m.maximo_52s - m.precio) / m.maximo_52s
        agregar(
            "cerca_maximos",
            0.5 * (2.0 * posicion - 1.0) + 0.5 * clamp(1.0 - caida / 0.20),
            f"A {caida * 100:.1f}% del maximo anual, "
            f"percentil {clamp(posicion, 0.0, 1.0) * 100:.0f} del rango",
        )

    # RSI: pico en 60, cayendo a -1 en 20 y en 100. La direccion ya la cubren
    # tendencia y momentum; aca solo se juzga si el movimiento es sostenible.
    if m.rsi is not None:
        estado = (
            "sobrecomprado" if m.rsi >= 70
            else "sobrevendido" if m.rsi <= 30
            else "zona neutral"
        )
        agregar("rsi", clamp(1.0 - abs(m.rsi - 60.0) / 20.0), f"RSI {m.rsi:.1f} ({estado})")

    if m.rating_tv is not None:
        agregar(
            "rating_tv",
            clamp(m.rating_tv),
            f"Rating tecnico de TradingView: {clamp(m.rating_tv):+.2f}",
        )

    # Calidad: el ADX mide fuerza de tendencia; la direccion sale de la
    # diferencia entre +DI y -DI.
    if m.adx is not None and m.di_mas is not None and m.di_menos is not None:
        fuerza = clamp((m.adx - 20.0) / 25.0, 0.0, 1.0)
        agregar(
            "calidad_tendencia",
            fuerza * squash(m.di_mas - m.di_menos, 15.0),
            f"ADX {m.adx:.1f} (+DI {m.di_mas:.1f} / -DI {m.di_menos:.1f})",
        )

    # Volumen: un pico solo es alcista si el precio subio con el. El mismo
    # pico contra un precio que cae es distribucion, asi que la direccion
    # invierte el signo en lugar de solo escalar la magnitud.
    if m.volumen_relativo is not None and m.cambio_reciente is not None:
        agregar(
            "volumen",
            squash(m.volumen_relativo - 1.0, 0.4) * clamp(m.cambio_reciente / 0.05),
            f"Volumen {m.volumen_relativo:.2f}x su base, "
            f"precio {m.cambio_reciente * 100:+.1f}%",
        )

    return salida


# ══════════════════════════════════════════════════════════════════════════
#  Puntaje, veredicto y tamaño de posicion
# ══════════════════════════════════════════════════════════════════════════

VEREDICTOS = ["COMPRA FUERTE", "COMPRA", "MANTENER", "REDUCIR", "VENTA"]


@dataclass
class Evaluacion:
    medicion: Medicion
    puntaje: float
    veredicto: str
    veredicto_previo: str
    confianza: float
    cobertura: float
    senales: List[Senal] = field(default_factory=list)
    advertencias: List[str] = field(default_factory=list)
    acciones: int = 0
    stop: Optional[float] = None
    objetivo: Optional[float] = None

    @property
    def puntaje_100(self) -> float:
        return (self.puntaje + 1.0) * 50.0

    @property
    def rebajado(self) -> bool:
        return self.veredicto != self.veredicto_previo


def _desvio(valores: List[float]) -> float:
    if len(valores) < 2:
        return 0.0
    media = sum(valores) / len(valores)
    return math.sqrt(sum((v - media) ** 2 for v in valores) / (len(valores) - 1))


def evaluar(m: Medicion) -> Evaluacion:
    senales = _senales(m)
    peso_total = sum(s.peso for s in senales)

    if peso_total <= 0:
        return Evaluacion(
            m, 0.0, "MANTENER", "MANTENER", 0.0, 0.0,
            advertencias=["Datos insuficientes"],
        )

    # Promedio ponderado sobre las senales que si tenian datos: a la que le
    # faltan, se le reparte el peso en vez de contarla como neutral.
    puntaje = clamp(sum(s.aporte for s in senales) / peso_total)
    cobertura = peso_total / sum(PESOS.values())

    if puntaje >= UMBRAL_COMPRA_FUERTE:
        indice = 0
    elif puntaje >= UMBRAL_COMPRA:
        indice = 1
    elif puntaje <= UMBRAL_VENTA:
        indice = 4
    elif puntaje <= UMBRAL_REDUCIR:
        indice = 3
    else:
        indice = 2
    previo = VEREDICTOS[indice]

    # Controles de riesgo: cada uno que se incumple baja un escalon.
    advertencias = []
    if m.volatilidad_anual is not None and m.volatilidad_anual > MAX_VOLATILIDAD_ANUAL:
        advertencias.append(
            f"Volatilidad {m.volatilidad_anual * 100:.0f}% supera el limite de "
            f"{MAX_VOLATILIDAD_ANUAL * 100:.0f}%"
        )
    if m.caida_actual is not None and m.caida_actual > MAX_CAIDA_DESDE_MAXIMOS:
        advertencias.append(
            f"Caida desde maximos {m.caida_actual * 100:.0f}% supera el limite de "
            f"{MAX_CAIDA_DESDE_MAXIMOS * 100:.0f}%"
        )
    if m.volumen_usd is not None and m.volumen_usd < MIN_VOLUMEN_DIARIO_USD:
        advertencias.append(f"Liquidez baja: {m.volumen_usd:,.0f} USD por dia")

    veredicto = VEREDICTOS[min(indice + len(advertencias), 4)]

    # La confianza mide cuanto se ponen de acuerdo las senales: ocho tirando
    # en direcciones opuestas dan un puntaje cerca de cero que parece un
    # MANTENER tranquilo pero en realidad es una discusion.
    confianza = clamp(1.0 - _desvio([s.puntaje for s in senales]), 0.0, 1.0) * cobertura

    evaluacion = Evaluacion(
        m, puntaje, veredicto, previo, confianza, cobertura, senales, advertencias
    )

    if veredicto in ("COMPRA FUERTE", "COMPRA") and m.atr and m.atr > 0:
        # El stop va a 2,5 ATR: una accion mas volatil recibe un stop mas
        # ancho y, por lo tanto, menos acciones para el mismo riesgo.
        distancia = MULTIPLO_ATR_STOP * m.atr
        if m.precio - distancia > 0:
            por_riesgo = int((CAPITAL * RIESGO_POR_OPERACION) // distancia)
            por_tope = int((CAPITAL * MAX_PORCENTAJE_POSICION) // m.precio)
            evaluacion.acciones = max(0, min(por_riesgo, por_tope))
            evaluacion.stop = m.precio - distancia
            evaluacion.objetivo = m.precio + OBJETIVO_BENEFICIO_RIESGO * distancia

    return evaluacion


# ══════════════════════════════════════════════════════════════════════════
#  Salida
# ══════════════════════════════════════════════════════════════════════════


def _barra(puntaje: float, ancho: int = 5) -> str:
    lleno = int(round(abs(puntaje) * ancho))
    if puntaje >= 0:
        return " " * ancho + "|" + "#" * lleno + " " * (ancho - lleno)
    return " " * (ancho - lleno) + "#" * lleno + "|" + " " * ancho


def mostrar_ranking(evaluaciones: List[Evaluacion], cuantas: int) -> None:
    print(f"\n{'#':>3}  {'SIMBOLO':<10} {'PRECIO':>10} {'PUNTAJE':>8}  "
          f"{'VEREDICTO':<14} {'CONF':>5} {'COB':>5}")
    print("-" * 68)
    for posicion, e in enumerate(evaluaciones[:cuantas], start=1):
        marca = " *" if e.rebajado else ""
        print(
            f"{posicion:>3}  {e.medicion.simbolo:<10} {e.medicion.precio:>10,.2f} "
            f"{e.puntaje_100:>8.1f}  {e.veredicto + marca:<14} "
            f"{e.confianza * 100:>4.0f}% {e.cobertura * 100:>4.0f}%"
        )
    if any(e.rebajado for e in evaluaciones[:cuantas]):
        print("\n  * veredicto rebajado por controles de riesgo")


def mostrar_detalle(e: Evaluacion) -> None:
    m = e.medicion
    titulo = f"{m.simbolo}" + (f" — {m.nombre}" if m.nombre else "")
    print("\n" + "=" * 72)
    print(f"{titulo}   {m.precio:,.2f}")
    print(f"Puntaje {e.puntaje_100:.1f}/100   Veredicto: {e.veredicto}", end="")
    if e.rebajado:
        print(f"   (antes de riesgo: {e.veredicto_previo})")
    else:
        print()
    print(f"Confianza {e.confianza * 100:.0f}%   Cobertura {e.cobertura * 100:.0f}%")
    print("-" * 72)

    for s in sorted(e.senales, key=lambda x: abs(x.aporte), reverse=True):
        print(
            f"{s.nombre:<18} {s.puntaje:+.2f} {_barra(s.puntaje)} "
            f"x{s.peso:.2f} = {s.aporte:+.3f}   {s.motivo}"
        )

    if e.advertencias:
        print("\nAdvertencias de riesgo:")
        for advertencia in e.advertencias:
            print(f"  ! {advertencia}")

    if e.acciones > 0 and e.stop and e.objetivo:
        importe = e.acciones * m.precio
        print(
            f"\nPlan: {e.acciones} acciones a {m.precio:,.2f} = {importe:,.2f} "
            f"({importe / CAPITAL * 100:.1f}% del capital)"
        )
        print(
            f"      Stop {e.stop:,.2f}   Objetivo {e.objetivo:,.2f}   "
            f"Riesgo maximo {e.acciones * (m.precio - e.stop):,.2f}"
        )


# ══════════════════════════════════════════════════════════════════════════
#  Programa principal
# ══════════════════════════════════════════════════════════════════════════


def main() -> None:
    if MIS_ACCIONES:
        print(f"Consultando {len(MIS_ACCIONES)} acciones en TradingView...")
        filas = consultar_acciones(MIS_ACCIONES, MERCADO)
    else:
        print(f"Escaneando las {CANTIDAD} acciones mas grandes de '{MERCADO}'...")
        filas = escanear_mercado(MERCADO, CANTIDAD)

    if not filas:
        print("TradingView no devolvio datos. Revisá el nombre del mercado.")
        return

    evaluaciones = []
    errores = 0
    for fila in filas:
        try:
            evaluaciones.append(evaluar(medir(fila)))
        except (ValueError, TypeError, ZeroDivisionError):
            errores += 1

    if SOLO_COMPRAS:
        evaluaciones = [
            e for e in evaluaciones if e.veredicto in ("COMPRA", "COMPRA FUERTE")
        ]

    evaluaciones.sort(key=lambda e: (e.puntaje, e.confianza), reverse=True)

    print(f"Listo: {len(evaluaciones)} acciones puntuadas"
          + (f", {errores} omitidas por datos incompletos" if errores else ""))

    if not evaluaciones:
        print("Ninguna accion pasa el filtro. Probá con SOLO_COMPRAS = False.")
        return

    mostrar_ranking(evaluaciones, MOSTRAR)
    for evaluacion in evaluaciones[:DETALLE_DE_LOS_MEJORES]:
        mostrar_detalle(evaluacion)

    print("\n" + "=" * 72)
    print("Esto es una herramienta de analisis, no asesoramiento financiero.")
    print("El rendimiento pasado no predice el futuro.")


main()
