from __future__ import annotations

from io import BytesIO
import base64
from pathlib import Path
from html.parser import HTMLParser
from urllib.parse import urljoin
import re
from typing import Optional

import numpy as np
import pandas as pd
import plotly.graph_objects as go
import requests
import streamlit as st
from sklearn.metrics import mean_absolute_error, mean_squared_error
from xgboost import XGBRegressor

# =====================================================================
# BUENAVENTURA · INTELIGENCIA DE VENTAS | Versión final
# Lee la base histórica en modo de solo lectura. No escribe en data/ ni outputs/.
# =====================================================================
st.set_page_config(
    page_title="Buenaventura | Pronóstico de ingresos",
    page_icon="⛏️",
    layout="wide",
    initial_sidebar_state="expanded",
)

APP_DIR = Path(__file__).resolve().parent
SALES_FILENAME = "base_ventas_buenaventura_2019Q1_2026Q2.csv"
WORLD_BANK_PAGE = "https://www.worldbank.org/en/research/commodity-markets"
WORLD_BANK_MONTHLY_XLSX = (
    "https://thedocs.worldbank.org/en/doc/74e8be41ceb20fa0da750cda2f6b9e4e-0050012026/related/CMO-Historical-Data-Monthly.xlsx"
)
OFFICIAL_LOGO_URL = "https://buenaventura.com/wp-content/uploads/2024/06/Logo-1.png"
OFFICIAL_INVESTOR_URL = "https://buenaventura.com/inversionista/"
FY2025_RESULTS_URL = "https://buenaventura.com/wp-content/uploads/2025/02/4Q25-BVN-Earnings-Press-Release-Vf.pdf"
Q1_2026_RESULTS_URL = "https://buenaventura.com/wp-content/uploads/2026/04/1Q26-BVN-Earnings-Press-Release.pdf"
Q2_2026_RESULTS_URL = "https://www.sec.gov/Archives/edgar/data/1013131/000110465926088868/tm2621768d1_6k.htm"
Q2_2026_FINANCIALS_URL = "https://buenaventura.com/wp-content/uploads/2026/04/2Q-Buenaventura-Consolidated-2026.pdf"

TARGET = "total_revenues_usd_millones"
FEATURES_INDEX = ["ventas_lag_1", "ventas_lag_4", "indice_metales_ponderado_lag_1"]
FEATURES_GOLD = ["ventas_lag_1", "ventas_lag_4", "precio_oro_lag_1_usd_oz"]
FEATURES_NO_EXT = ["ventas_lag_1", "ventas_lag_4"]
FEATURE_SETS = {
    "Índice ponderado de metales": FEATURES_INDEX,
    "Precio del oro": FEATURES_GOLD,
    "Sin variable externa": FEATURES_NO_EXT,
}
EXTENSION_META = {
    "Índice ponderado de metales": {
        "feature": "indice_metales_ponderado_lag_1",
        "market_col": "indice_metales_ponderado",
        "label": "Índice ponderado de metales",
        "unit": "base 100",
        "short": "Índice de metales",
    },
    "Precio del oro": {
        "feature": "precio_oro_lag_1_usd_oz",
        "market_col": "precio_oro_usd_oz",
        "label": "Precio del oro",
        "unit": "US$/onza troy",
        "short": "Precio del oro",
    },
}

# Proxy fija y documentada. Se calcula con volumen x precio realizado FY2025.
# No es una desagregación auditada de ingresos por metal.
FY2025_VOLUME_PRICE = {
    "gold": 109_387 * 3_547,
    "silver": 14_969_375 * 41.87,
    "lead": 17_941 * 1_912,
    "zinc": 23_980 * 2_844,
    "copper": 58_429 * 10_071,
}
_WEIGHT_TOTAL = sum(FY2025_VOLUME_PRICE.values())
METAL_WEIGHTS = {k: v / _WEIGHT_TOTAL for k, v in FY2025_VOLUME_PRICE.items()}
METAL_LABELS_ES = {"gold": "Oro", "silver": "Plata", "copper": "Cobre", "lead": "Plomo", "zinc": "Zinc"}
PRICE_COLS = ["gold", "silver", "copper", "lead", "zinc"]

# =====================================================================
# INTERFAZ · distribución inspirada en la propuesta visual aprobada
# =====================================================================
def _asset_uri(filename: str) -> str:
    """Convierte ilustraciones incluidas en el paquete en URI embebidas para el navegador."""
    path = APP_DIR / "assets" / filename
    try:
        raw = path.read_bytes()
        return "data:image/jpeg;base64," + base64.b64encode(raw).decode("ascii")
    except OSError:
        return ""


HERO_IMAGE_URI = _asset_uri("mining_header.jpg")
SIDEBAR_IMAGE_URI = _asset_uri("mining_sidebar.jpg")

st.markdown(
    f"""
    <style>
    :root {{
      --bvn-forest: #0D2E24;
      --bvn-deep: #12372C;
      --bvn-green: #26734D;
      --bvn-lime: #7DB844;
      --bvn-ink: #132E25;
      --bvn-muted: #52695D;
      --bvn-border: #DDE7DE;
      --bvn-bg: #F2F6F2;
    }}
    html {{ scroll-behavior: smooth; }}
    .stApp {{ background: var(--bvn-bg); color: var(--bvn-ink); }}
    [data-testid="stHeader"] {{ background: rgba(242,246,242,.97); }}
    [data-testid="stMainBlockContainer"], .block-container {{ max-width: 1660px; padding-top: .7rem; padding-bottom: 2.2rem; }}
    [data-testid="stSidebar"] {{
      background-color: #0D2E24;
      background-image: linear-gradient(180deg, rgba(9,37,29,.98) 0%, rgba(9,37,29,.88) 55%, rgba(9,37,29,.42) 100%), url('{SIDEBAR_IMAGE_URI}');
      background-position: center, center bottom; background-size: cover, 100% auto; background-repeat: no-repeat;
      border-right: 1px solid #234D3B;
    }}
    [data-testid="stSidebar"] * {{ color: #F5F8F4; }}
    [data-testid="stSidebar"] [data-testid="stMarkdownContainer"] p,
    [data-testid="stSidebar"] [data-testid="stCaptionContainer"] {{ color: #D2E0D5 !important; }}
    [data-testid="stSidebar"] a {{ text-decoration: none !important; }}
    .bvn-side-brand {{ padding: .4rem .1rem .85rem .1rem; margin-bottom: .65rem; border-bottom: 1px solid rgba(221,239,224,.2); }}
    .bvn-side-brand img {{ display:block; width:100%; max-width:220px; max-height:67px; object-fit:contain; object-position:left center; background:#FFFFFF; border-radius:9px; padding:.45rem; }}
    .bvn-side-wordmark {{ color:#FFFFFF !important; font-weight:850; font-size:.84rem; letter-spacing:.12em; margin-top:.7rem; }}
    .bvn-side-tagline {{ color:#C5DCCB !important; font-size:.7rem; letter-spacing:.16em; margin-top:.14rem; text-transform:uppercase; }}
    .bvn-nav {{ display:flex; flex-direction:column; gap:.35rem; margin-top:.75rem; }}
    .bvn-nav a {{ display:flex; align-items:center; gap:.7rem; color:#EDF5EF !important; padding:.7rem .75rem; border-radius:10px; border-left:3px solid transparent; font-size:.91rem; }}
    .bvn-nav a:first-child {{ background:rgba(255,255,255,.11); border-left-color:#7FCB91; font-weight:750; }}
    .bvn-nav a:hover {{ background:rgba(255,255,255,.12); }}
    .bvn-side-bottom {{ margin-top:1.3rem; border-top:1px solid rgba(221,239,224,.22); padding-top:.9rem; }}
    .bvn-side-bottom .motto {{ color:#D5E5D8; font-size:.72rem; letter-spacing:.17em; line-height:1.7; text-transform:uppercase; }}
    .bvn-side-bottom .side-note {{ color:#D0DED3; font-size:.74rem; line-height:1.5; margin-top:.75rem; }}
    .bvn-hero {{
      display:flex; align-items:center; min-height:158px; box-sizing:border-box; overflow:hidden;
      padding:1.35rem 1.55rem; margin:.1rem 0 .9rem 0; border-radius:18px;
      background-color:#12372C;
      background-image:linear-gradient(90deg, rgba(247,250,247,.99) 0%, rgba(247,250,247,.97) 34%, rgba(247,250,247,.73) 52%, rgba(13,43,32,.20) 100%), url('{HERO_IMAGE_URI}');
      background-position:center, right center; background-size:cover, cover; background-repeat:no-repeat;
      box-shadow:0 7px 22px rgba(16,39,31,.08); border:1px solid #E1E9E1;
    }}
    .bvn-hero-copy {{ max-width: 850px; padding-right:1.2rem; }}
    .bvn-kicker {{ color:#347548; font-size:.72rem; line-height:1.3; font-weight:850; letter-spacing:.18em; text-transform:uppercase; }}
    .bvn-hero h1 {{ color:#102F25; font-size:clamp(1.55rem,2.35vw,2.35rem); line-height:1.14; margin:.42rem 0 .55rem 0; font-weight:850; }}
    .bvn-hero p {{ color:#314A3E; font-size:.96rem; line-height:1.5; margin:0; max-width:810px; }}
    .bvn-chiprow {{ display:flex; gap:.45rem; flex-wrap:wrap; margin-top:.7rem; }}
    .bvn-chip {{ display:inline-flex; align-items:center; gap:.35rem; padding:.33rem .62rem; border:1px solid #D7E3D9; border-radius:999px; background:rgba(255,255,255,.88); color:#234D38; font-size:.75rem; font-weight:650; }}
    .bvn-legal {{ margin-top:.55rem; font-size:.76rem; color:#52685D; }}
    .section-kicker {{ color:#3F7A4C; font-size:.72rem; line-height:1.4; font-weight:850; letter-spacing:.13em; text-transform:uppercase; margin-top:.2rem; }}
    h1,h2,h3,h4 {{ color:var(--bvn-ink); }}
    h2 {{ margin-top:.25rem !important; margin-bottom:.4rem !important; }}
    h3 {{ font-size:1.07rem !important; }}
    p, li {{ color:#33483C; }}
    [data-testid="stMetric"] {{ box-sizing:border-box; height:100%; min-height:108px; padding:.85rem .95rem; background:#FFFFFF; border:1px solid var(--bvn-border); border-radius:14px; box-shadow:0 4px 13px rgba(22,50,35,.035); overflow:visible; }}
    [data-testid="stMetricLabel"] {{ color:#566C5F !important; white-space:normal !important; line-height:1.35 !important; font-size:.84rem !important; }}
    [data-testid="stMetricValue"] {{ color:#173D2C !important; font-size:clamp(1.22rem,1.65vw,1.8rem) !important; line-height:1.22 !important; font-weight:800 !important; white-space:normal !important; overflow-wrap:anywhere !important; }}
    [data-testid="stMetricDelta"] {{ white-space:normal !important; line-height:1.28 !important; font-size:.74rem !important; }}
    [data-testid="stAlert"] {{ border-radius:12px; border-width:1px; }}
    [data-testid="stAlert"] [data-testid="stMarkdownContainer"] p {{ color:#203B2D !important; line-height:1.5 !important; }}
    div.stButton > button, div.stDownloadButton > button, a[data-testid="stLinkButton"] {{ border-radius:10px; font-weight:750; min-height:2.65rem; }}
    div.stButton > button[kind="primary"], div.stDownloadButton > button[kind="primary"] {{ background:#216C45; border-color:#216C45; color:#FFFFFF; }}
    div[data-testid="stDataFrame"] {{ border:1px solid #DDE7DE; border-radius:11px; overflow:hidden; }}
    [data-testid="stExpander"] {{ border:1px solid #DDE7DE; border-radius:12px; background:rgba(255,255,255,.88); }}
    .bvn-small {{ font-size:.82rem; line-height:1.52; color:#5B6F63; }}
    .bvn-panel-title {{ display:flex; align-items:center; gap:.5rem; color:#143D2D; font-size:1.05rem; font-weight:820; margin-bottom:.6rem; }}
    .bvn-mini-icon {{ display:inline-flex; align-items:center; justify-content:center; width:1.7rem; height:1.7rem; border-radius:9px; background:#E8F2E7; color:#226C45; font-size:1rem; flex:0 0 1.7rem; }}
    .bvn-steps {{ display:flex; flex-direction:column; gap:.7rem; }}
    .bvn-step {{ display:flex; gap:.65rem; align-items:flex-start; padding:.6rem .65rem; border:1px solid #E4ECE4; border-radius:11px; background:#FAFCFA; }}
    .bvn-step-number {{ flex:0 0 2rem; width:2rem; height:2rem; display:flex; align-items:center; justify-content:center; border-radius:50%; background:#D5EBDD; color:#174A31; font-size:1rem; font-weight:850; }}
    .bvn-step-title {{ color:#203E2D; font-weight:800; font-size:.84rem; margin-bottom:.12rem; }}
    .bvn-step-text {{ color:#54675D; font-size:.78rem; line-height:1.4; }}
    .bvn-footer {{ border-top:1px solid #DDE6DE; margin-top:1.2rem; padding-top:.8rem; color:#62736A; font-size:.75rem; line-height:1.5; }}
    /* Visualizacion refinada de historia, pronostico y escenarios */
    .bvn-chart-note {{ color:#718477; font-size:.84rem; line-height:1.45; margin:.05rem 0 .3rem 0; }}
    .bvn-outlook-head {{ display:flex; align-items:center; justify-content:space-between; gap:.6rem; margin:.15rem 0 .75rem 0; padding:.65rem .75rem; border:1px solid #E0EAE1; border-radius:12px; background:linear-gradient(110deg,#F6FAF5,#FFFFFF); }}
    .bvn-outlook-head-label {{ color:#718477; font-size:.65rem; font-weight:850; letter-spacing:.11em; text-transform:uppercase; }}
    .bvn-outlook-head-value {{ color:#1C5B3D; font-size:.84rem; font-weight:850; white-space:nowrap; }}
    .bvn-outlook-card {{ position:relative; overflow:hidden; border:1px solid #DFE8E0; border-radius:15px; padding:1rem 1rem .95rem 1rem; margin:.62rem 0; background:#FFFFFF; box-shadow:0 5px 16px rgba(22,50,35,.045); }}
    .bvn-outlook-card:before {{ content:""; position:absolute; inset:0 auto 0 0; width:4px; background:#A8C99B; }}
    .bvn-outlook-card-primary:before {{ background:#2C8052; }}
    .bvn-outlook-card-final:before {{ background:#7DB844; }}
    .bvn-outlook-card-scenario:before {{ background:#A2B6AB; }}
    .bvn-outlook-label {{ display:flex; align-items:center; justify-content:space-between; gap:.55rem; color:#63786B; font-size:.78rem; line-height:1.3; font-weight:700; }}
    .bvn-outlook-period {{ display:inline-block; margin-top:.35rem; padding:.18rem .45rem; border-radius:6px; background:#F0F6EF; color:#287047; font-size:.72rem; font-weight:850; letter-spacing:.025em; }}
    .bvn-outlook-value {{ color:#173D2C; font-size:clamp(1.28rem,1.62vw,1.78rem); font-weight:900; letter-spacing:-.035em; line-height:1.22; margin:.35rem 0 .4rem 0; white-space:nowrap; overflow-wrap:normal; }}
    .bvn-outlook-value small {{ color:#607568; font-size:.78rem; font-weight:750; letter-spacing:0; }}
    .bvn-outlook-foot {{ display:flex; align-items:center; flex-wrap:wrap; gap:.4rem; color:#697D70; font-size:.75rem; line-height:1.4; }}
    .bvn-outlook-pill {{ display:inline-flex; align-items:center; gap:.22rem; border-radius:999px; padding:.2rem .48rem; font-size:.73rem; line-height:1.25; font-weight:800; white-space:nowrap; }}
    .bvn-pill-positive {{ color:#21653F; background:#DFF5E6; }}
    .bvn-pill-negative {{ color:#9A3D35; background:#FCE9E5; }}
    .bvn-pill-neutral {{ color:#436052; background:#EDF2EE; }}
    .bvn-outlook-meta {{ display:flex; align-items:flex-start; gap:.55rem; padding:.48rem .1rem; color:#738579; font-size:.77rem; line-height:1.4; }}
    .bvn-outlook-meta strong {{ display:block; color:#355544; font-weight:800; margin-bottom:.08rem; }}
    .bvn-outlook-meta-icon {{ flex:0 0 1.65rem; width:1.65rem; height:1.65rem; display:flex; align-items:center; justify-content:center; color:#27734A; background:#E7F1E7; border-radius:8px; font-size:.85rem; }}
    @media(max-width:1100px) {{ .bvn-outlook-value {{ font-size:1.45rem; }} .bvn-outlook-card {{ padding:.85rem; }} }}
    @media(max-width:1100px) {{ .bvn-hero {{ background-image:linear-gradient(90deg,rgba(247,250,247,.98),rgba(247,250,247,.83)),url('{HERO_IMAGE_URI}'); }} }}
    @media(max-width:850px) {{
      .bvn-hero {{ min-height:auto; padding:1.05rem; background-image:linear-gradient(90deg,rgba(247,250,247,.98),rgba(247,250,247,.9)),url('{HERO_IMAGE_URI}'); }}
      .bvn-hero h1 {{ font-size:1.55rem; }}
      .block-container,[data-testid="stMainBlockContainer"] {{ padding-left:.85rem; padding-right:.85rem; }}
    }}
    </style>
    """,
    unsafe_allow_html=True,
)

# =====================================================================
# DATOS Y FECHAS
# =====================================================================
def quarter_key(label: str) -> int:
    m = re.fullmatch(r"(\d{4})Q([1-4])", str(label).strip().upper())
    if not m:
        raise ValueError(f"Trimestre no válido: {label}")
    return int(m.group(1)) * 4 + int(m.group(2))


def key_to_quarter(key: int) -> str:
    year = (int(key) - 1) // 4
    q = (int(key) - 1) % 4 + 1
    return f"{year}Q{q}"


def next_quarter(label: str) -> str:
    return key_to_quarter(quarter_key(label) + 1)


def prior_quarter(label: str) -> str:
    return key_to_quarter(quarter_key(label) - 1)


def find_sales_file() -> Optional[Path]:
    """Busca la base ya existente; no la crea, copia, modifica ni sobrescribe."""
    home = Path.home()
    project = home / "Downloads" / "proyecto_xgboost_buenaventura"
    candidates = [
        APP_DIR / "data" / SALES_FILENAME,
        APP_DIR / SALES_FILENAME,
        APP_DIR.parent / SALES_FILENAME,
        APP_DIR.parent / "data" / SALES_FILENAME,
        project / "data" / SALES_FILENAME,
        project / SALES_FILENAME,
        home / "Downloads" / SALES_FILENAME,
    ]
    for path in candidates:
        if path.exists() and path.is_file():
            return path
    return None


@st.cache_data(show_spinner=False)
def load_sales(path_text: str) -> pd.DataFrame:
    df = pd.read_csv(path_text, encoding="utf-8-sig")
    required = {"trimestre", TARGET, "fuente_url"}
    missing = required - set(df.columns)
    if missing:
        raise ValueError(f"La base no contiene las columnas necesarias: {sorted(missing)}")
    df = df.copy()
    df["trimestre"] = df["trimestre"].astype(str).str.strip().str.upper()
    df["_key"] = df["trimestre"].map(quarter_key)
    df[TARGET] = pd.to_numeric(df[TARGET], errors="coerce")
    df = df.sort_values("_key").reset_index(drop=True)
    if df["trimestre"].duplicated().any():
        raise ValueError("La base contiene trimestres duplicados. No se modificó el archivo.")
    if df[TARGET].isna().any() or (df[TARGET] <= 0).any():
        raise ValueError("Hay ingresos vacíos o no positivos. Revisa la base sin reemplazar datos por estimaciones.")
    expected = list(range(int(df["_key"].min()), int(df["_key"].max()) + 1))
    if df["_key"].tolist() != expected:
        present = set(df["_key"])
        missing_q = [key_to_quarter(k) for k in expected if k not in present]
        raise ValueError(f"La serie no es trimestral continua. Trimestres faltantes: {missing_q}")
    if df["fuente_url"].fillna("").astype(str).str.strip().eq("").any():
        raise ValueError("Hay filas sin enlace de fuente. La base no fue modificada.")
    return df


def parse_month(value) -> Optional[pd.Period]:
    if pd.isna(value):
        return None
    if isinstance(value, (pd.Timestamp, np.datetime64)):
        return pd.Timestamp(value).to_period("M")
    if hasattr(value, "year") and hasattr(value, "month") and not isinstance(value, str):
        try:
            return pd.Period(year=int(value.year), month=int(value.month), freq="M")
        except Exception:
            pass
    text = str(value).strip()
    if re.fullmatch(r"\d{4}(?:\.0)?", text):
        return None
    match = re.fullmatch(r"(\d{4})\s*M\s*(\d{1,2})", text, re.I)
    if match:
        year, month = int(match.group(1)), int(match.group(2))
        if 1 <= month <= 12:
            return pd.Period(year=year, month=month, freq="M")
    match = re.fullmatch(r"(\d{4})[-/](\d{1,2})(?:[-/]\d{1,2})?", text)
    if match:
        year, month = int(match.group(1)), int(match.group(2))
        if 1 <= month <= 12:
            return pd.Period(year=year, month=month, freq="M")
    parsed = pd.to_datetime(text, errors="coerce")
    if not pd.isna(parsed):
        return pd.Timestamp(parsed).to_period("M")
    return None


def _parse_year(value) -> Optional[int]:
    try:
        number = int(float(str(value).strip()))
        return number if 1900 <= number <= 2100 else None
    except Exception:
        return None


def _parse_month_number(value) -> Optional[int]:
    if pd.isna(value):
        return None
    text = str(value).strip().lower()
    try:
        number = int(float(text))
        return number if 1 <= number <= 12 else None
    except Exception:
        pass
    names = {
        "jan": 1, "january": 1, "ene": 1, "enero": 1,
        "feb": 2, "february": 2, "febrero": 2,
        "mar": 3, "march": 3, "marzo": 3,
        "apr": 4, "april": 4, "abr": 4, "abril": 4,
        "may": 5, "mayo": 5, "jun": 6, "june": 6, "junio": 6,
        "jul": 7, "july": 7, "julio": 7, "aug": 8, "august": 8, "ago": 8, "agosto": 8,
        "sep": 9, "sept": 9, "september": 9, "set": 9, "septiembre": 9,
        "oct": 10, "october": 10, "octubre": 10,
        "nov": 11, "november": 11, "noviembre": 11,
        "dec": 12, "december": 12, "dic": 12, "diciembre": 12,
    }
    return next((number for name, number in names.items() if text == name or text.startswith(name + " ")), None)


def identify_period_and_metals(raw: pd.DataFrame) -> pd.DataFrame:
    """Detecta encabezados, periodo y series mensuales en el Excel mensual del Banco Mundial."""
    metals = tuple(PRICE_COLS)
    best_header, best_cols = None, {}
    for row_idx in range(min(35, len(raw))):
        found = {}
        for col_idx in range(raw.shape[1]):
            value = str(raw.iat[row_idx, col_idx]).strip().lower() if not pd.isna(raw.iat[row_idx, col_idx]) else ""
            for metal in metals:
                if value == metal or value.startswith((metal + ",", metal + " ", metal + "(")):
                    found[metal] = col_idx
        if len(found) > len(best_cols):
            best_header, best_cols = row_idx, found
    if best_header is None or len(best_cols) < len(metals):
        raise ValueError("No se identificaron los cinco encabezados de precios de metales en 'Monthly Prices'.")

    data_start = best_header + 1
    period_options = []
    for col_idx in range(min(5, raw.shape[1])):
        count = sum(parse_month(raw.iat[row_idx, col_idx]) is not None for row_idx in range(data_start, len(raw)))
        period_options.append((count, col_idx))
    period_count, period_col = max(period_options, default=(0, 0))
    rows = []

    if period_count >= 24:
        for row_idx in range(data_start, len(raw)):
            period = parse_month(raw.iat[row_idx, period_col])
            if period is None:
                continue
            record = {"mes": period}
            for metal, col_idx in best_cols.items():
                value = pd.to_numeric(raw.iat[row_idx, col_idx], errors="coerce")
                record[metal] = float(value) if pd.notna(value) and np.isfinite(float(value)) and float(value) > 0 else np.nan
            if sum(pd.notna(record.get(metal)) for metal in metals) >= 3:
                rows.append(record)
    else:
        for year_col in range(min(5, raw.shape[1])):
            for month_col in range(min(5, raw.shape[1])):
                if year_col == month_col:
                    continue
                built = []
                for row_idx in range(data_start, len(raw)):
                    year = _parse_year(raw.iat[row_idx, year_col])
                    month = _parse_month_number(raw.iat[row_idx, month_col])
                    if year is None or month is None:
                        continue
                    record = {"mes": pd.Period(year=year, month=month, freq="M")}
                    for metal, col_idx in best_cols.items():
                        value = pd.to_numeric(raw.iat[row_idx, col_idx], errors="coerce")
                        record[metal] = float(value) if pd.notna(value) and np.isfinite(float(value)) and float(value) > 0 else np.nan
                    built.append(record)
                if len(built) > len(rows):
                    rows = built

    if len(rows) < 24:
        raise ValueError(f"Solo se identificaron {len(rows)} meses válidos en la hoja mensual del Banco Mundial.")
    monthly = pd.DataFrame(rows).drop_duplicates("mes").sort_values("mes").reset_index(drop=True)
    for metal in metals:
        if metal not in monthly.columns or monthly[metal].notna().sum() < 24:
            raise ValueError(f"La serie mensual de {metal} tiene menos de 24 observaciones válidas.")
    return monthly


def current_world_bank_monthly_url() -> str:
    """Busca el archivo XLSX vigente en la página oficial, usando una URL de respaldo conocida."""
    class WorkbookLinkParser(HTMLParser):
        def __init__(self):
            super().__init__()
            self.links = []
        def handle_starttag(self, tag, attrs):
            if tag.lower() != "a":
                return
            attrs_map = dict(attrs)
            href = attrs_map.get("href", "")
            title = attrs_map.get("title", "")
            if "CMO-Historical-Data-Monthly.xlsx" in href or "CMO-Historical-Data-Monthly.xlsx" in title:
                self.links.append(urljoin(WORLD_BANK_PAGE, href))

    try:
        response = requests.get(WORLD_BANK_PAGE, timeout=12, headers={"User-Agent": "Mozilla/5.0"})
        response.raise_for_status()
        parser = WorkbookLinkParser()
        parser.feed(response.text)
        if parser.links:
            return parser.links[0]
    except Exception:
        pass
    return WORLD_BANK_MONTHLY_XLSX


@st.cache_data(ttl=43200, show_spinner=False)
def load_metal_index_quarterly() -> pd.DataFrame:
    """Descarga precios mensuales del Banco Mundial y crea medias trimestrales completas."""
    workbook_url = current_world_bank_monthly_url()
    response = requests.get(workbook_url, timeout=30, headers={"User-Agent": "Mozilla/5.0"})
    response.raise_for_status()
    workbook = pd.ExcelFile(BytesIO(response.content), engine="openpyxl")
    sheet = next((name for name in workbook.sheet_names if name.strip().lower() == "monthly prices"), None)
    if sheet is None:
        raise ValueError(f"No se encontró la hoja 'Monthly Prices'. Hojas disponibles: {workbook.sheet_names}")
    monthly = identify_period_and_metals(pd.read_excel(workbook, sheet_name=sheet, header=None, engine="openpyxl"))
    monthly["trimestre"] = monthly["mes"].map(lambda period: f"{period.year}Q{period.quarter}")

    aggregations = {"meses_observados": ("mes", "nunique")}
    for metal in PRICE_COLS:
        aggregations[metal] = (metal, "mean")
        aggregations[f"{metal}_n"] = (metal, "count")
    quarterly = monthly.groupby("trimestre", as_index=False).agg(**aggregations)
    quarterly["_key"] = quarterly["trimestre"].map(quarter_key)

    base_rows = quarterly[quarterly["trimestre"] == "2019Q1"]
    if base_rows.empty or base_rows.iloc[0][PRICE_COLS].isna().any():
        raise ValueError("Faltan precios completos para construir la base del índice (2019Q1).")
    base = base_rows.iloc[0]

    quarterly["indice_metales_ponderado"] = np.nan
    quarterly["precio_oro_usd_oz"] = np.nan
    complete_quarter = quarterly["meses_observados"].eq(3)
    complete_index = complete_quarter.copy()
    for metal in PRICE_COLS:
        complete_index &= quarterly[f"{metal}_n"].eq(3)
    complete_gold = complete_quarter & quarterly["gold_n"].eq(3)

    index_values = np.zeros(len(quarterly), dtype=float)
    for metal in PRICE_COLS:
        index_values += METAL_WEIGHTS[metal] * (quarterly[metal].astype(float) / float(base[metal])) * 100.0
    quarterly.loc[complete_index, "indice_metales_ponderado"] = index_values[complete_index.to_numpy()]
    quarterly.loc[complete_gold, "precio_oro_usd_oz"] = quarterly.loc[complete_gold, "gold"].astype(float)
    return quarterly.sort_values("_key").reset_index(drop=True)


def feature_frame(sales: pd.DataFrame, market: pd.DataFrame) -> pd.DataFrame:
    """Construye rezagos trimestrales sin rellenar las ventas históricas faltantes."""
    sales_part = sales[["trimestre", "_key", TARGET]].copy()
    keys = range(int(sales_part["_key"].min()) - 4, int(sales_part["_key"].max()) + 1)
    grid = pd.DataFrame({"_key": list(keys)})
    grid["trimestre"] = grid["_key"].map(key_to_quarter)
    grid = grid.merge(sales_part, on=["trimestre", "_key"], how="left", validate="one_to_one")
    market_cols = ["trimestre", "indice_metales_ponderado", "precio_oro_usd_oz"]
    grid = grid.merge(market[market_cols], on="trimestre", how="left", validate="one_to_one")
    grid["ventas_lag_1"] = grid[TARGET].shift(1)
    grid["ventas_lag_4"] = grid[TARGET].shift(4)
    grid["indice_metales_ponderado_lag_1"] = grid["indice_metales_ponderado"].shift(1)
    grid["precio_oro_lag_1_usd_oz"] = grid["precio_oro_usd_oz"].shift(1)
    return grid


# =====================================================================
# MODELO Y VALIDACIÓN TEMPORAL
# =====================================================================
def new_model() -> XGBRegressor:
    """XGBoost regularizado: predice el cambio trimestral y reduce el ajuste excesivo."""
    return XGBRegressor(
        objective="reg:squarederror",
        n_estimators=80,
        max_depth=2,
        learning_rate=0.04,
        min_child_weight=3,
        subsample=0.90,
        colsample_bytree=0.90,
        reg_lambda=10.0,
        random_state=42,
        n_jobs=1,
        tree_method="hist",
    )


def mae_rmse(actual, predicted) -> tuple[float, float]:
    return float(mean_absolute_error(actual, predicted)), float(np.sqrt(mean_squared_error(actual, predicted)))


def walk_forward_evaluation(
    features_df: pd.DataFrame,
    target_quarters: list[str],
    feature_cols: list[str],
) -> pd.DataFrame:
    """Predice un trimestre adelante; nunca usa ventas del trimestre que se está prediciendo."""
    records = []
    required = list(dict.fromkeys(feature_cols + [TARGET, "ventas_lag_1"]))
    for quarter in target_quarters:
        row = features_df.loc[features_df["trimestre"].eq(quarter)]
        if row.empty or row[required].isna().any(axis=None):
            continue
        row = row.iloc[[0]]
        prior = features_df.loc[features_df["_key"] < int(row["_key"].iloc[0])].copy()
        train = prior.dropna(subset=required)
        if len(train) < 16:
            continue
        model = new_model()
        delta = train[TARGET] - train["ventas_lag_1"]
        model.fit(train[feature_cols], delta)
        predicted_change = float(model.predict(row[feature_cols])[0])
        predicted_sales = max(0.0, float(row["ventas_lag_1"].iloc[0]) + predicted_change)
        records.append({
            "trimestre": quarter,
            "ventas_reales": float(row[TARGET].iloc[0]),
            "xgboost": predicted_sales,
            "cambio_xgboost": predicted_change,
            "ultimo_trimestre": float(row["ventas_lag_1"].iloc[0]),
            "estacional_lag_4": float(row["ventas_lag_4"].iloc[0]),
        })
    return pd.DataFrame(records)


def walk_forward_multistep_evaluation(
    features_df: pd.DataFrame,
    horizon: int = 4,
    holdout_origins: int = 5,
    feature_cols: Optional[list[str]] = None,
) -> pd.DataFrame:
    """Evalúa varios trimestres recursivos; en cada origen solo entrena con historia previa."""
    cols = feature_cols or FEATURES_INDEX
    base = features_df.dropna(subset=[TARGET, "ventas_lag_1", "ventas_lag_4"]).sort_values("_key").reset_index(drop=True)
    if len(base) < 22:
        raise ValueError(f"Solo hay {len(base)} filas útiles para el horizonte; se necesitan al menos 22.")
    max_origin = len(base) - horizon
    first_origin = max(16, max_origin - holdout_origins + 1)
    records = []
    external_cols = [col for col in cols if col not in {"ventas_lag_1", "ventas_lag_4"}]

    for origin_pos in range(first_origin, max_origin + 1):
        prior = base.iloc[:origin_pos].copy()
        required = list(dict.fromkeys(cols + [TARGET, "ventas_lag_1", "ventas_lag_4"]))
        train = prior.dropna(subset=required)
        if len(train) < 16:
            continue
        origin_row = base.iloc[origin_pos]
        external_value = None
        if external_cols:
            ext = external_cols[0]
            value = origin_row.get(ext, np.nan)
            if pd.notna(value):
                external_value = float(value)
            else:
                prior_values = prior[ext].dropna()
                if not prior_values.empty:
                    external_value = float(prior_values.iloc[-1])
            if external_value is None:
                continue

        model = new_model()
        model.fit(train[cols], train[TARGET] - train["ventas_lag_1"])
        sales_history = {str(r["trimestre"]): float(r[TARGET]) for _, r in prior.iterrows()}
        origin_label = str(prior["trimestre"].iloc[-1])
        persistence = float(prior[TARGET].iloc[-1])

        for step in range(1, horizon + 1):
            target_row = base.iloc[origin_pos + step - 1]
            quarter = str(target_row["trimestre"])
            q_key = int(target_row["_key"])
            previous = key_to_quarter(q_key - 1)
            seasonal_q = key_to_quarter(q_key - 4)
            lag1 = sales_history.get(previous)
            lag4 = sales_history.get(seasonal_q)
            if lag1 is None or lag4 is None:
                break
            row_values = {"ventas_lag_1": lag1, "ventas_lag_4": lag4}
            for ext in external_cols:
                row_values[ext] = external_value
            xrow = pd.DataFrame([row_values])[cols]
            predicted_change = float(model.predict(xrow)[0])
            prediction = max(0.0, lag1 + predicted_change)
            records.append({
                "origen": origin_label,
                "horizonte": step,
                "trimestre": quarter,
                "ventas_reales": float(target_row[TARGET]),
                "xgboost": prediction,
                "ultimo_trimestre": persistence,
                "estacional_lag_4": lag4,
            })
            # Solo agregamos la predicción, nunca la venta real futura.
            sales_history[quarter] = prediction

    result = pd.DataFrame(records)
    if result.empty:
        raise ValueError("No se generaron ventanas recursivas válidas para evaluar.")
    return result


def build_common_comparisons(features_df: pd.DataFrame) -> dict:
    """Prepara comparaciones con las mismas fechas de prueba para todos los métodos."""
    all_features = FEATURES_INDEX + FEATURES_GOLD
    common = features_df.dropna(subset=list(dict.fromkeys(all_features + [TARGET, "ventas_lag_1", "ventas_lag_4"])))
    target_quarters = common.sort_values("_key")["trimestre"].tail(6).tolist()
    if len(target_quarters) < 3:
        raise ValueError("No hay suficientes trimestres comunes para una comparación justa.")

    one_step = {
        "index": walk_forward_evaluation(features_df, target_quarters, FEATURES_INDEX),
        "gold": walk_forward_evaluation(features_df, target_quarters, FEATURES_GOLD),
        "plain": walk_forward_evaluation(features_df, target_quarters, FEATURES_NO_EXT),
    }
    shared_keys = ["trimestre"]
    one = one_step["index"][shared_keys + ["ventas_reales", "xgboost", "ultimo_trimestre", "estacional_lag_4"]].rename(columns={"xgboost": "xgboost_indice"})
    one = one.merge(one_step["gold"][shared_keys + ["xgboost"]].rename(columns={"xgboost": "xgboost_oro"}), on=shared_keys, how="inner", validate="one_to_one")
    one = one.merge(one_step["plain"][shared_keys + ["xgboost"]].rename(columns={"xgboost": "xgboost_sin_externa"}), on=shared_keys, how="inner", validate="one_to_one")
    if one.empty:
        raise ValueError("Los modelos no generaron fechas comunes en la evaluación a un trimestre.")

    multi_index = walk_forward_multistep_evaluation(features_df, 4, 5, FEATURES_INDEX)
    multi_gold = walk_forward_multistep_evaluation(features_df, 4, 5, FEATURES_GOLD)
    multi_plain = walk_forward_multistep_evaluation(features_df, 4, 5, FEATURES_NO_EXT)
    keycols = ["origen", "horizonte", "trimestre"]
    multi = multi_index[keycols + ["ventas_reales", "xgboost", "ultimo_trimestre", "estacional_lag_4"]].rename(columns={"xgboost": "xgboost_indice"})
    multi = multi.merge(multi_gold[keycols + ["xgboost"]].rename(columns={"xgboost": "xgboost_oro"}), on=keycols, how="inner", validate="one_to_one")
    multi = multi.merge(multi_plain[keycols + ["xgboost"]].rename(columns={"xgboost": "xgboost_sin_externa"}), on=keycols, how="inner", validate="one_to_one")
    if multi.empty:
        raise ValueError("Los modelos no generaron ventanas comunes en la evaluación recursiva.")
    return {"one_step": one, "multi_step": multi}


@st.cache_data(show_spinner=False)
def cached_comparisons(features_df: pd.DataFrame) -> dict:
    return build_common_comparisons(features_df)


def comparison_metrics(comparison: pd.DataFrame, multi_step: bool = False) -> pd.DataFrame:
    if multi_step:
        predictions = [
            ("XGBoost + índice de metales", "xgboost_indice", "Índice ponderado"),
            ("XGBoost + precio del oro", "xgboost_oro", "Precio del oro"),
            ("XGBoost sin variable externa", "xgboost_sin_externa", "Ninguna"),
            ("Referencia: último trimestre", "ultimo_trimestre", "Ninguna"),
            ("Referencia estacional", "estacional_lag_4", "Ninguna"),
        ]
    else:
        predictions = [
            ("XGBoost + índice de metales", "xgboost_indice", "Índice ponderado"),
            ("XGBoost + precio del oro", "xgboost_oro", "Precio del oro"),
            ("XGBoost sin variable externa", "xgboost_sin_externa", "Ninguna"),
            ("Referencia: último trimestre", "ultimo_trimestre", "Ninguna"),
            ("Referencia estacional", "estacional_lag_4", "Ninguna"),
        ]
    records = []
    for model_name, column, external_name in predictions:
        mae, rmse = mae_rmse(comparison["ventas_reales"], comparison[column])
        records.append({
            "Modelo": model_name,
            "Variable externa": external_name,
            "MAE (US$ M)": mae,
            "RMSE (US$ M)": rmse,
            "Pronósticos evaluados": len(comparison),
        })
    return pd.DataFrame(records).sort_values("MAE (US$ M)").reset_index(drop=True)


def choose_external_model(multi: pd.DataFrame) -> tuple[str, list[str], str]:
    """Elige entre las variantes XGBoost que incluyen una variable externa usando el MAE recursivo."""
    scores = {
        "Índice ponderado de metales": mae_rmse(multi["ventas_reales"], multi["xgboost_indice"])[0],
        "Precio del oro": mae_rmse(multi["ventas_reales"], multi["xgboost_oro"])[0],
    }
    selected_name = min(scores, key=scores.get)
    features = FEATURES_INDEX if selected_name == "Índice ponderado de metales" else FEATURES_GOLD
    return selected_name, features, EXTENSION_META[selected_name]["feature"]


def project_sales(
    sales: pd.DataFrame,
    market: pd.DataFrame,
    features_df: pd.DataFrame,
    horizon: int,
    scenario_pct: float,
    feature_cols: list[str],
    selected_external_name: str,
) -> tuple[pd.DataFrame, XGBRegressor]:
    """Entrena la variante seleccionada y construye el horizonte futuro recursivamente."""
    meta = EXTENSION_META[selected_external_name]
    feature_col = meta["feature"]
    market_col = meta["market_col"]
    train = features_df.dropna(subset=list(dict.fromkeys(feature_cols + [TARGET, "ventas_lag_1", "ventas_lag_4"]))).copy()
    if len(train) < 20:
        raise ValueError(f"Hay {len(train)} observaciones utilizables para entrenar; se necesitan al menos 20.")

    model = new_model()
    model.fit(train[feature_cols], train[TARGET] - train["ventas_lag_1"])
    sales_history = {r["trimestre"]: float(r[TARGET]) for _, r in sales.iterrows()}
    external_history = {
        str(row["trimestre"]): float(row[market_col])
        for _, row in market.iterrows()
        if pd.notna(row.get(market_col))
    }
    last_sales_quarter = str(sales["trimestre"].iloc[-1])
    next_q = next_quarter(last_sales_quarter)
    scenario_history: dict[str, float] = {}
    forecasts = []

    for _ in range(horizon):
        quarter = next_q
        q_key = quarter_key(quarter)
        external_quarter = prior_quarter(quarter)
        external_key = quarter_key(external_quarter)
        lag1_quarter = prior_quarter(quarter)
        lag4_quarter = key_to_quarter(q_key - 4)
        lag1 = sales_history.get(lag1_quarter)
        lag4 = sales_history.get(lag4_quarter)
        if lag1 is None or lag4 is None:
            raise ValueError(f"No se encontraron los rezagos de ventas necesarios para {quarter}.")

        if external_quarter in external_history:
            external_value = external_history[external_quarter]
            external_source = f"Observado · {external_quarter}"
        elif external_quarter in scenario_history:
            external_value = scenario_history[external_quarter]
            external_source = f"Escenario · {external_quarter}"
        else:
            prior_values = {**external_history, **scenario_history}
            earlier = [key for key in prior_values if quarter_key(key) < external_key]
            if not earlier:
                raise ValueError(f"No hay un precio anterior disponible para construir el escenario de {external_quarter}.")
            base_quarter = max(earlier, key=quarter_key)
            steps = external_key - quarter_key(base_quarter)
            external_value = prior_values[base_quarter] * ((1.0 + scenario_pct / 100.0) ** steps)
            scenario_history[external_quarter] = external_value
            external_source = f"Supuesto desde {base_quarter}"

        row_values = {
            "ventas_lag_1": lag1,
            "ventas_lag_4": lag4,
            feature_col: external_value,
        }
        xrow = pd.DataFrame([row_values])[feature_cols]
        predicted_change = float(model.predict(xrow)[0])
        predicted_sales = max(0.0, lag1 + predicted_change)
        forecasts.append({
            "trimestre": quarter,
            "pronostico_xgboost_usd_millones": predicted_sales,
            "cambio_estimado_usd_millones": predicted_change,
            "ventas_lag_1_usd_millones": lag1,
            "ventas_lag_4_usd_millones": lag4,
            "periodo_variable_externa": external_quarter,
            "variable_externa": selected_external_name,
            "valor_variable_externa_lag_1": external_value,
            "unidad_variable_externa": meta["unit"],
            "origen_variable_externa": external_source,
        })
        sales_history[quarter] = predicted_sales
        next_q = next_quarter(quarter)

    return pd.DataFrame(forecasts), model


# =====================================================================
# DATOS, MODELO Y DISTRIBUCIÓN DEL DASHBOARD
# =====================================================================
with st.sidebar:
    st.markdown(
        f"""
        <div class="bvn-side-brand">
          <img src="{OFFICIAL_LOGO_URL}" alt="Compañía de Minas Buenaventura" loading="eager">
          <div class="bvn-side-wordmark">INTELIGENCIA DE VENTAS</div>
          <div class="bvn-side-tagline">Pronóstico de ingresos</div>
        </div>
        <nav class="bvn-nav" aria-label="Navegación del dashboard">
          <a href="#top-dashboard">▥ &nbsp; Resumen</a>
          <a href="#forecast-section">⌁ &nbsp; Pronóstico</a>
          <a href="#validation-section">▤ &nbsp; Precisión del modelo</a>
          <a href="#methodology-section">◎ &nbsp; Metodología</a>
          <a href="#sources-section">⇩ &nbsp; Datos y fuentes</a>
        </nav>
        <div class="bvn-side-bottom">
          <div class="motto">Minería<br>que genera<br>un mundo mejor</div>
          <div class="side-note"><b>Modelo analítico</b><br>XGBoost + variables de precios de metales<br><br><b>Prototipo analítico · Revisión 1.1</b><br>No es una proyección oficial de la compañía.<br><br><span style="font-size:.65rem;opacity:.8">Build: 2026-10-20 · escenario sincronizado</span></div>
        </div>
        """,
        unsafe_allow_html=True,
    )

sales_file = find_sales_file()
if sales_file is None:
    st.error(
        f"No encuentro **{SALES_FILENAME}**. Coloca ese archivo existente en la carpeta `data` o en la raíz de `proyecto_xgboost_buenaventura`. No se ha creado una base nueva."
    )
    st.code("C:\\Users\\Estudio's\\Downloads\\proyecto_xgboost_buenaventura\\data\\" + SALES_FILENAME)
    st.link_button("Abrir portal oficial de inversionistas", OFFICIAL_INVESTOR_URL)
    st.stop()

try:
    sales = load_sales(str(sales_file))
except Exception as exc:
    st.error(f"No se pudo validar la base histórica existente: {exc}")
    st.stop()

with st.spinner("Consultando precios históricos completos de los metales en el Banco Mundial…"):
    try:
        market = load_metal_index_quarterly()
    except Exception as exc:
        st.error("No se pudo cargar la serie de precios del Banco Mundial. La base de ventas no se modificó.")
        st.caption(f"Detalle técnico: {exc}")
        st.link_button("Abrir fuente oficial de precios", WORLD_BANK_PAGE)
        st.info("Comprueba la conexión a Internet y pulsa «Actualizar escenario». La consulta se guarda temporalmente en caché para acelerar las siguientes ejecuciones.")
        st.stop()

features_df = feature_frame(sales, market)
usable = features_df.dropna(subset=FEATURES_INDEX + [TARGET]).copy()

try:
    comparisons = cached_comparisons(features_df)
    one = comparisons["one_step"]
    multi = comparisons["multi_step"]
    one_metrics = comparison_metrics(one, multi_step=False)
    multi_metrics = comparison_metrics(multi, multi_step=True)
    selected_model_name, selected_feature_cols, _ = choose_external_model(multi)
except Exception as exc:
    st.error(f"No se pudo completar la validación temporal: {exc}")
    st.stop()

last_sales = float(sales[TARGET].iloc[-1])
last_sales_quarter = str(sales["trimestre"].iloc[-1])
selected_label = "XGBoost + índice de metales" if selected_model_name == "Índice ponderado de metales" else "XGBoost + precio del oro"
selected_multi_row = multi_metrics.loc[multi_metrics["Modelo"].eq(selected_label)].iloc[0]
selected_one_row = one_metrics.loc[one_metrics["Modelo"].eq(selected_label)].iloc[0]
baseline_multi = multi_metrics[multi_metrics["Modelo"].isin(["Referencia: último trimestre", "Referencia estacional"])].sort_values("MAE (US$ M)").iloc[0]
baseline_one = one_metrics[one_metrics["Modelo"].isin(["Referencia: último trimestre", "Referencia estacional"])].sort_values("MAE (US$ M)").iloc[0]
latest_idx = market.dropna(subset=["indice_metales_ponderado"])
latest_gold = market.dropna(subset=["precio_oro_usd_oz"])
latest_index_row = latest_idx.iloc[-1] if not latest_idx.empty else None
latest_gold_row = latest_gold.iloc[-1] if not latest_gold.empty else None

# Controles de escenario en una banda horizontal, como en la propuesta aprobada.
st.markdown('<div id="top-dashboard"></div>', unsafe_allow_html=True)
st.markdown(
    f"""
    <section class="bvn-hero">
      <div class="bvn-hero-copy">
        <div class="bvn-kicker">BVN · INTELIGENCIA DE VENTAS</div>
        <h1>Pronóstico de ingresos trimestrales</h1>
        <p>Modelo experimental con XGBoost para estimar los ingresos de Buenaventura usando la historia trimestral y el comportamiento de los precios de metales.</p>
        <div class="bvn-chiprow">
          <span class="bvn-chip">▥ Datos de ingresos</span>
          <span class="bvn-chip">◷ Validación temporal</span>
          <span class="bvn-chip">◉ Precios del Banco Mundial</span>
        </div>
        <div class="bvn-legal">Prototipo analítico independiente · No es una proyección oficial de la compañía.</div>
      </div>
    </section>
    """,
    unsafe_allow_html=True,
)

with st.container(border=True):
    st.markdown('<div class="bvn-panel-title"><span class="bvn-mini-icon">⚙</span>Configuración del escenario</div>', unsafe_allow_html=True)
    cfg1, cfg2, cfg3 = st.columns([1.0, 1.55, .9], gap="medium")
    with cfg1:
        horizon = st.selectbox("Trimestres a proyectar", options=list(range(1, 9)), index=3, key="horizon_control")
    with cfg2:
        scenario_pct_widget = st.slider(
            f"Cambio trimestral supuesto de {selected_model_name.lower()}",
            min_value=-10.0, max_value=10.0, value=0.0, step=1.0, format="%.0f%%",
            help="Solo se aplica a trimestres sin un dato externo completo observado. 0% mantiene el último valor completo conocido.",
            key="scenario_pct_control",
        )
        # Fuente única para cálculo y presentación; evita depender de valores duplicados.
        scenario_pct_active = float(st.session_state["scenario_pct_control"])
        scenario_pct_label = f"{scenario_pct_active:+.0f}%"
        st.caption(f"Escenario activo en esta sesión: {scenario_pct_label} por trimestre.")
    with cfg3:
        if st.button("↻ Actualizar precios externos", type="primary", width="stretch", key="refresh_prices_main"):
            load_metal_index_quarterly.clear()
            st.rerun()
    st.caption("El escenario se recalcula al mover los controles. Este botón fuerza la actualización de los precios externos del Banco Mundial.")
    st.caption("El supuesto editable no es una predicción oficial del precio de los metales. Los datos de ingresos se leen sin modificarse.")

# Entrenar el modelo elegido y preparar proyecciones.
try:
    forecast, final_model = project_sales(
        sales=sales,
        market=market,
        features_df=features_df,
        horizon=int(horizon),
        scenario_pct=float(scenario_pct_active),
        feature_cols=selected_feature_cols,
        selected_external_name=selected_model_name,
    )
except Exception as exc:
    st.error(f"No se pudo calcular la proyección: {exc}")
    st.stop()

# Contrafactual explícito: escenario elegido frente al mismo modelo con 0%.
# No se inventa una elasticidad; se mide la diferencia real que devuelve XGBoost.
scenario_diagnostic_error = None
try:
    if abs(float(scenario_pct_active)) > 1e-9:
        forecast_reference, _ = project_sales(
            sales=sales,
            market=market,
            features_df=features_df,
            horizon=int(horizon),
            scenario_pct=0.0,
            feature_cols=selected_feature_cols,
            selected_external_name=selected_model_name,
        )
    else:
        forecast_reference = forecast.copy()
    scenario_income_effect = (
        forecast["pronostico_xgboost_usd_millones"].to_numpy(dtype=float)
        - forecast_reference["pronostico_xgboost_usd_millones"].to_numpy(dtype=float)
    )
except Exception as exc:
    forecast_reference = None
    scenario_income_effect = np.full(len(forecast), np.nan, dtype=float)
    scenario_diagnostic_error = f"{type(exc).__name__}: {exc}"

# La comparación se conserva también en la tabla exportable para comprobar el efecto por trimestre.
forecast["efecto_vs_escenario_0_usd_millones"] = scenario_income_effect

if int(horizon) > 4:
    st.warning(
        "El pronóstico supera los cuatro trimestres usados en la evaluación recursiva. "
        "Los trimestres 5 a 8 son extrapolaciones adicionales y no cuentan con la misma validación específica."
    )

first_row = forecast.iloc[0]
last_row = forecast.iloc[-1]
first_value = float(first_row["pronostico_xgboost_usd_millones"])
last_value = float(last_row["pronostico_xgboost_usd_millones"])

def format_usd_millions(value: float) -> str:
    """Formato espanol compacto: US$ 545,7 millones."""
    number = f"{value:,.1f}"
    number = number.replace(",", "TMPSEP").replace(".", ",").replace("TMPSEP", ".")
    return f"US$ {number} millones"

first_value_display = format_usd_millions(first_value)
last_value_display = format_usd_millions(last_value)
first_change_pct = ((first_value / last_sales) - 1.0) * 100.0
selected_recursive_rmse = float(selected_multi_row["RMSE (US$ M)"])
selected_recursive_mae = float(selected_multi_row["MAE (US$ M)"])

# Resumen ejecutivo: cuatro tarjetas, con etiquetas cortas para evitar cortes.
st.markdown('<div class="section-kicker">Resumen ejecutivo</div>', unsafe_allow_html=True)
kpi = st.columns(4, gap="medium")
kpi[0].metric("Último ingreso observado", f"US$ {last_sales:,.1f} M", last_sales_quarter)
kpi[1].metric("Cobertura histórica", f"{len(sales)} trimestres", f"{sales['trimestre'].iloc[0]} – {last_sales_quarter}")
kpi[2].metric("Modelo seleccionado · 4T", "XGBoost + oro" if selected_model_name == "Precio del oro" else "XGBoost + índice", f"MAE recursivo {selected_recursive_mae:,.1f} M")
if latest_index_row is not None:
    kpi[3].metric("Índice ponderado de metales", f"{float(latest_index_row['indice_metales_ponderado']):,.1f}", f"Base 2019Q1 = 100 · {latest_index_row['trimestre']}")
else:
    kpi[3].metric("Índice ponderado de metales", "No disponible")
st.caption(f"Último ingreso real disponible: {last_sales_quarter}. Los precios externos tienen su propio calendario de publicación; la tabla identifica el periodo de precio usado en cada pronóstico.")

# Gráfica principal y escenario futuro en dos columnas, igual que el concepto visual.
st.markdown('<div id="forecast-section"></div>', unsafe_allow_html=True)
chart_col, outlook_col = st.columns([2.0, 1.05], gap="medium")
with chart_col:
    with st.container(border=True):
        st.markdown('<div class="bvn-panel-title"><span class="bvn-mini-icon">▥</span>Histórico y pronóstico de ingresos trimestrales</div>', unsafe_allow_html=True)
        st.markdown('<div class="bvn-chart-note">Ingresos consolidados en US$ millones · La línea punteada separa los datos observados del pronóstico.</div>', unsafe_allow_html=True)
        history_view = sales[["trimestre", TARGET]].copy().rename(columns={TARGET: "Ingresos"})
        forecast_view = forecast[["trimestre", "pronostico_xgboost_usd_millones"]].copy().rename(columns={"pronostico_xgboost_usd_millones": "Ingresos"})
        hist_x = history_view["trimestre"].tolist()
        future_x = forecast_view["trimestre"].tolist()
        combined_labels = hist_x + future_x
        # Mostrar etiquetas alternas en la historia para conservar legibilidad; etiquetar todos los puntos futuros.
        tick_values = hist_x[::2]
        if last_sales_quarter not in tick_values:
            tick_values.append(last_sales_quarter)
        tick_values += future_x

        fig = go.Figure()
        # Resaltar suavemente la zona de proyección sin ocultar la trayectoria de los datos.
        if future_x:
            fig.add_vrect(
                x0=last_sales_quarter, x1=future_x[-1],
                fillcolor="rgba(125,184,68,0.055)", line_width=0, layer="below",
            )
        # Rango ilustrativo, basado en RMSE recursivo; no es un intervalo estadístico formal.
        upper = [max(0.0, float(v) + selected_recursive_rmse) for v in forecast_view["Ingresos"]]
        lower = [max(0.0, float(v) - selected_recursive_rmse) for v in forecast_view["Ingresos"]]
        if future_x:
            fig.add_trace(go.Scatter(
                x=future_x, y=lower, mode="lines", line=dict(width=0, color="rgba(125,184,68,0)"),
                hoverinfo="skip", showlegend=False, name="Límite inferior ilustrativo",
            ))
            fig.add_trace(go.Scatter(
                x=future_x, y=upper, mode="lines", line=dict(width=0, color="rgba(125,184,68,0)"),
                fill="tonexty", fillcolor="rgba(125,184,68,.18)",
                name="Rango ilustrativo ±RMSE", hoverinfo="skip",
            ))
        fig.add_trace(go.Scatter(
            x=history_view["trimestre"], y=history_view["Ingresos"], mode="lines+markers",
            name="Ingresos observados", line=dict(color="#174E38", width=2.8), marker=dict(size=5, color="#174E38"),
            hovertemplate="%{x}<br>US$ %{y:,.1f} millones<extra>Real</extra>",
        ))
        pred_x = [last_sales_quarter] + future_x
        pred_y = [last_sales] + [float(v) for v in forecast_view["Ingresos"]]
        fig.add_trace(go.Scatter(
            x=pred_x, y=pred_y, mode="lines+markers", name="Pronóstico XGBoost",
            line=dict(color="#78B83F", width=3, dash="dash"),
            marker=dict(size=8, symbol="diamond", color="#78B83F", line=dict(color="#FFFFFF", width=1)),
            hovertemplate="%{x}<br>US$ %{y:,.1f} millones<extra>Pronóstico</extra>",
        ))
        fig.add_vline(x=last_sales_quarter, line_width=1.4, line_dash="dot", line_color="#627F6D")
        fig.add_annotation(
            x=last_sales_quarter, y=1.02, xref="x", yref="paper",
            text="INICIO DEL PRONÓSTICO", showarrow=False, xanchor="right", yanchor="bottom",
            font=dict(color="#477957", size=9), bgcolor="rgba(251,253,251,.92)",
            bordercolor="#DDE8DC", borderwidth=1, borderpad=4,
        )
        fig.update_layout(
            height=390, template="plotly_white", paper_bgcolor="rgba(0,0,0,0)", plot_bgcolor="#FBFDFB",
            margin=dict(l=48, r=14, t=58, b=82),
            legend=dict(orientation="h", y=1.14, x=0, xanchor="left", yanchor="bottom", font=dict(color="#193D2D", size=11), bgcolor="rgba(255,255,255,.96)", bordercolor="#E1EAE2", borderwidth=1),
            font=dict(family="Arial, sans-serif", color="#2C4035", size=12), hovermode="x unified",
            hoverlabel=dict(bgcolor="#12372C", bordercolor="#12372C", font=dict(color="#FFFFFF", size=12)),
            # Fija explícitamente el orden temporal: los trazos de la banda de pronóstico
            # se añaden antes que la historia y, sin categoryarray, Plotly puede ordenar
            # las categorías por su primera aparición (dibujando el futuro a la izquierda).
            xaxis=dict(title=dict(text="Trimestre", font=dict(color="#163D2C", size=13)), tickfont=dict(color="#344A3D", size=10), tickmode="array", tickvals=tick_values, tickangle=-38, categoryorder="array", categoryarray=combined_labels, showgrid=False, automargin=True, showline=True, linecolor="#9BAFA1", title_standoff=14),
            yaxis=dict(title=dict(text="Ingresos totales (US$ millones)", font=dict(color="#163D2C", size=13)), tickfont=dict(color="#52695D", size=11), gridcolor="#E4ECE5", zeroline=False, tickformat=",.0f", rangemode="tozero", automargin=True, title_standoff=12, showline=False),
        )
        fig.update_xaxes(title=dict(text="Trimestre", font=dict(color="#163D2C", size=13)), tickfont=dict(color="#344A3D", size=10))
        fig.update_yaxes(title=dict(text="Ingresos totales (US$ millones)", font=dict(color="#163D2C", size=13)), tickfont=dict(color="#344A3D", size=11))
        st.plotly_chart(fig, width="stretch", theme=None, config={"displaylogo": False, "responsive": True})
        st.markdown('<div class="bvn-chart-note">La banda verde representa ±RMSE de validación recursiva; es orientativa, no un intervalo de confianza estadístico.</div>', unsafe_allow_html=True)

with outlook_col:
    with st.container(border=True):
        st.markdown('<div class="bvn-panel-title"><span class="bvn-mini-icon">◎</span>Escenario de ingresos futuros</div>', unsafe_allow_html=True)
        st.markdown(
            f'<div class="bvn-outlook-head"><span class="bvn-outlook-head-label">Horizonte proyectado</span><span class="bvn-outlook-head-value">{len(forecast)} trimestres</span></div>',
            unsafe_allow_html=True,
        )
        first_pill_class = "bvn-pill-positive" if first_change_pct >= 0 else "bvn-pill-negative"
        first_pill_icon = "↑" if first_change_pct >= 0 else "↓"
        last_change_pct = (last_value / last_sales - 1) * 100
        last_pill_class = "bvn-pill-positive" if last_change_pct >= 0 else "bvn-pill-negative"
        last_pill_icon = "↑" if last_change_pct >= 0 else "↓"
        st.markdown(
            f"""
            <div class="bvn-outlook-card bvn-outlook-card-primary">
              <div class="bvn-outlook-label"><span>Primera proyección</span><span class="bvn-outlook-period">{first_row['trimestre']}</span></div>
              <div class="bvn-outlook-value">{first_value_display}</div>
              <div class="bvn-outlook-foot"><span class="bvn-outlook-pill {first_pill_class}">{first_pill_icon} {first_change_pct:+.1f}%</span><span>frente al último ingreso real</span></div>
            </div>
            <div class="bvn-outlook-card bvn-outlook-card-final">
              <div class="bvn-outlook-label"><span>Último trimestre proyectado</span><span class="bvn-outlook-period">{last_row['trimestre']}</span></div>
              <div class="bvn-outlook-value">{last_value_display}</div>
              <div class="bvn-outlook-foot"><span class="bvn-outlook-pill {last_pill_class}">{last_pill_icon} {last_change_pct:+.1f}%</span><span>frente al último ingreso real</span></div>
            </div>
            <div class="bvn-outlook-card bvn-outlook-card-scenario">
              <div class="bvn-outlook-label"><span>Supuesto de variable externa</span><span class="bvn-outlook-period">Escenario editable</span></div>
              <div class="bvn-outlook-value">{scenario_pct_label} <small>por trimestre</small></div>
              <div class="bvn-outlook-foot"><span class="bvn-outlook-pill bvn-pill-neutral">Solo para precios faltantes</span></div>
            </div>
            <div class="bvn-outlook-meta"><span class="bvn-outlook-meta-icon">⌁</span><span><strong>Periodo cubierto</strong>{first_row['trimestre']} – {last_row['trimestre']}</span></div>
            <div class="bvn-outlook-meta"><span class="bvn-outlook-meta-icon">◈</span><span><strong>Variable externa</strong>{selected_model_name}</span></div>
            """,
            unsafe_allow_html=True,
        )
        if scenario_diagnostic_error is not None and abs(float(scenario_pct_active)) > 1e-9:
            st.warning("No se pudo comparar este escenario contra el caso de 0%. Revisa el mensaje técnico y no interpretes el efecto como cero.")
            st.caption(f"Detalle de diagnóstico: {scenario_diagnostic_error}")
        elif abs(float(scenario_pct_active)) > 1e-9 and forecast_reference is not None:
            if np.allclose(scenario_income_effect, 0.0, rtol=0.0, atol=0.01):
                st.warning(
                    "El escenario sí modifica los precios simulados del oro, pero XGBoost devuelve "
                    "los mismos ingresos que en el escenario de 0% para estos valores. El control "
                    "está conectado; el modelo no está mostrando sensibilidad local en este rango. "
                    "No interpretes esto como prueba de que el oro no afecta al negocio."
                )
            else:
                last_effect = float(scenario_income_effect[-1])
                st.info(
                    f"Efecto estimado frente al escenario 0% en {last_row['trimestre']}: "
                    f"US$ {last_effect:+,.2f} millones."
                )
        else:
            st.caption(
                "Efecto contra el escenario base de 0% disponible en la tabla detallada. "
                "Si es cero, el modelo no ha respondido a este cambio de precio en ese horizonte."
            )

# Tabla detallada e interpretación lado a lado.
st.markdown('<div class="section-kicker">Pronóstico detallado</div>', unsafe_allow_html=True)
table_col, explain_col = st.columns([2.15, 1.0], gap="medium")
with table_col:
    with st.container(border=True):
        st.markdown('<div class="bvn-panel-title"><span class="bvn-mini-icon">▤</span>Pronóstico trimestral detallado</div>', unsafe_allow_html=True)
        external_title = "Índice metales (base 100)" if selected_model_name == "Índice ponderado de metales" else "Precio del oro (US$/oz)"
        table = pd.DataFrame({
            "Trimestre": forecast["trimestre"],
            "Pronóstico (US$ M)": forecast["pronostico_xgboost_usd_millones"].map(lambda v: f"{v:,.2f}"),
            "Cambio (US$ M)": forecast["cambio_estimado_usd_millones"].map(lambda v: f"{v:+,.2f}"),
            "Ingreso previo (US$ M)": forecast["ventas_lag_1_usd_millones"].map(lambda v: f"{v:,.2f}"),
            "Mismo trim. año previo": forecast["ventas_lag_4_usd_millones"].map(lambda v: f"{v:,.2f}"),
            external_title: forecast["valor_variable_externa_lag_1"].map(lambda v: f"{v:,.2f}"),
            "Efecto vs 0% (US$ M)": forecast["efecto_vs_escenario_0_usd_millones"].map(lambda v: f"{v:+,.2f}" if pd.notna(v) else "No calculado"),
            "Periodo y origen del dato": forecast.apply(lambda row: f"{row['periodo_variable_externa']} · {row['origen_variable_externa']}", axis=1),
        })
        st.dataframe(table, width="stretch", hide_index=True, height=min(310, 48 + 36 * len(table)))
        st.download_button(
            "⬇ Descargar pronóstico en CSV",
            data=forecast.to_csv(index=False).encode("utf-8-sig"),
            file_name="pronostico_ventas_buenaventura_xgboost.csv",
            mime="text/csv", type="primary", width="content",
        )
with explain_col:
    with st.container(border=True):
        st.markdown('<div class="bvn-panel-title"><span class="bvn-mini-icon">☼</span>¿Cómo interpretar este modelo?</div>', unsafe_allow_html=True)
        st.markdown(
            """
            <div class="bvn-steps">
              <div class="bvn-step"><div class="bvn-step-number">1</div><div><div class="bvn-step-title">Usa la historia del negocio</div><div class="bvn-step-text">Analiza los ingresos trimestrales y los precios de los metales como variable externa.</div></div></div>
              <div class="bvn-step"><div class="bvn-step-number">2</div><div><div class="bvn-step-title">Genera un pronóstico recursivo</div><div class="bvn-step-text">Cada trimestre proyectado se usa como entrada para el siguiente.</div></div></div>
              <div class="bvn-step"><div class="bvn-step-number">3</div><div><div class="bvn-step-title">Resultados experimentales</div><div class="bvn-step-text">Son escenarios para análisis, no estimaciones oficiales de Buenaventura.</div></div></div>
            </div>
            """,
            unsafe_allow_html=True,
        )
        st.markdown('<div class="bvn-divider"></div>', unsafe_allow_html=True)
        st.markdown(f"**MAE recursivo del modelo elegido:** US$ {selected_recursive_mae:,.1f} millones. Un MAE menor indica menor error medio.")
        if selected_recursive_mae <= float(baseline_multi["MAE (US$ M)"]):
            st.success(f"En la evaluación a cuatro trimestres, XGBoost superó a la mejor referencia sencilla ({baseline_multi['Modelo']}).")
        else:
            st.warning(f"En la evaluación a cuatro trimestres, la referencia sencilla ({baseline_multi['Modelo']}) tuvo menor MAE. Interpreta el pronóstico con cautela.")

# La validación completa permanece disponible sin saturar el resumen ejecutivo.
st.markdown('<div id="validation-section"></div>', unsafe_allow_html=True)
with st.expander("Ver precisión histórica y comparación de modelos", expanded=True):
    st.markdown("### Evaluación a un trimestre")
    st.markdown("Todos los métodos se comparan en las mismas fechas. El MAE mide el error absoluto medio en US$ millones; menor es mejor.")
    one_display = one_metrics.copy()
    one_display["MAE (US$ M)"] = one_display["MAE (US$ M)"].map(lambda value: f"{value:,.1f}")
    one_display["RMSE (US$ M)"] = one_display["RMSE (US$ M)"].map(lambda value: f"{value:,.1f}")
    st.dataframe(one_display, width="stretch", hide_index=True)
    one_delta = ((float(selected_one_row["MAE (US$ M)"]) / float(baseline_one["MAE (US$ M)"])) - 1.0) * 100.0
    if one_delta <= 0:
        st.success(f"A un trimestre, el modelo seleccionado igualó o superó a la mejor referencia (diferencia de MAE {one_delta:+.1f}%).")
    else:
        st.warning(
            f"Limitación relevante: a un trimestre, la variante XGBoost seleccionada para el horizonte de cuatro trimestres "
            f"tuvo {one_delta:.1f}% más MAE que la mejor referencia sencilla "
            f"({selected_one_row['MAE (US$ M)']:.1f} frente a {baseline_one['MAE (US$ M)']:.1f} US$ millones). "
            "Para decisiones a un solo trimestre, la referencia sencilla fue más precisa. "
            "La selección de XGBoost responde al horizonte recursivo de cuatro trimestres, no a todos los horizontes."
        )
    with st.expander("Ver predicciones históricas individuales"):
        one_detail = one.rename(columns={
            "trimestre": "Trimestre", "ventas_reales": "Ingresos reales (US$ M)",
            "xgboost_indice": "XGBoost + índice (US$ M)", "xgboost_oro": "XGBoost + oro (US$ M)",
            "xgboost_sin_externa": "XGBoost sin variable externa (US$ M)",
            "ultimo_trimestre": "Último trimestre (US$ M)", "estacional_lag_4": "Mismo trimestre año anterior (US$ M)",
        })
        st.dataframe(one_detail, width="stretch", hide_index=True)

    st.markdown("### Evaluación recursiva a cuatro trimestres")
    multi_display = multi_metrics.copy()
    multi_display["MAE (US$ M)"] = multi_display["MAE (US$ M)"].map(lambda value: f"{value:,.1f}")
    multi_display["RMSE (US$ M)"] = multi_display["RMSE (US$ M)"].map(lambda value: f"{value:,.1f}")
    st.dataframe(multi_display, width="stretch", hide_index=True)
    best_xgb_label = str(multi_metrics[multi_metrics["Modelo"].str.startswith("XGBoost")].sort_values("MAE (US$ M)").iloc[0]["Modelo"])
    if selected_recursive_mae <= float(baseline_multi["MAE (US$ M)"]):
        st.success(f"La variante externa seleccionada ({selected_model_name}) logró un MAE menor o igual que la mejor referencia sencilla en la validación a cuatro trimestres.")
    else:
        st.warning(f"La mejor referencia sencilla ({baseline_multi['Modelo']}) supera al modelo seleccionado en esta evaluación. El resultado se conserva como experimental.")
    if best_xgb_label != selected_label:
        st.info(f"La variante XGBoost con menor MAE absoluto en la prueba recursiva fue: {best_xgb_label}. La proyección principal conserva la mejor variante XGBoost con variable externa, según la regla del proyecto.")
    with st.expander("Ver las ventanas de validación recursiva"):
        multi_detail = multi.rename(columns={
            "origen": "Trimestre de origen", "horizonte": "Paso", "trimestre": "Trimestre evaluado", "ventas_reales": "Ingresos reales (US$ M)",
            "xgboost_indice": "XGBoost + índice (US$ M)", "xgboost_oro": "XGBoost + oro (US$ M)",
            "xgboost_sin_externa": "XGBoost sin externa (US$ M)", "ultimo_trimestre": "Persistencia (US$ M)",
            "estacional_lag_4": "Referencia estacional (US$ M)",
        })
        st.dataframe(multi_detail, width="stretch", hide_index=True)

with st.expander("¿Por qué variaron tanto los ingresos recientes? · Reportes de Buenaventura"):
    st.markdown(
        f"""
        Los reportes oficiales muestran movimientos pronunciados: los ingresos del **4T25 fueron US$ 623.4 millones**, frente a US$ 299.6 millones en 4T24; el **1T26 fue US$ 624.6 millones**, frente a US$ 307.7 millones en 1T25. Los comunicados relacionan el crecimiento con cambios en precios realizados y volúmenes vendidos.

        El precio de los metales es solo una parte de la explicación: la producción, los volúmenes vendidos, la mezcla de minerales, las paradas operativas y los resultados de negocios asociados también pueden influir. Por eso un índice de precios no explica todas las variaciones de ingresos.

        Fuentes: [Resultados FY2025/4T25]({FY2025_RESULTS_URL}) · [Resultados 1T26]({Q1_2026_RESULTS_URL}) · [Estados financieros consolidados 2T26]({Q2_2026_FINANCIALS_URL}).
        """
    )

st.markdown('<div id="methodology-section"></div>', unsafe_allow_html=True)
with st.expander("Metodología y cómo leer los resultados"):
    st.markdown(
        """
        **Ingresos observados:** son los valores históricos de la base; no son estimaciones del algoritmo.

        **Pronóstico XGBoost:** el modelo estima el cambio trimestral y lo suma al ingreso del trimestre anterior. Para los siguientes pasos se reutilizan las ventas predichas, no ventas futuras reales.

        **Rezago 1:** ingreso del trimestre inmediatamente anterior. **Rezago 4:** ingreso del mismo trimestre del año previo.

        **MAE:** error absoluto medio en US$ millones. **RMSE:** penaliza más los errores grandes, también en US$ millones.

        **Rango verde de la gráfica:** banda ilustrativa basada en ±RMSE de la evaluación recursiva; no es un intervalo estadístico formal ni una garantía.

        **Advertencia metodológica:** solo hay 30 trimestres históricos. La evaluación temporal evita usar ventas futuras durante entrenamiento, pero no elimina la incertidumbre ni garantiza que XGBoost supere una referencia sencilla.
        """
    )

st.markdown('<div id="sources-section"></div>', unsafe_allow_html=True)
with st.expander("Datos y fuentes"):
    weights_text = "\n".join(f"- {METAL_LABELS_ES[metal]}: {weight * 100:.2f}%" for metal, weight in METAL_WEIGHTS.items())
    st.markdown(
        f"""
        **Variable objetivo:** `total_revenues_usd_millones`, ingresos trimestrales en US$ millones. El modelo no utiliza EBITDA como variable objetivo.

        **Base histórica:** `{sales_file}`. La aplicación la lee en modo de solo lectura, no la edita ni la reemplaza.

        **Variable externa elegida:** {selected_model_name}. Se selecciona entre XGBoost + índice ponderado y XGBoost + precio del oro según el menor MAE recursivo a cuatro trimestres. XGBoost sin variable externa y dos referencias sencillas se mantienen como controles.

        **Índice ponderado:** precios trimestrales del Banco Mundial para oro, plata, cobre, plomo y zinc; cada metal se normaliza a 100 en 2019Q1. Las ponderaciones son una aproximación calculada con volumen × precio realizado FY2025; no son participación auditada de ingresos por metal y no deben interpretarse como mezcla oficial de ventas. Los insumos de los metales pueden tener distintos alcances operativos, por lo que este índice es un proxy analítico.

        **Ponderaciones usadas:**
        {weights_text}

        **Trazabilidad de precios:** la aplicación consulta el XLSX mensual vigente desde la página del Banco Mundial y calcula medias trimestrales con los tres meses observados. El resultado de precios se mantiene en caché 12 horas; cuando la fuente externa se actualiza, puede cambiar el resultado. Para reproducir una corrida histórica exactamente, es necesario archivar también la versión del XLSX utilizada.

        **Control adicional de la base:** [Libro de auditoría Excel: serie, control anual y diccionario](https://github.com/dlgeronimom-stack/buenaventura-sales-intelligence/blob/main/data/base_ventas_buenaventura_2019Q1_2026Q2.xlsx) · [Notas de trazabilidad](https://github.com/dlgeronimom-stack/buenaventura-sales-intelligence/blob/main/data/LEEME_base_ventas_buenaventura.md).

        **Fuentes:** [Portal de inversionistas de Buenaventura]({OFFICIAL_INVESTOR_URL}) · [Resultados FY2025/4T25]({FY2025_RESULTS_URL}) · [Resultados 1T26]({Q1_2026_RESULTS_URL}) · [Resultados 2T26 (comunicado oficial SEC 6-K)]({Q2_2026_RESULTS_URL}) · [Estados financieros 2T26]({Q2_2026_FINANCIALS_URL}) · [Banco Mundial — Commodity Markets / Pink Sheet]({WORLD_BANK_PAGE}).
        """
    )

st.markdown(
    f"""
    <div class="bvn-footer">
      <strong>BUENAVENTURA · INTELIGENCIA DE VENTAS</strong><br>
      Prototipo analítico independiente para aprendizaje y análisis. Se usa el logotipo público y una ilustración minera decorativa generada para el diseño; no es un producto oficial de Buenaventura.<br>
      Base histórica sin modificaciones · Último trimestre de ingresos: {last_sales_quarter} · Fuente de precios externos: Banco Mundial.
    </div>
    """,
    unsafe_allow_html=True,
)
