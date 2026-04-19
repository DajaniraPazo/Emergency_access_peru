"""
Constantes de rutas, CRS y funciones auxiliares compartidas.
"""
from pathlib import Path
import numpy as np
import logging

# ---------------------------------------------------------------------------
# Rutas del proyecto
# ---------------------------------------------------------------------------
ROOT = Path(__file__).resolve().parents[1]
RAW_DIR = ROOT / "data" / "Raw"
PROCESSED_DIR = ROOT / "data" / "processed"
FIGURES_DIR = ROOT / "output" / "figures"
TABLES_DIR = ROOT / "output" / "tables"

for _d in [RAW_DIR, PROCESSED_DIR, FIGURES_DIR, TABLES_DIR]:
    _d.mkdir(parents=True, exist_ok=True)

# ---------------------------------------------------------------------------
# Sistemas de referencia de coordenadas
# ---------------------------------------------------------------------------
CRS_GEO = "EPSG:4326"       # WGS84 – coordenadas geográficas
CRS_PROJ = "EPSG:32718"     # UTM Zona 18S – métrico para Perú (distancias en metros)

# ---------------------------------------------------------------------------
# Mapeo flexible de nombres de columnas (lower-strip) → nombre estándar
# Permite adaptarse a distintas versiones de los datasets de MINSA/INEI.
# ---------------------------------------------------------------------------
IPRESS_COL_MAP = {
    "codigo_renaes": "id_ipress",
    "codigo": "id_ipress",
    "nombre": "nombre_ipress",
    "nombre_ipress": "nombre_ipress",
    "departamento": "departamento",
    "provincia": "provincia",
    "distrito": "distrito",
    "ubigeo": "ubigeo",
    "ubi_geo": "ubigeo",
    "categoria": "categoria",
    "latitud": "latitud",
    "lat": "latitud",
    "longitud": "longitud",
    "lon": "longitud",
    "lng": "longitud",
    "estado": "estado",
}

EMERGENCIAS_COL_MAP = {
    "codigo_renaes": "id_ipress",
    "codigo": "id_ipress",
    "nombre_ipress": "nombre_ipress",
    "nombre": "nombre_ipress",
    "departamento": "departamento",
    "provincia": "provincia",
    "distrito": "distrito",
    "ubigeo": "ubigeo",
    "anio": "anio",
    "año": "anio",
    "year": "anio",
    "mes": "mes",
    "month": "mes",
    "periodo": "periodo",
    "emergencias_atendidas": "emergencias",
    "emergencias": "emergencias",
    "atenciones": "emergencias",
    "total_emergencias": "emergencias",
    "n_emergencias": "emergencias",
    "atencion_emergencia": "emergencias",
}

CENTROS_COL_MAP = {
    "ubigeo": "ubigeo",
    "ubi_geo": "ubigeo",
    "nombre_ccpp": "nombre_ccpp",
    "nombre": "nombre_ccpp",
    "centro_poblado": "nombre_ccpp",
    "departamento": "departamento",
    "provincia": "provincia",
    "distrito": "distrito",
    "latitud": "latitud",
    "lat": "latitud",
    "longitud": "longitud",
    "lon": "longitud",
    "lng": "longitud",
    "tipo": "tipo",
    "poblacion": "poblacion",
    "pob_total": "poblacion",
    "altitud": "altitud",
}

DISTRITOS_COL_MAP = {
    "ubigeo": "ubigeo",
    "iddist": "ubigeo",
    "cod_dist": "ubigeo",
    "departamen": "departamento",
    "departamento": "departamento",
    "provincia": "provincia",
    "distrito": "distrito",
}

# ---------------------------------------------------------------------------
# Funciones auxiliares
# ---------------------------------------------------------------------------

def setup_logging(name: str = "emergency_access") -> logging.Logger:
    logging.basicConfig(
        level=logging.INFO,
        format="%(asctime)s [%(levelname)s] %(name)s – %(message)s",
        datefmt="%H:%M:%S",
    )
    return logging.getLogger(name)


def apply_col_map(df, col_map: dict):
    """Normaliza nombres de columnas y aplica el mapa de renombrado."""
    df.columns = [c.lower().strip() for c in df.columns]
    rename = {k: v for k, v in col_map.items() if k in df.columns}
    return df.rename(columns=rename)


def safe_normalize(series):
    """Min-max normalización a [0, 1]. Devuelve 0 si todos los valores son iguales."""
    mn, mx = series.min(), series.max()
    if mx == mn:
        return series * 0.0
    return (series - mn) / (mx - mn)


def quintile_label(score: float) -> str:
    """Etiqueta de quintil basada en un score normalizado 0-1."""
    if score >= 0.80:
        return "Muy alto"
    elif score >= 0.60:
        return "Alto"
    elif score >= 0.40:
        return "Medio"
    elif score >= 0.20:
        return "Bajo"
    else:
        return "Muy bajo"


QUINTILE_COLORS = {
    "Muy alto": "#1a9641",
    "Alto": "#a6d96a",
    "Medio": "#ffffbf",
    "Bajo": "#fdae61",
    "Muy bajo": "#d7191c",
}

QUINTILE_ORDER = ["Muy alto", "Alto", "Medio", "Bajo", "Muy bajo"]
