"""
Limpieza y preprocesamiento de los cuatro datasets.
Cada función recibe el DataFrame crudo y devuelve uno limpio y estándar.
"""
import pandas as pd
import numpy as np
import geopandas as gpd
from src.utils import (
    IPRESS_COL_MAP, EMERGENCIAS_COL_MAP, CENTROS_COL_MAP, DISTRITOS_COL_MAP,
    PROCESSED_DIR, CRS_GEO, apply_col_map, setup_logging
)

log = setup_logging("cleaning")

# ---------------------------------------------------------------------------
# IPRESS
# ---------------------------------------------------------------------------

def clean_ipress(df: pd.DataFrame) -> pd.DataFrame:
    """
    Limpia el dataset de establecimientos IPRESS.

    Decisiones de limpieza:
    - Se renombran columnas a nombres estándar mediante IPRESS_COL_MAP.
    - Se eliminan registros sin latitud o longitud (no pueden localizarse).
    - Se filtran solo establecimientos ACTIVOS (si existe la columna 'estado').
    - Se eliminan duplicados por id_ipress.
    - Se estandariza ubigeo a 6 dígitos con cero a la izquierda.
    - Las coordenadas se validan dentro del bbox aproximado de Perú.
    """
    df = apply_col_map(df.copy(), IPRESS_COL_MAP)

    required = ["latitud", "longitud"]
    for col in required:
        if col not in df.columns:
            raise ValueError(
                f"Columna requerida '{col}' no encontrada en IPRESS. "
                f"Columnas disponibles: {list(df.columns)}"
            )

    n_inicial = len(df)

    # Convertir coordenadas a numérico
    df["latitud"] = pd.to_numeric(df["latitud"], errors="coerce")
    df["longitud"] = pd.to_numeric(df["longitud"], errors="coerce")

    # Eliminar filas sin coordenadas
    df = df.dropna(subset=["latitud", "longitud"])

    # Bbox de Perú: lat [-18.4, -0.0], lon [-81.5, -68.7]
    mask_valid = (
        df["latitud"].between(-18.4, 0.0) &
        df["longitud"].between(-81.5, -68.7)
    )
    df = df[mask_valid].copy()
    log.info(f"IPRESS: {n_inicial} → {len(df)} filas tras filtrar coords inválidas")

    # Filtrar solo activos
    if "estado" in df.columns:
        df["estado"] = df["estado"].astype(str).str.upper().str.strip()
        n_antes = len(df)
        df = df[df["estado"].isin(["ACTIVO", "1", "ACTIVE"])].copy()
        log.info(f"IPRESS: {n_antes} → {len(df)} filas tras filtrar ACTIVOS")

    # Eliminar duplicados
    if "id_ipress" in df.columns:
        df = df.drop_duplicates(subset=["id_ipress"])

    # Estandarizar ubigeo
    if "ubigeo" in df.columns:
        df["ubigeo"] = df["ubigeo"].astype(str).str.zfill(6)

    df = df.reset_index(drop=True)
    log.info(f"IPRESS limpio: {len(df)} registros")
    return df


# ---------------------------------------------------------------------------
# EMERGENCIAS
# ---------------------------------------------------------------------------

def clean_emergencias(df: pd.DataFrame) -> pd.DataFrame:
    """
    Limpia y agrega el dataset de producción de emergencias por IPRESS.

    Decisiones de limpieza:
    - Se renombran columnas mediante EMERGENCIAS_COL_MAP.
    - Se convierte la columna de emergencias a numérico.
    - Se eliminan filas con emergencias negativas o nulas.
    - Si existe columna 'periodo' (YYYYMM), se extrae el año.
    - Se agrega el total de emergencias por id_ipress (suma histórica).
    - El ubigeo se estandariza a 6 dígitos.
    """
    df = apply_col_map(df.copy(), EMERGENCIAS_COL_MAP)

    if "emergencias" not in df.columns:
        candidates = [c for c in df.columns if "emerg" in c.lower() or "atenc" in c.lower()]
        if candidates:
            df = df.rename(columns={candidates[0]: "emergencias"})
            log.warning(f"Se usó '{candidates[0]}' como columna de emergencias.")
        else:
            raise ValueError(
                "No se encontró columna de emergencias. "
                f"Columnas disponibles: {list(df.columns)}"
            )

    # Extraer año desde 'periodo' si no hay columna 'anio'
    if "anio" not in df.columns and "periodo" in df.columns:
        df["periodo"] = df["periodo"].astype(str)
        df["anio"] = pd.to_numeric(df["periodo"].str[:4], errors="coerce")

    df["emergencias"] = pd.to_numeric(df["emergencias"], errors="coerce")
    df = df[df["emergencias"].notna() & (df["emergencias"] >= 0)].copy()

    # Agrupación por establecimiento
    group_cols = ["id_ipress"]
    if "ubigeo" in df.columns:
        df["ubigeo"] = df["ubigeo"].astype(str).str.zfill(6)

    agg_dict = {"emergencias": "sum"}
    if "nombre_ipress" in df.columns:
        agg_dict["nombre_ipress"] = "first"
    if "ubigeo" in df.columns:
        agg_dict["ubigeo"] = "first"
    if "departamento" in df.columns:
        agg_dict["departamento"] = "first"

    df_agg = df.groupby(group_cols, as_index=False).agg(agg_dict)
    df_agg = df_agg.rename(columns={"emergencias": "total_emergencias"})

    log.info(f"Emergencias limpias: {len(df_agg)} establecimientos con datos")
    return df_agg


# ---------------------------------------------------------------------------
# CENTROS POBLADOS
# ---------------------------------------------------------------------------

def clean_centros_poblados(df: pd.DataFrame) -> pd.DataFrame:
    """
    Limpia el dataset de centros poblados del INEI.

    Decisiones de limpieza:
    - Renombrado estándar de columnas.
    - Eliminación de filas sin coordenadas.
    - Validación de coordenadas dentro del bbox de Perú.
    - Estandarización de ubigeo a 6 dígitos.
    - Eliminación de duplicados por nombre + ubigeo.
    """
    df = apply_col_map(df.copy(), CENTROS_COL_MAP)

    for col in ["latitud", "longitud"]:
        if col not in df.columns:
            raise ValueError(
                f"Columna '{col}' no encontrada en Centros Poblados. "
                f"Columnas disponibles: {list(df.columns)}"
            )

    n_inicial = len(df)
    df["latitud"] = pd.to_numeric(df["latitud"], errors="coerce")
    df["longitud"] = pd.to_numeric(df["longitud"], errors="coerce")
    df = df.dropna(subset=["latitud", "longitud"])

    mask_valid = (
        df["latitud"].between(-18.4, 0.0) &
        df["longitud"].between(-81.5, -68.7)
    )
    df = df[mask_valid].copy()
    log.info(f"Centros Poblados: {n_inicial} → {len(df)} filas tras filtrar coords")

    if "ubigeo" in df.columns:
        df["ubigeo"] = df["ubigeo"].astype(str).str.zfill(6)

    dedup_cols = [c for c in ["ubigeo", "nombre_ccpp"] if c in df.columns]
    if dedup_cols:
        df = df.drop_duplicates(subset=dedup_cols)

    df = df.reset_index(drop=True)
    log.info(f"Centros Poblados limpios: {len(df)} registros")
    return df


# ---------------------------------------------------------------------------
# DISTRITOS
# ---------------------------------------------------------------------------

def clean_distritos(gdf: gpd.GeoDataFrame) -> gpd.GeoDataFrame:
    """
    Limpia el GeoDataFrame de distritos.

    Decisiones de limpieza:
    - Renombrado estándar de columnas.
    - Estandarización de ubigeo a 6 dígitos.
    - Eliminación de geometrías nulas o inválidas.
    - Reproyección a CRS_GEO si es necesario.
    """
    gdf = gdf.copy()
    gdf.columns = [c.lower().strip() for c in gdf.columns]
    rename = {k: v for k, v in DISTRITOS_COL_MAP.items() if k in gdf.columns}
    gdf = gdf.rename(columns=rename)

    if "ubigeo" not in gdf.columns:
        # Intentar construir ubigeo desde columnas separadas
        candidates = {
            "iddpto": 2, "idprov": 4, "iddist": 6,
            "dep": 2, "prov": 4, "dist": 6,
        }
        log.warning(
            "No se encontró columna 'ubigeo'. "
            f"Columnas disponibles: {list(gdf.columns)}"
        )

    if "ubigeo" in gdf.columns:
        gdf["ubigeo"] = gdf["ubigeo"].astype(str).str.zfill(6)

    # Eliminar geometrías inválidas
    n_antes = len(gdf)
    gdf = gdf[~gdf.geometry.isna()].copy()
    gdf = gdf[gdf.geometry.is_valid | gdf.geometry.buffer(0).is_valid].copy()
    gdf.geometry = gdf.geometry.buffer(0)  # corrige geometrías menores inválidas
    log.info(f"Distritos: {n_antes} → {len(gdf)} tras limpiar geometrías")

    if gdf.crs is None:
        from src.utils import CRS_GEO
        gdf = gdf.set_crs(CRS_GEO)
    elif gdf.crs.to_epsg() != 4326:
        gdf = gdf.to_crs(CRS_GEO)

    return gdf


# ---------------------------------------------------------------------------
# Pipeline completo
# ---------------------------------------------------------------------------

def run_cleaning_pipeline(
    df_ipress, df_emergencias, df_centros, gdf_distritos
):
    """
    Ejecuta la limpieza de los 4 datasets y guarda los resultados en processed/.
    Devuelve los 4 objetos limpios.
    """
    log.info("=== Iniciando pipeline de limpieza ===")

    ipress_clean = clean_ipress(df_ipress)
    emerg_clean = clean_emergencias(df_emergencias)
    centros_clean = clean_centros_poblados(df_centros)
    distritos_clean = clean_distritos(gdf_distritos)

    # Guardar en processed/
    ipress_clean.to_csv(PROCESSED_DIR / "ipress_clean.csv", index=False)
    emerg_clean.to_csv(PROCESSED_DIR / "emergencias_clean.csv", index=False)
    centros_clean.to_csv(PROCESSED_DIR / "centros_poblados_clean.csv", index=False)
    distritos_clean.to_file(PROCESSED_DIR / "distritos_clean.gpkg", driver="GPKG")

    log.info("=== Limpieza completa. Archivos guardados en data/processed/ ===")
    return ipress_clean, emerg_clean, centros_clean, distritos_clean


def generate_data_dictionary() -> pd.DataFrame:
    """Genera un diccionario de variables estándar del proyecto."""
    rows = [
        ("ipress_clean.csv", "id_ipress", "str", "Código RENAES del establecimiento"),
        ("ipress_clean.csv", "nombre_ipress", "str", "Nombre del establecimiento de salud"),
        ("ipress_clean.csv", "ubigeo", "str", "Código ubigeo de 6 dígitos (distrittal)"),
        ("ipress_clean.csv", "departamento", "str", "Nombre del departamento"),
        ("ipress_clean.csv", "provincia", "str", "Nombre de la provincia"),
        ("ipress_clean.csv", "distrito", "str", "Nombre del distrito"),
        ("ipress_clean.csv", "categoria", "str", "Categoría del establecimiento (I-1 … III-E)"),
        ("ipress_clean.csv", "latitud", "float", "Latitud WGS84 (negativa en hemisferio sur)"),
        ("ipress_clean.csv", "longitud", "float", "Longitud WGS84 (negativa, Perú)"),
        ("emergencias_clean.csv", "id_ipress", "str", "Código RENAES del establecimiento"),
        ("emergencias_clean.csv", "total_emergencias", "int", "Total de atenciones de emergencia registradas"),
        ("emergencias_clean.csv", "ubigeo", "str", "Código ubigeo distrital"),
        ("centros_poblados_clean.csv", "ubigeo", "str", "Código ubigeo del distrito al que pertenece"),
        ("centros_poblados_clean.csv", "nombre_ccpp", "str", "Nombre del centro poblado"),
        ("centros_poblados_clean.csv", "latitud", "float", "Latitud WGS84"),
        ("centros_poblados_clean.csv", "longitud", "float", "Longitud WGS84"),
        ("distritos_clean.gpkg", "ubigeo", "str", "Código ubigeo distrital de 6 dígitos"),
        ("distritos_clean.gpkg", "departamento", "str", "Nombre del departamento"),
        ("distritos_clean.gpkg", "distrito", "str", "Nombre del distrito"),
        ("distritos_clean.gpkg", "geometry", "Polygon", "Límite distrital en WGS84"),
    ]
    return pd.DataFrame(rows, columns=["archivo", "variable", "tipo", "descripcion"])
