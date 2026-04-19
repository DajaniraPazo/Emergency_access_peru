"""
Cálculo del Índice de Acceso a Servicios de Emergencia (IASE) a nivel distrital.

Metodología:
El IASE combina tres componentes normalizados:

  A. Disponibilidad de establecimientos (facility_score):
     n_ipress / max(n_ipress) → refleja cuántos establecimientos tiene el distrito.

  B. Actividad de emergencias (activity_score):
     total_emergencias / max(total_emergencias) → refleja el volumen de atenciones.

  C. Acceso espacial (access_score):
     1 - normalized(dist_media_km) → mayor distancia = menor acceso.
     Para distritos sin centros poblados registrados, se asigna 0 (sin datos de acceso).

Especificaciones:
  - Baseline: pesos iguales (1/3 cada componente)
  - Alternativa: mayor peso en acceso espacial (A=0.25, B=0.25, C=0.50)
    Justificación: en emergencias médicas el tiempo de traslado es el factor
    más crítico; este peso penaliza más fuertemente a distritos con centros
    poblados alejados de cualquier establecimiento.
"""
import numpy as np
import pandas as pd
import geopandas as gpd
from src.utils import safe_normalize, quintile_label, QUINTILE_ORDER, TABLES_DIR, setup_logging

log = setup_logging("metrics")

# Pesos de cada especificación
WEIGHTS_BASELINE = {"facility": 1 / 3, "activity": 1 / 3, "access": 1 / 3}
WEIGHTS_ALTERNATIVE = {"facility": 0.25, "activity": 0.25, "access": 0.50}


def compute_components(dist_gdf: gpd.GeoDataFrame) -> gpd.GeoDataFrame:
    """
    Normaliza los tres componentes del IASE a escala [0, 1].
    Agrega columnas: facility_score, activity_score, access_score.
    """
    gdf = dist_gdf.copy()

    gdf["facility_score"] = safe_normalize(gdf["n_ipress"].fillna(0))

    emerg_col = "total_emergencias" if "total_emergencias" in gdf.columns else None
    if emerg_col:
        gdf["activity_score"] = safe_normalize(gdf[emerg_col].fillna(0))
    else:
        log.warning("Sin datos de emergencias; activity_score = 0.")
        gdf["activity_score"] = 0.0

    if "dist_media_km" in gdf.columns:
        dist_norm = safe_normalize(gdf["dist_media_km"].fillna(gdf["dist_media_km"].max()))
        gdf["access_score"] = 1 - dist_norm
        gdf.loc[gdf["n_centros"] == 0, "access_score"] = 0.0
    else:
        log.warning("Sin datos de distancia; access_score = 0.")
        gdf["access_score"] = 0.0

    return gdf


def compute_iase_baseline(dist_gdf: gpd.GeoDataFrame) -> gpd.GeoDataFrame:
    """
    Especificación baseline: pesos iguales (1/3 por componente).
    Agrega columnas: IASE_base, clase_base.
    """
    gdf = compute_components(dist_gdf)
    w = WEIGHTS_BASELINE
    gdf["IASE_base"] = (
        w["facility"] * gdf["facility_score"] +
        w["activity"] * gdf["activity_score"] +
        w["access"] * gdf["access_score"]
    )
    gdf["clase_base"] = gdf["IASE_base"].apply(quintile_label)
    log.info("IASE baseline calculado.")
    return gdf


def compute_iase_alternative(dist_gdf: gpd.GeoDataFrame) -> gpd.GeoDataFrame:
    """
    Especificación alternativa: mayor peso en acceso espacial (A=0.25, B=0.25, C=0.50).
    Agrega columnas: IASE_alt, clase_alt.
    """
    gdf = dist_gdf.copy()
    if "facility_score" not in gdf.columns:
        gdf = compute_components(gdf)
    w = WEIGHTS_ALTERNATIVE
    gdf["IASE_alt"] = (
        w["facility"] * gdf["facility_score"] +
        w["activity"] * gdf["activity_score"] +
        w["access"] * gdf["access_score"]
    )
    gdf["clase_alt"] = gdf["IASE_alt"].apply(quintile_label)
    log.info("IASE alternativa calculada.")
    return gdf


def compute_full_iase(dist_gdf: gpd.GeoDataFrame) -> gpd.GeoDataFrame:
    """Calcula ambas especificaciones en un solo paso."""
    gdf = compute_iase_baseline(dist_gdf)
    gdf = compute_iase_alternative(gdf)
    return gdf


def classify_districts(gdf: gpd.GeoDataFrame, score_col: str = "IASE_base") -> gpd.GeoDataFrame:
    """
    Clasifica distritos en quintiles y añade columna de rango numérico.
    """
    gdf = gdf.copy()
    label_col = score_col.replace("IASE_", "clase_")
    if label_col not in gdf.columns:
        gdf[label_col] = gdf[score_col].apply(quintile_label)

    rank_col = score_col + "_rank"
    gdf[rank_col] = gdf[score_col].rank(ascending=False, method="min").astype(int)
    return gdf


def compare_specifications(gdf: gpd.GeoDataFrame) -> pd.DataFrame:
    """
    Compara los resultados del baseline vs la especificación alternativa.

    Devuelve un DataFrame con:
    - ubigeo, departamento, distrito (si existen)
    - IASE_base, IASE_alt
    - rank_base, rank_alt
    - rank_change: cambio de posición (positivo = mejora de rango en alternativa)
    - clase_base, clase_alt
    - clase_change: si la clase cambió entre especificaciones
    """
    id_cols = [c for c in ["ubigeo", "departamento", "provincia", "distrito"] if c in gdf.columns]
    score_cols = [c for c in ["IASE_base", "IASE_alt", "clase_base", "clase_alt"] if c in gdf.columns]

    df = gdf[id_cols + score_cols].copy()

    if "IASE_base" in df.columns:
        df["rank_base"] = df["IASE_base"].rank(ascending=False, method="min").astype(int)
    if "IASE_alt" in df.columns:
        df["rank_alt"] = df["IASE_alt"].rank(ascending=False, method="min").astype(int)

    if "rank_base" in df.columns and "rank_alt" in df.columns:
        df["rank_change"] = df["rank_base"] - df["rank_alt"]  # positivo = sube en alternativa

    if "clase_base" in df.columns and "clase_alt" in df.columns:
        df["clase_change"] = df["clase_base"] != df["clase_alt"]
        n_changed = df["clase_change"].sum()
        pct_changed = n_changed / len(df) * 100
        log.info(
            f"Comparación: {n_changed} distritos ({pct_changed:.1f}%) "
            "cambiaron de clase entre baseline y alternativa."
        )

    df = df.sort_values("IASE_base", ascending=False).reset_index(drop=True)

    df.to_csv(TABLES_DIR / "comparison_baseline_vs_alternative.csv", index=False)
    log.info("Tabla de comparación guardada en output/tables/")
    return df


def get_top_bottom(
    gdf: gpd.GeoDataFrame,
    score_col: str = "IASE_base",
    n: int = 10,
) -> tuple:
    """
    Retorna los n distritos mejor y peor servidos según score_col.
    Devuelve (top_df, bottom_df).
    """
    sorted_df = gdf.sort_values(score_col, ascending=False).reset_index(drop=True)
    top = sorted_df.head(n).copy()
    bottom = sorted_df.tail(n).copy()
    return top, bottom


def save_district_table(gdf: gpd.GeoDataFrame) -> None:
    """Guarda la tabla distrital final en output/tables/."""
    cols = [c for c in gdf.columns if c != "geometry"]
    df = gdf[cols].copy()
    df.to_csv(TABLES_DIR / "district_iase_table.csv", index=False)
    log.info("Tabla distrital final guardada en output/tables/district_iase_table.csv")
