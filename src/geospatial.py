"""
Pipeline geoespacial: GeoDataFrames, joins espaciales y cálculo de distancias.

Decisión de CRS:
- Se usa EPSG:4326 (WGS84) para almacenamiento y visualización.
- Se reprojecta a EPSG:32718 (UTM Zona 18S) para cálculos de distancia en metros,
  ya que esta proyección minimiza la distorsión métrica para el territorio peruano.
"""
import numpy as np
import pandas as pd
import geopandas as gpd
from scipy.spatial import cKDTree
from src.utils import CRS_GEO, CRS_PROJ, PROCESSED_DIR, setup_logging

log = setup_logging("geospatial")


# ---------------------------------------------------------------------------
# Construcción de GeoDataFrames
# ---------------------------------------------------------------------------

def make_geodataframe(
    df: pd.DataFrame,
    lon_col: str = "longitud",
    lat_col: str = "latitud",
    crs: str = CRS_GEO,
) -> gpd.GeoDataFrame:
    """Convierte un DataFrame con coordenadas en un GeoDataFrame de puntos."""
    gdf = gpd.GeoDataFrame(
        df,
        geometry=gpd.points_from_xy(df[lon_col], df[lat_col]),
        crs=crs,
    )
    return gdf


# ---------------------------------------------------------------------------
# Joins espaciales
# ---------------------------------------------------------------------------

def join_points_to_districts(
    gdf_points: gpd.GeoDataFrame,
    gdf_districts: gpd.GeoDataFrame,
    district_ubigeo_col: str = "ubigeo",
) -> gpd.GeoDataFrame:
    """
    Asigna cada punto al distrito que lo contiene (sjoin 'within').
    Si un punto cae fuera de todos los distritos (bordes o errores de coord.),
    se intenta con 'nearest' como fallback.
    """
    gdf_points = gdf_points.to_crs(CRS_GEO)
    gdf_districts = gdf_districts.to_crs(CRS_GEO)

    # Join principal: 'within'
    joined = gpd.sjoin(
        gdf_points,
        gdf_districts[[district_ubigeo_col, "geometry"]],
        how="left",
        predicate="within",
    )
    joined = joined.drop(columns=["index_right"], errors="ignore")

    # Fallback nearest para puntos sin distrito asignado
    sin_distrito = joined[joined[district_ubigeo_col].isna()].drop(
        columns=[district_ubigeo_col], errors="ignore"
    )
    if len(sin_distrito) > 0:
        log.info(
            f"{len(sin_distrito)} puntos sin distrito por 'within'; "
            "aplicando fallback nearest."
        )
        fallback = gpd.sjoin_nearest(
            sin_distrito,
            gdf_districts[[district_ubigeo_col, "geometry"]],
            how="left",
        )
        fallback = fallback.drop(columns=["index_right"], errors="ignore")
        joined.update(fallback[[district_ubigeo_col]])

    assigned = joined[district_ubigeo_col].notna().sum()
    log.info(f"Asignados {assigned}/{len(gdf_points)} puntos a distritos")
    return joined


# ---------------------------------------------------------------------------
# Distancia al establecimiento más cercano
# ---------------------------------------------------------------------------

def compute_nearest_facility_distance(
    gdf_centros: gpd.GeoDataFrame,
    gdf_ipress: gpd.GeoDataFrame,
    crs_proj: str = CRS_PROJ,
) -> np.ndarray:
    """
    Para cada centro poblado, calcula la distancia euclidiana (en km)
    al establecimiento IPRESS más cercano, usando un cKDTree.

    Se reprojecta a UTM Zona 18S para obtener distancias métricas precisas.
    Devuelve un array de distancias en km, una por cada centro poblado.
    """
    centros_proj = gdf_centros.to_crs(crs_proj)
    ipress_proj = gdf_ipress.to_crs(crs_proj)

    ipress_coords = np.column_stack([
        ipress_proj.geometry.x.values,
        ipress_proj.geometry.y.values,
    ])
    centros_coords = np.column_stack([
        centros_proj.geometry.x.values,
        centros_proj.geometry.y.values,
    ])

    tree = cKDTree(ipress_coords)
    distances_m, _ = tree.query(centros_coords, k=1, workers=-1)
    distances_km = distances_m / 1000.0

    log.info(
        f"Distancias calculadas: mediana={np.median(distances_km):.1f} km, "
        f"máximo={np.max(distances_km):.1f} km"
    )
    return distances_km


# ---------------------------------------------------------------------------
# Resumen a nivel distrital
# ---------------------------------------------------------------------------

def build_district_geodataframe(
    ipress_clean: pd.DataFrame,
    emergencias_clean: pd.DataFrame,
    centros_clean: pd.DataFrame,
    distritos_clean: gpd.GeoDataFrame,
) -> gpd.GeoDataFrame:
    """
    Construye el GeoDataFrame distrital consolidado con:
    - n_ipress: número de establecimientos IPRESS en el distrito
    - total_emergencias: suma de atenciones de emergencia del distrito
    - n_centros: número de centros poblados en el distrito
    - dist_media_km: distancia media (km) de los centros poblados al IPRESS más cercano
    - dist_max_km: distancia máxima (km) al IPRESS más cercano
    - pct_centros_lejanos: porcentaje de centros poblados a más de 10 km del IPRESS más cercano

    Manejo de distritos sin datos: se rellenan con 0 / NaN según la variable.
    """
    log.info("Construyendo GeoDataFrame distrital...")

    # GeoDataFrames de puntos
    gdf_ipress = make_geodataframe(ipress_clean)
    gdf_centros = make_geodataframe(centros_clean)

    # Asignar IPRESS a distritos
    ipress_with_dist = join_points_to_districts(gdf_ipress, distritos_clean)
    if "ubigeo_right" in ipress_with_dist.columns and "ubigeo" not in ipress_with_dist.columns:
        ipress_with_dist = ipress_with_dist.rename(columns={"ubigeo_right": "ubigeo"})

    # Asignar centros poblados a distritos
    centros_with_dist = join_points_to_districts(gdf_centros, distritos_clean)
    if "ubigeo_right" in centros_with_dist.columns and "ubigeo" not in centros_with_dist.columns:
        centros_with_dist = centros_with_dist.rename(columns={"ubigeo_right": "ubigeo"})

    # Calcular distancias: centros → IPRESS más cercano
    distances_km = compute_nearest_facility_distance(gdf_centros, gdf_ipress)
    centros_with_dist = centros_with_dist.copy()
    centros_with_dist["dist_nearest_km"] = distances_km

    # Agregar IPRESS por distrito
    ipress_ubigeo_col = _resolve_ubigeo_col(ipress_with_dist)
    ipress_agg = (
        ipress_with_dist.groupby(ipress_ubigeo_col)
        .size()
        .reset_index(name="n_ipress")
        .rename(columns={ipress_ubigeo_col: "ubigeo"})
    )

    # Agregar emergencias (unir con ipress y luego al distrito)
    emerg_agg = None
    if "id_ipress" in ipress_with_dist.columns and "id_ipress" in emergencias_clean.columns:
        ipress_emerg = ipress_with_dist[[ipress_ubigeo_col, "id_ipress"]].merge(
            emergencias_clean[["id_ipress", "total_emergencias"]],
            on="id_ipress",
            how="left",
        )
        emerg_agg = (
            ipress_emerg.groupby(ipress_ubigeo_col)["total_emergencias"]
            .sum()
            .reset_index()
            .rename(columns={ipress_ubigeo_col: "ubigeo"})
        )
    elif "ubigeo" in emergencias_clean.columns:
        emerg_agg = (
            emergencias_clean.groupby("ubigeo")["total_emergencias"]
            .sum()
            .reset_index()
        )

    # Agregar centros poblados por distrito
    centros_ubigeo_col = _resolve_ubigeo_col(centros_with_dist)
    centros_agg = (
        centros_with_dist.groupby(centros_ubigeo_col)
        .agg(
            n_centros=("dist_nearest_km", "count"),
            dist_media_km=("dist_nearest_km", "mean"),
            dist_max_km=("dist_nearest_km", "max"),
            pct_centros_lejanos=("dist_nearest_km", lambda x: (x > 10).mean() * 100),
        )
        .reset_index()
        .rename(columns={centros_ubigeo_col: "ubigeo"})
    )

    # Construir GDF distrital base
    dist_gdf = distritos_clean.copy()
    if "ubigeo" not in dist_gdf.columns:
        raise ValueError("El GDF de distritos no tiene columna 'ubigeo'.")

    dist_gdf = dist_gdf.merge(ipress_agg, on="ubigeo", how="left")
    if emerg_agg is not None:
        dist_gdf = dist_gdf.merge(emerg_agg, on="ubigeo", how="left")
    else:
        dist_gdf["total_emergencias"] = np.nan
    dist_gdf = dist_gdf.merge(centros_agg, on="ubigeo", how="left")

    # Rellenar NaN con 0 para conteos
    dist_gdf["n_ipress"] = dist_gdf["n_ipress"].fillna(0).astype(int)
    dist_gdf["total_emergencias"] = dist_gdf["total_emergencias"].fillna(0)
    dist_gdf["n_centros"] = dist_gdf["n_centros"].fillna(0).astype(int)

    # Distritos sin centros poblados: distancia indefinida → se deja NaN
    log.info(
        f"GDF distrital construido: {len(dist_gdf)} distritos, "
        f"{dist_gdf['n_ipress'].sum()} IPRESS asignados, "
        f"{dist_gdf['n_centros'].sum()} centros poblados asignados"
    )

    # Guardar
    dist_gdf.to_file(PROCESSED_DIR / "district_summary.gpkg", driver="GPKG")
    dist_gdf.drop(columns="geometry").to_csv(
        PROCESSED_DIR / "district_summary.csv", index=False
    )
    log.info("GDF distrital guardado en data/processed/")

    return dist_gdf


def _resolve_ubigeo_col(gdf: gpd.GeoDataFrame) -> str:
    """Detecta la columna de ubigeo en el GDF resultante del join."""
    for candidate in ["ubigeo", "ubigeo_right", "ubigeo_left"]:
        if candidate in gdf.columns:
            return candidate
    raise ValueError(f"No se encontró columna ubigeo en {list(gdf.columns)}")
