"""
Funciones de carga de los cuatro datasets crudos requeridos.
Cada función devuelve un DataFrame o GeoDataFrame sin transformar.
"""
import pandas as pd
import geopandas as gpd
from pathlib import Path
from src.utils import RAW_DIR, CRS_GEO, setup_logging

log = setup_logging("data_loader")


def _read_csv_flexible(filepath: Path, **kwargs) -> pd.DataFrame:
    """Intenta leer un CSV con UTF-8 y luego con latin-1 si falla."""
    for enc in ("utf-8", "latin-1", "utf-8-sig"):
        try:
            if "sep" not in kwargs:
                sample = open(filepath, "r", encoding=enc).read(500)
                kwargs["sep"] = ";" if sample.count(";") > sample.count(",") else ","
            df = pd.read_csv(filepath, encoding=enc, low_memory=False, **kwargs)
            log.info(f"Cargado {filepath.name} con encoding={enc} ({len(df)} filas)")
            return df
        except UnicodeDecodeError:
            continue
    raise ValueError(f"No se pudo leer {filepath.name} con ningún encoding soportado.")


def load_ipress(filename: str = "ipress.csv") -> pd.DataFrame:
    """
    Carga el dataset de establecimientos de salud IPRESS de MINSA.
    Archivo esperado: data/Raw/ipress.csv
    Fuente: https://datos.gob.pe/dataset/ipress
    """
    path = RAW_DIR / filename
    if not path.exists():
        raise FileNotFoundError(
            f"No se encontró '{path}'.\n"
            "Descarga el dataset IPRESS desde MINSA/datos.gob.pe y "
            f"guárdalo como data/Raw/{filename}"
        )
    ext = path.suffix.lower()
    if ext in (".xlsx", ".xls"):
        df = pd.read_excel(path)
        log.info(f"Cargado {path.name} desde Excel ({len(df)} filas)")
    else:
        df = _read_csv_flexible(path, sep=',')
    return df


def load_emergencias(filename: str = "emergencias.csv") -> pd.DataFrame:
    """
    Carga el dataset de Producción Asistencial en Emergencia por IPRESS.
    Archivo esperado: data/Raw/emergencias.csv
    Fuente: datos.gob.pe – Producción Asistencial en Emergencia por IPRESS
    """
    path = RAW_DIR / filename
    if not path.exists():
        raise FileNotFoundError(
            f"No se encontró '{path}'.\n"
            "Descarga el dataset de emergencias desde datos.gob.pe y "
            f"guárdalo como data/Raw/{filename}"
        )
    ext = path.suffix.lower()
    if ext in (".xlsx", ".xls"):
        df = pd.read_excel(path)
        log.info(f"Cargado {path.name} desde Excel ({len(df)} filas)")
    else:
        df = _read_csv_flexible(path)
    return df


def load_centros_poblados(filename: str = "centros_poblados.shp") -> pd.DataFrame:
    """
    Carga el dataset de Centros Poblados del IGN (shapefile).
    Archivo esperado: data/Raw/centros_poblados.shp
    """
    path = RAW_DIR / filename
    if not path.exists():
        raise FileNotFoundError(
            f"No se encontró '{path}'.\n"
            "Descarga el dataset de Centros Poblados y "
            f"guárdalo como data/Raw/{filename}"
        )
    import geopandas as gpd
    gdf = gpd.read_file(path)
    df = pd.DataFrame(gdf.drop(columns="geometry"))
    df = df.rename(columns={
        "X": "longitud",
        "Y": "latitud",
        "NOM_POBLAD": "nombre_ccpp",
        "DEP": "departamento",
        "PROV": "provincia",
        "DIST": "distrito",
        "CÓD_INT": "ubigeo",
    })
    log.info(f"Cargado {path.name}: {len(df)} centros poblados")
    return df


def load_distritos(filename: str = "DISTRITOS.shp") -> gpd.GeoDataFrame:
    """
    Carga el shapefile de límites distritales de Perú.
    Archivo esperado: data/Raw/DISTRITOS.shp (+ .dbf, .shx, .prj)
    Fuente: MINSA GIS / IGN
    """
    path = RAW_DIR / filename
    if not path.exists():
        raise FileNotFoundError(
            f"No se encontró '{path}'.\n"
            "Descarga el shapefile DISTRITOS.shp y coloca todos sus "
            f"archivos (.shp, .dbf, .shx, .prj) en data/Raw/"
        )
    gdf = gpd.read_file(path)
    if gdf.crs is None:
        gdf = gdf.set_crs(CRS_GEO)
        log.warning("El shapefile no tenía CRS definido. Se asignó EPSG:4326.")
    elif gdf.crs.to_epsg() != 4326:
        gdf = gdf.to_crs(CRS_GEO)
        log.info(f"CRS reproyectado a {CRS_GEO}")
    log.info(f"Cargado {path.name}: {len(gdf)} distritos, CRS={gdf.crs.to_epsg()}")
    return gdf
