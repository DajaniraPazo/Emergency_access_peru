"""
Script principal que ejecuta todo el pipeline de análisis.
Uso: python run_pipeline.py

Requiere que los datos crudos estén en data/Raw/:
  - ipress.csv
  - emergencias.csv
  - centros_poblados.csv
  - DISTRITOS.shp (.dbf, .shx, .prj)
"""
import sys
from pathlib import Path
sys.path.insert(0, str(Path(__file__).parent))

from src.utils import setup_logging
from src.data_loader import load_ipress, load_emergencias, load_centros_poblados, load_distritos
from src.cleaning import run_cleaning_pipeline, generate_data_dictionary
from src.geospatial import build_district_geodataframe, make_geodataframe
from src.metrics import compute_full_iase, compare_specifications, classify_districts, save_district_table
from src.visualization import run_visualization_pipeline
from src.utils import TABLES_DIR

log = setup_logging("pipeline")


def main():
    log.info("=" * 60)
    log.info("PIPELINE: Acceso a Emergencias en Perú")
    log.info("=" * 60)

    # -----------------------------------------------------------------------
    # TAREA 1 – Carga de datos crudos
    # -----------------------------------------------------------------------
    log.info("PASO 1: Cargando datasets crudos...")
    df_ipress = load_ipress()
    df_emergencias = load_emergencias()
    df_centros = load_centros_poblados()
    gdf_distritos = load_distritos()

    # -----------------------------------------------------------------------
    # TAREA 1 – Limpieza y preprocesamiento
    # -----------------------------------------------------------------------
    log.info("PASO 2: Limpieza y preprocesamiento...")
    ipress_c, emerg_c, centros_c, dist_c = run_cleaning_pipeline(
        df_ipress, df_emergencias, df_centros, gdf_distritos
    )

    # Guardar diccionario de datos
    data_dict = generate_data_dictionary()
    data_dict.to_csv(TABLES_DIR / "data_dictionary.csv", index=False)
    log.info("Diccionario de datos guardado.")

    # -----------------------------------------------------------------------
    # TAREA 2 – Pipeline geoespacial
    # -----------------------------------------------------------------------
    log.info("PASO 3: Construyendo GeoDataFrame distrital...")
    district_gdf = build_district_geodataframe(ipress_c, emerg_c, centros_c, dist_c)

    # -----------------------------------------------------------------------
    # TAREA 3 – Cálculo del IASE (baseline + alternativa)
    # -----------------------------------------------------------------------
    log.info("PASO 4: Calculando métricas distritales (IASE)...")
    district_gdf = compute_full_iase(district_gdf)
    district_gdf = classify_districts(district_gdf, "IASE_base")
    district_gdf = classify_districts(district_gdf, "IASE_alt")

    comparison_df = compare_specifications(district_gdf)
    save_district_table(district_gdf)

    # -----------------------------------------------------------------------
    # TAREA 4 & 5 – Visualizaciones estáticas y mapas
    # -----------------------------------------------------------------------
    log.info("PASO 5: Generando visualizaciones...")
    ipress_gdf = make_geodataframe(ipress_c)
    run_visualization_pipeline(district_gdf, comparison_df, ipress_gdf)

    log.info("=" * 60)
    log.info("PIPELINE COMPLETO. Resultados en:")
    log.info("  data/processed/   – datos limpios")
    log.info("  output/figures/   – gráficos y mapas")
    log.info("  output/tables/    – tablas CSV")
    log.info("Para ver la app: streamlit run app.py")
    log.info("=" * 60)


if __name__ == "__main__":
    main()
