"""
Streamlit App – Emergency Healthcare Access Inequality in Peru
Ejecutar: streamlit run app.py
"""
import warnings
warnings.filterwarnings("ignore")

import os
import sys
from pathlib import Path

# Asegurar que src/ esté en el path
sys.path.insert(0, str(Path(__file__).parent))

import streamlit as st
import pandas as pd
import geopandas as gpd
import matplotlib.pyplot as plt
import folium
from streamlit_folium import st_folium

from src.utils import PROCESSED_DIR, FIGURES_DIR, TABLES_DIR, QUINTILE_COLORS, QUINTILE_ORDER

# ---------------------------------------------------------------------------
# Configuración de página
# ---------------------------------------------------------------------------
st.set_page_config(
    page_title="Acceso a Emergencias en Perú",
    page_icon="🏥",
    layout="wide",
    initial_sidebar_state="expanded",
)

# ---------------------------------------------------------------------------
# Carga de datos procesados (cacheada)
# ---------------------------------------------------------------------------

@st.cache_data(show_spinner="Cargando datos procesados...")
def load_district_gdf():
    path = PROCESSED_DIR / "district_summary.gpkg"
    if not path.exists():
        return None
    return gpd.read_file(path)

@st.cache_data(show_spinner="Cargando IPRESS...")
def load_ipress_gdf():
    path = PROCESSED_DIR / "ipress_clean.csv"
    if not path.exists():
        return None
    df = pd.read_csv(path, dtype={"ubigeo": str})
    import geopandas as gpd
    gdf = gpd.GeoDataFrame(
        df,
        geometry=gpd.points_from_xy(df["longitud"], df["latitud"]),
        crs="EPSG:4326",
    )
    return gdf

@st.cache_data(show_spinner="Cargando tabla distrital...")
def load_district_table():
    path = TABLES_DIR / "district_iase_table.csv"
    if not path.exists():
        path2 = PROCESSED_DIR / "district_summary.csv"
        if not path2.exists():
            return None
        return pd.read_csv(path2, dtype={"ubigeo": str})
    return pd.read_csv(path, dtype={"ubigeo": str})

@st.cache_data(show_spinner="Cargando comparación de especificaciones...")
def load_comparison():
    path = TABLES_DIR / "comparison_baseline_vs_alternative.csv"
    if not path.exists():
        return None
    return pd.read_csv(path, dtype={"ubigeo": str})

# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

DATA_READY = False

def _check_data():
    dist_gdf = load_district_gdf()
    return dist_gdf is not None and len(dist_gdf) > 0

def _show_image(filename: str):
    path = FIGURES_DIR / filename
    if path.exists():
        st.image(str(path), use_column_width=True)
    else:
        st.warning(f"Imagen no generada aún: {filename}")

def _show_no_data():
    st.error(
        "**Los datos procesados no están disponibles.**\n\n"
        "Ejecuta primero el pipeline completo:\n"
        "```bash\n"
        "python run_pipeline.py\n"
        "```\n\n"
        "Asegúrate de que los archivos de datos crudos estén en `data/Raw/`."
    )

# ---------------------------------------------------------------------------
# Sidebar
# ---------------------------------------------------------------------------

with st.sidebar:
    st.title("🏥 Acceso a Emergencias – Perú")
    st.markdown("---")
    st.markdown(
        "**Proyecto de análisis geoespacial** del acceso a servicios de emergencia "
        "a nivel distrital en el Perú."
    )
    st.markdown("---")
    st.caption("Curso: Python / Data Science")
    st.caption("Datasets: MINSA IPRESS · Emergencias · INEI Centros Poblados · DISTRITOS.shp")

    data_ok = _check_data()
    if data_ok:
        st.success("✅ Datos procesados disponibles")
    else:
        st.warning("⚠️ Datos no procesados aún. Ejecuta `python run_pipeline.py`.")

# ---------------------------------------------------------------------------
# Tabs
# ---------------------------------------------------------------------------

tab1, tab2, tab3, tab4 = st.tabs([
    "📋 Datos & Metodología",
    "📊 Análisis Estático",
    "🗺️ Resultados Geoespaciales",
    "🔍 Exploración Interactiva",
])

# ===========================================================================
# TAB 1 – Datos & Metodología
# ===========================================================================
with tab1:
    st.header("Datos y Metodología")

    col1, col2 = st.columns([1, 1])

    with col1:
        st.subheader("Planteamiento del problema")
        st.markdown(
            """
            El acceso a servicios de emergencia en salud varía significativamente entre
            los **1,874 distritos del Perú**. Factores como la densidad de establecimientos,
            la actividad asistencial registrada, y la distancia física desde los centros
            poblados hacia los establecimientos determinan qué tan bien atendida está la
            población ante una emergencia médica.

            **Pregunta central:** ¿Qué distritos aparecen en condiciones relativamente
            mejores o peores de acceso a servicios de emergencia, y qué evidencia lo sustenta?
            """
        )

        st.subheader("Fuentes de datos")
        st.markdown(
            """
            | Dataset | Fuente | Variables clave |
            |---------|--------|-----------------|
            | IPRESS | MINSA / datos.gob.pe | Establecimientos, coordenadas, categoría |
            | Producción de emergencias | datos.gob.pe | Atenciones de emergencia por IPRESS |
            | Centros Poblados | INEI | Coordenadas de asentamientos |
            | DISTRITOS.shp | MINSA GIS / IGN | Límites distritales |
            """
        )

    with col2:
        st.subheader("Decisiones de limpieza")
        st.markdown(
            """
            - **IPRESS:** se eliminaron registros sin coordenadas válidas dentro del bounding box
              de Perú (lat: -18.4 a 0, lon: -81.5 a -68.7). Se filtraron solo establecimientos activos.
            - **Emergencias:** se agruparon por establecimiento (suma histórica total).
            - **Centros Poblados:** se descartaron los sin coordenadas o fuera del territorio peruano.
            - **Distritos:** se corrigieron geometrías inválidas con `buffer(0)`.
            - **CRS:** todos los datos se cargan en **EPSG:4326 (WGS84)** para visualización.
              Para el cálculo de distancias se reprojecta a **EPSG:32718 (UTM Zona 18S)**,
              que minimiza la distorsión métrica en el territorio peruano.
            """
        )

        st.subheader("Construcción del IASE")
        st.markdown(
            """
            El **Índice de Acceso a Servicios de Emergencia (IASE)** combina tres componentes
            normalizados a [0, 1]:

            | Componente | Variable base | Dirección |
            |------------|---------------|-----------|
            | **Disponibilidad** | N° de IPRESS en el distrito | Mayor = mejor |
            | **Actividad** | Total emergencias atendidas | Mayor = mejor |
            | **Acceso espacial** | 1 – dist_media_km normalizada | Mayor = mejor |

            **Especificación baseline:** pesos iguales (1/3 cada componente).

            **Especificación alternativa:** mayor peso en acceso espacial (facility=0.25,
            activity=0.25, access=0.50). Justificación: en emergencias médicas el tiempo
            de traslado es el factor más crítico para la sobrevivencia del paciente.
            """
        )

    st.markdown("---")
    st.subheader("Limitaciones")
    st.markdown(
        """
        - La distancia calculada es **euclidiana (línea recta)** proyectada, no tiempo de traslado real
          por carretera o accidentes geográficos.
        - Los datos de emergencias pueden tener **subregistro** en zonas con menor conectividad al sistema SUSALUD.
        - No se incorpora información de **población** por distrito (por falta de acceso directo al censo),
          por lo que los índices son absolutos, no per cápita.
        - Centros poblados muy pequeños pueden no estar registrados en el dataset del INEI.
        - La categoría del establecimiento (I-1 a III-E) no se pondera en el índice actual.
        """
    )

    # Diccionario de datos
    st.markdown("---")
    st.subheader("Diccionario de variables")
    from src.cleaning import generate_data_dictionary
    try:
        dict_df = generate_data_dictionary()
        st.dataframe(dict_df, use_container_width=True, hide_index=True)
    except Exception as e:
        st.info(f"Diccionario no disponible: {e}")


# ===========================================================================
# TAB 2 – Análisis Estático
# ===========================================================================
with tab2:
    st.header("Análisis Estático")

    if not _check_data():
        _show_no_data()
    else:
        dist_gdf = load_district_gdf()
        comparison_df = load_comparison()

        st.subheader("Q1 – Disponibilidad territorial de IPRESS")
        st.markdown(
            "Distribución del número de establecimientos IPRESS por distrito. "
            "La distribución es fuertemente asimétrica: pocos distritos concentran muchos "
            "establecimientos mientras la mayoría tiene muy pocos o ninguno."
        )
        _show_image("01_facility_distribution.png")

        st.markdown("---")
        st.subheader("Q1 – Actividad de emergencias")
        st.markdown(
            "Top 20 distritos por atenciones de emergencia y relación entre disponibilidad "
            "y actividad. La correlación imperfecta indica que algunos distritos tienen "
            "muchos establecimientos pero poca actividad registrada (posible subregistro o "
            "infrautilización), y viceversa."
        )
        _show_image("02_emergency_activity.png")

        st.markdown("---")
        st.subheader("Q2 – Acceso espacial de los centros poblados")
        st.markdown(
            "Distribución de la distancia media (km) al IPRESS más cercano y porcentaje "
            "de centros poblados a más de 10 km. Revela que gran parte de los centros "
            "poblados en distritos rurales y remotos están muy lejos de cualquier establecimiento."
        )
        _show_image("03_distance_distribution.png")

        st.markdown("---")
        st.subheader("Q3 – Coherencia de los componentes del IASE")
        st.markdown(
            "Correlación entre disponibilidad, actividad y acceso espacial. "
            "Baja correlación entre componentes justifica incluirlos todos: "
            "miden aspectos distintos del acceso."
        )
        _show_image("04_component_correlation.png")

        st.markdown("---")
        st.subheader("Q3 – Ranking distrital (IASE baseline)")
        st.markdown(
            "Distritos con mayor y menor acceso a servicios de emergencia según el IASE baseline. "
            "Los distritos con mejor acceso típicamente están en zonas urbanas o capitales de provincia."
        )
        _show_image("05_top_bottom_districts.png")

        st.markdown("---")
        st.subheader("Q4 – Sensibilidad metodológica")
        st.markdown(
            "Comparación entre baseline y especificación alternativa. "
            "El scatter muestra concordancia general, pero el histograma de cambio de rango "
            "indica que algunos distritos se ven significativamente afectados al darle más peso "
            "al acceso espacial, especialmente distritos con muchos IPRESS pero mal distribuidos."
        )
        _show_image("06_baseline_vs_alternative.png")

        # Tabla resumen
        if comparison_df is not None:
            st.markdown("---")
            st.subheader("Tabla: cambios de clase entre especificaciones")
            if "clase_change" in comparison_df.columns:
                changed = comparison_df[comparison_df["clase_change"] == True]
                n_changed = len(changed)
                pct = n_changed / len(comparison_df) * 100
                st.metric("Distritos que cambiaron de clase", f"{n_changed} ({pct:.1f}%)")
                st.dataframe(
                    changed[
                        [c for c in ["ubigeo", "distrito", "departamento",
                                     "IASE_base", "IASE_alt", "clase_base", "clase_alt",
                                     "rank_change"] if c in changed.columns]
                    ].head(30),
                    use_container_width=True, hide_index=True,
                )


# ===========================================================================
# TAB 3 – Resultados Geoespaciales
# ===========================================================================
with tab3:
    st.header("Resultados Geoespaciales")

    if not _check_data():
        _show_no_data()
    else:
        dist_gdf = load_district_gdf()
        dist_table = load_district_table()

        col1, col2, col3 = st.columns(3)
        if dist_table is not None:
            with col1:
                n_sin_ipress = (dist_table["n_ipress"] == 0).sum() if "n_ipress" in dist_table.columns else "N/D"
                st.metric("Distritos sin IPRESS", str(n_sin_ipress))
            with col2:
                if "dist_media_km" in dist_table.columns:
                    med_dist = dist_table["dist_media_km"].median()
                    st.metric("Distancia mediana al IPRESS (km)", f"{med_dist:.1f}")
            with col3:
                if "IASE_base" in dist_table.columns:
                    st.metric("Score IASE promedio", f"{dist_table['IASE_base'].mean():.3f}")

        st.markdown("---")
        st.subheader("Mapa 1 – Densidad de IPRESS por distrito")
        st.markdown("Mayor intensidad de color = más establecimientos en el distrito.")
        _show_image("map_01_facility_density.png")

        st.markdown("---")
        st.subheader("Mapa 2 – Distancia media al IPRESS más cercano")
        st.markdown("Distritos en rojo tienen centros poblados muy alejados de cualquier establecimiento.")
        _show_image("map_02_distance_access.png")

        st.markdown("---")
        st.subheader("Mapa 3 – IASE Baseline (pesos iguales)")
        _show_image("map_03_IASE_base.png")

        st.markdown("---")
        st.subheader("Mapa 4 – Comparación baseline vs especificación alternativa")
        _show_image("map_04_comparison.png")

        # Tabla distrital filtrable
        st.markdown("---")
        st.subheader("Tabla distrital completa")
        if dist_table is not None:
            dept_list = ["Todos"] + sorted(dist_table["departamento"].dropna().unique().tolist()) \
                if "departamento" in dist_table.columns else ["Todos"]
            dept_filter = st.selectbox("Filtrar por departamento", dept_list)

            show_df = dist_table.copy()
            if dept_filter != "Todos" and "departamento" in show_df.columns:
                show_df = show_df[show_df["departamento"] == dept_filter]

            display_cols = [c for c in [
                "ubigeo", "departamento", "provincia", "distrito",
                "n_ipress", "total_emergencias", "n_centros",
                "dist_media_km", "IASE_base", "clase_base", "IASE_alt", "clase_alt",
            ] if c in show_df.columns]

            st.dataframe(
                show_df[display_cols].sort_values("IASE_base", ascending=False)
                if "IASE_base" in show_df.columns else show_df[display_cols],
                use_container_width=True, hide_index=True,
            )
            csv = show_df[display_cols].to_csv(index=False).encode("utf-8")
            st.download_button(
                "⬇️ Descargar tabla (CSV)", csv,
                file_name="district_iase_filtered.csv", mime="text/csv",
            )


# ===========================================================================
# TAB 4 – Exploración Interactiva
# ===========================================================================
with tab4:
    st.header("Exploración Interactiva")

    if not _check_data():
        _show_no_data()
    else:
        dist_gdf = load_district_gdf()
        ipress_gdf = load_ipress_gdf()
        comparison_df = load_comparison()

        map_option = st.radio(
            "Selecciona el mapa a visualizar:",
            ["IASE Baseline", "IASE Alternativa", "Establecimientos IPRESS"],
            horizontal=True,
        )

        from src.visualization import folium_iase_map, folium_facilities_map

        if map_option == "IASE Baseline":
            st.markdown(
                "**Mapa interactivo del IASE Baseline.** "
                "Haz clic sobre un distrito para ver sus indicadores. "
                "Verde = mejor acceso, Rojo = peor acceso."
            )
            if "IASE_base" in dist_gdf.columns:
                m = folium_iase_map(dist_gdf, "IASE_base")
                st_folium(m, width=1100, height=650)
            else:
                st.warning("IASE baseline no calculado. Ejecuta el pipeline primero.")

        elif map_option == "IASE Alternativa":
            st.markdown(
                "**Mapa interactivo del IASE Alternativa** (mayor peso en acceso espacial). "
                "Compara con el baseline para ver qué distritos cambian de posición."
            )
            if "IASE_alt" in dist_gdf.columns:
                m = folium_iase_map(dist_gdf, "IASE_alt")
                st_folium(m, width=1100, height=650)
            else:
                st.warning("IASE alternativa no calculada. Ejecuta el pipeline primero.")

        elif map_option == "Establecimientos IPRESS":
            st.markdown(
                "**Mapa de establecimientos IPRESS** sobre fondo de IASE baseline. "
                "Los marcadores están agrupados; haz zoom para verlos individualmente."
            )
            if ipress_gdf is not None:
                m = folium_facilities_map(ipress_gdf, dist_gdf)
                st_folium(m, width=1100, height=650)
            else:
                st.warning("Datos IPRESS no encontrados en data/processed/ipress_clean.csv")

        # Comparación interactiva baseline vs alternativa
        if comparison_df is not None and "IASE_base" in comparison_df.columns:
            st.markdown("---")
            st.subheader("Q4 – Comparación por departamento: baseline vs alternativa")
            if "departamento" in comparison_df.columns:
                dept_sel = st.selectbox(
                    "Departamento", sorted(comparison_df["departamento"].dropna().unique()),
                    key="dept_comp"
                )
                df_dept = comparison_df[comparison_df["departamento"] == dept_sel].copy()
                id_col = next((c for c in ["distrito", "ubigeo"] if c in df_dept.columns), "ubigeo")
                df_plot = df_dept.nlargest(20, "IASE_base")

                fig, ax = plt.subplots(figsize=(10, 6))
                x = range(len(df_plot))
                ax.bar([i - 0.2 for i in x], df_plot["IASE_base"], 0.4,
                       label="Baseline", color="#2171b5", alpha=0.85)
                if "IASE_alt" in df_plot.columns:
                    ax.bar([i + 0.2 for i in x], df_plot["IASE_alt"], 0.4,
                           label="Alternativa", color="#e6550d", alpha=0.85)
                ax.set_xticks(list(x))
                ax.set_xticklabels(df_plot[id_col].astype(str), rotation=45, ha="right", fontsize=8)
                ax.set_ylabel("Score IASE")
                ax.set_title(f"IASE Baseline vs Alternativa – {dept_sel}")
                ax.legend()
                plt.tight_layout()
                st.pyplot(fig)
                plt.close()
            else:
                st.info("Sin columna 'departamento' en los datos de comparación.")
