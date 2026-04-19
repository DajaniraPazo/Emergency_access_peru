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
        st.image(str(path))
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
    st.caption("Datos: centros de salud IPRESS (MINSA) · Emergencias · Comunidades INEI · Límites distritales")

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
            los **1,874 distritos del Perú**. Factores como la cantidad de centros de salud
            (IPRESS), el volumen de atenciones registradas, y la distancia física desde las
            comunidades (centros poblados) hacia los establecimientos determinan qué tan bien
            atendida está la población ante una emergencia médica.

            **Pregunta central:** ¿Qué distritos tienen mejores o peores condiciones de
            acceso a servicios de emergencia, y qué evidencia lo respalda?
            """
        )

        st.subheader("Fuentes de datos")
        st.markdown(
            """
            | Conjunto de datos | Fuente | Variables principales |
            |-------------------|--------|-----------------------|
            | IPRESS (centros de salud) | MINSA / datos.gob.pe | Nombre, ubicación GPS, categoría del establecimiento |
            | Producción de emergencias | datos.gob.pe | N° de atenciones de emergencia por centro de salud |
            | Centros Poblados (comunidades) | INEI | Nombre y coordenadas de cada asentamiento |
            | DISTRITOS.shp (límites distritales) | MINSA GIS / IGN | Polígonos de los 1,874 distritos del Perú |
            """
        )

    with col2:
        st.subheader("Decisiones de limpieza")
        st.markdown(
            """
            - **IPRESS (centros de salud):** se descartaron registros sin coordenadas GPS válidas
              dentro del territorio peruano. Se mantuvieron solo los establecimientos activos.
            - **Emergencias:** se sumaron todas las atenciones históricas por cada centro de salud.
            - **Centros Poblados (comunidades):** se eliminaron los que no tienen coordenadas
              o están fuera del territorio peruano.
            - **Distritos:** se corrigieron automáticamente los límites geográficos con problemas
              de geometría.
            - **Sistema de coordenadas:** se usa **WGS84 (EPSG:4326)** para mostrar los mapas.
              Para calcular distancias en kilómetros se usa **UTM Zona 18S (EPSG:32718)**,
              que reduce la distorsión de medidas en el territorio peruano.
            """
        )

        st.subheader("Construcción del IASE")
        st.markdown(
            """
            El **IASE (Índice de Acceso a Servicios de Emergencia)** combina tres componentes,
            cada uno normalizado entre 0 y 1 (donde 1 = mejor situación):

            | Componente | Qué mide | Interpretación |
            |------------|----------|----------------|
            | **Disponibilidad** | N° de centros de salud (IPRESS) en el distrito | Más centros = mejor |
            | **Actividad** | Total de emergencias atendidas | Más atenciones = mejor |
            | **Acceso espacial** | Distancia promedio inversa al centro de salud más cercano | Más cerca = mejor |

            **Versión base:** los tres componentes tienen el mismo peso (1/3 cada uno).

            **Versión alternativa:** se le da más peso a la distancia (distancia: 50%,
            disponibilidad: 25%, actividad: 25%). Justificación: en emergencias médicas,
            el tiempo de traslado es el factor más crítico para salvar una vida.
            """
        )

    st.markdown("---")
    st.subheader("Limitaciones")
    st.markdown(
        """
        - La distancia calculada es **en línea recta**, no tiempo de traslado real por carretera
          o considerando accidentes geográficos como ríos o montañas.
        - Los datos de emergencias pueden estar **incompletos** en zonas con menor acceso al
          sistema de registro del MINSA/SUSALUD.
        - No se incluye información de **población** por distrito (por falta de acceso al censo),
          por lo que el índice mide volumen absoluto, no acceso per cápita.
        - Algunas comunidades muy pequeñas (centros poblados) pueden no estar registradas en el
          dataset del INEI.
        - El nivel de complejidad del centro de salud (categorías I-1 a III-E) no está ponderado
          en el índice actual: un puesto de salud rural y un hospital cuentan igual.
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

        st.subheader("Q1 – ¿Cuántos centros de salud (IPRESS) tiene cada distrito?")
        st.markdown(
            "Distribución del número de centros de salud (IPRESS) por distrito. "
            "La distribución es muy desigual: unos pocos distritos concentran muchos "
            "establecimientos mientras la gran mayoría tiene muy pocos o ninguno."
        )
        _show_image("01_facility_distribution.png")

        st.markdown("---")
        st.subheader("Q1 – ¿Qué tan activos son los centros de salud en atención de emergencias?")
        st.markdown(
            "Top 20 distritos por total de atenciones de emergencia registradas, y relación "
            "entre cantidad de centros de salud y atenciones. La relación imperfecta entre "
            "ambas variables indica que algunos distritos tienen muchos centros de salud "
            "pero pocas atenciones registradas (posible subregistro o baja utilización), "
            "y viceversa."
        )
        _show_image("02_emergency_activity.png")

        st.markdown("---")
        st.subheader("Q2 – ¿Qué tan cerca están las comunidades de un centro de salud?")
        st.markdown(
            "Distribución de la distancia promedio (km) al centro de salud (IPRESS) más cercano "
            "y porcentaje de comunidades (centros poblados) a más de 10 km. "
            "Revela que una gran parte de las comunidades en distritos rurales y remotos "
            "está muy lejos de cualquier establecimiento de salud."
        )
        _show_image("03_distance_distribution.png")

        st.markdown("---")
        st.subheader("Q3 – ¿Los tres componentes del índice (IASE) miden cosas distintas?")
        st.markdown(
            "Relación entre disponibilidad (N° de centros de salud), actividad (emergencias "
            "atendidas) y acceso espacial (distancia al centro de salud más cercano). "
            "Si los componentes tienen poca correlación entre sí, significa que cada uno "
            "aporta información nueva y vale la pena incluir los tres en el índice."
        )
        _show_image("04_component_correlation.png")

        st.markdown("---")
        st.subheader("Q3 – Ranking de distritos: ¿quiénes tienen mejor y peor acceso?")
        st.markdown(
            "Distritos con mayor y menor acceso a servicios de emergencia según el IASE "
            "(índice de acceso a emergencias) en su versión base. "
            "Los distritos mejor posicionados suelen ser zonas urbanas o capitales de provincia."
        )
        _show_image("05_top_bottom_districts.png")

        st.markdown("---")
        st.subheader("Q4 – ¿Cambian los resultados al ajustar los pesos del índice?")
        st.markdown(
            "Comparación entre la versión base y la versión alternativa del IASE "
            "(índice de acceso a emergencias). "
            "El diagrama de puntos muestra que ambas versiones coinciden en términos generales, "
            "pero el histograma de cambios de posición revela que algunos distritos suben o "
            "bajan mucho al darle más peso a la distancia, especialmente aquellos con muchos "
            "centros de salud (IPRESS) pero mal distribuidos en su territorio."
        )
        _show_image("06_baseline_vs_alternative.png")

        # Tabla resumen
        if comparison_df is not None:
            st.markdown("---")
            st.subheader("Tabla: distritos que cambiaron de categoría de acceso al ajustar la fórmula")
            if "clase_change" in comparison_df.columns:
                changed = comparison_df[comparison_df["clase_change"] == True]
                n_changed = len(changed)
                pct = n_changed / len(comparison_df) * 100
                st.metric("Distritos que cambiaron de categoría de acceso", f"{n_changed} ({pct:.1f}%)")
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
                st.metric("Distritos sin ningún centro de salud (IPRESS)", str(n_sin_ipress))
            with col2:
                if "dist_media_km" in dist_table.columns:
                    med_dist = dist_table["dist_media_km"].median()
                    st.metric("Distancia promedio al centro de salud más cercano (km)", f"{med_dist:.1f}")
            with col3:
                if "IASE_base" in dist_table.columns:
                    st.metric("Puntuación promedio del índice de acceso (IASE)", f"{dist_table['IASE_base'].mean():.3f}")

        st.markdown("---")
        st.subheader("Mapa 1 – N° de centros de salud (IPRESS) por distrito")
        st.markdown("Mayor intensidad de color = más centros de salud en el distrito.")
        _show_image("map_01_facility_density.png")

        st.markdown("---")
        st.subheader("Mapa 2 – Distancia promedio al centro de salud más cercano")
        st.markdown(
            "Distritos en rojo tienen comunidades (centros poblados) muy alejadas de cualquier "
            "centro de salud (IPRESS)."
        )
        _show_image("map_02_distance_access.png")

        st.markdown("---")
        st.subheader("Mapa 3 – Índice de acceso a emergencias (IASE) – versión base")
        _show_image("map_03_IASE_base.png")

        st.markdown("---")
        st.subheader("Mapa 4 – Comparación entre versión base y versión alternativa del índice")
        _show_image("map_04_comparison.png")

        # Tabla distrital filtrable
        st.markdown("---")
        st.subheader("Tabla completa de indicadores por distrito")
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
            [
                "IASE base (pesos iguales)",
                "IASE alternativa (mayor peso en distancia)",
                "Centros de salud (IPRESS)",
            ],
            horizontal=True,
        )

        from src.visualization import folium_iase_map, folium_facilities_map

        if map_option == "IASE base (pesos iguales)":
            st.markdown(
                "**Mapa interactivo del IASE base** (índice con pesos iguales entre los tres componentes). "
                "Haz clic sobre un distrito para ver sus indicadores. "
                "Verde = mejor acceso, Rojo = menor acceso."
            )
            if "IASE_base" in dist_gdf.columns:
                m = folium_iase_map(dist_gdf, "IASE_base")
                st_folium(m, width=1100, height=650)
            else:
                st.warning("Índice IASE base no calculado. Ejecuta el pipeline primero.")

        elif map_option == "IASE alternativa (mayor peso en distancia)":
            st.markdown(
                "**Mapa interactivo del IASE alternativa** (mayor peso en la distancia al centro de salud). "
                "Compara con la versión base para ver qué distritos cambian de posición."
            )
            if "IASE_alt" in dist_gdf.columns:
                m = folium_iase_map(dist_gdf, "IASE_alt")
                st_folium(m, width=1100, height=650)
            else:
                st.warning("Índice IASE alternativa no calculado. Ejecuta el pipeline primero.")

        elif map_option == "Centros de salud (IPRESS)":
            st.markdown(
                "**Mapa de centros de salud (IPRESS)** sobre el fondo del índice de acceso base. "
                "Los marcadores están agrupados; haz zoom para verlos individualmente."
            )
            if ipress_gdf is not None:
                m = folium_facilities_map(ipress_gdf, dist_gdf)
                st_folium(m, width=1100, height=650)
            else:
                st.warning("Datos de centros de salud no encontrados en data/processed/ipress_clean.csv")

        # Comparación interactiva baseline vs alternativa
        if comparison_df is not None and "IASE_base" in comparison_df.columns:
            st.markdown("---")
            st.subheader("Q4 – Comparación por departamento: versión base vs versión alternativa del índice")
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
                       label="Versión base", color="#2171b5", alpha=0.85)
                if "IASE_alt" in df_plot.columns:
                    ax.bar([i + 0.2 for i in x], df_plot["IASE_alt"], 0.4,
                           label="Versión alternativa", color="#e6550d", alpha=0.85)
                ax.set_xticks(list(x))
                ax.set_xticklabels(df_plot[id_col].astype(str), rotation=45, ha="right", fontsize=8)
                ax.set_ylabel("Puntuación del índice de acceso (IASE)")
                ax.set_title(f"Índice de acceso (IASE): versión base vs alternativa – {dept_sel}")
                ax.legend()
                plt.tight_layout()
                st.pyplot(fig)
                plt.close()
            else:
                st.info("Sin columna 'departamento' en los datos de comparación.")
