"""
Visualizaciones estáticas (matplotlib/seaborn) y mapas (GeoPandas/Folium).

Criterio de selección de gráficos:
- Cada gráfico responde directamente a una de las 4 preguntas analíticas.
- Se prefieren visualizaciones que muestren distribución + ranking simultáneamente.
- Los mapas coropléticos permiten identificar patrones geográficos que los gráficos no muestran.
- Los mapas interactivos (Folium) permiten explorar casos individuales con tooltips.
"""
import warnings
warnings.filterwarnings("ignore")

import numpy as np
import pandas as pd
import geopandas as gpd
import matplotlib.pyplot as plt
import matplotlib.patches as mpatches
import seaborn as sns
import folium
from folium.plugins import MarkerCluster
import json

from src.utils import (
    FIGURES_DIR, QUINTILE_COLORS, QUINTILE_ORDER, setup_logging
)

log = setup_logging("visualization")
sns.set_theme(style="whitegrid", palette="muted", font_scale=1.1)


# ===========================================================================
# GRÁFICOS ESTÁTICOS – matplotlib / seaborn
# ===========================================================================

def plot_facility_distribution(dist_gdf: gpd.GeoDataFrame, save: bool = True):
    """
    Distribución del número de IPRESS por distrito (histograma + boxplot).
    Responde Q1: disponibilidad territorial de establecimientos.
    Elegido sobre un simple bar chart porque muestra la asimetría de la distribución,
    que es clave para entender la desigualdad.
    """
    fig, axes = plt.subplots(1, 2, figsize=(13, 5))

    data = dist_gdf["n_ipress"].dropna()

    axes[0].hist(data, bins=40, color="#2171b5", edgecolor="white", alpha=0.85)
    axes[0].axvline(data.median(), color="#d62728", linestyle="--", label=f"Mediana: {data.median():.0f}")
    axes[0].set_title("Distribución: N° de IPRESS por distrito")
    axes[0].set_xlabel("N° de establecimientos IPRESS")
    axes[0].set_ylabel("N° de distritos")
    axes[0].legend()

    axes[1].boxplot(data, vert=True, patch_artist=True,
                    boxprops=dict(facecolor="#a6cee3"), medianprops=dict(color="#d62728", linewidth=2))
    axes[1].set_title("Dispersión: N° de IPRESS por distrito")
    axes[1].set_ylabel("N° de establecimientos IPRESS")

    plt.suptitle("Q1 – Disponibilidad territorial de establecimientos IPRESS", fontweight="bold")
    plt.tight_layout()
    if save:
        fig.savefig(FIGURES_DIR / "01_facility_distribution.png", dpi=150, bbox_inches="tight")
        log.info("Guardado: 01_facility_distribution.png")
    return fig


def plot_emergency_activity(dist_gdf: gpd.GeoDataFrame, save: bool = True):
    """
    Top 20 distritos por actividad de emergencias + scatter facilities vs emergencias.
    Responde Q1: actividad asistencial de emergencia.
    El bar chart de top-20 y el scatter juntos revelan si muchos establecimientos
    implican más atenciones (correlación o anomalías).
    """
    fig, axes = plt.subplots(1, 2, figsize=(14, 6))

    id_col = next((c for c in ["distrito", "ubigeo"] if c in dist_gdf.columns), None)
    top20 = dist_gdf.nlargest(20, "total_emergencias")

    # Bar chart top 20
    axes[0].barh(
        top20[id_col].astype(str) if id_col else top20.index.astype(str),
        top20["total_emergencias"],
        color="#2ca02c", alpha=0.85,
    )
    axes[0].invert_yaxis()
    axes[0].set_title("Top 20 distritos – Total emergencias")
    axes[0].set_xlabel("Total atenciones de emergencia")

    # Scatter: n_ipress vs emergencias
    axes[1].scatter(
        dist_gdf["n_ipress"],
        dist_gdf["total_emergencias"],
        alpha=0.4, s=15, color="#1f77b4",
    )
    axes[1].set_xlabel("N° de IPRESS")
    axes[1].set_ylabel("Total emergencias atendidas")
    axes[1].set_title("IPRESS vs Actividad de emergencias")

    plt.suptitle("Q1 – Actividad asistencial de emergencia por distrito", fontweight="bold")
    plt.tight_layout()
    if save:
        fig.savefig(FIGURES_DIR / "02_emergency_activity.png", dpi=150, bbox_inches="tight")
        log.info("Guardado: 02_emergency_activity.png")
    return fig


def plot_distance_distribution(dist_gdf: gpd.GeoDataFrame, save: bool = True):
    """
    Distribución de la distancia media (km) de los centros poblados al IPRESS más cercano.
    Responde Q2: acceso espacial de los asentamientos.
    El histograma + curva KDE muestra cuántos distritos tienen acceso deficiente
    (cola derecha larga), que una tabla no captura.
    """
    fig, axes = plt.subplots(1, 2, figsize=(13, 5))

    data = dist_gdf["dist_media_km"].dropna()

    sns.histplot(data, bins=40, kde=True, ax=axes[0], color="#e6550d")
    axes[0].axvline(data.median(), color="#2171b5", linestyle="--",
                    label=f"Mediana: {data.median():.1f} km")
    axes[0].set_title("Distancia media al IPRESS más cercano")
    axes[0].set_xlabel("Distancia media (km)")
    axes[0].set_ylabel("N° de distritos")
    axes[0].legend()

    # % de centros a más de 10 km
    pct_data = dist_gdf["pct_centros_lejanos"].dropna()
    sns.histplot(pct_data, bins=30, kde=False, ax=axes[1], color="#756bb1")
    axes[1].axvline(50, color="#d62728", linestyle="--", label="50% de CCPP a >10 km")
    axes[1].set_title("% de centros poblados a más de 10 km de una IPRESS")
    axes[1].set_xlabel("% de centros poblados lejanos")
    axes[1].set_ylabel("N° de distritos")
    axes[1].legend()

    plt.suptitle("Q2 – Acceso espacial de los centros poblados a servicios de emergencia", fontweight="bold")
    plt.tight_layout()
    if save:
        fig.savefig(FIGURES_DIR / "03_distance_distribution.png", dpi=150, bbox_inches="tight")
        log.info("Guardado: 03_distance_distribution.png")
    return fig


def plot_component_correlation(dist_gdf: gpd.GeoDataFrame, save: bool = True):
    """
    Matriz de correlación y pairplot de los tres componentes del IASE.
    Responde Q3: coherencia interna de los indicadores.
    Permite identificar si los tres componentes miden cosas distintas
    (baja correlación) o redundantes (alta correlación).
    """
    comp_cols = [c for c in ["facility_score", "activity_score", "access_score"] if c in dist_gdf.columns]
    if not comp_cols:
        log.warning("No se encontraron columnas de componentes para el pairplot.")
        return None

    df = dist_gdf[comp_cols].dropna()
    labels = {"facility_score": "Disponibilidad", "activity_score": "Actividad", "access_score": "Acceso espacial"}
    df = df.rename(columns=labels)

    fig, axes = plt.subplots(1, 2, figsize=(13, 5))

    corr = df.corr()
    mask = np.triu(np.ones_like(corr, dtype=bool))
    sns.heatmap(corr, ax=axes[0], annot=True, fmt=".2f", cmap="RdYlGn",
                vmin=-1, vmax=1, mask=mask, square=True, linewidths=0.5)
    axes[0].set_title("Correlación entre componentes del IASE")

    axes[1].scatter(df.iloc[:, 0], df.iloc[:, 1], alpha=0.3, s=10, color="#1f77b4")
    axes[1].set_xlabel(df.columns[0])
    axes[1].set_ylabel(df.columns[1])
    axes[1].set_title(f"{df.columns[0]} vs {df.columns[1]}")

    plt.suptitle("Q3 – Coherencia entre componentes del índice IASE", fontweight="bold")
    plt.tight_layout()
    if save:
        fig.savefig(FIGURES_DIR / "04_component_correlation.png", dpi=150, bbox_inches="tight")
        log.info("Guardado: 04_component_correlation.png")
    return fig


def plot_top_bottom_districts(dist_gdf: gpd.GeoDataFrame, score_col: str = "IASE_base",
                               n: int = 15, save: bool = True):
    """
    Ranking horizontal: top-N y bottom-N distritos según IASE.
    Responde Q3: identificación de los más y menos atendidos.
    Más efectivo que una tabla para mostrar brechas de escala entre extremos.
    """
    id_col = next((c for c in ["distrito", "ubigeo"] if c in dist_gdf.columns), "ubigeo")
    dept_col = "departamento" if "departamento" in dist_gdf.columns else None

    sorted_gdf = dist_gdf.sort_values(score_col, ascending=False).reset_index(drop=True)
    top = sorted_gdf.head(n)
    bottom = sorted_gdf.tail(n).sort_values(score_col, ascending=True)

    def _label(row):
        name = str(row[id_col])
        return f"{name} ({row[dept_col][:4]})" if dept_col and pd.notna(row[dept_col]) else name

    fig, axes = plt.subplots(1, 2, figsize=(14, 7))

    axes[0].barh([_label(r) for _, r in top.iterrows()], top[score_col], color="#1a9641", alpha=0.85)
    axes[0].invert_yaxis()
    axes[0].set_title(f"Top {n} – Mejor acceso a emergencias")
    axes[0].set_xlabel(score_col)

    axes[1].barh([_label(r) for _, r in bottom.iterrows()], bottom[score_col], color="#d7191c", alpha=0.85)
    axes[1].invert_yaxis()
    axes[1].set_title(f"Bottom {n} – Peor acceso a emergencias")
    axes[1].set_xlabel(score_col)

    plt.suptitle("Q3 – Ranking distrital de acceso a servicios de emergencia (IASE)", fontweight="bold")
    plt.tight_layout()
    if save:
        fig.savefig(FIGURES_DIR / "05_top_bottom_districts.png", dpi=150, bbox_inches="tight")
        log.info("Guardado: 05_top_bottom_districts.png")
    return fig


def plot_baseline_vs_alternative(comparison_df: pd.DataFrame, save: bool = True):
    """
    Scatter baseline vs alternativa + distribución de cambios de rango.
    Responde Q4: sensibilidad metodológica.
    El scatter muestra si los dos índices están de acuerdo globalmente;
    el histograma de rank_change cuantifica cuántos distritos cambian de posición
    significativamente al cambiar los pesos.
    """
    fig, axes = plt.subplots(1, 2, figsize=(13, 5))

    if "IASE_base" in comparison_df.columns and "IASE_alt" in comparison_df.columns:
        axes[0].scatter(comparison_df["IASE_base"], comparison_df["IASE_alt"],
                        alpha=0.3, s=10, color="#1f77b4")
        mn = min(comparison_df["IASE_base"].min(), comparison_df["IASE_alt"].min())
        mx = max(comparison_df["IASE_base"].max(), comparison_df["IASE_alt"].max())
        axes[0].plot([mn, mx], [mn, mx], "r--", linewidth=1, label="Línea 45°")
        axes[0].set_xlabel("IASE Baseline (pesos iguales)")
        axes[0].set_ylabel("IASE Alternativa (mayor peso acceso)")
        axes[0].set_title("Concordancia entre especificaciones")
        axes[0].legend()

    if "rank_change" in comparison_df.columns:
        sns.histplot(comparison_df["rank_change"], bins=40, kde=False,
                     ax=axes[1], color="#9467bd")
        axes[1].axvline(0, color="red", linestyle="--", label="Sin cambio")
        axes[1].set_title("Cambio de rango: Baseline → Alternativa")
        axes[1].set_xlabel("Cambio de rango (positivo = mejora)")
        axes[1].set_ylabel("N° de distritos")
        axes[1].legend()

    plt.suptitle("Q4 – Sensibilidad metodológica: baseline vs especificación alternativa", fontweight="bold")
    plt.tight_layout()
    if save:
        fig.savefig(FIGURES_DIR / "06_baseline_vs_alternative.png", dpi=150, bbox_inches="tight")
        log.info("Guardado: 06_baseline_vs_alternative.png")
    return fig


# ===========================================================================
# MAPAS ESTÁTICOS – GeoPandas
# ===========================================================================

def _choropleth(gdf, col, title, cmap, ax, label=""):
    missing = gdf[col].isna()
    gdf[~missing].plot(column=col, ax=ax, cmap=cmap, legend=True,
                       legend_kwds={"shrink": 0.6, "label": label},
                       linewidth=0.2, edgecolor="grey")
    if missing.any():
        gdf[missing].plot(ax=ax, color="lightgrey", linewidth=0.2, edgecolor="grey")
    ax.set_title(title, fontsize=12, fontweight="bold")
    ax.axis("off")


def map_facility_density(dist_gdf: gpd.GeoDataFrame, save: bool = True):
    """Mapa coroplético: número de IPRESS por distrito."""
    fig, ax = plt.subplots(figsize=(9, 12))
    _choropleth(dist_gdf, "n_ipress", "N° de IPRESS por distrito", "YlOrRd", ax, "N° IPRESS")
    plt.tight_layout()
    if save:
        fig.savefig(FIGURES_DIR / "map_01_facility_density.png", dpi=150, bbox_inches="tight")
        log.info("Guardado: map_01_facility_density.png")
    return fig


def map_distance_access(dist_gdf: gpd.GeoDataFrame, save: bool = True):
    """Mapa coroplético: distancia media al IPRESS más cercano."""
    fig, ax = plt.subplots(figsize=(9, 12))
    _choropleth(dist_gdf, "dist_media_km", "Distancia media al IPRESS más cercano (km)",
                "RdYlGn_r", ax, "km")
    plt.tight_layout()
    if save:
        fig.savefig(FIGURES_DIR / "map_02_distance_access.png", dpi=150, bbox_inches="tight")
        log.info("Guardado: map_02_distance_access.png")
    return fig


def map_iase(dist_gdf: gpd.GeoDataFrame, score_col: str = "IASE_base",
             clase_col: str = "clase_base", save: bool = True):
    """Mapa coroplético: IASE con clasificación por quintiles."""
    fig, axes = plt.subplots(1, 2, figsize=(18, 12))

    # Mapa continuo
    _choropleth(dist_gdf, score_col,
                f"{score_col} – Score continuo", "RdYlGn", axes[0], "Score")

    # Mapa discreto por clase
    if clase_col in dist_gdf.columns:
        color_map = QUINTILE_COLORS
        gdf = dist_gdf.copy()
        gdf["_color"] = gdf[clase_col].map(color_map).fillna("lightgrey")
        gdf.plot(color=gdf["_color"], ax=axes[1], linewidth=0.2, edgecolor="grey")
        patches = [mpatches.Patch(color=color_map[c], label=c) for c in QUINTILE_ORDER]
        axes[1].legend(handles=patches, loc="lower left", title="Clase IASE", fontsize=9)
        axes[1].set_title(f"{clase_col} – Clasificación por quintiles", fontsize=12, fontweight="bold")
        axes[1].axis("off")

    plt.suptitle("Q3 – Índice de Acceso a Servicios de Emergencia (IASE)", fontsize=14, fontweight="bold")
    plt.tight_layout()
    if save:
        fig.savefig(FIGURES_DIR / f"map_03_{score_col}.png", dpi=150, bbox_inches="tight")
        log.info(f"Guardado: map_03_{score_col}.png")
    return fig


def map_comparison(dist_gdf: gpd.GeoDataFrame, save: bool = True):
    """Mapas lado a lado: baseline vs alternativa."""
    fig, axes = plt.subplots(1, 2, figsize=(18, 12))
    for ax, col, title in [
        (axes[0], "IASE_base", "Baseline\n(pesos iguales 1/3)"),
        (axes[1], "IASE_alt", "Alternativa\n(acceso: 50%, disp: 25%, act: 25%)"),
    ]:
        if col in dist_gdf.columns:
            _choropleth(dist_gdf, col, title, "RdYlGn", ax, "Score")
        else:
            ax.set_title(f"{col} no calculado")
            ax.axis("off")

    plt.suptitle("Q4 – Comparación baseline vs especificación alternativa", fontsize=14, fontweight="bold")
    plt.tight_layout()
    if save:
        fig.savefig(FIGURES_DIR / "map_04_comparison.png", dpi=150, bbox_inches="tight")
        log.info("Guardado: map_04_comparison.png")
    return fig


# ===========================================================================
# MAPAS INTERACTIVOS – Folium
# ===========================================================================

def _dist_tooltip_cols(gdf):
    candidates = ["distrito", "departamento", "ubigeo", "n_ipress",
                  "total_emergencias", "dist_media_km", "IASE_base", "clase_base"]
    return [c for c in candidates if c in gdf.columns]


def folium_iase_map(dist_gdf: gpd.GeoDataFrame, score_col: str = "IASE_base") -> folium.Map:
    """
    Mapa interactivo Folium del IASE con coroplético y tooltips por distrito.
    """
    gdf = dist_gdf.to_crs("EPSG:4326").copy()
    if score_col not in gdf.columns:
        log.warning(f"Columna {score_col} no encontrada en gdf.")
        return folium.Map(location=[-9.19, -75.0], zoom_start=5)

    center = [-9.19, -75.0]
    m = folium.Map(location=center, zoom_start=5, tiles="CartoDB positron")

    tooltip_cols = _dist_tooltip_cols(gdf)

    folium.Choropleth(
        geo_data=gdf.__geo_interface__,
        data=gdf,
        columns=["ubigeo", score_col] if "ubigeo" in gdf.columns else [gdf.index, score_col],
        key_on="feature.properties.ubigeo" if "ubigeo" in gdf.columns else "feature.id",
        fill_color="RdYlGn",
        fill_opacity=0.7,
        line_opacity=0.2,
        legend_name=f"IASE ({score_col})",
        name="IASE",
    ).add_to(m)

    # Tooltips individuales
    style_fn = lambda x: {"fillOpacity": 0, "weight": 0}
    tooltip = folium.GeoJsonTooltip(
        fields=tooltip_cols,
        aliases=[c.replace("_", " ").title() for c in tooltip_cols],
        localize=True,
    )
    folium.GeoJson(
        gdf[tooltip_cols + ["geometry"]],
        style_function=style_fn,
        tooltip=tooltip,
        name="Info distritos",
    ).add_to(m)

    folium.LayerControl().add_to(m)
    return m


def folium_facilities_map(
    ipress_gdf: gpd.GeoDataFrame,
    dist_gdf: gpd.GeoDataFrame,
) -> folium.Map:
    """
    Mapa interactivo con marcadores agrupados de IPRESS sobre coroplético de IASE.
    """
    center = [-9.19, -75.0]
    m = folium.Map(location=center, zoom_start=5, tiles="CartoDB positron")

    # Coroplético de IASE si está disponible
    if "IASE_base" in dist_gdf.columns and "ubigeo" in dist_gdf.columns:
        gdf_4326 = dist_gdf.to_crs("EPSG:4326")
        folium.Choropleth(
            geo_data=gdf_4326.__geo_interface__,
            data=gdf_4326,
            columns=["ubigeo", "IASE_base"],
            key_on="feature.properties.ubigeo",
            fill_color="RdYlGn",
            fill_opacity=0.5,
            line_opacity=0.1,
            legend_name="IASE Baseline",
            name="IASE",
        ).add_to(m)

    # Markers de IPRESS
    cluster = MarkerCluster(name="Establecimientos IPRESS").add_to(m)
    ipress_4326 = ipress_gdf.to_crs("EPSG:4326") if ipress_gdf.crs else ipress_gdf

    for _, row in ipress_4326.iterrows():
        try:
            lat, lon = row.geometry.y, row.geometry.x
            nombre = row.get("nombre_ipress", row.get("nombre", "IPRESS"))
            categoria = row.get("categoria", "N/D")
            popup_txt = f"<b>{nombre}</b><br>Categoría: {categoria}"
            folium.CircleMarker(
                location=[lat, lon], radius=4,
                color="#2171b5", fill=True, fill_opacity=0.7,
                popup=folium.Popup(popup_txt, max_width=250),
            ).add_to(cluster)
        except Exception:
            continue

    folium.LayerControl().add_to(m)
    return m


def save_folium_map(m: folium.Map, filename: str) -> str:
    """Guarda un mapa Folium en output/figures/ y devuelve la ruta."""
    path = FIGURES_DIR / filename
    m.save(str(path))
    log.info(f"Mapa Folium guardado: {path}")
    return str(path)


# ===========================================================================
# Pipeline completo de visualizaciones
# ===========================================================================

def run_visualization_pipeline(dist_gdf: gpd.GeoDataFrame,
                                comparison_df: pd.DataFrame,
                                ipress_gdf: gpd.GeoDataFrame = None):
    """Genera y guarda todas las visualizaciones estáticas."""
    log.info("=== Generando visualizaciones ===")
    plot_facility_distribution(dist_gdf)
    plot_emergency_activity(dist_gdf)
    plot_distance_distribution(dist_gdf)
    plot_component_correlation(dist_gdf)
    plot_top_bottom_districts(dist_gdf)
    plot_baseline_vs_alternative(comparison_df)
    map_facility_density(dist_gdf)
    map_distance_access(dist_gdf)
    if "IASE_base" in dist_gdf.columns:
        map_iase(dist_gdf, "IASE_base", "clase_base")
    if "IASE_alt" in dist_gdf.columns:
        map_iase(dist_gdf, "IASE_alt", "clase_alt")
    map_comparison(dist_gdf)
    log.info("=== Visualizaciones completadas ===")
