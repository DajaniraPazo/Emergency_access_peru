# Emergency Healthcare Access Inequality in Peru

Geospatial analytics pipeline to analyze emergency healthcare access inequality across districts in Peru, combining health facility data, emergency care activity, populated centers, and district boundaries.

---

## What does this project do?

This project builds a **district-level Emergency Healthcare Access Index (IASE)** for all 1,874 districts of Peru. It combines four public datasets to answer: *which districts appear better or worse served in emergency healthcare, and what evidence supports that conclusion?*

---

## Main analytical goal

Identify spatial inequalities in emergency healthcare access at the district level by constructing a composite index that integrates:
- Health facility availability (number of IPRESS per district)
- Emergency care activity (total emergency consultations recorded)
- Spatial accessibility (average distance from populated centers to the nearest health facility)

---

## Datasets used

| Dataset | Source | Purpose |
|---------|--------|---------|
| IPRESS establishments | MINSA / datos.gob.pe | Facility locations, category, status |
| Emergency production by IPRESS | datos.gob.pe | Emergency consultations volume per facility |
| Populated Centers (Centros Poblados) | INEI / datos.gob.pe | Settlement coordinates for spatial access |
| DISTRITOS.shp | MINSA GIS / IGN | District boundary polygons |

### Where to download the data

1. **IPRESS** → [datos.gob.pe – IPRESS MINSA](https://datos.gob.pe/dataset/ipress) → Save as `data/Raw/ipress.csv`
2. **Emergencias** → [datos.gob.pe – Producción Asistencial Emergencia](https://datos.gob.pe/dataset/produccion-asistencial) → Save as `data/Raw/emergencias.csv`
3. **Centros Poblados** → [datos.gob.pe – Centros Poblados](https://www.datosabiertos.gob.pe) → Save as `data/Raw/centros_poblados.csv`
4. **DISTRITOS.shp** → MINSA GIS or IGN → Place all files (`.shp`, `.dbf`, `.shx`, `.prj`) in `data/Raw/`

---

## Data cleaning decisions

- **IPRESS:** Dropped records without valid coordinates within Peru's bounding box (lat: -18.4 to 0, lon: -81.5 to -68.7). Filtered to active establishments only.
- **Emergency production:** Aggregated by facility (sum across all periods). Negative or null values dropped.
- **Populated centers:** Dropped records without coordinates or outside Peru's bbox. Duplicates removed by name + ubigeo.
- **Districts:** Invalid geometries fixed with `buffer(0)`. Null geometries dropped.
- **CRS handling:** All data stored in **EPSG:4326 (WGS84)**. Distance calculations use **EPSG:32718 (UTM Zone 18S)** to minimize metric distortion for Peru's territory.

---

## District-level metric construction

### IASE – Índice de Acceso a Servicios de Emergencia

Three components, each normalized to [0, 1]:

| Component | Variable | Direction |
|-----------|----------|-----------|
| `facility_score` | Number of IPRESS in district | Higher = better |
| `activity_score` | Total emergency consultations | Higher = better |
| `access_score` | 1 − normalized(mean distance to nearest IPRESS) | Higher = better |

**Baseline specification:** Equal weights (1/3 each)
```
IASE_base = (1/3)·facility_score + (1/3)·activity_score + (1/3)·access_score
```

**Alternative specification:** Higher weight on spatial access
```
IASE_alt = 0.25·facility_score + 0.25·activity_score + 0.50·access_score
```
*Justification: In medical emergencies, transport time is the most critical factor for patient survival. This specification penalizes districts where populated centers are far from any facility.*

**Classification:** Districts classified into quintiles (Muy bajo / Bajo / Medio / Alto / Muy alto).

---

## How to install dependencies

```bash
pip install -r requirements.txt
```

Recommended: use a virtual environment.

```bash
python -m venv .venv
source .venv/bin/activate   # Windows: .venv\Scripts\activate
pip install -r requirements.txt
```

---

## How to run the processing pipeline

1. Place all raw data files in `data/Raw/` (see dataset links above).
2. Run:

```bash
python run_pipeline.py
```

This will:
- Clean and standardize all 4 datasets → `data/processed/`
- Build the district-level GeoDataFrame with spatial joins
- Calculate IASE baseline and alternative
- Generate all static charts and maps → `output/figures/`
- Save district tables → `output/tables/`

---

## How to run the Streamlit app

```bash
streamlit run app.py
```

The app has 4 tabs:
1. **Datos & Metodología** – problem statement, sources, cleaning, methodology, limitations
2. **Análisis Estático** – matplotlib/seaborn charts with interpretations
3. **Resultados Geoespaciales** – static maps and filterable district table
4. **Exploración Interactiva** – Folium interactive maps and baseline vs alternative comparison

---

## Repository structure

```
emergency_access_peru/
├── app.py                    # Streamlit app (4 tabs)
├── run_pipeline.py           # Main pipeline script
├── requirements.txt          # Python dependencies
├── README.md
│
├── src/
│   ├── utils.py              # Paths, CRS, column mappings, helper functions
│   ├── data_loader.py        # Load raw datasets
│   ├── cleaning.py           # Clean and standardize datasets
│   ├── geospatial.py         # Spatial joins, distance calculations, district GDF
│   ├── metrics.py            # IASE computation (baseline + alternative)
│   └── visualization.py     # Static charts, GeoPandas maps, Folium maps
│
├── data/
│   ├── Raw/                  # Original downloaded files
│   └── processed/            # Cleaned outputs (.csv, .gpkg)
│
├── output/
│   ├── figures/              # Static charts and maps (.png) + Folium (.html)
│   └── tables/               # District-level tables (.csv)
│
└── video/
    └── link.txt              # Link to explanatory video
```

---

## Principales hallazgos

- **55 distritos** no tienen ningún centro de salud registrado
- Distancia mediana al centro de salud más cercano: **3.6 km**
- Puntuación IASE promedio nacional: **0.161**
- **Mejor acceso:** Arequipa, Callería (Ucayali), ICA, Lima y Mazamari-Pangoa (Junín)
- **Peor acceso:** Chisquilla y Shipasbamba (Amazonas), Calango y Andajes (Lima sierra)
- La especificación alternativa penaliza más fuertemente a zonas rurales dispersas donde los centros poblados están lejos de cualquier establecimiento de salud

---

## Main limitations

- **Euclidean distance**, not road travel time — underestimates real access difficulty in mountainous or jungle areas.
- **No population data** — indices are absolute, not per capita.
- **Emergency data underreporting** — facilities with poor connectivity to SUSALUD may appear to have low activity.
- **Populated center coverage** — very small settlements may be absent from the INEI dataset.
- **Facility category not weighted** — a category I-1 post is treated equally to a category III-E hospital.

---

## GitHub workflow

- Work on feature branches (e.g., `feature/geospatial-pipeline`, `feature/streamlit-app`)
- Merge to `main` via Pull Request only
- Never commit directly to `main`
