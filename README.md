# Music Genre Classification via Spotify Audio Features

Automatic music genre classification using Spotify's pre-computed audio features and machine learning. This project performs exploratory data analysis (EDA) and establishes a K-Nearest Neighbors baseline model across 114 genre labels.

---

## Research Question

> *Can songs be automatically classified into musical genres based on extracted audio features such as danceability, energy, loudness, tempo, and acousticness — and which features carry the most discriminative signal?*

---

## Dataset

**[Spotify Tracks Dataset](https://www.kaggle.com/datasets/maharshipandya/-spotify-tracks-dataset)** via Kaggle

- 114,000 track-genre records
- 114 genre labels (1,000 tracks each — perfectly balanced)
- 13 numerical audio descriptors per track

---

## Project Structure

```
├── spotify_genre_classification_eda7.ipynb   # Main notebook
└── README.md
```

---

## Notebook Overview

| Section | Description |
|---|---|
| 1. Import Libraries | Dependencies and plot configuration |
| 2. Load & Inspect | Dataset structure, dtypes, genre counts |
| 3. Data Cleaning | Null handling, duplicates, outlier inspection |
| 4. Feature Engineering | 6 composite audio features |
| 5. EDA | Distributions, correlations, discriminability |
| 6. Prepare for Modeling | Stratified sampling, encoding, scaling |
| 7. Baseline Model — KNN | Hyperparameter tuning, evaluation |
| 8. Summary & Next Steps | Key findings and Module 24 roadmap |

---

## Feature Engineering

Six composite features were created to improve genre separation:

| Feature | Formula | Rationale |
|---|---|---|
| `energy_valence_ratio` | `energy / (valence + 0.01)` | Separates high-energy dark (metal) vs. happy (pop) |
| `acoustic_energy_diff` | `acousticness - energy` | Classical/folk vs. electronic |
| `dance_tempo_product` | `danceability × tempo` | Uptempo groove vs. slow dance |
| `speech_instrumental_diff` | `speechiness - instrumentalness` | Hip-hop vs. ambient/classical |
| `duration_min` | `duration_ms / 60000` | Human-readable duration |
| `loudness_norm` | `(loudness + 60) / 60` | Re-scales loudness to [0, 1] |

---

## Key EDA Findings

- **Top discriminative features:** `acousticness`, `energy`, `instrumentalness`, `speechiness`, `loudness_norm`
- **Classical** stands apart with high acousticness and instrumentalness — easiest to classify
- **Metal / Heavy-Metal / Black-Metal** share nearly identical energy profiles — hardest cluster to separate
- **Hip-hop / R&B** overlap in danceability and speechiness; boundary is cultural rather than acoustic
- **16,299 tracks** appear under 2+ genre labels, reflecting real-world multi-label music categorization

---

## Baseline Model Results

**Algorithm:** K-Nearest Neighbors (KNN) with Euclidean distance  
**Tuning:** 5-fold stratified cross-validation over K = 1–20  
**Best K:** 18

| Metric | Score |
|---|---|
| Test Accuracy | See notebook output |
| Macro F1 Score | See notebook output |
| Random Chance Baseline | 0.88% (1/114) |

---

## Installation

```bash
git clone https://github.com/<your-username>/<repo-name>.git
cd <repo-name>
pip install -r requirements.txt
```

### Dependencies

```
numpy
pandas
matplotlib
seaborn
scikit-learn
```

---

## Usage

1. Download the dataset from [Kaggle](https://www.kaggle.com/datasets/maharshipandya/-spotify-tracks-dataset) and place `spotify_dataset.csv` in the project root (or update the path in cell 2).
2. Run the notebook top-to-bottom:

```bash
jupyter notebook spotify_genre_classification_eda7.ipynb
```

---

## Next Steps (Module 24)

- Compare KNN baseline against SVM and ensemble methods (Random Forest, Gradient Boosting)
- Investigate genre clustering and multi-label classification strategies
- Explore dimensionality reduction (PCA, UMAP) for visualization and performance

---

## Author

**Brian Cabalic** — Capstone Project, Module 16: EDA & Baseline Model
