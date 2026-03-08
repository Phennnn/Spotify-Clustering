# 🎵 Spotify Audio Clustering: Context-Aware Music Organization

> Unsupervised machine learning approach to discovering listening contexts from audio features

[![Python](https://img.shields.io/badge/Python-3.8+-blue.svg)](https://www.python.org/)
[![scikit-learn](https://img.shields.io/badge/scikit--learn-1.0+-orange.svg)](https://scikit-learn.org/)
[![License: MIT](https://img.shields.io/badge/License-MIT-green.svg)](LICENSE)

## 📖 Overview

This project applies unsupervised machine learning to automatically organize 32,833 Spotify tracks into **8 distinct listening contexts** based solely on audio features—without using genre labels. The system discovers natural groupings like "gym music," "study vibes," and "sad songs" directly from acoustic properties.

**Key Achievement:** Demonstrates that functional listening contexts can be derived from content-based audio analysis, challenging traditional genre-based music organization.

## 🎯 Problem Statement

Traditional music organization relies on genre labels (Pop, Rock, EDM), which:
- Are subjective and inconsistent
- Don't capture listening context (workout vs. relaxation)
- Fail to represent mood-based similarity

**Our approach:** Use unsupervised learning to discover mood-based clusters from 12 Spotify audio features.

## 🔬 Methodology

### 1. **Data Collection**
- Dataset: 32,833 tracks from Spotify API (via Kaggle)
- Features: 12 numerical audio attributes (danceability, energy, valence, acousticness, etc.)
- No genre labels used during clustering (unsupervised approach)

### 2. **Preprocessing**
- Removed all metadata (genres, artist names) to ensure purely unsupervised learning
- Applied StandardScaler (Z-score normalization) to handle heterogeneous feature scales
- Critical for distance-based algorithms like K-Means

### 3. **Dimensionality Reduction (PCA)**
- Reduced 12D → 2D for visualization (retained 28.92% variance)
- Note: Clustering performed on full 12-dimensional data for accuracy

### 4. **Clustering**
- **K-Means (k=8):** Primary clustering algorithm
  - Chose k using Elbow Method
  - Parameters: `n_clusters=8, random_state=42, n_init=10`
- **DBSCAN:** Outlier detection
  - Parameters: `eps=0.5, min_samples=5`
  - Identified 38 anomalous tracks

### 5. **Validation**
- Cluster centroid analysis for interpretability
- Built Gradio web app for qualitative testing
- Cross-referenced with ground-truth genres (withheld during training)

## 🎨 Results

### Discovered Clusters

| Cluster | Name | Key Characteristics |
|---------|------|---------------------|
| 0 | Gym Phonk / Sigma Mode | High energy (0.79), fast tempo (137 BPM) |
| 1 | Y2K Nostalgia (Happy) | Moderate energy (0.62), positive vibes |
| 2 | **Depresso Espresso** | **High acousticness (0.61), low energy (0.40)** |
| 3 | Glitchcore / Gaming | High instrumentalness (0.76) - no vocals |
| 4 | Rave / Speed Garage | High danceability (0.72), party mode |
| 5 | Villain Arc (Angst) | High energy (0.77), dark intensity |
| 6 | NPC Background Vibes | Moderate all features, ambient |
| 7 | **Main Character Energy** | **High valence (0.69), high danceability (0.76)** |

**Key Finding:** Algorithm successfully isolated "sad music" (Cluster 2: high acousticness, low energy) and "happy upbeat" (Cluster 7: high valence, high danceability) using only numerical features—no mood labels provided!

### Visualizations

**PCA Scatter Plot (8 Clusters):**
- Clear visual separation between clusters
- Each dot = 1 song, each color = 1 cluster
- Proves natural boundaries exist in audio feature space

**Elbow Method:**
- Tested k=1 to k=10
- Identified optimal k=8 (balance of granularity and interpretability)

**Correlation Heatmap:**
- Energy ↔ Loudness: r=0.68 (loud songs feel intense)
- Energy ↔ Acousticness: r=-0.54 (acoustic songs are calmer)

## 🚀 Demo Application

**Gradio Web App:** Interactive recommendation system
- **Input:** Song name
- **Output:** 
  - Identified cluster ("Main Character Energy," "Depresso Espresso," etc.)
  - 5 similar song recommendations from same cluster
- **How it works:** Content-based filtering using cluster membership

### Try It Yourself
```python
python app.py
```
Open browser at `http://localhost:7860`

Example queries:
- "Happier" → Main Character Energy cluster
- "Someone Like You" → Depresso Espresso cluster

## 🛠️ Technologies

**Languages & Libraries:**
- Python 3.8+
- pandas, numpy (data manipulation)
- scikit-learn (ML algorithms)
- matplotlib, seaborn (visualization)
- Gradio (web deployment)

**Algorithms:**
- K-Means Clustering
- DBSCAN (Density-Based Spatial Clustering)
- PCA (Principal Component Analysis)

## 📊 Project Structure

```
spotify-clustering/
├── notebooks/
│   └── Spotify_Clustering_Final.ipynb    # Main analysis notebook
├── data/
│   └── spotify_songs.csv                  # Dataset (download link below)
├── src/
│   ├── preprocessing.py                   # Data cleaning & scaling
│   ├── clustering.py                      # K-Means & DBSCAN implementation
│   └── visualization.py                   # Plotting functions
├── app.py                                 # Gradio web application
├── requirements.txt                       # Python dependencies
├── README.md                              # This file
└── LICENSE                                # MIT License

```

## 🚦 Getting Started

### Prerequisites
```bash
Python 3.8 or higher
pip (Python package manager)
```

### Installation

1. **Clone the repository**
```bash
git clone https://github.com/yourusername/spotify-clustering.git
cd spotify-clustering
```

2. **Install dependencies**
```bash
pip install -r requirements.txt
```

3. **Download dataset**
- Dataset: [Spotify Songs Dataset on Kaggle](https://www.kaggle.com/datasets/joebeachcapital/30000-spotify-songs)
- Place `spotify_songs.csv` in the `data/` folder

4. **Run the notebook**
```bash
jupyter notebook notebooks/Spotify_Clustering_Final.ipynb
```

5. **Launch web app**
```bash
python app.py
```

## 📈 Key Metrics

| Metric | Value |
|--------|-------|
| **Dataset Size** | 32,833 songs |
| **Features** | 12 audio attributes |
| **Clusters** | 8 |
| **PCA Variance Retained** | 28.92% (2 components) |
| **Outliers Detected** | 38 (0.12% of dataset) |
| **Correlation (Energy-Loudness)** | r=0.68 |

## 🎓 Research Insights

1. **Audio Features Encode Mood:** Clusters formed from low-level features correspond to high-level listening contexts (bridging the "semantic gap" in MIR)

2. **Genre Labels Are Imprecise:** Pop songs scattered across 6+ clusters, showing genre doesn't capture functional use

3. **Unsupervised Learning Works:** Successfully reconstructed human listening patterns without any labeled training data

## 🔮 Future Work

- [ ] Multi-modal fusion (lyrics + audio + metadata)
- [ ] Temporal modeling for dynamic song structure
- [ ] Personalized clustering based on user history
- [ ] Deep learning embeddings (autoencoders)
- [ ] Cross-cultural validation (non-Western music)

## 📚 References

- Spotify Web API: https://developer.spotify.com/documentation/web-api/
- Kaggle Dataset: https://www.kaggle.com/datasets/joebeachcapital/30000-spotify-songs
- Music Information Retrieval (MIR): [ISMIR Conference](https://www.ismir.net/)

## 👥 Authors

**Stephen Ace F. Sy** & **James Adrian Castro**
- Course: Machine Learning Final Project
- Date: February 2025

## 📄 License

This project is licensed under the MIT License - see the [LICENSE](LICENSE) file for details.

## 🙏 Acknowledgments

- Spotify for providing the audio analysis API
- Kaggle community for dataset curation
- scikit-learn developers for excellent ML tools

---

⭐ **Star this repo if you found it helpful!**

📧 **Questions?** Open an issue or contact: [your.email@example.com]

🔗 **Live Demo:** [Link to deployed Gradio app if available]
