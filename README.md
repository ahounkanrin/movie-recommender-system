# Movie Recommender System [![Streamlit App](https://static.streamlit.io/badges/streamlit_badge_black_red.svg)](https://movie-recommender-ahounkanrin.streamlit.app/)

This repository contains the code for building a movie recommender system deployed as an interactive web application with **Streamlit**.

The Model uses matrix factorisation and is optimised using alternating least square.

## Data Sources

This recommender system uses two main data sources:

* **Rating Data:** The model is trained using the **MovieLens 32M movie ratings.** dataset.
* **Movie Metadata:** Movie details (posters, titles, release year) were retrieved using the **TMDB API**.

## Repository Structure

| File Name | Description |
| :--- | :--- |
| **`data_parser.py`** | Script for processing the MovieLens data |
| **`train.py`** | Trains the recommendation model |
| **`train_parallel.py`** | A parallelised and JIT-optimised version of `train.py` (using Numba) |
| **`train_with_genres_features.py`** | Adds movie genres embeddings to the the model |
| **`train_final_model.py`** | Trains the deployed model on the entire data available |
| **`make_recommendation.py`** | Makes recommendations for a given user |
| **`app.py`** | Streamlit application file |
---

## Getting Started Locally

To set up and run this project on your local machine, follow these steps:

### 1. Clone the Repository

```bash
git clone https://github.com/ahounkanrin/movie-recommender-system.git
cd movie-recommender-system
```
### 2. Install Dependencies

```bash 
pip install -r requirements.txt
```

### 3. Run the Streamlit App Locally
```bash
streamlit run app.py
```

---

## Deployed Application

Click on the badge below to visit the deployed application.

[![Streamlit App](https://static.streamlit.io/badges/streamlit_badge_black_red.svg)](https://movie-recommender-ahounkanrin.streamlit.app/)

---
