# Philosophy vs Pop Culture — Semantic Concept Divergence

An NLP project analyzing how the meaning of core philosophical concepts 
differs between academic philosophy and modern pop culture discourse.

---

## Problem

When philosophers talk about concepts like "identity", "truth", or "meaning", 
and when those same words are used in everyday internet conversations — 
are they actually referring to the same ideas?

This project investigates that question using natural language processing.

---

## Data

Two distinct corpora were constructed:

- **Pop Culture (~100,000+ rows)**
  - Reddit comments
  - YouTube comments
  - Instagram captions and comments
  - TikTok data

- **Philosophy (~16,000+ rows)**
  - Extracted philosophical texts (Jung, Nietzsche, Russell)
  - Philosophy-focused discussions

---

## Method

1. Clean and normalize text across all sources
2. Generate sentence embeddings using `all-MiniLM-L6-v2`
3. Filter text by core concepts (e.g., identity, truth, love)
4. Compute centroid embeddings for each concept per corpus
5. Measure similarity using cosine similarity
6. Define divergence as:

   divergence = 1 - cosine_similarity

---

## Concepts Analyzed

identity, meaning, death, self, love, god, power, truth, freedom

---

## Observations

Preliminary results suggest that concepts tied to internal experience 
(e.g., identity, meaning, self) exhibit higher semantic divergence 
between philosophy and pop culture than externally-oriented concepts 
(e.g., truth, freedom).

Further statistical validation is required.

---

## Outputs

- Divergence ranking per concept
- PCA visualization of concept embeddings
- Comparative semantic analysis across domains

---

## Project Structure

├── data/
│   ├── final/
│   │   └── philosophy_vs_popculture.csv
│   ├── philosophy_texts/
│   └── pop_texts/
│
├── notebooks/
│   ├── analysis/
│   │   └── custome_concept_divergence.ipynb
│   ├── Data Cleaning/
│   └── Data Collection/
│
├── outputs/
│   ├── concept_divergence_results.csv
│   ├── divergence_bar_chart.png
│   ├── pca_all_concepts.png
│   └── master_results.csv
│
├── requirements.txt
└── README.md


---

## How to Run

```bash
pip install -r requirements.txt

Then open:

notebooks/03_concept_divergence.ipynb



Tools
Python
Pandas, NumPy
SentenceTransformers
scikit-learn
Matplotlib
