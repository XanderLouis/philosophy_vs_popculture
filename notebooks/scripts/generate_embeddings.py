import sys
print("Python:", sys.executable)
print("Starting...")
from sentence_transformers import SentenceTransformer
import pandas as pd
import numpy as np
import os
import time

NEW      = r'C:\Users\USER\Desktop\Projects\philosophy_vs_popculture_clean'
MODEL    = r'C:\Users\USER\Desktop\Projects\philosophy_and_popculture\notebooks\models\all-MiniLM-L6-v2'
OUT      = f'{NEW}/outputs'
os.makedirs(OUT, exist_ok=True)

# ── Load data ──────────────────────────────────────────────────────────
df = pd.read_csv(f'{NEW}/data/final/philosophy_vs_popculture_final.csv')
print(f"Loaded: {df.shape}")
print(df['source_type'].value_counts())

# ── Load model ─────────────────────────────────────────────────────────
print(f"\nLoading model from {MODEL}...")
model = SentenceTransformer(MODEL)
print("✅ Model loaded")

# ── Embed ──────────────────────────────────────────────────────────────
philo_texts = df[df['source_type'] == 'philosophy']['comment'].tolist()
pop_texts   = df[df['source_type'] == 'pop_culture']['comment'].tolist()

print(f"\nEmbedding {len(philo_texts):,} philosophy chunks...")
t0 = time.time()
philo_emb = model.encode(philo_texts, batch_size=64, show_progress_bar=True)
print(f"Done in {(time.time()-t0)/60:.1f} min — shape: {philo_emb.shape}")

print(f"\nEmbedding {len(pop_texts):,} pop culture comments...")
t0 = time.time()
pop_emb = model.encode(pop_texts, batch_size=64, show_progress_bar=True)
print(f"Done in {(time.time()-t0)/60:.1f} min — shape: {pop_emb.shape}")

# ── Save ───────────────────────────────────────────────────────────────
np.save(f'{OUT}/philo_embeddings.npy', philo_emb)
np.save(f'{OUT}/pop_embeddings.npy',   pop_emb)

print(f"\n✅ Saved:")
print(f"  philo_embeddings.npy — {philo_emb.shape}")
print(f"  pop_embeddings.npy   — {pop_emb.shape}")