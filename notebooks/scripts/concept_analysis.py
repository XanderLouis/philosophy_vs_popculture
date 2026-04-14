import numpy as np
import pandas as pd
from sklearn.metrics.pairwise import cosine_similarity
from sklearn.decomposition import PCA
import matplotlib.pyplot as plt
import os

NEW = r'C:\Users\USER\Desktop\Projects\philosophy_vs_popculture_clean'
OUT = f'{NEW}/outputs'

# ── Load embeddings and data ───────────────────────────────────────────
print("Loading embeddings...")
philo_emb = np.load(f'{OUT}/philo_embeddings.npy')
pop_emb   = np.load(f'{OUT}/pop_embeddings.npy')

df = pd.read_csv(f'{NEW}/data/final/philosophy_vs_popculture_final.csv')
philo_df = df[df['source_type'] == 'philosophy'].reset_index(drop=True)
pop_df   = df[df['source_type'] == 'pop_culture'].reset_index(drop=True)

print(f"Philosophy: {philo_emb.shape}")
print(f"Pop culture: {pop_emb.shape}")

# ── Concepts to analyze ────────────────────────────────────────────────
concepts = ['truth', 'freedom', 'meaning', 'identity', 
            'love', 'death', 'self', 'god', 'power', 'justice']

# ── For each concept: find top-N most relevant chunks ─────────────────
from sentence_transformers import SentenceTransformer
MODEL = r'C:\Users\USER\Desktop\Projects\philosophy_and_popculture\notebooks\models\all-MiniLM-L6-v2'
model = SentenceTransformer(MODEL)
print("✅ Model loaded\n")

results = []
concept_embeddings = {}

for concept in concepts:
    print(f"Analyzing: {concept}")
    
    # Embed the concept word itself
    concept_vec = model.encode([concept])  # shape (1, 384)
    concept_embeddings[concept] = concept_vec
    
    # Similarity of every chunk to the concept
    philo_sims = cosine_similarity(concept_vec, philo_emb)[0]
    pop_sims   = cosine_similarity(concept_vec, pop_emb)[0]
    
    # Top 500 most relevant chunks per side
    top_n = 500
    philo_top_idx = np.argsort(philo_sims)[-top_n:]
    pop_top_idx   = np.argsort(pop_sims)[-top_n:]
    
    philo_top_emb = philo_emb[philo_top_idx]
    pop_top_emb   = pop_emb[pop_top_idx]
    
    # Save per-concept embeddings
    np.save(f'{OUT}/{concept}_philo_emb.npy', philo_top_emb)
    np.save(f'{OUT}/{concept}_pop_emb.npy',   pop_top_emb)
    
    # Centroid similarity (how close are the two clouds?)
    philo_centroid = philo_top_emb.mean(axis=0)
    pop_centroid   = pop_top_emb.mean(axis=0)
    centroid_sim   = cosine_similarity([philo_centroid], [pop_centroid])[0][0]
    divergence     = 1 - centroid_sim
    
    # Internal coherence (how tight is each cluster?)
    philo_coherence = cosine_similarity(philo_top_emb).mean()
    pop_coherence   = cosine_similarity(pop_top_emb).mean()
    
    # Top context words
    philo_top_texts = philo_df.iloc[philo_top_idx]['comment'].tolist()
    pop_top_texts   = pop_df.iloc[pop_top_idx]['comment'].tolist()
    
    from collections import Counter
    import re
    stopwords = set(['the','a','an','and','or','but','in','on','at','to',
                     'for','of','with','is','it','i','you','he','she','we',
                     'they','this','that','be','are','was','were','have',
                     'has','had','not','as','by','from','so','if','do','my'])
    
    def top_words(texts, n=10):
        words = re.findall(r'\b[a-z]{3,}\b', ' '.join(texts).lower())
        filtered = [w for w in words if w not in stopwords and w != concept]
        return [w for w,_ in Counter(filtered).most_common(n)]
    
    results.append({
        'concept': concept,
        'philo_count': len(philo_top_idx),
        'pop_count': len(pop_top_idx),
        'centroid_similarity': round(centroid_sim, 4),
        'divergence': round(divergence, 4),
        'philo_coherence': round(philo_coherence, 4),
        'pop_coherence': round(pop_coherence, 4),
        'philo_context': ', '.join(top_words(philo_top_texts)),
        'pop_context': ', '.join(top_words(pop_top_texts)),
    })
    print(f"  divergence={divergence:.4f}  philo_coherence={philo_coherence:.4f}  pop_coherence={pop_coherence:.4f}")

# ── Save results ───────────────────────────────────────────────────────
results_df = pd.DataFrame(results).sort_values('divergence', ascending=False)
results_df.to_csv(f'{OUT}/master_results.csv', index=False)
print(f"\n✅ Results saved")
print(results_df[['concept','divergence','centroid_similarity']].to_string(index=False))




# ── Divergence bar chart ───────────────────────────────────────────────
results_df = pd.read_csv(f'{OUT}/master_results.csv').sort_values('divergence', ascending=True)

fig, ax = plt.subplots(figsize=(10, 6))
colors = ['#c0392b' if d > 0.25 else '#2980b9' for d in results_df['divergence']]
bars = ax.barh(results_df['concept'], results_df['divergence'], color=colors)
ax.set_xlabel('Divergence (1 - cosine similarity)', fontsize=12)
ax.set_title('Philosopher vs Pop Culture: Concept Divergence', fontsize=14, fontweight='bold')
ax.axvline(x=0.25, color='gray', linestyle='--', alpha=0.5, label='threshold')
for bar, val in zip(bars, results_df['divergence']):
    ax.text(val + 0.005, bar.get_y() + bar.get_height()/2, 
            f'{val:.3f}', va='center', fontsize=10)
plt.tight_layout()
plt.savefig(f'{OUT}/divergence_bar_chart.png', dpi=150)
plt.show()
print("✅ Chart saved")