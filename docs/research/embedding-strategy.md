# Research: Strategie d'Embedding

> Resultats de la recherche sur la strategie d'embedding pour hora-graph-core.

---

## Decision : External-First

hora-graph-core **ne genere pas** les embeddings par defaut. L'utilisateur fournit les vecteurs.

### Pourquoi

| Raison | Detail |
|--------|--------|
| **Zero-dep** | ONNX Runtime = 50MB+ de dep, incompatible avec la philosophie |
| **Flexibilite** | L'utilisateur choisit son modele (OpenAI, Cohere, local, etc.) |
| **Portabilite** | ONNX ne fonctionne pas partout (WASM, edge devices) |
| **Separation des responsabilites** | hora = storage + search, pas inference |

### API

```rust
// L'utilisateur fournit l'embedding
hora.add_entity("concept", "auth", props, Some(&embedding))?;
hora.vector_search(&query_embedding, 10)?;

// Pas d'embedding = pas de vector search (BM25 uniquement)
hora.add_entity("concept", "auth", props, None)?;
hora.text_search("authentication", 10)?;

// Mode text-only (embedding_dims=0)
let config = HoraConfig { embedding_dims: 0, ..default() };
```

---

## Feature optionnelle : `embedder`

Pour les utilisateurs qui veulent de l'embedding integre :

```toml
[features]
embedder = ["dep:ort"]
```

```rust
#[cfg(feature = "embedder")]
pub fn add_entity_with_text(
    &mut self,
    entity_type: &str,
    name: &str,
    text: &str,  // sera embed automatiquement
    props: Option<Properties>,
) -> Result<EntityId> {
    let embedding = self.embedder.as_ref()
        .ok_or(HoraError::EmbedderNotConfigured)?
        .embed(text)?;
    self.add_entity(entity_type, name, props, Some(&embedding))
}
```

### Modeles ONNX recommandes

| Modele | Dims | Taille | Qualite | Use case |
|--------|------|--------|---------|----------|
| all-MiniLM-L6-v2 | 384 | 90MB | Bonne | General purpose, rapide |
| nomic-embed-text-v1.5 | 768 | 274MB | Tres bonne | Meilleur rapport qualite/taille |
| bge-base-en-v1.5 | 768 | 438MB | Excellente | Anglais, benchmark leader |
| multilingual-e5-small | 384 | 470MB | Bonne | Multilingue |

**Recommandation :** all-MiniLM-L6-v2 (384d) pour commencer. Dimension basse = moins de memoire, plus rapide.

---

## Dimensions d'embedding

### Impact sur la performance

| Dims | Memoire/1M vecs | Cosine 100K | Storage |
|------|-----------------|-------------|---------|
| 128 | 512 MB | ~1ms | Compact |
| **384** | **1.5 GB** | **~3ms** | **Bon compromis** |
| 768 | 3 GB | ~6ms | Lourd |
| 1536 | 6 GB | ~12ms | Tres lourd |

### Configuration

```rust
pub struct HoraConfig {
    pub embedding_dims: u16,  // 0 = pas de vector search
    // ...
}
```

- `embedding_dims = 0` : mode text-only, pas de vector store
- `embedding_dims = 384` : defaut recommande
- `embedding_dims = 768` : pour les modeles plus riches
- `embedding_dims = 1536` : pour OpenAI ada-002

**L'embedding_dims est fixe a la creation du fichier .hora** et ne peut pas etre change apres (car les pages VectorData sont dimensionnees pour).

---

## BM25 zero-dep

Meme sans embeddings, hora offre le text search via BM25.

### Tokenizer zero-dep

```rust
pub fn tokenize(text: &str) -> Vec<String> {
    text.to_lowercase()
        .split(|c: char| !c.is_alphanumeric())
        .filter(|s| !s.is_empty() && s.len() > 1)
        .filter(|s| !STOP_WORDS.contains(s))
        .map(|s| s.to_string())
        .collect()
}

const STOP_WORDS: &[&str] = &[
    "the", "a", "an", "is", "are", "was", "were", "be", "been",
    "being", "have", "has", "had", "do", "does", "did", "will",
    "would", "could", "should", "may", "might", "shall", "can",
    "to", "of", "in", "for", "on", "with", "at", "by", "from",
    "as", "into", "through", "during", "before", "after", "above",
    "below", "between", "out", "off", "over", "under", "again",
    "further", "then", "once", "and", "but", "or", "nor", "not",
    "so", "it", "its", "this", "that", "these", "those",
];
```

### BM25+ scoring

```
score(q, d) = Σ IDF(t) × (tf(t,d) × (k1 + 1)) / (tf(t,d) + k1 × (1 - b + b × |d|/avgdl)) + δ
```

Parametres :
- k1 = 1.2 (saturation TF)
- b = 0.75 (normalisation longueur)
- δ = 1.0 (BM25+ : boost les termes rares)

### Posting lists en VByte

```rust
pub fn vbyte_encode(n: u32, buf: &mut Vec<u8>) {
    let mut val = n;
    while val >= 0x80 {
        buf.push((val as u8) | 0x80);
        val >>= 7;
    }
    buf.push(val as u8);
}

pub fn vbyte_decode(buf: &[u8]) -> (u32, usize) {
    let mut result = 0u32;
    let mut shift = 0;
    for (i, &byte) in buf.iter().enumerate() {
        result |= ((byte & 0x7F) as u32) << shift;
        if byte & 0x80 == 0 {
            return (result, i + 1);
        }
        shift += 7;
    }
    panic!("invalid vbyte");
}
```

Les posting lists stockent des **deltas** (differences entre doc IDs consecutifs) en VByte → compression ~50%.

---

## Combinaison Vector + BM25 : RRF

### Reciprocal Rank Fusion

```
RRF_score(d) = Σ 1 / (k + rank_i(d))
```

Avec k = 60 (constante standard).

### Poids configurables

```rust
pub struct SearchOpts {
    pub top_k: usize,          // defaut: 10
    pub vector_weight: f32,    // defaut: 0.7
    pub bm25_weight: f32,      // defaut: 0.3
    pub activation_boost: bool, // defaut: true
}
```

### Flow complet

```
1. vector_search(query_embedding, k*2) → ranked list V
2. text_search(query_text, k*2) → ranked list T
3. RRF fusion :
   ∀ doc in V ∪ T :
     score = vector_weight / (60 + rank_V(doc))
           + bm25_weight / (60 + rank_T(doc))
4. Si activation_boost :
   score *= (1 + 0.1 * activation(doc))
5. Trier par score, retourner top-k
```

---

## Matrice de decision

| Scenario | embedding_dims | Feature embedder | Strategy |
|----------|---------------|-----------------|----------|
| Agent IA avec API embedding | 384-1536 | Non | External embeddings |
| App standalone, pas d'API | 0 | Non | Text-only (BM25) |
| App standalone, besoin vector | 384 | Oui | ONNX local |
| Browser (WASM) | 384 | Non | External (API call) |
| Dev/test rapide | 0 | Non | Text-only |

---

*Recherche effectuee le 2026-03-02.*
