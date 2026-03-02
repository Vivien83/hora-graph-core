# 05 — Search Engine

> SIMD vector search, BM25, HNSW, Hybrid RRF.
> Integre les resultats de recherche SIMD et HNSW.

---

## Architecture search

```
search(query_text, query_embedding, opts)
  │
  ├─ Vector leg (si embedding fourni)
  │   ├─ < 10K vecs : brute-force SIMD
  │   └─ >= 10K vecs : HNSW index
  │
  ├─ BM25 leg (si texte fourni)
  │   └─ Inverted index, VByte posting lists
  │
  └─ RRF Fusion
      └─ score = Σ 1/(k + rank_i) × weight_i
```

---

## 1. SIMD Cosine Similarity

### Ecosysteme Rust SIMD (2025-2026)

| Approche | Status | Recommandation |
|----------|--------|---------------|
| `std::simd` (portable) | **Nightly-only, non stabilise** | Ne pas utiliser |
| `std::arch` (intrinsics) | **Stable depuis Rust 1.27** | Notre choix |
| `wide` crate | Stable, types SIMD portables | Alternative si zero-dep impossible |

**Decision : `std::arch` pur.** Zero dep, stable, controle total.

### Dispatch multi-architecture

```rust
use std::sync::OnceLock;

type CosineFn = fn(&[f32], &[f32]) -> f32;
static COSINE_FN: OnceLock<CosineFn> = OnceLock::new();

pub fn cosine_similarity(a: &[f32], b: &[f32]) -> f32 {
    let f = COSINE_FN.get_or_init(|| {
        #[cfg(target_arch = "x86_64")]
        if is_x86_feature_detected!("avx2") && is_x86_feature_detected!("fma") {
            return |a, b| unsafe { cosine_avx2(a, b) };
        }
        #[cfg(target_arch = "aarch64")]
        { return |a, b| unsafe { cosine_neon(a, b) }; }
        cosine_scalar
    });
    f(a, b)
}
```

### AVX2 — 8 f32 par instruction, 4 accumulateurs

```rust
#[target_feature(enable = "avx2,fma")]
unsafe fn cosine_avx2(a: &[f32], b: &[f32]) -> f32 {
    // 4 accumulateurs pour masquer la latence FMA (4 cycles)
    // _mm256_fmadd_ps : a*b + c en 1 instruction, throughput 0.5 cycles
    // Boucle unrolled x4 : 32 floats par iteration
    // 384 / 32 = 12 iterations
    // Reduction : permute2f128 + hadd
}
```

**Pourquoi 4 accumulateurs :** latence FMA = 4 cycles, throughput = 0.5 cycles. Pour saturer le pipeline : `4 / 0.5 = 8` accumulateurs ideaux, 4 est le minimum pratique.

### NEON — 4 f32 par instruction

```rust
#[target_feature(enable = "neon")]
unsafe fn cosine_neon(a: &[f32], b: &[f32]) -> f32 {
    // vfmaq_f32 : FMA natif, toujours disponible sur AArch64
    // vaddvq_f32 : horizontal sum en 1 instruction (AArch64 uniquement)
    // Pas de detection runtime necessaire (NEON = toujours present)
}
```

### WASM SIMD — 4 f32, pas de FMA

```rust
#[cfg(all(target_arch = "wasm32", target_feature = "simd128"))]
fn cosine_wasm(a: &[f32], b: &[f32]) -> f32 {
    // f32x4_mul + f32x4_add (pas de FMA natif)
    // Perf : ~2-3x scalar, 4x plus lent qu'AVX2
    // Detection : cote JS via WebAssembly.validate()
}
```

### Optimisation cle : vecteurs pre-normalises

Si les embeddings sont normalises (||v|| = 1, cas des sentence transformers) :
```
cosine(a, b) = dot(a, b)
```
Elimine le calcul des 2 normes → **2x plus rapide**.

**Decision :** Stocker un flag `normalized: bool` dans le header. Si true, utiliser dot product au lieu de cosine.

### Performance estimee

| Architecture | 384-dim | 100K vecs brute-force | Methode |
|-------------|---------|----------------------|---------|
| AVX2 + FMA | ~150ns/pair | ~15ms (single-thread) | 4 acc, unroll x4 |
| NEON (M1) | ~200ns/pair | ~20ms | 4 acc, vfmaq |
| WASM SIMD | ~500ns/pair | ~50ms | sans FMA |
| Scalar | ~1μs/pair | ~100ms | auto-vectorise partiellement |

---

## 2. BM25 Inverted Index

### Zero dependency

L'index BM25 est 100% stdlib Rust. Tokenization par `char::is_alphanumeric()`, stop words hardcodes, VByte maison.

### Formule BM25+

```
score(D, Q) = Σ_q∈Q  IDF(q) × (tf(q,D) × (k1 + 1)) / (tf(q,D) + k1 × (1 - b + b × |D|/avgdl))

IDF(q) = ln((N - df(q) + 0.5) / (df(q) + 0.5) + 1)
```

**Parametres :** k1 = 1.2, b = 0.75 (standards)

### Tokenization

```rust
pub fn tokenize(text: &str) -> Vec<String> {
    text.to_lowercase()
        .split(|c: char| !c.is_alphanumeric())
        .filter(|t| t.len() > 1 && !STOP_WORDS.contains(t))
        .map(String::from)
        .collect()
}
```

Pas de stemming (ajouterait une dep). Suffisant pour un KG ou les termes sont souvent des noms propres et techniques.

### VByte encoding (20 lignes)

```rust
pub fn encode_vbyte(mut n: u32, buf: &mut Vec<u8>) {
    loop {
        let byte = (n & 0x7F) as u8;
        n >>= 7;
        if n == 0 { buf.push(byte | 0x80); break; }
        buf.push(byte);
    }
}

pub fn decode_vbyte(bytes: &[u8], offset: usize) -> (u32, usize) {
    let mut result: u32 = 0;
    let mut shift = 0;
    let mut i = offset;
    loop {
        let byte = bytes[i];
        result |= ((byte & 0x7F) as u32) << shift;
        i += 1;
        if byte & 0x80 != 0 { break; }
        shift += 7;
    }
    (result, i - offset)
}
```

### Posting lists delta-encodees

```
Terme "authentication" → docs [3, 7, 42, 100]
Deltas : [3, 4, 35, 58]
Chaque posting : (delta_doc_id: VByte, tf: VByte)
Compression ~68% vs raw u32
```

### IDF : recompute lazy

L'IDF change a chaque insert (N et df changent). Recomputer a chaque insert = O(vocab_size).

**Solution :** marquer les termes affectes comme dirty, recomputer au query time.

### Performance cible

| Operation | Cible |
|-----------|-------|
| Insert (tokenize + append posting) | < 10μs |
| Search (3 termes, 100K docs) | < 2ms |
| Index size | ~10 bytes/token posting |

---

## 3. HNSW (Hierarchical Navigable Small World)

### Quand activer

```
< 10K vecteurs  → brute-force SIMD (toujours plus rapide)
>= 10K vecteurs → HNSW index (O(log n) vs O(n))
```

Switch automatique a l'insertion du 10 000e vecteur.

### Parametres

| Parametre | Defaut | Impact |
|-----------|--------|--------|
| M | 16 | Connexions par noeud. Plus = meilleur recall, plus de RAM |
| M_max0 | 32 (2×M) | Connexions a la couche 0. **Critique pour la qualite** |
| ef_construction | 200 | Qualite du graphe a la construction |
| ef_search | 50-200 | Qualite de la recherche. Ajustable par query |
| mL | 1/ln(M) ≈ 0.36 | Multiplicateur de niveau |

### Structure

```rust
pub struct HnswIndex {
    nodes: Vec<HnswNode>,      // node_id → donnees du noeud
    vectors: Vec<f32>,         // flat: node_id * dim .. (node_id+1)*dim
    deleted: Vec<bool>,        // soft-delete bitmap
    entry_point: Option<u32>,  // noeud au niveau max
    max_layer: u8,
    dim: usize,
    params: HnswParams,
}

pub struct HnswNode {
    pub level: u8,
    pub neighbors: Vec<SmallVec<[u32; 32]>>, // par couche
}
```

### Algorithme INSERT (paper Malkov & Yashunin 2018)

```
1. Generer level = floor(-ln(U(0,1)) × mL)
2. Phase 1 : greedy descent depuis max_layer vers level+1 (ef=1)
3. Phase 2 : beam search + connexion aux couches level..0 (ef=ef_construction)
4. Pour chaque voisin : pruning si > M_max connections
5. Si level > max_layer → nouveau entry point
```

### Heuristique de selection de voisins (Algorithm 4)

La simple selection top-M est sous-optimale. L'heuristique diversite :
- Pour chaque candidat e, verifier si e est plus proche de q que de tout voisin deja selectionne
- Si oui → garder (diversifie les directions)
- Si non → ecarder (redondant)
- Remplir avec les ecartes si < M voisins

Gain : **+5-10% recall** vs top-M simple.

### Deletion : soft-delete + rebuild

Pas de suppression physique dans HNSW. Tombstone bitmap.
Rebuild complet quand > 15% de deleted nodes.

### Performance attendue

| Metrique | Valeur | Conditions |
|----------|--------|-----------|
| Recall@10 ef=50 | ~85-90% | M=16, 1M vecs |
| Recall@10 ef=200 | ~95-98% | M=16, 1M vecs |
| QPS ef=50 (SIMD) | ~20K-50K | Single thread, AVX2 |
| Build 1M vecs | 5-20 min | Single thread |
| RAM overhead | ~160 MB | M=16, 1M vecs (hors raw data) |

### Dependencies minimales

```toml
ordered-float = "4"  # Ord sur f32 pour BinaryHeap
smallvec = "1"       # SmallVec pour neighbor lists inline
rand = "0.8"         # SmallRng pour generation de niveaux
```

---

## 4. Hybrid Search — Reciprocal Rank Fusion (RRF)

### Formule

```
RRF_score(d) = Σ_r  weight_r / (k + rank_r(d))

r = chaque "leg" de recherche (vector, BM25)
k = 60 (constante de smoothing)
rank_r(d) = rang du document d dans les resultats de la leg r
weight_r = poids de la leg (vector: 0.7, BM25: 0.3)
```

### Pourquoi RRF et pas CombMNZ/CombSUM

| Methode | Avantage | Inconvenient |
|---------|----------|-------------|
| CombSUM | Simple | Sensible aux echelles de scores |
| CombMNZ | Favorise les docs trouves par les 2 legs | Sensible aux echelles |
| **RRF** | **Insensible aux echelles** | Ignore les scores absolus |

RRF est insensible aux echelles car il utilise les **rangs** et non les scores. Parfait pour fusionner cosine similarity [0,1] et BM25 [0, ∞).

### Integration avec activation ACT-R

Le score final integre l'activation bio-inspiree :

```rust
pub fn search(&self, query_text: Option<&str>, query_embedding: Option<&[f32]>,
              opts: SearchOpts) -> Result<Vec<SearchHit>> {
    let k_rrf = 60;

    // Legs
    let vec_results = query_embedding.map(|e| self.vector_search(e, opts.top_k * 3));
    let bm25_results = query_text.map(|q| self.bm25_index.search(q, opts.top_k * 3));

    // RRF fusion
    let mut scores: HashMap<EntityId, f64> = HashMap::new();
    if let Some(ref vr) = vec_results {
        for (rank, hit) in vr.iter().enumerate() {
            *scores.entry(hit.id).or_default() += 0.7 / (k_rrf + rank + 1) as f64;
        }
    }
    if let Some(ref br) = bm25_results {
        for (rank, hit) in br.iter().enumerate() {
            *scores.entry(hit.id).or_default() += 0.3 / (k_rrf + rank + 1) as f64;
        }
    }

    // Boost par activation ACT-R
    for (id, score) in &mut scores {
        let activation = self.get_activation(*id);
        let retrieval_prob = sigmoid_retrieval(activation, self.config.tau, self.config.s);
        *score *= retrieval_prob; // P(i) dans [0, 1]
    }

    // Filtre dark nodes
    scores.retain(|id, _| !self.is_dark(*id) || opts.include_dark);

    // Top-k
    let mut ranked: Vec<_> = scores.into_iter().collect();
    ranked.sort_by(|a, b| b.1.partial_cmp(&a.1).unwrap());
    ranked.truncate(opts.top_k);

    // Side-effect : record access
    for &(id, _) in &ranked {
        self.record_access(id)?;
    }

    Ok(ranked.into_iter().map(|(id, score)| SearchHit { id, score }).collect())
}
```

---

## 5. Embedding Strategy

### Decision : externe par defaut

hora-graph-core **ne compute jamais d'embeddings**. Il stocke et recherche des vecteurs pre-calcules.

C'est le modele SQLite : SQLite stocke des donnees, il ne les genere pas.

### API

```rust
// Rust : accepter &[f32] brut
fn add_entity(&mut self, ..., embedding: Option<&[f32]>) -> Result<EntityId>;
fn vector_search(&self, query_embedding: &[f32], k: usize) -> Result<Vec<SearchHit>>;

// Node.js : Float32Array zero-copy via napi-rs
fn add_entity(&mut self, ..., embedding: Option<Float32Array>) -> Result<u64>;
```

### Trait Embedder (optionnel, helper)

```rust
pub trait Embedder: Send + Sync {
    fn embed(&self, text: &str) -> Result<Vec<f32>>;
    fn embed_batch(&self, texts: &[&str]) -> Result<Vec<Vec<f32>>>;
    fn dimensions(&self) -> usize;
}
```

L'appelant implemente ce trait avec son provider (OpenAI, Ollama, transformers.js...).
HoraCore ne contient pas d'Embedder. Le wiring se fait cote application.

### Feature flag optionnel : `embedder`

```toml
[features]
embedder = ["dep:ort", "dep:tokenizers"]  # ONNX Runtime, all-MiniLM-L6-v2 (23MB int8)
```

Pour les utilisateurs qui veulent un embedding embedded. **Pas dans le chemin zero-dep par defaut.**

### BM25 sans embedding (mode text-only)

`embedding_dims = 0` dans la config → mode BM25-only :
- Pas de pages vecteur allouees
- Pas de SIMD code path active
- `vector_search()` retourne erreur
- `search(text, None)` fonctionne normalement

---

## Decisions prises

| Decision | Justification |
|----------|--------------|
| `std::arch` pour SIMD | Seule option stable zero-dep |
| OnceLock pour dispatch | Zero overhead apres premier appel |
| 4 accumulateurs FMA | Masque la latence pipeline (4 cycles) |
| Pre-normalisation optionnelle | 2x speedup si embeddings normalises |
| BM25 zero-dep | Tokenizer stdlib, VByte maison, posting delta |
| HNSW a 10K+ | Crossover brute-force/HNSW empirique |
| RRF pour fusion | Insensible aux echelles, simple, efficace |
| Embeddings externes | Zero dep, flexibility maximale |
| embedding_dims=0 valid | Mode text-only sans overhead vecteur |

---

*Document cree le 2026-03-02. Integre les resultats de recherche SIMD, HNSW et embedding.*
