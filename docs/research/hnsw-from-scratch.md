# Research: HNSW from Scratch en Rust

> Resultats de la recherche sur l'implementation de HNSW (Hierarchical Navigable Small World).

---

## References

- Malkov & Yashunin (2018). "Efficient and Robust Approximate Nearest Neighbor using Hierarchical Navigable Small World Graphs"
- hnswlib (C++ reference implementation) : https://github.com/nmslib/hnswlib
- instant-distance (Rust) : https://github.com/instant-labs/instant-distance
- usearch (C++/Rust) : https://github.com/unum-cloud/usearch

---

## Algorithme HNSW

### Principe

Un graphe multi-couches ou chaque noeud est connecte a ses voisins les plus proches.
Les couches superieures sont plus sparses (moins de noeuds) → navigation rapide.
La couche 0 contient tous les noeuds → recherche precise.

```
Layer 3:  [A] ──────────────── [E]          (tres sparse)
Layer 2:  [A] ── [C] ── [E] ── [G]         (sparse)
Layer 1:  [A] [B] [C] [D] [E] [F] [G]      (dense)
Layer 0:  [A] [B] [C] [D] [E] [F] [G] [H]  (tous les noeuds)
```

### Insertion

```
insert(q, max_layer):
  1. Choisir le layer max pour q : l = floor(-ln(random()) * mL)
     mL = 1/ln(M)
  2. Depuis l'entry point, descendre greedy de layer_max a l+1
  3. Pour chaque layer de l a 0 :
     - Chercher ef_construction voisins les plus proches
     - Connecter q aux M meilleurs (heuristique de selection)
     - Si un voisin a > M_max connexions, pruner
```

### Recherche

```
search(query, k, ef):
  1. Depuis l'entry point, descendre greedy de layer_max a layer 1
  2. A layer 0 : greedy search avec ef candidats
  3. Retourner les k plus proches parmi les ef candidats
```

### Heuristique de selection des voisins

Deux options :
- **Simple** : garder les M plus proches
- **Heuristique** : diversifier (eviter que tous les voisins soient dans le meme cluster)

L'heuristique est meilleure pour le recall mais plus lente a construire.

---

## Parametres

| Parametre | Defaut | Description | Impact |
|-----------|--------|-------------|--------|
| `M` | 16 | Connexions par noeud (layer >0) | Plus = meilleur recall, plus de memoire |
| `M_max0` | 32 (2×M) | Connexions max a layer 0 | Layer 0 est la plus dense |
| `ef_construction` | 200 | Candidats pendant l'insertion | Plus = meilleur index, plus lent a construire |
| `ef_search` | 100 | Candidats pendant la recherche | Plus = meilleur recall, plus lent |
| `mL` | 1/ln(M) | Facteur de normalisation des layers | Standard |

### Regle de pouce

- `M = 16` est un bon defaut pour la plupart des use cases
- `ef_construction >= 2 * M` pour un bon index
- `ef_search >= k` (sinon impossible d'avoir k resultats)
- `ef_search = k * 5` pour un bon recall (>95%)

---

## Structure de donnees en Rust

```rust
pub struct HnswIndex {
    layers: Vec<HnswLayer>,       // layers[0] = bottom, layers[n] = top
    entry_point: Option<u32>,      // index du point d'entree (layer max)
    max_layer: usize,              // layer max actuel
    params: HnswParams,
}

pub struct HnswLayer {
    neighbors: Vec<Vec<(u32, f32)>>,  // pour chaque noeud : (voisin_id, distance)
}
```

**Optimisation memoire :**
- `Vec<Vec<...>>` a beaucoup d'overhead (pointeur + len + cap par noeud)
- Alternative : CSR-like flat storage avec offsets
- Ou : `SmallVec<[u32; M]>` pour eviter les allocations

---

## Persistence de l'index HNSW

### Option 1 : Rebuild a chaque open
- Simple mais O(n·log(n)) a l'ouverture
- 1M vecteurs ≈ 30-60 secondes de rebuild
- **Inacceptable pour un open rapide**

### Option 2 : Serialiser les neighbor lists
- Stocker les neighbor lists dans des pages dediees
- Format : `[node_id: u32, layer: u8, neighbors: [(u32, f32); M]]`
- Open = charger les pages en memoire
- **Decision retenue**

### Format de page HNSW

```
HnswPage:
  [PageHeader (8B)]
  [entry_point: u32]
  [max_layer: u8]
  [node_count: u32]
  entries: [
    node_id: u32
    max_layer: u8
    layer0_neighbors: [(u32, f32); M_max0]  // toujours present
    layerN_neighbors: [(u32, f32); M]        // pour chaque layer > 0
  ]
```

---

## Quand utiliser HNSW vs brute-force

| N vecteurs | Strategie | Raison |
|-----------|-----------|--------|
| < 1K | Brute-force | HNSW overhead pas justifie |
| 1K - 10K | Brute-force SIMD | Encore rapide en brute-force |
| **10K - 100K** | **Seuil de decision** | **Benchmark pour decider** |
| > 100K | HNSW | Brute-force trop lent |
| > 1M | HNSW + PQ | Memoire et vitesse |

**Decision :** auto-switch a 10K vecteurs. Feature flag `hnsw`.

---

## Recall vs Speed tradeoff

Avec les parametres standard (M=16, ef_construction=200) :

| ef_search | Recall@100 | Latency (1M, 384d) |
|-----------|-----------|---------------------|
| 50 | ~85% | ~0.3ms |
| 100 | ~93% | ~0.5ms |
| 200 | ~97% | ~1.0ms |
| 500 | ~99% | ~2.5ms |

Pour hora, recall > 95% est l'objectif. ef_search = 200 est un bon defaut.

---

## Alternatives evaluees

### IVF-Flat (Inverted File)
- Partition les vecteurs en clusters (k-means)
- Recherche dans les nProbe clusters les plus proches
- Plus simple que HNSW mais recall plus bas

### ANNOY (Spotify)
- Random projection trees
- Bon pour les gros datasets (billions)
- Immutable : pas d'ajout apres construction
- **Incompatible avec hora** (on a besoin d'insertion dynamique)

### Vamana (DiskANN)
- Optimise pour le disk (pas tout en RAM)
- Plus complexe que HNSW
- Pertinent si hora depasse la RAM (v2.0+)

**Decision :** HNSW est le meilleur compromis pour hora (dynamique, bon recall, bien documente).

---

*Recherche effectuee le 2026-03-02.*
