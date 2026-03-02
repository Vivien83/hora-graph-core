# Research: ACT-R Implementation pour Knowledge Graph

> Resultats de la recherche approfondie sur l'implementation computationnelle d'ACT-R.

---

## Sources principales

- Anderson, J.R. (1991). "The Adaptive Character of Thought"
- Anderson & Lebiere (1998). "The Atomic Components of Thought"
- Petrov, A.A. (2006). "Computationally Efficient Approximation of the Base-Level Learning Equation"
- Nader et al. (2000). "Fear memories require protein synthesis in the amygdala for reconsolidation"
- Tononi & Cirelli (2006). "Sleep function and synaptic homeostasis" (SHY hypothesis)
- Expertium (2024). "The FSRS Algorithm"

---

## BLL (Base-Level Learning)

### Equation
```
B(i) = ln( Σⱼ₌₁ⁿ  tⱼ^(-d) )
```
- d = 0.5 (decay exponent, Anderson 1991)
- tⱼ = temps ecoule depuis le j-ieme acces (en secondes)

### Probleme de stockage
Stocker tous les timestamps → O(n) par entite. Pour un noeud accede 10000 fois → 80KB juste pour les timestamps.

### Solution retenue : Petrov 2006 Hybrid

Conserver les **10 derniers** timestamps exacts (bosse recente) + agreger l'historique.

L'approximation optimized learning (Anderson & Lebiere 1998) pour l'historique :
```
B_history ≈ ln( n_old / (1-d) · L^(d-1) )
```

La bosse Petrov : les derniers acces sont calcules exactement car ils contribuent le plus au score.

**Avantages :**
- O(1) memoire par entite (fixe: 10 timestamps + 2 compteurs)
- Erreur < 5% par rapport au calcul exact
- Capture correctement la bosse post-acces

**Inconvenients :**
- L'evacuation du 11eme acces perd en precision (agrege dans historical_sum)
- Le historical_sum est une approximation cumulative qui peut deriver

### Stabilite numerique

Pour des deltas tres grands (>10^8 secondes ≈ 3 ans), `t^(-0.5)` → 0. Risque d'underflow.

Solution : **log-sum-exp trick**
```
ln(Σ tⱼ^(-d)) = ln(Σ exp(-d·ln(tⱼ)))
```
Utiliser max + sum_exp pour eviter underflow/overflow.

---

## Spreading Activation

### Implementation sur graphe

BFS avec attenuation. A chaque hop, l'activation propagee diminue selon le fan effect.

Fan effect formula :
```
Sⱼᵢ = S - ln(fan)
```
Avec S = 1.6 (standard ACT-R), inhibition quand fan > e^1.6 ≈ 4.95.

**Implication architecturale :** les hubs (noeuds avec beaucoup de connexions) propagent de l'**inhibition**. C'est biologiquement correct (interference) mais contre-intuitif.

### Representation CSR pour la performance

CSR (Compressed Sparse Row) est optimal pour le BFS :
- Acces sequentiel aux voisins d'un noeud = excellent pour le cache
- O(1) pour trouver les voisins d'un noeud
- O(degree) pour iterer sur les voisins

En v0.1, utiliser Vec<Edge> + HashMap<EntityId, Vec<EdgeId>> (simpler).
En v0.5, passer au CSR page-based.

---

## FSRS

### Power law decay
```
R(t, S) = (1 + F · t/S)^(-w₂₀)
F = 0.9^(-1/w₂₀) - 1
```

### Complementarite avec ACT-R

ACT-R = **scoring** (qui est pertinent maintenant ?)
FSRS = **scheduling** (quand revoir ?)

Les deux utilisent une power law mais avec des objectifs differents. Pas de conflit theorique.

---

## Reconsolidation

### Neuroscience
- Fenetre post-reactivation : 4-6 heures
- Protein synthesis dependent
- Inhibition de Rac1 previent l'oubli
- Memoire destabilisee peut etre modifiee ou effacee

### Extrapolation computationnelle
- Fenetre de 5h : apres un acces fort, le noeud est "labile"
- Pendant la fenetre : les mises a jour ont plus d'impact
- Apres la fenetre : le noeud est "restabilise" avec un boost

**Decision :** fenetre configurable (defaut 5h = 18000s), car l'extrapolation au computationnel est approximative.

---

## Dark Nodes (Rac1-inspired)

### Biologie
- Rac1 depolymerise l'actine des epines dendritiques
- La spine se retracte → synapse affaiblie
- L'engram complet reste mais est inaccessible
- Inhiber Rac1 restaure la memoire

### Mapping computationnel
- Activation sous seuil pendant >7 jours → dark
- Edges attenues (×0.1) mais pas supprimes
- Recovery possible par stimulation externe forte
- GC physique apres 90 jours d'inactivite

**Decision :** seuil configurable, fenetre 7j = extrapolation (biologie = minutes-heures).

---

## Dream Cycle (CLS - Complementary Learning Systems)

### SHY Hypothesis (Tononi & Cirelli 2006)
- Pendant le sommeil : downscaling global de ~20%
- Homeostatique : evite la saturation des synapses
- Les memoires fortes survivent, les faibles sont oubliees

### Interleaved Replay
- Mix ancien/recent (30/70) = consolidation des deux
- Replay compresse : pas tout l'episode, juste les entites cles
- Re-activation = record_access → BLL boost

### CLS Transfer
- Episodes repetes → extraire les faits recurrents
- Creer des faits "semantiques" (sans contexte episodique)
- = decontextualisation progressive (episodique → semantique)

### Memory Linking
- Co-allocation temporelle : entites creees dans la meme fenetre
- Creer des liens implicites
- Fenetre de 6h basee sur la coallocation cellulaire (neuroscience)

---

*Recherche effectuee le 2026-03-02.*
