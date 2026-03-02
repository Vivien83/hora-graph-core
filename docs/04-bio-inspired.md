# 04 — Modele Bio-Inspire

> ACT-R, CLS, dark nodes, dream cycle, reconsolidation.
> Chaque formule est sourcee, chaque decision est justifiee.

---

## Vue d'ensemble : cerveau → hora

| Cerveau | Mecanisme | hora-graph-core | Phase |
|---------|-----------|-----------------|-------|
| Engram (CREB) | Selection competitive 10-30% | Encodage selectif, dedup agressive | v0.1 |
| Dentate Gyrus | Pattern separation | Embedding orthogonalisation | v0.2 |
| CA3 | Pattern completion (Hopfield) | Graph expansion depuis cue partielle | v0.1 |
| Spreading activation | Propagation avec decay/distance | BFS avec attenuation par hop | v0.3 |
| ACT-R BLL | `B = ln(Σ tⱼ^(-d))`, d≈0.5 | Score d'activation par noeud | v0.3 |
| Fan Effect | Plus d'associations = dilution | `Sⱼᵢ = S - ln(fan)` | v0.3 |
| Rac1 pathway | Oubli actif moleculaire | Dark nodes (silencieux, recuperables) | v0.3 |
| CLS fast/slow | Hippocampe/neocortex | Episodique → semantique | v0.4 |
| Reconsolidation | Acces destabilise 4-6h | Update window apres acces | v0.3 |
| Memory linking | 6h co-allocation window | Liens temporels auto | v0.4 |
| SHY (sommeil) | Downscaling homeostatique | Dream cycle: normalisation | v0.4 |
| Sharp-wave ripples | Replay compresse | Consolidation interleaved | v0.4 |
| Amygdale | Salience emotionnelle | emotional_weight sur activation | v0.3 |
| Working memory | 4±1 chunks | Context window management | v0.3 |
| Episodique→semantique | Decontextualisation progressive | Transformation par repetition | v0.4 |
| FSRS | `R = (1 + t/(S·w₂₀))^(-1)` | Scheduling de renforcement | v0.3 |

---

## 1. ACT-R Base-Level Learning (BLL)

### Equation exacte

```
B(i) = ln( Σⱼ₌₁ⁿ  tⱼ^(-d) )
```

- `tⱼ` = temps ecoule (secondes) depuis le j-ieme acces
- `d` = decay exponent = **0.5** (power law of forgetting, Anderson 1991)
- `n` = nombre total d'acces

### Probleme de stockage

Stocker tous les timestamps : O(n) memoire par entite. Inacceptable pour des noeuds accedes des milliers de fois.

### Solution : Approximation hybride Petrov 2006

**Principe :** conserver les N derniers timestamps exacts (bosse recente), agreger l'historique.

L'approximation Anderson & Lebiere 1998 (optimized learning) pour l'historique :
```
B_historique ≈ ln( n / (1-d) · L^(d-1) )
```
Ou `n` = acces historiques, `L` = lifetime.

La bosse Petrov (dernier acces) est calculee exactement.

### Implementation

```rust
pub struct ActivationState {
    pub created_at: f64,           // epoch seconds
    pub historical_count: u32,     // acces avant la fenetre recente
    pub historical_sum: f64,       // Σ tⱼ^(-d) pour les anciens acces
    pub recent_accesses: [f64; 10], // timestamps des 10 derniers acces
    pub recent_count: u8,          // combien de slots utilises (0-10)
    pub decay_d: f32,              // 0.5 par defaut
    pub cached_activation: f32,    // dernier score calcule
    pub dirty: bool,               // besoin de recalcul ?
}

impl ActivationState {
    /// Calculer B(i) = ln(Σ tⱼ^(-d))
    pub fn compute_activation(&self, now: f64) -> f64 {
        let mut sum = self.historical_sum;
        let d = self.decay_d as f64;

        // Acces recents : calcul exact (bosse Petrov)
        for i in 0..self.recent_count as usize {
            let delta = (now - self.recent_accesses[i]).max(0.001);
            sum += delta.powf(-d);
        }

        if sum <= 0.0 { return f64::NEG_INFINITY; }
        sum.ln()
    }

    /// Enregistrer un acces
    pub fn record_access(&mut self, now: f64) {
        let d = self.decay_d as f64;

        if self.recent_count >= 10 {
            // Evacuer le plus ancien vers l'historique agrege
            let oldest = self.recent_accesses[0];
            let delta = (now - oldest).max(0.001);
            self.historical_sum += delta.powf(-d);
            self.historical_count += 1;
            // Shift left
            self.recent_accesses.copy_within(1..10, 0);
            self.recent_accesses[9] = now;
        } else {
            self.recent_accesses[self.recent_count as usize] = now;
            self.recent_count += 1;
        }
        self.dirty = true;
    }
}
```

### Stabilite numerique

Pour les vieilles memoires (tⱼ tres grand), `tⱼ^(-0.5)` → 0. Risque d'underflow.

Solution : log-sum-exp pour eviter underflow/overflow.

```rust
fn stable_bll_sum(deltas: &[f64], d: f64) -> f64 {
    let log_terms: Vec<f64> = deltas.iter()
        .filter(|&&dt| dt > 0.0)
        .map(|&dt| -d * dt.max(1.0).ln())
        .collect();

    if log_terms.is_empty() { return f64::NEG_INFINITY; }

    let max_log = log_terms.iter().cloned().fold(f64::NEG_INFINITY, f64::max);
    let sum_exp: f64 = log_terms.iter().map(|&l| (l - max_log).exp()).sum();
    max_log + sum_exp.ln()  // = ln(Σ tⱼ^(-d))
}
```

### Parametres exposes

| Parametre | Defaut | Configurable | Description |
|-----------|--------|-------------|-------------|
| `decay_d` | 0.5 | Oui | Vitesse d'oubli. 0.5 = power law standard |
| `recent_window_size` | 10 | Oui | Timestamps exacts conserves |
| `initial_activation` | 0.0 | Oui | Activation a la creation |

**Source :** Petrov 2006, "Computationally Efficient Approximation of the Base-Level Learning Equation"

---

## 2. Spreading Activation

### Formule ACT-R

```
A(i) = B(i) + Σⱼ(Wⱼ · Sⱼᵢ) + ε

Sⱼᵢ = S - ln(fan_i)

ou :
  S = force associative max (defaut: 1.6)
  fan_i = out-degree du noeud source j (nombre d'associations)
  Wⱼ = W_total / n_sources (poids reparti equitablement)
  ε = bruit logistique(0, s²π²/3)
```

### Fan Effect : seuil d'inhibition

Avec S=1.6, quand `fan > e^S ≈ 5` :
- `Sⱼᵢ = 1.6 - ln(fan)` devient **negatif**
- L'activation propage de l'**inhibition** au lieu du renforcement
- C'est voulu : modelise l'interference mnemonique

### Implementation sur CSR

```rust
pub fn spread_activation(
    &self,
    sources: &[(EntityId, f64)],  // (noeud source, poids)
    params: &SpreadingParams,
) -> HashMap<EntityId, f64> {
    let mut activations: HashMap<EntityId, f64> = HashMap::new();
    let mut frontier: Vec<(EntityId, f64, u8)> = Vec::new();

    // Init : sources dans la frontiere
    for &(id, w) in sources {
        frontier.push((id, w, 0));
    }

    while let Some((node, incoming, depth)) = frontier.pop() {
        if depth > params.max_depth { continue; }
        if incoming.abs() < params.cutoff { continue; }

        *activations.entry(node).or_insert(0.0) += incoming;

        // Propager aux voisins (lecture CSR = sequentielle, cache-friendly)
        let neighbors = self.edge_store.outgoing(node);
        let fan = neighbors.len();
        if fan == 0 { continue; }

        let s_ji = params.s_max - (fan as f64).ln();
        let w_j = params.w_total / sources.len() as f64;
        let outgoing = w_j * s_ji;

        if outgoing.abs() > params.cutoff {
            for &(target, _edge_id) in neighbors {
                frontier.push((target, outgoing, depth + 1));
            }
        }
    }

    activations
}
```

### Parametres

| Parametre | Defaut | Description |
|-----------|--------|-------------|
| `s_max` | 1.6 | Force associative max |
| `w_total` | 1.0 | Activation totale des sources |
| `max_depth` | 3 | Profondeur max BFS |
| `cutoff` | 0.01 | Seuil min de propagation |
| `noise_s` | 0.25 | Ecart-type du bruit logistique |

---

## 3. Retrieval Probability

### Sigmoide ACT-R

```
P(i) = 1 / (1 + exp(-(A(i) - τ) / s))

A(i) = B(i) + spreading + ε
τ = seuil de recuperation
s = bruit / sensibilite
```

Quand `A(i) = τ` : P = 50%. Quand `s → 0` : tout-ou-rien.

### Calibration

| Parametre | Defaut ACT-R | Range raisonnable | Impact |
|-----------|-------------|-------------------|--------|
| `τ` (tau) | -0.5 | [-2.0, 2.0] | Seuil de rappel. Bas = plus de rappels |
| `s` (noise) | 0.4 | [0.1, 1.5] | Sensibilite. Haut = courbe aplatie |
| `F` (latency) | 0.63 | [0.1, 2.0] | T = F·exp(-A) en ms |

### Usage dans hora

Le retrieval probability sert a **ponderer les resultats de recherche** :
- Un noeud avec A(i) eleve = plus probable d'etre pertinent
- Le score final combine RRF (hybrid search) × P(i) (activation)
- Les dark nodes ont P(i) ≈ 0 → exclus naturellement

---

## 4. FSRS — Scheduling de renforcement

### Relation avec ACT-R

| Concept | ACT-R | FSRS |
|---------|-------|------|
| Oubli | `tⱼ^(-d)` power law | `R = (1 + F·t/S)^(-w₂₀)` power law |
| Renforcement | +1 terme dans Σ | Stabilite S augmente |
| Difficulte | Bruit s + seuil τ | Parametre D [1,10] |
| Usage | Scoring temps-reel | Planification (quand revoir ?) |

### Formule retrievability

```
R(t, S) = (1 + F · t/S)^(-w₂₀)

F = 0.9^(-1/w₂₀) - 1
w₂₀ ≈ 0.2 (exposant personalise)
t = jours depuis la derniere revue
S = stabilite en jours
```

### Combinaison hybride ACT-R + FSRS

ACT-R repond a **"qui recuperer maintenant ?"** (temps-reel).
FSRS repond a **"quand planifier la prochaine revue ?"** (scheduling).

```rust
pub fn total_activation(&self, now: f64) -> f64 {
    let bll = self.activation.compute_activation(now);
    let spreading = self.spreading_cache;
    let fsrs_modulator = (self.current_retrievability(now) - 0.5) * 2.0;
    bll + spreading + fsrs_modulator
}

pub fn next_review_interval_days(&self, r_desired: f64) -> f64 {
    let w20 = 0.2;
    let f = 0.9_f64.powf(-1.0 / w20) - 1.0;
    self.stability / f * (r_desired.powf(-1.0 / w20) - 1.0)
}
```

**Source :** Expertium 2024, "The FSRS Algorithm"

---

## 5. Reconsolidation Window

### Neuroscience

- Fenetre post-reactivation : **4-6 heures** (protein synthesis dependent)
- Destabilisation requiert reactivation + degradation proteique
- Deux phases : destabilisation rapide → restabilisation lente (6h+)
- La fenetre varie selon la force de la memoire et le contexte

**Source :** Nader et al. 2000, PMC2948875

### State machine

```
┌─────────┐  reactivation forte   ┌─────────┐
│  Stable │ ─────────────────────> │  Labile │
│         │                        │  (4-6h) │
└─────────┘                        └────┬────┘
     ^                                  │
     │  restabilisation (6h)            │ fenetre expiree
     │  + boost stabilite               │
     │                                  v
     │                            ┌──────────────┐
     └─────────────────────────── │ Restabilizing │
                                  └──────────────┘

     reactivation faible : pas de destabilisation, juste BLL record

     ┌─────────┐  activation < seuil pendant >7j  ┌──────┐
     │  Stable │ ────────────────────────────────> │ Dark │
     │         │                                   │      │
     └─────────┘  <─── reactivation forte ──────── └──────┘
                        (recovery)
```

### Etats

```rust
pub enum MemoryState {
    Stable,
    Labile { destabilized_at: f64, strength: f64 },
    Restabilizing { started_at: f64, duration_secs: f64 },
    Dark { silenced_at: f64, peak_activation: f64 },
}
```

### Parametres

| Parametre | Defaut | Description |
|-----------|--------|-------------|
| `labile_window_secs` | 18000 (5h) | Duree de la fenetre plastique |
| `restabilization_secs` | 21600 (6h) | Duree de restabilisation |
| `destabilization_threshold` | 0.5 | Force min pour destabiliser |
| `restabilization_boost` | 1.2 | Boost FSRS apres restabilisation |

---

## 6. Dark Nodes (Oubli actif, Rac1-inspired)

### Mecanisme biologique → computationnel

| Biologie (Rac1) | hora |
|------------------|------|
| Rac1 depolymerise l'actine des spines | Noeud sous le seuil = silenced |
| Spine retraction, synapse affaiblie | Edge weights attenues (×0.1) |
| Inhibition de Rac1 restaure la memoire | Recovery par reactivation forte |
| Engram complet reste inaccessible | Node persiste, juste masked du search |
| Apoptose cellulaire (tres long) | GC apres 90j d'inactivite totale |

### Pipeline dark node

```
1. dark_node_pass() — appele periodiquement (24h)
   ∀ entity :
     activation = compute_activation(now)
     si activation < silencing_threshold
       ET last_access > silencing_delay
       → memory_state = Dark
       → attenue edges sortants ×0.1

2. search() — exclut les dark nodes par defaut
   si opts.include_dark == true → inclut les dark nodes

3. attempt_recovery(entity_id, external_cue)
   si external_activation >= recovery_threshold
     → memory_state = Labile (destabilise pour re-encodage)
     → restaure edges sortants
     → record_access()

4. gc_dead_nodes() — appele manuellement ou au compact()
   ∀ dark node :
     si silenced_at > 90j
       ET aucun acces depuis
       → marque pour suppression physique
```

### Parametres

| Parametre | Defaut | Description |
|-----------|--------|-------------|
| `silencing_threshold` | -2.0 | Activation sous laquelle = dark |
| `silencing_delay_secs` | 604800 (7j) | Delai avant silencing |
| `recovery_threshold` | 1.5 | Activation externe pour recovery |
| `gc_eligible_after_secs` | 7776000 (90j) | Delai avant eligibilite GC |
| `edge_attenuation_factor` | 0.1 | Multiplicateur edges dark |

---

## 7. Dream Cycle (Consolidation CLS)

### Sequence complete

```
dream_cycle() :
  1. SHY Downscaling
     ∀ entity: activation *= 0.78
     (homeostatique, evite la saturation)

  2. Interleaved Replay
     Selectionner mix episodes anciens (30%) + recents (70%)
     Re-activer les entites mentionnees dans ces episodes

  3. CLS Transfer
     ∀ episode avec consolidation_count >= 3 :
       Extraire les faits recurrents
       Creer/renforcer le fait semantique correspondant
       Incrementer consolidation_count

  4. Memory Linking
     ∀ paire d'entites co-creees dans fenetre 6h :
       Creer un edge "temporally_linked" si n'existe pas

  5. Dark Node Check
     ∀ entity: si activation < dark_threshold → mark Dark

  6. GC optionnel
     Dark nodes > 90j sans acces → candidate for delete
```

### Parametres consolidation

| Parametre | Defaut | Description |
|-----------|--------|-------------|
| `enabled` | true | Dream cycle on/off |
| `interval_secs` | 86400 (24h) | Frequence du cycle |
| `downscaling_factor` | 0.78 | SHY: reduction d'activation |
| `replay_mix_old` | 0.30 | % d'episodes anciens dans le replay |
| `cls_repetition_threshold` | 3 | Repetitions avant CLS transfer |
| `linking_window_secs` | 21600 (6h) | Fenetre de co-allocation |
| `max_replay_items` | 100 | Episodes par cycle max |

---

## Decisions prises

| Decision | Justification |
|----------|--------------|
| Approximation Petrov pour BLL | O(1) memoire par entite au lieu de O(n_acces) |
| 10 timestamps recents exacts | Capture la bosse post-acces avec precision |
| Fan effect avec S=1.6 | Standard ACT-R, inhibition au-dela de 5 connexions |
| Spreading limite a depth 3 | Au-dela, activation negligeable avec fan effect |
| FSRS comme modulateur | Complementaire a ACT-R, pas redondant |
| Reconsolidation configurable | Le defaut (5h) est biologique mais ajustable |
| Dark = masked, pas delete | Reversible jusqu'au GC a 90j |
| Dream cycle idempotent | Relancer ne change pas le resultat |

---

## Limites connues

- La fenetre Rac1 biologique opere sur minutes-heures, pas jours → le silencing delay 7j est une extrapolation computationnelle
- La calibration τ/s necessite des donnees empiriques specifiques au KG
- Le CLS transfer est simplifie par rapport aux vrais replay corticaux (pas de gradient descent)
- La combinaison ACT-R + FSRS est nouvelle, pas de litterature validant ce couplage specifique

---

*Document cree le 2026-03-02. Integre les resultats de recherche ACT-R.*
