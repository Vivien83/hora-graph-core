# 06 — Modele Bi-Temporel

> Deux axes temporels : valid_at (monde reel) et created_at (systeme).
> Chaque fait a une fenetre de validite, jamais de suppression logique.

---

## Pourquoi bi-temporel ?

### Probleme

Un knowledge graph classique (Neo4j, Kuzu) stocke l'etat courant. Quand un fait change, l'ancien est perdu.

Exemple :
- "Alice travaille chez Acme" (vrai de 2020 a 2023)
- "Alice travaille chez BigCorp" (vrai depuis 2023)

Sans temporal : on ne peut pas repondre a "ou travaillait Alice en 2021 ?"

### Solution : deux axes

| Axe | Nom | Semantique | Controle |
|-----|-----|-----------|----------|
| **World time** | `valid_at` / `invalid_at` | Quand le fait est vrai dans le monde reel | L'utilisateur |
| **System time** | `created_at` | Quand le fait a ete enregistre dans hora | Le systeme |

```
                    valid_at ──────────────────>

created_at │  ┌──────────────────────────────┐
           │  │ "Alice works at Acme"         │
           │  │ valid_at: 2020-01  invalid_at: 2023-06 │
           │  └──────────────────────────────┘
           │
           │  ┌──────────────────────────────────────┐
           │  │ "Alice works at BigCorp"              │
           │  │ valid_at: 2023-06  invalid_at: MAX    │
           │  └──────────────────────────────────────┘
           ▼
```

---

## Representation dans Edge

```rust
#[repr(C)]
pub struct Edge {
    pub id: EdgeId,           // u64
    pub source: EntityId,     // u64
    pub target: EntityId,     // u64
    pub relation_type: u8,    // index dans la table de types
    pub confidence: u8,       // 0-255, mappe sur 0.0-1.0
    pub valid_at: i64,        // epoch millis — debut de validite monde
    pub invalid_at: i64,      // epoch millis — fin de validite monde (0 = toujours valide)
    pub created_at: i64,      // epoch millis — quand insere dans hora (systeme)
    pub description_offset: u32, // offset dans string pool
    pub description_len: u16,    // longueur description
    pub _padding: [u8; 2],       // alignement
}
// Total: 56 bytes
```

### Semantique des timestamps

| Champ | Valeur par defaut | Signification |
|-------|-------------------|---------------|
| `valid_at` | `now()` | Le fait est vrai depuis maintenant |
| `invalid_at` | `0` | Le fait est toujours valide (pas de date de fin) |
| `created_at` | `now()` (auto) | Timestamp systeme d'insertion |

### `invalid_at` = 0 signifie "pas encore invalide"

Convention : `invalid_at == 0` → le fait est actuellement valide.
C'est plus cache-friendly que `i64::MAX` (pas de constante magique) et se teste avec une simple comparaison a zero.

---

## Operations temporelles

### `invalidate_fact(fact_id)` — Marquer un fait comme invalide

```rust
pub fn invalidate_fact(&mut self, fact_id: EdgeId) -> Result<()> {
    let mut inner = self.inner.write().unwrap();
    let edge = inner.storage.get_edge(fact_id)?
        .ok_or(HoraError::EdgeNotFound(fact_id))?;

    if edge.invalid_at != 0 {
        return Err(HoraError::AlreadyInvalidated(fact_id));
    }

    inner.storage.update_edge(fact_id, EdgeUpdate {
        invalid_at: Some(now_millis()),
        ..Default::default()
    })?;

    Ok(())
}
```

**Important :** `invalidate_fact()` ne supprime rien. Le fait reste dans la base avec sa fenetre de validite. C'est la suppression "logique" bi-temporelle.

### `delete_fact(fact_id)` — Suppression physique

Pour les cas ou le fait doit vraiment disparaitre (RGPD, erreur de saisie).

```rust
pub fn delete_fact(&mut self, fact_id: EdgeId) -> Result<()> {
    let mut inner = self.inner.write().unwrap();
    inner.storage.delete_edge(fact_id)?;
    Ok(())
}
```

### `facts_at(timestamp)` — Etat du monde a un instant t

```rust
pub fn facts_at(&self, t: i64) -> Result<Vec<Edge>> {
    let inner = self.inner.read().unwrap();
    inner.storage.scan_edges_temporal(|edge| {
        edge.valid_at <= t && (edge.invalid_at == 0 || edge.invalid_at > t)
    })
}
```

### `timeline(entity_id)` — Historique complet d'une entite

```rust
pub fn timeline(&self, entity_id: EntityId) -> Result<Vec<Edge>> {
    let inner = self.inner.read().unwrap();
    let mut edges = inner.storage.scan_edges_for_entity(entity_id)?;
    edges.sort_by_key(|e| e.valid_at);
    Ok(edges)
}
```

---

## Index temporel

### Structure : B+ tree sur valid_at

```
Index temporel = B+ tree triant les EdgeId par valid_at

Requete facts_at(t) :
  1. B+ tree seek : trouver la premiere entree avec valid_at <= t
  2. Scanner sequentiellement
  3. Filtrer : invalid_at == 0 || invalid_at > t

Complexite : O(log n) + O(k) ou k = nombre de faits valides a l'instant t
```

### Pages TemporalIndex

```
TemporalIndex page layout:
  ┌──────────────────────────────────────────┐
  │ PageHeader (8B)                          │
  │ Entry_0: valid_at(8B) edge_id(8B) = 16B │
  │ Entry_1: ...                              │
  │ ...                                       │
  └──────────────────────────────────────────┘
  Capacite: 4088 / 16 = 255 entrees par page
```

### Optimisation : interval tree (v0.5+)

Pour les requetes de type "quels faits sont valides entre t1 et t2", un interval tree serait plus efficace. En v0.1, le scan avec filtre suffit pour < 100K edges.

---

## Bi-temporel et ACT-R

L'activation ACT-R utilise `created_at` (pas `valid_at`) pour le calcul BLL. Raison : l'activation modelise la memoire du systeme, pas la validite du monde reel.

```
record_access() utilise now() → impacte BLL via ActivationState
invalidate_fact() utilise now() pour invalid_at → impacte facts_at() et timeline()
```

Les deux sont orthogonaux : un fait invalide peut encore avoir une haute activation (acces recent), et un fait valide peut avoir une faible activation (jamais consulte).

---

## Requetes temporelles avancees (v0.5+)

### Point-in-time query

```rust
// Etat du graphe au 1er janvier 2024
let snapshot = hora.facts_at(timestamp("2024-01-01"))?;
```

### Range query

```rust
// Tous les faits valides entre t1 et t2
let facts = hora.facts_between(t1, t2)?;
```

### Diff query

```rust
// Qu'est-ce qui a change entre t1 et t2 ?
let diff = hora.facts_diff(t1, t2)?;
// → { added: [...], invalidated: [...] }
```

### As-of query (bi-temporel complet)

```rust
// Etat du monde a valid_time, tel que connu a system_time
let facts = hora.facts_as_of(valid_time, system_time)?;
```

---

## Interaction avec d'autres modules

| Module | Utilise valid_at | Utilise created_at | Utilise invalid_at |
|--------|-----------------|-------------------|-------------------|
| `search/` | Filtre optionnel `at_time` | Non | Exclure invalides |
| `memory/activation` | Non | Oui (pour BLL timing) | Non |
| `memory/consolidation` | Non | Oui (age des episodes) | Non |
| `core/dedup` | Non | Non | Check si l'existant est invalide |
| `core/episode` | Non | Oui (episode timestamp) | Non |

---

## Decisions prises

| Decision | Justification |
|----------|--------------|
| `invalid_at = 0` pour "valide" | Plus naturel que i64::MAX, zero-check rapide |
| Pas de delete logique distinct | `invalidate_fact()` est le delete logique |
| B+ tree sur valid_at | Suffisant pour v0.1, interval tree en v0.5 |
| created_at auto-set | L'utilisateur ne controle pas le system time |
| BLL utilise created_at | L'activation modelise la memoire systeme |

---

*Document cree le 2026-03-02.*
