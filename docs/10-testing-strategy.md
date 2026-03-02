# 10 — Strategie de Tests

> Tests a chaque couche. Property-based pour la serialisation.
> Snapshot testing pour la regression API.

---

## Pyramide de tests

```
        ┌─────────────┐
        │  Integration │  5-10 tests
        │  (roundtrip) │  Fichier .hora complet
        ├─────────────┤
        │  Component   │  20-30 tests
        │  (module)    │  Storage, Search, Memory
        ├─────────────┤
        │    Unit      │  100+ tests
        │  (function)  │  Chaque fonction publique
        └─────────────┘
```

---

## Tests unitaires par module

### `core/types.rs`

| Test | Verifie |
|------|---------|
| `test_entity_id_display` | EntityId formatte correctement |
| `test_config_default` | HoraConfig::default() a des valeurs sensees |
| `test_property_value_types` | String, Int, Float, Bool conversions |

### `core/entity.rs`

| Test | Verifie |
|------|---------|
| `test_entity_creation` | Champs initialises correctement |
| `test_entity_with_properties` | Properties stockees et recuperables |
| `test_entity_update` | Update partiel fonctionne |
| `test_entity_id_auto_increment` | IDs ne se repetent jamais |

### `core/edge.rs`

| Test | Verifie |
|------|---------|
| `test_edge_creation` | Source/target/relation corrects |
| `test_edge_with_confidence` | Confidence 0-255 mappe sur 0.0-1.0 |
| `test_edge_temporal_defaults` | valid_at=now, invalid_at=0, created_at=now |
| `test_edge_invalidation` | invalid_at set, edge toujours present |

### `core/episode.rs`

| Test | Verifie |
|------|---------|
| `test_episode_creation` | Source, entity_ids, fact_ids stockes |
| `test_episode_ordering` | Episodes tries par timestamp |

### `core/dedup.rs`

| Test | Verifie |
|------|---------|
| `test_exact_name_match` | "hora-engine" == "hora-engine" |
| `test_case_insensitive_match` | "Hora Engine" → normalise → match |
| `test_jaccard_similarity` | Tokens communs detectes |
| `test_cosine_dedup` | Embeddings proches (>0.92) = dedup |
| `test_different_entities` | Noms et embeddings differents = pas dedup |

### `search/vector.rs`

| Test | Verifie |
|------|---------|
| `test_cosine_similarity_basic` | cos([1,0],[0,1]) = 0.0 |
| `test_cosine_similarity_identical` | cos(a,a) = 1.0 |
| `test_cosine_avx2_vs_scalar` | Memes resultats (tolerance 1e-5) |
| `test_cosine_neon_vs_scalar` | Memes resultats (tolerance 1e-5) |
| `test_vector_search_topk` | Retourne exactement k resultats |
| `test_vector_search_dimension_mismatch` | Erreur DimensionMismatch |
| `test_vector_search_empty` | 0 vecteurs → resultats vides |

### `search/bm25.rs`

| Test | Verifie |
|------|---------|
| `test_tokenizer_basic` | "Hello World" → ["hello", "world"] |
| `test_tokenizer_stopwords` | "the", "a", "is" filtres |
| `test_bm25_single_term` | Terme exact trouve le document |
| `test_bm25_tf_ranking` | Plus d'occurrences = score plus haut |
| `test_bm25_idf_ranking` | Terme rare = score plus haut |
| `test_bm25_no_text_entity` | Entite sans texte invisible au BM25 |
| `test_vbyte_roundtrip` | Encode/decode u32 identique |

### `search/hybrid.rs`

| Test | Verifie |
|------|---------|
| `test_rrf_fusion_basic` | Les deux legs combinees |
| `test_rrf_both_legs_higher` | Trouve par 2 legs = score plus haut |
| `test_rrf_single_leg` | Trouve par 1 seule leg = present mais score bas |
| `test_rrf_text_only_mode` | embedding_dims=0 fonctionne |
| `test_rrf_activation_boost` | Noeud actif booste dans le ranking |

### `search/traversal.rs`

| Test | Verifie |
|------|---------|
| `test_bfs_depth_0` | Retourne seulement le noeud source |
| `test_bfs_depth_1` | Retourne source + voisins directs |
| `test_bfs_depth_2` | A→B→C→D, depth=2 → {A,B,C} pas D |
| `test_bfs_cycle` | Cycle A→B→A ne boucle pas |
| `test_timeline_ordering` | Faits tries par valid_at |
| `test_facts_at_bitemporal` | Filtre correct valid_at/invalid_at |

### `memory/activation.rs`

| Test | Verifie |
|------|---------|
| `test_bll_single_access` | Activation apres 1 acces |
| `test_bll_decay` | Activation diminue avec le temps |
| `test_bll_multiple_accesses` | Plus d'acces = activation plus haute |
| `test_petrov_approximation` | Hybrid = exact (tolerance 5%) |
| `test_record_access_overflow` | >10 acces evacue vers historique |
| `test_spreading_activation` | Voisins recoivent de l'activation |
| `test_fan_effect_inhibition` | Fan > 5 → activation negative |

### `memory/dark_nodes.rs`

| Test | Verifie |
|------|---------|
| `test_dark_node_detection` | Activation < seuil → dark |
| `test_dark_node_excluded_search` | Dark node invisible au search |
| `test_dark_node_recovery` | Reactivation forte → recovery |
| `test_dark_node_gc` | >90j dark → eligible GC |

### `temporal/bitemporal.rs`

| Test | Verifie |
|------|---------|
| `test_facts_at_basic` | Filtre correct |
| `test_facts_at_invalidated` | Fait invalide exclu |
| `test_timeline_complete` | Tous les faits d'une entite |
| `test_timeline_ordered` | Tri par valid_at |

### `storage/memory.rs`

| Test | Verifie |
|------|---------|
| `test_memory_crud_entity` | Insert/get/update/delete |
| `test_memory_crud_edge` | Insert/get/delete |
| `test_memory_stats` | Compteurs corrects |

### `storage/embedded/`

| Test | Verifie |
|------|---------|
| `test_persistence_roundtrip` | Create→flush→close→open→verify |
| `test_wal_recovery` | Crash simule → recovery |
| `test_corrupted_magic` | Magic invalide → erreur |
| `test_version_check` | Version incompatible → erreur |
| `test_checkpoint` | WAL→fichier principal |
| `test_concurrent_readers` | Readers multi-thread OK |

---

## Property-based testing (proptest)

### Ou utiliser proptest

| Domaine | Propriete testee |
|---------|-----------------|
| Serialisation binaire | `from_bytes(to_bytes(x)) == x` pour tout x |
| B+ tree | Insert N random keys → get chacune retrouve la bonne valeur |
| String pool | Toute string inseree est recuperable identique |
| VByte encoding | `decode(encode(n)) == n` pour tout u32 |
| CRC32 | Toute modification d'1 bit change le checksum |
| BM25 tokenizer | Tokenize est idempotent : `tokenize(s)` toujours meme resultat |

### Exemple proptest

```rust
use proptest::prelude::*;

proptest! {
    #[test]
    fn entity_roundtrip(
        name in "[a-zA-Z0-9_-]{1,100}",
        entity_type in "[a-z]{1,20}",
    ) {
        let bytes = entity.to_bytes();
        let restored = Entity::from_bytes(&bytes).unwrap();
        prop_assert_eq!(restored.name, name);
        prop_assert_eq!(restored.entity_type, entity_type);
    }

    #[test]
    fn vbyte_roundtrip(n in 0u32..u32::MAX) {
        let mut buf = Vec::new();
        vbyte_encode(n, &mut buf);
        let decoded = vbyte_decode(&buf).unwrap();
        prop_assert_eq!(decoded, n);
    }
}
```

---

## Snapshot testing (insta)

### Quand utiliser insta

- Output d'API stabilisee (JSON de search results, stats)
- Format binaire du file header
- TypeScript generated types

```rust
use insta::assert_snapshot;

#[test]
fn test_search_output_format() {
    let mut hora = test_hora_with_data();
    let results = hora.search(Some("auth"), None, SearchOpts::default()).unwrap();
    assert_snapshot!(format!("{:#?}", results));
}

#[test]
fn test_stats_format() {
    let hora = test_hora_with_data();
    let stats = hora.stats().unwrap();
    assert_snapshot!(format!("{:#?}", stats));
}
```

---

## Tests d'integration

### Scenario complet : cycle de vie

```rust
#[test]
fn test_full_lifecycle() {
    let dir = tempdir().unwrap();
    let path = dir.path().join("test.hora");
    let config = HoraConfig { embedding_dims: 4, ..default() };

    // Phase 1 : Creer et populer
    {
        let mut hora = HoraCore::open(&path, config).unwrap();
        let a = hora.add_entity("project", "hora", None, None).unwrap();
        let b = hora.add_entity("language", "Rust", None, None).unwrap();
        hora.add_fact(a, b, "built_with", "hora is built with Rust", None).unwrap();
        hora.flush().unwrap();
    }

    // Phase 2 : Rouvrir et verifier
    {
        let hora = HoraCore::open(&path, config).unwrap();
        let stats = hora.stats().unwrap();
        assert_eq!(stats.entities, 2);
        assert_eq!(stats.facts, 1);
    }

    // Phase 3 : Modifier et compacter
    {
        let mut hora = HoraCore::open(&path, config).unwrap();
        let entities: Vec<_> = hora.scan_entities().unwrap().collect();
        hora.delete_entity(entities[0].id).unwrap();
        hora.compact().unwrap();
        hora.flush().unwrap();
    }

    // Phase 4 : Verifier apres compaction
    {
        let hora = HoraCore::open(&path, config).unwrap();
        let stats = hora.stats().unwrap();
        assert_eq!(stats.entities, 1);
        assert_eq!(stats.facts, 0); // cascade delete
    }
}
```

### Scenario : crash recovery

```rust
#[test]
fn test_crash_recovery() {
    let dir = tempdir().unwrap();
    let path = dir.path().join("test.hora");

    // Creer et flusher
    {
        let mut hora = HoraCore::open(&path, config).unwrap();
        hora.add_entity("project", "hora", None, None).unwrap();
        hora.flush().unwrap();

        // Ajouter sans flusher (simule crash)
        hora.add_entity("language", "Rust", None, None).unwrap();
        // drop sans flush → WAL contient le 2eme entity
    }

    // Recovery
    {
        let hora = HoraCore::open(&path, config).unwrap();
        let stats = hora.stats().unwrap();
        // Le WAL devrait etre rejoue → 2 entities
        assert_eq!(stats.entities, 2);
    }
}
```

---

## Benchmarks comme tests

Chaque benchmark a un seuil de regression :

```rust
// Si le bench depasse 2x le seuil, c'est un test fail
#[test]
fn test_no_perf_regression_insert() {
    let start = Instant::now();
    for i in 0..10_000 {
        hora.add_entity("test", &format!("entity_{}", i), None, None).unwrap();
    }
    let elapsed = start.elapsed();
    // 10K inserts en < 1 seconde = >10K ops/sec minimum
    assert!(elapsed < Duration::from_secs(1),
        "insert regression: {:?} for 10K ops", elapsed);
}
```

---

## CI Pipeline

```yaml
# .github/workflows/ci.yml
steps:
  - cargo clippy -- -D warnings
  - cargo test
  - cargo test --features hnsw
  - cargo bench --no-run  # verifie la compilation
  - cargo doc --no-deps
```

---

## Decisions prises

| Decision | Justification |
|----------|--------------|
| proptest pour serialisation | Le format binaire est critique, fuzzing detecte les edge cases |
| insta pour API output | Detecte les regressions involontaires |
| Pas de mocking | On teste les vrais backends (Memory, fichier temp) |
| Seuils de perf dans les tests | Previent les regressions silencieuses |
| Tests par module isolable | Chaque module testable independamment |

---

*Document cree le 2026-03-02.*
