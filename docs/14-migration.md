# 14 — Migration & Evolution

> Comment evoluer le format sans casser les fichiers existants.
> Comment migrer depuis d'autres systemes.

---

## Evolution du format .hora

### Versioning

Le file header contient :
- `format_version: u16` — version du format du fichier
- `min_read_version: u16` — version minimale pour lire ce fichier

### Strategie de compatibilite

| Changement | Version bump | Compatibilite |
|------------|-------------|---------------|
| Ajout d'un nouveau type de page | Non (si optionnel) | Readers anciens ignorent les pages inconnues |
| Changement de layout d'une page | `format_version++` | Migration automatique a l'ouverture |
| Ajout de champs au header (zone reserved) | Non | Les reserved bytes sont prevus pour ca |
| Changement de la taille des structures | `format_version++` | Migration obligatoire |
| Suppression d'un feature | `min_read_version++` | Anciens readers ne peuvent plus ouvrir |

### Regles

1. **Jamais de breaking change dans une minor version** (v0.1 → v0.2 OK, v0.1.1 → v0.1.2 jamais)
2. **Les reserved bytes dans le header sont pour les evolutions futures**
3. **Migration automatique :** a l'ouverture, si `format_version` est ancien, migrer in-place (ou copie + rename)
4. **Rollback possible :** fournir un outil `hora-migrate` qui peut downgrader (si possible)

### Sequence de migration

```
fn open(path, config):
  header = read_header(path)

  if header.format_version > CURRENT_VERSION:
    return Err(VersionMismatch)  // fichier trop recent

  if header.format_version < CURRENT_VERSION:
    if header.format_version < MIN_SUPPORTED_VERSION:
      return Err(VersionTooOld)  // trop ancien, migration impossible

    // Migration automatique
    migrate(path, header.format_version, CURRENT_VERSION)?

  // Ouvrir normalement
```

### Migrations prevues

| De | Vers | Changement | Migration |
|----|------|-----------|-----------|
| v1 (v0.1) | v2 (v0.5) | Page-based B+ tree | Rewrite complet (export → import) |
| v2 (v0.5) | v3 (v1.0) | Ajout colonnes activation | In-place (ajout pages ActivationLog) |

---

## Import depuis d'autres systemes

### Format d'echange : JSON Lines

```jsonl
{"type":"entity","id":"e1","entity_type":"project","name":"hora","properties":{"language":"Rust"}}
{"type":"entity","id":"e2","entity_type":"concept","name":"knowledge graph","properties":{}}
{"type":"edge","source":"e1","target":"e2","relation":"is_a","description":"hora is a knowledge graph","valid_at":"2024-01-01T00:00:00Z"}
{"type":"episode","source":"import","entities":["e1","e2"],"facts":["f1"]}
```

### Import depuis Neo4j

```
Neo4j → APOC export → JSON/CSV → hora-import
```

```bash
# Depuis Neo4j (Cypher + APOC)
CALL apoc.export.json.all("export.json", {})

# Vers hora (CLI)
hora-import --from neo4j --input export.json --output memory.hora
```

Mapping Neo4j → hora :
| Neo4j | hora |
|-------|------|
| Node | Entity |
| Relationship | Edge (fact) |
| Label | entity_type |
| Relationship type | relation_type |
| Properties | properties |
| (absent) | valid_at, activation |

### Import depuis Mem0

```python
# Extraire les memories de Mem0
memories = mem0.get_all()

# Convertir en hora format
for m in memories:
    hora.add_entity("memory", m.text, {"source": "mem0"}, m.embedding)
```

### Import depuis CSV (generique)

```bash
hora-import --from csv \
  --entities entities.csv \
  --edges edges.csv \
  --output memory.hora
```

Format entities.csv :
```csv
id,type,name,prop_language,prop_stars
1,project,hora,Rust,42
2,language,Rust,,
```

Format edges.csv :
```csv
source_id,target_id,relation,description,valid_at
1,2,built_with,"hora is built with Rust",2024-01-01
```

---

## Export depuis hora

### Export complet

```rust
hora.export_jsonl("export.jsonl")?;
```

### Export partiel (sous-graphe)

```rust
let subgraph = hora.traverse(start_id, TraverseOpts { depth: 3, ..default() })?;
hora.export_subgraph(&subgraph, "partial.jsonl")?;
```

### Export vers Neo4j

```bash
hora-export --to neo4j --input memory.hora --output import.cypher
```

Genere un script Cypher :
```cypher
CREATE (n1:project {name: "hora", language: "Rust"});
CREATE (n2:language {name: "Rust"});
MATCH (a:project {name: "hora"}), (b:language {name: "Rust"})
CREATE (a)-[:built_with {description: "..."}]->(b);
```

---

## Backward compatibility promises

### Ce qu'on garantit

| Depuis la version | Garantie |
|-------------------|----------|
| v1.0 | Format .hora compatible pendant au moins 2 major versions |
| v1.0 | API Rust stable (pas de breaking change sans major bump) |
| v1.0 | Types TypeScript generes stables |

### Ce qu'on ne garantit PAS avant v1.0

- Le format .hora peut changer entre v0.x sans migration automatique
- L'API Rust peut changer (ajout de parametres, renommage)
- Les benchmarks peuvent varier

### Strategie pour les early adopters (v0.x)

```
1. Exporter en JSON Lines avant upgrade
2. Upgrade hora-graph-core
3. Re-importer
```

C'est acceptable car avant v1.0 = "developer preview".

---

## CLI hora-migrate

En v1.0+ :

```bash
# Verifier la version d'un fichier
hora-migrate info memory.hora

# Migrer vers la derniere version
hora-migrate upgrade memory.hora

# Migrer vers une version specifique
hora-migrate upgrade --to-version 3 memory.hora

# Creer un backup avant migration
hora-migrate upgrade --backup memory.hora
# → cree memory.hora.bak.v2

# Verifier l'integrite apres migration
hora-migrate verify memory.hora
```

---

*Document cree le 2026-03-02.*
