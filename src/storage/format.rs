//! Binary serialization format for .hora files (v0.1 simple format).
//!
//! Layout:
//!   [FileHeader: 48 bytes]
//!   [Entity]*entity_count
//!   [Edge]*edge_count
//!   [Episode]*episode_count
//!
//! All multi-byte integers are little-endian. Strings are length-prefixed (u32 + bytes).

use std::io::{self, Read, Write};

use crate::core::edge::Edge;
use crate::core::entity::Entity;
use crate::core::episode::Episode;
use crate::core::types::{
    EdgeId, EntityId, EpisodeSource, Properties, PropertyValue,
};
use crate::error::{HoraError, Result};

const MAGIC: [u8; 4] = *b"HORA";
const FORMAT_VERSION: u16 = 1;

/// Metadata stored in the file header.
pub struct FileHeader {
    pub embedding_dims: u16,
    pub next_entity_id: u64,
    pub next_edge_id: u64,
    pub next_episode_id: u64,
    pub entity_count: u32,
    pub edge_count: u32,
    pub episode_count: u32,
}

// ── Write helpers ──────────────────────────────────────────

fn write_u8(w: &mut impl Write, v: u8) -> io::Result<()> {
    w.write_all(&[v])
}
fn write_u16(w: &mut impl Write, v: u16) -> io::Result<()> {
    w.write_all(&v.to_le_bytes())
}
fn write_u32(w: &mut impl Write, v: u32) -> io::Result<()> {
    w.write_all(&v.to_le_bytes())
}
fn write_u64(w: &mut impl Write, v: u64) -> io::Result<()> {
    w.write_all(&v.to_le_bytes())
}
fn write_i64(w: &mut impl Write, v: i64) -> io::Result<()> {
    w.write_all(&v.to_le_bytes())
}
fn write_f32(w: &mut impl Write, v: f32) -> io::Result<()> {
    w.write_all(&v.to_le_bytes())
}
fn write_str(w: &mut impl Write, s: &str) -> io::Result<()> {
    write_u32(w, s.len() as u32)?;
    w.write_all(s.as_bytes())
}

// ── Read helpers ───────────────────────────────────────────

fn read_u8(r: &mut impl Read) -> io::Result<u8> {
    let mut buf = [0u8; 1];
    r.read_exact(&mut buf)?;
    Ok(buf[0])
}
fn read_u16(r: &mut impl Read) -> io::Result<u16> {
    let mut buf = [0u8; 2];
    r.read_exact(&mut buf)?;
    Ok(u16::from_le_bytes(buf))
}
fn read_u32(r: &mut impl Read) -> io::Result<u32> {
    let mut buf = [0u8; 4];
    r.read_exact(&mut buf)?;
    Ok(u32::from_le_bytes(buf))
}
fn read_u64(r: &mut impl Read) -> io::Result<u64> {
    let mut buf = [0u8; 8];
    r.read_exact(&mut buf)?;
    Ok(u64::from_le_bytes(buf))
}
fn read_i64(r: &mut impl Read) -> io::Result<i64> {
    let mut buf = [0u8; 8];
    r.read_exact(&mut buf)?;
    Ok(i64::from_le_bytes(buf))
}
fn read_f32(r: &mut impl Read) -> io::Result<f32> {
    let mut buf = [0u8; 4];
    r.read_exact(&mut buf)?;
    Ok(f32::from_le_bytes(buf))
}
fn read_string(r: &mut impl Read) -> io::Result<String> {
    let len = read_u32(r)? as usize;
    let mut buf = vec![0u8; len];
    r.read_exact(&mut buf)?;
    String::from_utf8(buf).map_err(|e| io::Error::new(io::ErrorKind::InvalidData, e))
}

// ── PropertyValue tags ─────────────────────────────────────

const TAG_STRING: u8 = 0;
const TAG_INT: u8 = 1;
const TAG_FLOAT: u8 = 2;
const TAG_BOOL: u8 = 3;

fn write_property_value(w: &mut impl Write, v: &PropertyValue) -> io::Result<()> {
    match v {
        PropertyValue::String(s) => {
            write_u8(w, TAG_STRING)?;
            write_str(w, s)?;
        }
        PropertyValue::Int(n) => {
            write_u8(w, TAG_INT)?;
            write_i64(w, *n)?;
        }
        PropertyValue::Float(f) => {
            write_u8(w, TAG_FLOAT)?;
            w.write_all(&f.to_le_bytes())?;
        }
        PropertyValue::Bool(b) => {
            write_u8(w, TAG_BOOL)?;
            write_u8(w, u8::from(*b))?;
        }
    }
    Ok(())
}

fn read_property_value(r: &mut impl Read) -> io::Result<PropertyValue> {
    let tag = read_u8(r)?;
    match tag {
        TAG_STRING => Ok(PropertyValue::String(read_string(r)?)),
        TAG_INT => Ok(PropertyValue::Int(read_i64(r)?)),
        TAG_FLOAT => {
            let mut buf = [0u8; 8];
            r.read_exact(&mut buf)?;
            Ok(PropertyValue::Float(f64::from_le_bytes(buf)))
        }
        TAG_BOOL => {
            let b = read_u8(r)?;
            Ok(PropertyValue::Bool(b != 0))
        }
        _ => Err(io::Error::new(
            io::ErrorKind::InvalidData,
            format!("unknown property tag: {}", tag),
        )),
    }
}

pub(crate) fn write_properties(w: &mut impl Write, props: &Properties) -> io::Result<()> {
    write_u32(w, props.len() as u32)?;
    for (key, val) in props {
        write_str(w, key)?;
        write_property_value(w, val)?;
    }
    Ok(())
}

pub(crate) fn read_properties(r: &mut impl Read) -> io::Result<Properties> {
    let count = read_u32(r)? as usize;
    let mut map = Properties::with_capacity(count);
    for _ in 0..count {
        let key = read_string(r)?;
        let val = read_property_value(r)?;
        map.insert(key, val);
    }
    Ok(map)
}

// ── EpisodeSource ──────────────────────────────────────────

fn episode_source_to_u8(s: &EpisodeSource) -> u8 {
    match s {
        EpisodeSource::Conversation => 0,
        EpisodeSource::Document => 1,
        EpisodeSource::Api => 2,
    }
}

fn u8_to_episode_source(v: u8) -> io::Result<EpisodeSource> {
    match v {
        0 => Ok(EpisodeSource::Conversation),
        1 => Ok(EpisodeSource::Document),
        2 => Ok(EpisodeSource::Api),
        _ => Err(io::Error::new(
            io::ErrorKind::InvalidData,
            format!("unknown episode source: {}", v),
        )),
    }
}

// ── Entity ─────────────────────────────────────────────────

fn write_entity(w: &mut impl Write, e: &Entity) -> io::Result<()> {
    write_u64(w, e.id.0)?;
    write_str(w, &e.entity_type)?;
    write_str(w, &e.name)?;
    write_properties(w, &e.properties)?;
    match &e.embedding {
        Some(emb) => {
            write_u8(w, 1)?;
            write_u32(w, emb.len() as u32)?;
            for &v in emb {
                write_f32(w, v)?;
            }
        }
        None => {
            write_u8(w, 0)?;
        }
    }
    write_i64(w, e.created_at)?;
    Ok(())
}

fn read_entity(r: &mut impl Read) -> io::Result<Entity> {
    let id = EntityId(read_u64(r)?);
    let entity_type = read_string(r)?;
    let name = read_string(r)?;
    let properties = read_properties(r)?;
    let has_embedding = read_u8(r)?;
    let embedding = if has_embedding != 0 {
        let len = read_u32(r)? as usize;
        let mut emb = Vec::with_capacity(len);
        for _ in 0..len {
            emb.push(read_f32(r)?);
        }
        Some(emb)
    } else {
        None
    };
    let created_at = read_i64(r)?;
    Ok(Entity {
        id,
        entity_type,
        name,
        properties,
        embedding,
        created_at,
    })
}

// ── Edge ───────────────────────────────────────────────────

fn write_edge(w: &mut impl Write, e: &Edge) -> io::Result<()> {
    write_u64(w, e.id.0)?;
    write_u64(w, e.source.0)?;
    write_u64(w, e.target.0)?;
    write_str(w, &e.relation_type)?;
    write_str(w, &e.description)?;
    write_f32(w, e.confidence)?;
    write_i64(w, e.valid_at)?;
    write_i64(w, e.invalid_at)?;
    write_i64(w, e.created_at)?;
    Ok(())
}

fn read_edge(r: &mut impl Read) -> io::Result<Edge> {
    let id = EdgeId(read_u64(r)?);
    let source = EntityId(read_u64(r)?);
    let target = EntityId(read_u64(r)?);
    let relation_type = read_string(r)?;
    let description = read_string(r)?;
    let confidence = read_f32(r)?;
    let valid_at = read_i64(r)?;
    let invalid_at = read_i64(r)?;
    let created_at = read_i64(r)?;
    Ok(Edge {
        id,
        source,
        target,
        relation_type,
        description,
        confidence,
        valid_at,
        invalid_at,
        created_at,
    })
}

// ── Episode ────────────────────────────────────────────────

fn write_episode(w: &mut impl Write, ep: &Episode) -> io::Result<()> {
    write_u64(w, ep.id)?;
    write_u8(w, episode_source_to_u8(&ep.source))?;
    write_str(w, &ep.session_id)?;
    write_u32(w, ep.entity_ids.len() as u32)?;
    for eid in &ep.entity_ids {
        write_u64(w, eid.0)?;
    }
    write_u32(w, ep.fact_ids.len() as u32)?;
    for fid in &ep.fact_ids {
        write_u64(w, fid.0)?;
    }
    write_i64(w, ep.created_at)?;
    write_u32(w, ep.consolidation_count)?;
    Ok(())
}

fn read_episode(r: &mut impl Read) -> io::Result<Episode> {
    let id = read_u64(r)?;
    let source = u8_to_episode_source(read_u8(r)?)?;
    let session_id = read_string(r)?;
    let entity_count = read_u32(r)? as usize;
    let mut entity_ids = Vec::with_capacity(entity_count);
    for _ in 0..entity_count {
        entity_ids.push(EntityId(read_u64(r)?));
    }
    let fact_count = read_u32(r)? as usize;
    let mut fact_ids = Vec::with_capacity(fact_count);
    for _ in 0..fact_count {
        fact_ids.push(EdgeId(read_u64(r)?));
    }
    let created_at = read_i64(r)?;
    let consolidation_count = read_u32(r)?;
    Ok(Episode {
        id,
        source,
        session_id,
        entity_ids,
        fact_ids,
        created_at,
        consolidation_count,
    })
}

// ── Top-level serialize / deserialize ──────────────────────

/// Serialize a complete graph state to a writer.
pub fn serialize(
    w: &mut impl Write,
    header: &FileHeader,
    entities: &[Entity],
    edges: &[Edge],
    episodes: &[Episode],
) -> Result<()> {
    // Header (48 bytes)
    w.write_all(&MAGIC)?;
    write_u16(w, FORMAT_VERSION)?;
    write_u16(w, header.embedding_dims)?;
    write_u64(w, header.next_entity_id)?;
    write_u64(w, header.next_edge_id)?;
    write_u64(w, header.next_episode_id)?;
    write_u32(w, header.entity_count)?;
    write_u32(w, header.edge_count)?;
    write_u32(w, header.episode_count)?;
    w.write_all(&[0u8; 4])?; // reserved

    // Entities
    for entity in entities {
        write_entity(w, entity)?;
    }

    // Edges
    for edge in edges {
        write_edge(w, edge)?;
    }

    // Episodes
    for episode in episodes {
        write_episode(w, episode)?;
    }

    Ok(())
}

/// Returned data from deserializing a .hora file.
pub struct DeserializedGraph {
    pub header: FileHeader,
    pub entities: Vec<Entity>,
    pub edges: Vec<Edge>,
    pub episodes: Vec<Episode>,
}

/// Deserialize a complete graph state from a reader.
pub fn deserialize(r: &mut impl Read) -> Result<DeserializedGraph> {
    // Magic
    let mut magic = [0u8; 4];
    r.read_exact(&mut magic)?;
    if magic != MAGIC {
        return Err(HoraError::InvalidFile {
            reason: "not a .hora file (bad magic)",
        });
    }

    // Version
    let version = read_u16(r)?;
    if version != FORMAT_VERSION {
        return Err(HoraError::VersionMismatch {
            file_version: version,
            min_supported: FORMAT_VERSION,
            max_supported: FORMAT_VERSION,
        });
    }

    let embedding_dims = read_u16(r)?;
    let next_entity_id = read_u64(r)?;
    let next_edge_id = read_u64(r)?;
    let next_episode_id = read_u64(r)?;
    let entity_count = read_u32(r)?;
    let edge_count = read_u32(r)?;
    let episode_count = read_u32(r)?;

    // Reserved
    let mut reserved = [0u8; 4];
    r.read_exact(&mut reserved)?;

    let header = FileHeader {
        embedding_dims,
        next_entity_id,
        next_edge_id,
        next_episode_id,
        entity_count,
        edge_count,
        episode_count,
    };

    // Entities
    let mut entities = Vec::with_capacity(entity_count as usize);
    for _ in 0..entity_count {
        entities.push(read_entity(r)?);
    }

    // Edges
    let mut edges = Vec::with_capacity(edge_count as usize);
    for _ in 0..edge_count {
        edges.push(read_edge(r)?);
    }

    // Episodes
    let mut episodes = Vec::with_capacity(episode_count as usize);
    for _ in 0..episode_count {
        episodes.push(read_episode(r)?);
    }

    Ok(DeserializedGraph {
        header,
        entities,
        edges,
        episodes,
    })
}
