# Research: SIMD Cross-Platform en Rust

> Resultats de la recherche sur les strategies SIMD portables en Rust stable.

---

## Etat de l'art (mars 2026)

### `std::arch` (stable)
- **Stable depuis Rust 1.27** pour x86/x86_64
- **Stable pour AArch64** (NEON) depuis Rust 1.59
- **Stable pour WASM SIMD** depuis Rust 1.60 (target_feature "simd128")
- Chaque architecture a son propre module : `std::arch::x86_64`, `std::arch::aarch64`
- Acces direct aux intrinsics hardware

### `std::simd` (NIGHTLY ONLY)
- **Toujours nightly** en mars 2026
- API portable : meme code pour toutes les architectures
- Ne pas utiliser pour un projet visant stable

**Decision :** utiliser `std::arch` (stable) avec dispatch runtime.

---

## Strategies de dispatch

### 1. Compile-time dispatch (`cfg(target_arch)`)
```rust
#[cfg(target_arch = "x86_64")]
fn cosine_simd(a: &[f32], b: &[f32]) -> f32 {
    unsafe { cosine_avx2(a, b) }
}

#[cfg(target_arch = "aarch64")]
fn cosine_simd(a: &[f32], b: &[f32]) -> f32 {
    unsafe { cosine_neon(a, b) }
}
```
**Probleme :** ne detecte pas si le CPU supporte reellement AVX2 (vieux x86_64 sans AVX2).

### 2. Runtime dispatch via `is_x86_feature_detected!` + OnceLock
```rust
use std::sync::OnceLock;

type CosineFn = fn(&[f32], &[f32]) -> f32;

static COSINE_FN: OnceLock<CosineFn> = OnceLock::new();

fn cosine(a: &[f32], b: &[f32]) -> f32 {
    let f = COSINE_FN.get_or_init(|| {
        #[cfg(target_arch = "x86_64")]
        {
            if is_x86_feature_detected!("avx2") && is_x86_feature_detected!("fma") {
                return cosine_avx2 as CosineFn;
            }
        }
        #[cfg(target_arch = "aarch64")]
        {
            return cosine_neon as CosineFn;
        }
        cosine_scalar as CosineFn
    });
    f(a, b)
}
```
**Avantage :** detecte les features reelles du CPU. OnceLock = initialise une seule fois.
**Overhead :** 1 indirection (function pointer) = negligeable.

**Decision retenue :** Option 2 (runtime dispatch via OnceLock).

---

## Implementations par architecture

### AVX2 (x86_64)

```rust
#[target_feature(enable = "avx2,fma")]
unsafe fn cosine_avx2(a: &[f32], b: &[f32]) -> f32 {
    use std::arch::x86_64::*;

    let n = a.len();
    let mut dot = _mm256_setzero_ps();   // 4 accumulateurs
    let mut dot2 = _mm256_setzero_ps();
    let mut norm_a = _mm256_setzero_ps();
    let mut norm_b = _mm256_setzero_ps();

    let chunks = n / 16;  // 16 f32 par iteration (2 × 256-bit)
    for i in 0..chunks {
        let offset = i * 16;
        let va1 = _mm256_loadu_ps(a.as_ptr().add(offset));
        let vb1 = _mm256_loadu_ps(b.as_ptr().add(offset));
        let va2 = _mm256_loadu_ps(a.as_ptr().add(offset + 8));
        let vb2 = _mm256_loadu_ps(b.as_ptr().add(offset + 8));

        dot = _mm256_fmadd_ps(va1, vb1, dot);
        dot2 = _mm256_fmadd_ps(va2, vb2, dot2);
        norm_a = _mm256_fmadd_ps(va1, va1, norm_a);
        norm_b = _mm256_fmadd_ps(vb1, vb1, norm_b);
    }

    // Horizontal sum + scalar remainder...
    // dot + dot2 → single sum
    // return dot_sum / (norm_a_sum.sqrt() * norm_b_sum.sqrt())
}
```

**Performance :** 16 f32/iteration avec FMA. Pour 384d : ~24 iterations = ~48 instructions.

### NEON (AArch64)

```rust
#[target_feature(enable = "neon")]
unsafe fn cosine_neon(a: &[f32], b: &[f32]) -> f32 {
    use std::arch::aarch64::*;

    let n = a.len();
    let mut dot = vdupq_n_f32(0.0);   // 4 × f32
    let mut norm_a = vdupq_n_f32(0.0);
    let mut norm_b = vdupq_n_f32(0.0);

    let chunks = n / 4;
    for i in 0..chunks {
        let offset = i * 4;
        let va = vld1q_f32(a.as_ptr().add(offset));
        let vb = vld1q_f32(b.as_ptr().add(offset));

        dot = vfmaq_f32(dot, va, vb);     // FMA natif sur NEON
        norm_a = vfmaq_f32(norm_a, va, va);
        norm_b = vfmaq_f32(norm_b, vb, vb);
    }

    let dot_sum = vaddvq_f32(dot);         // horizontal sum
    let na_sum = vaddvq_f32(norm_a);
    let nb_sum = vaddvq_f32(norm_b);

    dot_sum / (na_sum.sqrt() * nb_sum.sqrt())
}
```

**Note :** NEON FMA (vfmaq_f32) est natif sur ARMv8. `vaddvq_f32` pour le horizontal sum.

### WASM SIMD

```rust
#[target_feature(enable = "simd128")]
unsafe fn cosine_wasm(a: &[f32], b: &[f32]) -> f32 {
    use std::arch::wasm32::*;

    // 128-bit SIMD = 4 × f32
    // PAS de FMA → mul + add separes
    // ~4x plus lent que AVX2
}
```

### Scalar fallback

```rust
fn cosine_scalar(a: &[f32], b: &[f32]) -> f32 {
    let mut dot = 0.0f32;
    let mut norm_a = 0.0f32;
    let mut norm_b = 0.0f32;

    for i in 0..a.len() {
        dot += a[i] * b[i];
        norm_a += a[i] * a[i];
        norm_b += b[i] * b[i];
    }

    dot / (norm_a.sqrt() * norm_b.sqrt())
}
```

---

## Optimisations supplementaires

### Pre-normalisation
Si on sait que les vecteurs sont normalises (||v|| = 1), le cosine = dot product.
→ Supprimer le calcul des normes = 33% plus rapide.

Config : `HoraConfig::pre_normalize_embeddings = true`

### Batch cosine
Pour vector_search(query, k) sur N vecteurs :
- Batch load : charger les vecteurs par blocs cache-friendly
- Prefetch : `_mm_prefetch` pour le prochain bloc
- Partial sort : maintenir un min-heap de taille k au lieu de trier tous les scores

### Alignement memoire
AVX2 fonctionne mieux avec des donnees alignees a 32 bytes.
`_mm256_load_ps` (aligne) vs `_mm256_loadu_ps` (non-aligne) : ~5% de difference.

Decision : aligner les vecteurs a 32 bytes dans le VectorData page.

---

## Benchmarks attendus

| Architecture | 384d cosine | 100K top-100 |
|-------------|------------|-------------|
| AVX2 + FMA | ~50ns | ~3ms |
| NEON | ~80ns | ~5ms |
| WASM SIMD | ~200ns | ~15ms |
| Scalar | ~400ns | ~30ms |

---

*Recherche effectuee le 2026-03-02.*
