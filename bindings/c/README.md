# hora-graph-core C FFI

Bio-inspired embedded knowledge graph engine — C-compatible FFI via [cbindgen](https://github.com/mozilla/cbindgen).

## Build

```bash
cd bindings/c
cargo build --release
```

Produces:
- `libhora_graph_ffi.a` (static library)
- `libhora_graph_ffi.dylib` / `.so` / `.dll` (dynamic library)
- `hora_graph_core.h` (auto-generated header)

## Usage

```c
#include "hora_graph_core.h"

int main(void) {
    HoraCore *graph = hora_core_new();

    uint64_t alice = hora_core_add_entity(graph, "person");
    uint64_t bob = hora_core_add_entity(graph, "person");

    hora_core_add_edge(graph, alice, bob, "knows");

    hora_core_free(graph);
    return 0;
}
```

## Supported platforms

- Linux x86_64 / aarch64
- macOS x86_64 / arm64 (universal)
- Windows x86_64

## License

MIT OR Apache-2.0
