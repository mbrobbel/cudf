# Rust Bindings for libcudf

## Build & Test

```sh
# All commands run from rust/ directory
pixi run cargo check --tests          # type-check everything
pixi run cargo clippy --all-targets --all-features -- -Dwarnings
pixi run cargo fmt --check
pixi run cargo test                   # requires GPU
```

All three crates (rmm-sys, cudf-sys, cudf) must pass all four commands.

## Workspace Layout

- **rmm-sys** — CXX FFI for RMM (device queries). Links `librmm` + `libcudart`.
- **cudf-sys** — CXX FFI for libcudf. Links `libcudf`. C++ wrappers live in `cudf_sys` namespace.
- **cudf** — Safe Rust API. `#![forbid(unsafe_code)]` — all unsafe lives in cudf-sys.

Future crates (cudf-io, cudf-derive) will be added as siblings.

## Linting

### cudf crate — strict
Deny: `clippy::pedantic`, `clippy::as_conversions`, `clippy::shadow_reuse`,
`clippy::shadow_same`, `clippy::shadow_unrelated`,
`clippy::undocumented_unsafe_blocks`, `clippy::unwrap_used`, `missing_docs`.

### cudf-sys / rmm-sys — relaxed where FFI demands it
Allow `clippy::as_conversions` and `clippy::unwrap_used` where the FFI boundary requires it.
Still deny `clippy::undocumented_unsafe_blocks` — every `unsafe` block needs a `// SAFETY:` comment.

### rustfmt
Default configuration. No `.rustfmt.toml`.

## API Design

### Type Safety
- **Columns are untyped.** Safe accessors return `Result`/`Option` with runtime type checks.
  Do NOT use generic `Column<T>`.
- **Nullability** is checked at runtime via `has_nulls()`, not at the type level.
- **Scalar** uses an enum approach.
- **Extension traits** must be sealed (sealed trait pattern).
- **Shared behavior** across types (Column, Table, Scalar) goes in traits.

### Builder Pattern & `GpuOp` Trait
All GPU operations use builders with `.call()` as the terminal method.
`.stream()` is an optional builder method (defaults to the default stream).
Even simple operations go through builders — consistency over brevity.

Every builder implements the `GpuOp` trait (`cudf::stream::GpuOp`):
```rust
pub trait GpuOp: Sized {
    type Output;
    fn stream(self, stream: Stream) -> Self;
    fn call(self) -> Result<Self::Output>;
}
```

Call sites must `use cudf::stream::GpuOp;` (no prelude re-export).
`Stream` implements `Default` (returns the default CUDA stream).

### Stream / GPU Sync
Within a single stream, operations are ordered — explicit sync is only needed for
cross-stream work or host readback. Type-level sync semantics are a future design topic.

### Naming
Idiomatic Rust names first: `len()` not `num_rows()`.
Add `#[doc(alias = "cpp_name")]` for the corresponding C++ name so it is searchable.

## CXX Bridge Organization (cudf-sys)

Split the `#[cxx::bridge]` into **multiple modules in separate files**, one per domain.
Do not keep a single monolithic bridge file.

### Module layout — follow the C++ namespaces

Mirror the libcudf header/namespace structure so that finding the FFI binding for a
C++ function is intuitive. The mapping:

| C++ location | cudf-sys file |
|---|---|
| `cudf/types.hpp`, `cudf/column/`, `cudf/table/`, `cudf/scalar/` | `core.rs` |
| `cudf::` top-level (`sorting.hpp`, `copying.hpp`, `unary.hpp`, …) | `sorting.rs`, `copying.rs`, `unary.rs`, … |
| `cudf::strings::` | `strings.rs` |
| `cudf::lists::` | `lists.rs` |
| `cudf::io::` | `io.rs` |
| `cudf::join::` | `join.rs` |
| `cudf::hashing::` | `hashing.rs` |
| `cudf::dictionary::` | `dictionary.rs` |
| `cudf::structs::` | `structs.rs` |

`core.rs` holds shared enums (`TypeId`, `Order`, `NullOrder`, …), opaque type
declarations (`Column`, `Table`, `Scalar`, `column_view`, `table_view`, `DataType`,
`Stream`), and fundamental accessors (constructors, field getters, scalar factories).

Closely related top-level headers can share a file when it makes sense
(e.g. `round.rs` could fold into `unary.rs`), but default to one file per header.

### C++ wrapper pattern

All C++ wrappers live in the `cudf_sys` namespace. Conventions:
- RAII via `UniquePtr` — C++ wrapper classes own the underlying libcudf object.
- `std::call_once` for lazy view initialization (e.g. caching a `column_view`).
- Wrapper functions are free functions (`column_size(const Column&)`) not methods,
  because CXX maps `extern "C++"` functions, not member functions.

### Conventions

- Domain bridges import shared types from core via `type Column = crate::core::ffi::Column;`.
- Each `src/{module}.rs` has a matching `cpp/{module}.cpp` (includes its generated `.rs.h`).
- Use `cxx_build::bridges([...])` (plural) in build.rs.
- New FFI functions go in the domain module they belong to, not core.
- New shared types needed by multiple domains go in `core.rs`.
- Keep each bridge file ~200–400 lines.

## Error Handling

- Hand-written `Error` enum. No thiserror.
- Mark `#[non_exhaustive]` for forward compatibility.
- Minimal context for now; operation/type context can be added later.

## Unsafe Code

- **cudf**: `#![forbid(unsafe_code)]`. Zero unsafe blocks.
- **cudf-sys**: `unsafe` is allowed but every block requires a `// SAFETY:` comment
  (`deny(clippy::undocumented_unsafe_blocks)`).
- **Send/Sync**: Implement in cudf-sys wrapper types so the cudf crate never needs `unsafe` for this.

## Safe API Helpers (cudf crate)

- `scalar_to_ffi` / `scalar_from_ffi` convert between the safe `Scalar` enum and
  `cudf_sys::Scalar`. Use these at the boundary, not raw FFI calls.
- `to_orders()` / `to_null_orders()` convert `&[i32]` slices to cudf enum vectors.
- Always implement standard Rust traits where they make sense (`Debug`, `Display`,
  `Clone`, `PartialEq`, `From`/`TryFrom`, `Index`, `IntoIterator`, etc.).
  These can be added on top of domain-specific traits and inherent methods.

## Module Organization

### cudf-sys
See [CXX Bridge Organization](#cxx-bridge-organization-cudf-sys) — follows C++ namespaces.

### cudf (safe crate)
- Follow idiomatic Rust grouping, not the C++ layout.
- File-named modules (`module_name.rs`), not `mod.rs`.
- Submodule imports only — `use cudf::column::Column`, NOT `use cudf::Column`.
  No root-level re-exports.

## Documentation

- `deny(missing_docs)` on all crates. Every public item gets a doc comment.
- **cudf-sys**: Always link to the corresponding C++ API.
- **cudf**: The Rust API should stand on its own; C++ references are not required.

## Testing

- Inline `#[cfg(test)] mod tests` for unit tests; `tests/` directory for integration tests.
- Tests use the **safe API only** — no reaching into `cudf_sys::ffi`.
  If something cannot be tested via the safe API, the safe API has a gap that should be filled.

## Dependencies

Minimal: only `cxx` + internal workspace deps. No thiserror, bitflags, or other utility crates.
Hand-write everything.

## Feature Flags

- Current: `vendored`.
- Planned: `serde`, `arrow-interop`, `io`.

## Git Workflow

Conventional commits: `feat:`, `fix:`, `refactor:`, `docs:`, `test:`, `chore:`, etc.

**Important:** The git repo root is also the home directory. Always use `git add` with
specific paths — never `git add -A`.

## Versioning

Version `0.0.0+26.04.00` — the build metadata tracks the libcudf release via the
root `VERSION` file. Bump the metadata when upgrading the underlying libcudf.

## SPDX Headers

Every file must have:
```
// SPDX-FileCopyrightText: Copyright (c) 2026, NVIDIA CORPORATION.
// SPDX-License-Identifier: Apache-2.0
```
(Use `#` comment syntax for TOML/Cargo files.)
